from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import func
from sqlmodel import Session, select

from kaliok.execution import ExecutionContext, apply_execution_context
from kaliok.storage.models import (
    ContentBlock,
    DocumentVersion,
    NormalizedContentUnit,
    NormalizedContentUnitSource,
    Page,
    ProcessingRun,
    utc_now,
)


PROCESS_TYPE = "content_normalization"
ENGINE = "kaliok"
ENGINE_VERSION = "block-to-unit-v1"


@dataclass(frozen=True)
class ContentNormalizationResult:
    processing_run_id: UUID
    document_version_id: UUID
    unit_count: int


class ContentNormalizationService:
    """Create one normalized unit for each non-empty current content block."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def normalize(
        self,
        document_version_id: UUID,
        *,
        execution_context: ExecutionContext | None = None,
    ) -> ContentNormalizationResult:
        version = self._session.get(DocumentVersion, document_version_id)
        if version is None:
            raise ValueError(f"DocumentVersion inconnue : {document_version_id}.")

        run = ProcessingRun(
            document_version_id=version.id,
            process_type=PROCESS_TYPE,
            status="running",
            engine=ENGINE,
            engine_version=ENGINE_VERSION,
        )
        apply_execution_context(self._session, run, execution_context)
        self._session.add(run)
        self._session.flush()

        metrics = {
            "source_block_count": 0,
            "normalized_unit_count": 0,
            "skipped_empty_block_count": 0,
        }

        try:
            blocks = self._current_blocks(version.id)
            skipped_empty_count = sum(not block.content.strip() for block in blocks)
            usable_blocks = [block for block in blocks if block.content.strip()]
            metrics = {
                "source_block_count": len(blocks),
                "normalized_unit_count": len(usable_blocks),
                "skipped_empty_block_count": skipped_empty_count,
            }
            if not usable_blocks:
                raise ValueError(
                    "Aucun ContentBlock exploitable dans la perception courante "
                    f"de la DocumentVersion {version.id}."
                )

            with self._session.begin_nested():
                for unit_index, block in enumerate(usable_blocks):
                    self._persist_unit(
                        version_id=version.id,
                        run_id=run.id,
                        unit_index=unit_index,
                        block=block,
                    )
                self._session.flush()
        except Exception as error:
            run.status = "failed"
            run.completed_at = utc_now()
            run.metrics = metrics
            run.error_message = str(error)
            self._session.add(run)
            self._session.flush()
            raise

        run.status = "completed"
        run.completed_at = utc_now()
        run.metrics = metrics
        self._session.add(run)
        self._session.flush()

        return ContentNormalizationResult(
            processing_run_id=run.id,
            document_version_id=version.id,
            unit_count=len(usable_blocks),
        )

    def _current_blocks(self, version_id: UUID) -> list[ContentBlock]:
        reading_order = func.coalesce(
            ContentBlock.reading_order,
            ContentBlock.block_index,
        )
        return list(
            self._session.exec(
                select(ContentBlock)
                .join(Page, ContentBlock.page_id == Page.id)
                .where(
                    Page.document_version_id == version_id,
                    Page.perception_processing_run_id.is_not(None),
                    ContentBlock.processing_run_id
                    == Page.perception_processing_run_id,
                )
                .order_by(
                    Page.page_number,
                    reading_order,
                    ContentBlock.block_index,
                    ContentBlock.id,
                )
            ).all()
        )

    def _persist_unit(
        self,
        *,
        version_id: UUID,
        run_id: UUID,
        unit_index: int,
        block: ContentBlock,
    ) -> None:
        unit = NormalizedContentUnit(
            document_version_id=version_id,
            processing_run_id=run_id,
            unit_index=unit_index,
            content_type=block.block_type,
            content=block.content,
            source_reference=None,
            source_unit_id=str(block.id),
            parent_unit_id=None,
        )
        self._session.add(unit)
        self._session.flush()
        self._session.add(
            NormalizedContentUnitSource(
                normalized_content_unit_id=unit.id,
                content_block_id=block.id,
                source_order=0,
            )
        )
