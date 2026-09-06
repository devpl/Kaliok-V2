from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from uuid import UUID

from sqlalchemy import func
from sqlmodel import Session, select

from kaliok.execution import ExecutionContext, apply_execution_context
from kaliok.hashing import canonical_json_hash
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
PERCEPTION_PROCESS_TYPE = "document_extraction"


@dataclass(frozen=True)
class ContentNormalizationResult:
    processing_run_id: UUID
    document_version_id: UUID
    unit_count: int


class ContentNormalizationService:
    """Create one normalized unit for each non-empty source content block."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def normalize(
        self,
        document_version_id: UUID,
        *,
        perception_processing_run_id: UUID | None = None,
        execution_context: ExecutionContext | None = None,
        pipeline_metadata: Mapping[str, object] | None = None,
    ) -> ContentNormalizationResult:
        version = self._session.get(DocumentVersion, document_version_id)
        if version is None:
            raise ValueError(f"DocumentVersion inconnue : {document_version_id}.")

        source_blocks: list[ContentBlock] | None = None
        configuration: dict[str, object] = {
            "normalization_strategy": "block-to-unit",
            "normalization_version": ENGINE_VERSION,
        }
        if perception_processing_run_id is not None:
            source_blocks = self._perception_blocks(
                version.id,
                perception_processing_run_id,
            )
            configuration["perception_processing_run_id"] = str(
                perception_processing_run_id
            )
        if execution_context is not None:
            configuration["execution_environment"] = execution_context.environment
        if pipeline_metadata is not None:
            pipeline_snapshot = dict(pipeline_metadata)
            try:
                canonical_json_hash(pipeline_snapshot)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "Les métadonnées pipeline doivent être sérialisables en JSON."
                ) from error
            configuration["pipeline"] = pipeline_snapshot

        run = ProcessingRun(
            document_version_id=version.id,
            process_type=PROCESS_TYPE,
            status="running",
            engine=ENGINE,
            engine_version=ENGINE_VERSION,
            configuration=configuration,
        )
        apply_execution_context(self._session, run, execution_context)
        if execution_context is None:
            run.configuration_hash = canonical_json_hash(run.configuration)
        self._session.add(run)
        self._session.flush()

        metrics = {
            "source_block_count": 0,
            "normalized_unit_count": 0,
            "skipped_empty_block_count": 0,
        }

        try:
            blocks = (
                source_blocks
                if source_blocks is not None
                else self._current_blocks(version.id)
            )
            skipped_empty_count = sum(not block.content.strip() for block in blocks)
            usable_blocks = [block for block in blocks if block.content.strip()]
            metrics = {
                "source_block_count": len(blocks),
                "normalized_unit_count": len(usable_blocks),
                "skipped_empty_block_count": skipped_empty_count,
            }
            if not usable_blocks:
                raise ValueError(
                    "Aucun ContentBlock exploitable dans la perception sélectionnée "
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

    def _perception_blocks(
        self,
        version_id: UUID,
        perception_processing_run_id: UUID,
    ) -> list[ContentBlock]:
        perception_run = self._session.get(
            ProcessingRun,
            perception_processing_run_id,
        )
        if perception_run is None:
            raise ValueError(
                "ProcessingRun de perception inconnu : "
                f"{perception_processing_run_id}."
            )
        if perception_run.process_type != PERCEPTION_PROCESS_TYPE:
            raise ValueError(
                "Le run fourni n'est pas un run de perception "
                f"{PERCEPTION_PROCESS_TYPE}."
            )
        if perception_run.status != "completed":
            raise ValueError("Le run de perception doit être completed.")
        if perception_run.document_version_id != version_id:
            raise ValueError(
                "Le run de perception appartient à une autre DocumentVersion."
            )

        reading_order = func.coalesce(
            ContentBlock.reading_order,
            ContentBlock.block_index,
        )
        blocks = list(
            self._session.exec(
                select(ContentBlock)
                .join(Page, ContentBlock.page_id == Page.id)
                .where(
                    Page.document_version_id == version_id,
                    ContentBlock.processing_run_id == perception_processing_run_id,
                )
                .order_by(
                    Page.page_number,
                    reading_order,
                    ContentBlock.block_index,
                    ContentBlock.id,
                )
            ).all()
        )
        if not blocks:
            raise ValueError(
                "Aucun ContentBlock associé au run de perception "
                f"{perception_processing_run_id}."
            )
        return blocks

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
