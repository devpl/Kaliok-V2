from __future__ import annotations

from uuid import UUID

from sqlmodel import Session, select

from kaliok.execution import ExecutionContext, apply_execution_context
from kaliok.ingestion.types import (
    Identifier,
    IngestionRequest,
    IngestionResult,
    NormalizedContentUnit,
    NormalizedDocument,
)
from kaliok.normalization import ContentNormalizationService
from kaliok.storage.models import (
    ContentBlock,
    ContentBlockFragment,
    Document,
    DocumentVersion,
    NormalizedContentUnit as StoredContentUnit,
    Page,
    ProcessingRun,
    Source,
    utc_now,
)


class NormalizedContentConflictError(RuntimeError):
    pass


class PostgresDocumentStore:
    """Persist normalized document identity and version metadata atomically."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def store(
        self,
        request: IngestionRequest,
        document: NormalizedDocument,
        *,
        execution_context: ExecutionContext | None = None,
    ) -> IngestionResult:
        self._validate(document)

        with self._session.begin_nested():
            source_id = self._resolve_source_id(request.source_id)
            stored_document = self._resolve_document(
                request,
                document,
                source_id,
            )
            existing_version = self._find_version(
                stored_document.id,
                document.content_hash,
            )
            if existing_version is not None:
                if self._uses_common_representation(document):
                    return self._result(
                        existing_version,
                        document,
                        status="already_exists",
                    )
                self._ensure_content(existing_version, document.units)
                return self._result(
                    existing_version,
                    document,
                    status="already_exists",
                )

            versions = self._versions(stored_document.id)
            for version in versions:
                if version.is_current:
                    version.is_current = False
                    self._session.add(version)

            version = DocumentVersion(
                document_id=stored_document.id,
                version_number=self._next_version_number(versions),
                filename=document.filename,
                mime_type=document.mime_type,
                file_hash=document.content_hash,
                file_size=document.file_size,
                storage_uri=document.storage_uri,
                page_count=(1 if self._uses_common_representation(document) else document.page_count),
                document_type=document.document_type,
                document_subtype=document.document_subtype,
                version_status="active",
                processing_status="pending",
                readability_status="unknown",
                is_current=True,
            )
            self._session.add(version)
            self._session.flush()
            normalization_run_id = None
            if self._uses_common_representation(document):
                normalization_run_id = self._store_common_content(
                    version,
                    document.units,
                    execution_context=execution_context,
                )
                version.processing_status = "completed"
                version.readability_status = "readable" if document.units else "unreadable"
                version.processed_at = utc_now()
                self._session.add(version)
            else:
                self._store_content(version.id, document.units)

            return self._result(
                version,
                document,
                status="created",
                processing_run_id=normalization_run_id,
            )

    def _store_common_content(
        self,
        version: DocumentVersion,
        units: tuple[NormalizedContentUnit, ...],
        *,
        execution_context: ExecutionContext | None = None,
    ) -> UUID:
        perception_run = ProcessingRun(
            document_version_id=version.id,
            process_type="document_extraction",
            status="completed",
            engine="kaliok-txt",
            engine_version="paragraphs-v1",
            metrics={"source_block_count": len(units)},
            completed_at=utc_now(),
        )
        apply_execution_context(self._session, perception_run, execution_context)
        self._session.add(perception_run)
        self._session.flush()
        page = Page(
            document_version_id=version.id,
            page_number=1,
            page_status="active",
            width=None,
            height=None,
            has_native_text=bool(units),
            native_text_length=sum(len(unit.content) for unit in units),
            readability_status="readable" if units else "unreadable",
            perception_mode="logical_text",
            ocr_required=False,
            ocr_performed=False,
            perception_processing_run_id=perception_run.id,
            extra_data={"logical_page": True, "source_format": "text/plain"},
        )
        self._session.add(page)
        self._session.flush()
        for unit in units:
            block = ContentBlock(
                page_id=page.id,
                processing_run_id=perception_run.id,
                block_index=unit.order,
                reading_order=unit.order,
                block_type="paragraph",
                content=unit.content,
                extraction_method="plain_text",
                extraction_engine="kaliok-txt",
                extraction_engine_version="paragraphs-v1",
            )
            self._session.add(block)
            self._session.flush()
            self._session.add(
                ContentBlockFragment(
                    content_block_id=block.id,
                    page_id=page.id,
                    fragment_index=0,
                    reading_order=unit.order,
                    content=unit.content,
                    coordinate_system=None,
                )
            )
        self._session.flush()
        return ContentNormalizationService(self._session).normalize(
            version.id,
            execution_context=execution_context,
        ).processing_run_id

    @staticmethod
    def _uses_common_representation(document: NormalizedDocument) -> bool:
        return document.source.source_type == "plain_text"

    def _resolve_source_id(self, source_id: Identifier | None) -> UUID | None:
        if source_id is None:
            return None
        resolved_id = self._uuid(source_id, field_name="source_id")
        source = self._session.get(Source, resolved_id)
        if source is None:
            raise ValueError(f"Source inconnue : {resolved_id}.")
        return source.id

    def _resolve_document(
        self,
        request: IngestionRequest,
        normalized: NormalizedDocument,
        source_id: UUID | None,
    ) -> Document:
        if request.document_id is not None:
            document_id = self._uuid(
                request.document_id,
                field_name="document_id",
            )
            stored = self._session.exec(
                select(Document)
                .where(Document.id == document_id)
                .with_for_update()
            ).one_or_none()
            if stored is None:
                raise ValueError(f"Document inconnu : {document_id}.")
            if source_id is not None and stored.source_id != source_id:
                raise ValueError(
                    "Le document demandé n'appartient pas à la source fournie."
                )
            return stored

        stored = Document(
            source_id=source_id,
            external_id=request.source.external_id,
            title=normalized.title or request.source.name,
            document_family=normalized.document_family,
            status="active",
            language=normalized.language,
        )
        self._session.add(stored)
        self._session.flush()
        return stored

    def _find_version(
        self,
        document_id: UUID,
        content_hash: str,
    ) -> DocumentVersion | None:
        return self._session.exec(
            select(DocumentVersion).where(
                DocumentVersion.document_id == document_id,
                DocumentVersion.file_hash == content_hash,
            )
        ).first()

    def _versions(self, document_id: UUID) -> list[DocumentVersion]:
        return list(
            self._session.exec(
                select(DocumentVersion).where(
                    DocumentVersion.document_id == document_id
                )
            ).all()
        )

    def _ensure_content(
        self,
        version: DocumentVersion,
        units: tuple[NormalizedContentUnit, ...],
    ) -> None:
        stored = self._stored_content(version.id)
        if not stored and units:
            self._store_content(version.id, units)
            return
        if not self._content_matches(stored, units):
            raise NormalizedContentConflictError(
                "La version existante possède un contenu normalisé différent."
            )

    def _stored_content(self, version_id: UUID) -> list[StoredContentUnit]:
        return list(
            self._session.exec(
                select(StoredContentUnit)
                .where(StoredContentUnit.document_version_id == version_id)
                .order_by(StoredContentUnit.unit_index)
            ).all()
        )

    def _store_content(
        self,
        version_id: UUID,
        units: tuple[NormalizedContentUnit, ...],
    ) -> None:
        stored_by_source_id: dict[str, StoredContentUnit] = {}
        pairs: list[tuple[StoredContentUnit, NormalizedContentUnit]] = []

        for unit in units:
            stored = StoredContentUnit(
                document_version_id=version_id,
                unit_index=unit.order,
                content_type=unit.content_type,
                content=unit.content,
                source_reference=unit.source_reference,
                source_unit_id=unit.source_unit_id,
            )
            self._session.add(stored)
            pairs.append((stored, unit))
            if unit.source_unit_id is not None:
                stored_by_source_id[unit.source_unit_id] = stored

        self._session.flush()

        for stored, unit in pairs:
            if unit.parent_source_unit_id is None:
                continue
            parent = stored_by_source_id[unit.parent_source_unit_id]
            stored.parent_unit_id = parent.id
            self._session.add(stored)

        self._session.flush()

    @staticmethod
    def _content_matches(
        stored: list[StoredContentUnit],
        units: tuple[NormalizedContentUnit, ...],
    ) -> bool:
        if len(stored) != len(units):
            return False
        source_id_by_stored_id = {
            unit.id: unit.source_unit_id for unit in stored
        }
        return all(
            persisted.unit_index == incoming.order
            and persisted.content_type == incoming.content_type
            and persisted.content == incoming.content
            and persisted.source_reference == incoming.source_reference
            and persisted.source_unit_id == incoming.source_unit_id
            and source_id_by_stored_id.get(persisted.parent_unit_id)
            == incoming.parent_source_unit_id
            for persisted, incoming in zip(stored, units)
        )

    @staticmethod
    def _next_version_number(versions: list[DocumentVersion]) -> int:
        return max((version.version_number for version in versions), default=0) + 1

    @staticmethod
    def _validate(document: NormalizedDocument) -> None:
        if not document.filename.strip():
            raise ValueError("NormalizedDocument.filename est requis.")
        if not document.storage_uri.strip():
            raise ValueError("NormalizedDocument.storage_uri est requis.")
        if not document.content_hash.strip():
            raise ValueError("NormalizedDocument.content_hash est requis.")
        if document.file_size is not None and document.file_size < 0:
            raise ValueError("NormalizedDocument.file_size ne peut pas être négatif.")
        if document.page_count is not None and document.page_count < 0:
            raise ValueError("NormalizedDocument.page_count ne peut pas être négatif.")
        orders = [unit.order for unit in document.units]
        if orders != list(range(len(document.units))):
            raise ValueError(
                "Les unités normalisées doivent avoir un ordre continu depuis 0."
            )
        source_unit_ids = [
            unit.source_unit_id
            for unit in document.units
            if unit.source_unit_id is not None
        ]
        if len(source_unit_ids) != len(set(source_unit_ids)):
            raise ValueError("Les source_unit_id doivent être uniques.")
        known_source_ids = set(source_unit_ids)
        for unit in document.units:
            if not unit.content_type.strip():
                raise ValueError("Le content_type d'une unité est requis.")
            if (
                unit.parent_source_unit_id is not None
                and unit.parent_source_unit_id not in known_source_ids
            ):
                raise ValueError(
                    "Chaque parent_source_unit_id doit référencer une unité connue."
                )
            if unit.parent_source_unit_id == unit.source_unit_id:
                raise ValueError("Une unité ne peut pas être son propre parent.")

    @staticmethod
    def _uuid(value: Identifier, *, field_name: str) -> UUID:
        if isinstance(value, UUID):
            return value
        try:
            return UUID(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{field_name} doit être un UUID valide.") from error

    @staticmethod
    def _result(
        version: DocumentVersion,
        normalized: NormalizedDocument,
        *,
        status: str,
        processing_run_id: UUID | None = None,
    ) -> IngestionResult:
        return IngestionResult(
            document_id=version.document_id,
            document_version_id=version.id,
            processing_run_id=processing_run_id,
            detected_source=normalized.source,
            status=status,
        )
