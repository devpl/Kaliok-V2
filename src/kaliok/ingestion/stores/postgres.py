from __future__ import annotations

from uuid import UUID

from sqlmodel import Session, select

from kaliok.ingestion.types import (
    Identifier,
    IngestionRequest,
    IngestionResult,
    NormalizedDocument,
)
from kaliok.storage.models import Document, DocumentVersion, Source


class PostgresDocumentStore:
    """Persist normalized document identity and version metadata atomically."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def store(
        self,
        request: IngestionRequest,
        document: NormalizedDocument,
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
                page_count=document.page_count,
                document_type=document.document_type,
                document_subtype=document.document_subtype,
                version_status="active",
                processing_status="pending",
                readability_status="unknown",
                is_current=True,
            )
            self._session.add(version)
            self._session.flush()

            return self._result(version, document, status="created")

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
    ) -> IngestionResult:
        return IngestionResult(
            document_id=version.document_id,
            document_version_id=version.id,
            processing_run_id=None,
            detected_source=normalized.source,
            status=status,
        )
