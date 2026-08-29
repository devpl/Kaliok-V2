from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from sqlmodel import Session, select

from kaliok.rag.types import ExtractedDocument, Provenance, RetrievalUnit
from kaliok.storage.models import (
    Document,
    DocumentVersion,
    NormalizedContentUnit,
)


@dataclass(frozen=True)
class NormalizedContentReference:
    document_id: UUID | None = None
    document_version_id: UUID | None = None

    def __post_init__(self) -> None:
        if self.document_id is None and self.document_version_id is None:
            raise ValueError("Un Document.id ou DocumentVersion.id est requis.")


@dataclass(frozen=True)
class NormalizedSourceUnit:
    id: UUID
    order: int
    content_type: str
    content: str
    source_unit_id: str | None
    parent_unit_id: UUID | None


class NormalizedContentProvider:
    """Read already normalized content; never reopen the original source."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def provide(self, reference: object) -> ExtractedDocument:
        if not isinstance(reference, NormalizedContentReference):
            raise TypeError("NormalizedContentReference attendu.")
        document, version = self._resolve(reference)
        stored_units = list(
            self._session.exec(
                select(NormalizedContentUnit)
                .where(
                    NormalizedContentUnit.document_version_id == version.id
                )
                .order_by(NormalizedContentUnit.unit_index)
            ).all()
        )
        if not stored_units:
            raise ValueError(
                f"La version {version.id} ne possède aucun contenu normalisé."
            )
        units = tuple(
            NormalizedSourceUnit(
                id=unit.id,
                order=unit.unit_index,
                content_type=unit.content_type,
                content=unit.content,
                source_unit_id=unit.source_unit_id,
                parent_unit_id=unit.parent_unit_id,
            )
            for unit in stored_units
        )
        return ExtractedDocument(
            content=units,
            provenance=Provenance(
                document_id=document.id,
                document_version_id=version.id,
                representation="normalized_content_units",
            ),
            metadata={
                "filename": version.filename,
                "version_number": version.version_number,
            },
        )

    def _resolve(
        self,
        reference: NormalizedContentReference,
    ) -> tuple[Document, DocumentVersion]:
        if reference.document_version_id is not None:
            version = self._session.get(
                DocumentVersion, reference.document_version_id
            )
            if version is None:
                raise ValueError(
                    f"DocumentVersion inconnue : {reference.document_version_id}."
                )
            if (
                reference.document_id is not None
                and version.document_id != reference.document_id
            ):
                raise ValueError("La version n'appartient pas au document demandé.")
        else:
            version = self._session.exec(
                select(DocumentVersion).where(
                    DocumentVersion.document_id == reference.document_id,
                    DocumentVersion.is_current.is_(True),
                )
            ).one_or_none()
            if version is None:
                raise ValueError(
                    "Le document ne possède pas une unique version courante."
                )
        document = self._session.get(Document, version.document_id)
        if document is None:
            raise ValueError(f"Document inconnu : {version.document_id}.")
        return document, version


class NormalizedContentRepresentationBuilder:
    """Use each ordered normalized unit as one retrieval unit."""

    def build(self, document: ExtractedDocument) -> tuple[RetrievalUnit, ...]:
        if not isinstance(document.content, tuple) or not all(
            isinstance(unit, NormalizedSourceUnit) for unit in document.content
        ):
            raise TypeError("Contenu normalisé PostgreSQL attendu.")
        return tuple(
            RetrievalUnit(
                unit_id=unit.id,
                text=unit.content,
                provenance=Provenance(
                    document_id=document.provenance.document_id,
                    document_version_id=(
                        document.provenance.document_version_id
                    ),
                    source_ids=(unit.id,),
                    representation="normalized_content_unit",
                    metadata={
                        "normalized_content_unit_id": unit.id,
                        "source_unit_id": unit.source_unit_id,
                        "unit_index": unit.order,
                        "content_type": unit.content_type,
                        "parent_unit_id": unit.parent_unit_id,
                    },
                ),
                metadata={
                    "source_unit_id": unit.source_unit_id,
                    "unit_index": unit.order,
                    "content_type": unit.content_type,
                },
            )
            for unit in document.content
        )
