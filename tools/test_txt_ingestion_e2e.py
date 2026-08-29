"""Manual end-to-end check of the TXT ingestion path against PostgreSQL."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Sequence
from uuid import UUID

from sqlmodel import Session, select

from kaliok.ingestion import (
    DeclaredMediaTypeDetector,
    IngestionOrchestrator,
    IngestionRequest,
    NormalizedDocument,
    SourceIngestorSelector,
    SourceReference,
)
from kaliok.ingestion.ingestors.txt import (
    PLAIN_TEXT_MEDIA_TYPE,
    PLAIN_TEXT_SOURCE_TYPE,
    TxtSourceIngestor,
)
from kaliok.ingestion.stores.postgres import PostgresDocumentStore
from kaliok.observability import ObservabilityEvent
from kaliok.storage.database import create_database_engine
from kaliok.storage.models import (
    Document,
    DocumentVersion,
    NormalizedContentUnit,
)


class ConsoleObserver:
    """Display the events already emitted by IngestionOrchestrator."""

    def emit(self, event: ObservabilityEvent) -> None:
        duration = (
            "-"
            if event.duration_ms is None
            else f"{event.duration_ms:.2f} ms"
        )
        print(
            f"[event] {event.event_name} | execution={event.execution_id} "
            f"| correlation={event.correlation_id} | duration={duration} "
            f"| success={event.success}"
        )


class CapturingDocumentStore:
    """Transparent diagnostic wrapper around the real PostgreSQL store."""

    def __init__(self, store: PostgresDocumentStore) -> None:
        self._store = store
        self.normalized_document: NormalizedDocument | None = None

    def store(self, request, document):
        self.normalized_document = document
        return self._store.store(request, document)


@dataclass(frozen=True)
class PreviousState:
    version_ids: frozenset[UUID]
    current_version_id: UUID | None
    maximum_version_number: int


def _previous_state(session: Session, document_id: UUID | None) -> PreviousState:
    if document_id is None:
        return PreviousState(frozenset(), None, 0)
    document = session.get(Document, document_id)
    if document is None:
        raise ValueError(f"Document inconnu : {document_id}.")
    versions = list(
        session.exec(
            select(DocumentVersion).where(
                DocumentVersion.document_id == document_id
            )
        ).all()
    )
    current = [version.id for version in versions if version.is_current]
    if len(current) > 1:
        raise RuntimeError("Le document possède plusieurs versions courantes.")
    return PreviousState(
        version_ids=frozenset(version.id for version in versions),
        current_version_id=current[0] if current else None,
        maximum_version_number=max(
            (version.version_number for version in versions), default=0
        ),
    )


def _load_persisted(
    session: Session,
    document_id: UUID,
    document_version_id: UUID,
) -> tuple[Document, DocumentVersion, list[NormalizedContentUnit], list[DocumentVersion]]:
    document = session.get(Document, document_id)
    version = session.get(DocumentVersion, document_version_id)
    if document is None or version is None:
        raise RuntimeError("Document ou DocumentVersion introuvable après commit.")
    units = list(
        session.exec(
            select(NormalizedContentUnit)
            .where(
                NormalizedContentUnit.document_version_id
                == document_version_id
            )
            .order_by(NormalizedContentUnit.unit_index)
        ).all()
    )
    versions = list(
        session.exec(
            select(DocumentVersion)
            .where(DocumentVersion.document_id == document_id)
            .order_by(DocumentVersion.version_number)
        ).all()
    )
    return document, version, units, versions


def _validate_units(
    stored: list[NormalizedContentUnit],
    normalized: NormalizedDocument,
) -> None:
    expected = normalized.units
    if len(stored) != len(expected):
        raise AssertionError(
            f"Unités persistées : {len(stored)} ; attendues : {len(expected)}."
        )
    source_id_by_id = {unit.id: unit.source_unit_id for unit in stored}
    for persisted, incoming in zip(stored, expected):
        actual_parent_source_id = source_id_by_id.get(persisted.parent_unit_id)
        if (
            persisted.unit_index != incoming.order
            or persisted.content_type != incoming.content_type
            or persisted.source_unit_id != incoming.source_unit_id
            or actual_parent_source_id != incoming.parent_source_unit_id
            or persisted.content != incoming.content
        ):
            raise AssertionError(
                f"L'unité persistée d'ordre {persisted.unit_index} "
                "ne correspond pas à l'unité normalisée."
            )


def _validate_scenario(
    requested_document_id: UUID | None,
    previous: PreviousState,
    version: DocumentVersion,
    versions: list[DocumentVersion],
) -> str:
    current = [item for item in versions if item.is_current]
    if len(current) != 1 or current[0].id != version.id:
        raise AssertionError("La version retournée n'est pas l'unique version courante.")
    if requested_document_id is None:
        if version.version_number != 1:
            raise AssertionError("Une première ingestion doit créer la version 1.")
        return "A - première ingestion"
    if version.id in previous.version_ids:
        if {item.id for item in versions} != set(previous.version_ids):
            raise AssertionError("La réingestion identique a créé une version.")
        return "B - contenu identique, version réutilisée"
    if len(versions) != len(previous.version_ids) + 1:
        raise AssertionError("La modification doit créer exactement une version.")
    if version.version_number != previous.maximum_version_number + 1:
        raise AssertionError("Le numéro de version n'a pas été incrémenté.")
    if previous.current_version_id is not None:
        old_current = next(
            item for item in versions if item.id == previous.current_version_id
        )
        if old_current.is_current:
            raise AssertionError("L'ancienne version est encore courante.")
    return "C - contenu modifié, nouvelle version"


def _display(
    scenario: str,
    document: Document,
    version: DocumentVersion,
    units: list[NormalizedContentUnit],
    total_seconds: float,
) -> None:
    print("\nValidation PostgreSQL")
    print(f"Scénario                  : {scenario}")
    print(f"Document.id               : {document.id}")
    print(f"Document.source_id        : {document.source_id}")
    print(f"DocumentVersion.id        : {version.id}")
    print(f"Numéro de version         : {version.version_number}")
    print(f"is_current                : {version.is_current}")
    print(f"file_hash                 : {version.file_hash}")
    print(f"file_size                 : {version.file_size}")
    print(f"mime_type                 : {version.mime_type}")
    print(f"processing_status         : {version.processing_status}")
    print(f"NormalizedContentUnit     : {len(units)}")
    for unit in units:
        print("\n--- unité normalisée ---")
        print(f"ordre                     : {unit.unit_index}")
        print(f"content_type              : {unit.content_type}")
        print(f"source_unit_id            : {unit.source_unit_id}")
        print(f"parent_unit_id            : {unit.parent_unit_id}")
        print("contenu :")
        print(unit.content)
    print(f"\nDurée totale ingestion    : {total_seconds:.3f} s")


def run(
    txt_path: Path,
    *,
    document_id: UUID | None = None,
    source_id: UUID | None = None,
    external_id: str | None = None,
) -> None:
    path = txt_path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Fichier TXT introuvable : {path}.")

    source = SourceReference(
        name=path.name,
        uri=path.as_uri(),
        media_type=PLAIN_TEXT_MEDIA_TYPE,
        size=path.stat().st_size,
        external_id=external_id,
    )
    request = IngestionRequest(
        source=source,
        source_id=source_id,
        document_id=document_id,
    )
    engine = create_database_engine()
    try:
        with Session(engine) as session:
            previous = _previous_state(session, document_id)
            capturing_store = CapturingDocumentStore(
                PostgresDocumentStore(session)
            )
            orchestrator = IngestionOrchestrator(
                detector=DeclaredMediaTypeDetector(
                    {PLAIN_TEXT_MEDIA_TYPE: PLAIN_TEXT_SOURCE_TYPE}
                ),
                ingestor_selector=SourceIngestorSelector(
                    [TxtSourceIngestor()]
                ),
                document_store=capturing_store,
                observer=ConsoleObserver(),
            )
            started = perf_counter()
            try:
                result = orchestrator.ingest(request)
                session.commit()
            except Exception:
                session.rollback()
                raise
            elapsed = perf_counter() - started
            normalized = capturing_store.normalized_document
            if normalized is None:
                raise RuntimeError("Aucun NormalizedDocument n'a été produit.")

        result_document_id = UUID(str(result.document_id))
        result_version_id = UUID(str(result.document_version_id))
        with Session(engine) as verification_session:
            document, version, units, versions = _load_persisted(
                verification_session,
                result_document_id,
                result_version_id,
            )
            _validate_units(units, normalized)
            scenario = _validate_scenario(
                document_id,
                previous,
                version,
                versions,
            )
            _display(scenario, document, version, units, elapsed)
            print(f"Statut ingestion           : {result.status}")
            print("Validation                 : OK")
    finally:
        engine.dispose()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Valide manuellement l'ingestion TXT réelle sur PostgreSQL."
    )
    parser.add_argument("txt_path", type=Path, help="Chemin du fichier TXT local.")
    parser.add_argument(
        "--document-id",
        type=UUID,
        help="Document existant à réingérer pour les scénarios B et C.",
    )
    parser.add_argument(
        "--source-id",
        type=UUID,
        help="Source PostgreSQL existante à rattacher au document.",
    )
    parser.add_argument(
        "--external-id",
        help="Identifiant externe à conserver lors de la création du document.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        run(
            args.txt_path,
            document_id=args.document_id,
            source_id=args.source_id,
            external_id=args.external_id,
        )
    except Exception as error:
        print(f"ERREUR : {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
