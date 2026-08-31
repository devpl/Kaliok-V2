from __future__ import annotations

from sqlmodel import Session

from kaliok.ingestion.detection import (
    DeclaredMediaTypeDetector,
    SourceIngestorSelector,
)
from kaliok.ingestion.ingestors.txt import (
    PLAIN_TEXT_MEDIA_TYPE,
    PLAIN_TEXT_SOURCE_TYPE,
    TxtSourceIngestor,
)
from kaliok.ingestion.orchestrator import IngestionOrchestrator
from kaliok.ingestion.stores.postgres import PostgresDocumentStore
from kaliok.observability.base import Observer


def create_ingestion_orchestrator(
    session: Session,
    *,
    observer: Observer | None = None,
) -> IngestionOrchestrator:
    return IngestionOrchestrator(
        detector=DeclaredMediaTypeDetector(
            {
                PLAIN_TEXT_MEDIA_TYPE: PLAIN_TEXT_SOURCE_TYPE,
            }
        ),
        ingestor_selector=SourceIngestorSelector(
            [
                TxtSourceIngestor(),
            ]
        ),
        document_store=PostgresDocumentStore(session),
        observer=observer,
    )