from __future__ import annotations

from typing import Protocol

from kaliok.ingestion.types import (
    DetectedSource,
    IngestionRequest,
    IngestionResult,
    NormalizedDocument,
    SourceReference,
)


class SourceDetector(Protocol):
    def detect(self, source: SourceReference) -> DetectedSource: ...


class SourceIngestor(Protocol):
    def supports(self, source: DetectedSource) -> bool: ...

    def ingest(
        self,
        request: IngestionRequest,
        source: DetectedSource,
    ) -> NormalizedDocument: ...


class DocumentStore(Protocol):
    def store(
        self,
        request: IngestionRequest,
        document: NormalizedDocument,
    ) -> IngestionResult: ...
