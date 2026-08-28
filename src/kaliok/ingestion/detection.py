from __future__ import annotations

from collections.abc import Iterable

from kaliok.ingestion.base import SourceIngestor
from kaliok.ingestion.types import DetectedSource


class NoSourceIngestorError(LookupError):
    pass


class SourceIngestorSelector:
    def __init__(self, ingestors: Iterable[SourceIngestor]) -> None:
        self._ingestors = tuple(ingestors)

    def select(self, source: DetectedSource) -> SourceIngestor:
        for ingestor in self._ingestors:
            if ingestor.supports(source):
                return ingestor
        raise NoSourceIngestorError(
            f"Aucun ingestor ne prend en charge le type {source.source_type!r}."
        )
