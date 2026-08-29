from __future__ import annotations

from collections.abc import Iterable, Mapping

from kaliok.ingestion.base import SourceIngestor
from kaliok.ingestion.types import DetectedSource, SourceReference


class SourceDetectionError(ValueError):
    pass


class DeclaredMediaTypeDetector:
    """Resolve a source type from an explicitly declared media type."""

    def __init__(self, source_types_by_media_type: Mapping[str, str]) -> None:
        self._source_types_by_media_type = dict(source_types_by_media_type)

    def detect(self, source: SourceReference) -> DetectedSource:
        media_type = source.media_type
        if media_type is None or not media_type.strip():
            raise SourceDetectionError(
                "SourceReference.media_type est requis pour la détection déclarative."
            )
        source_type = self._source_types_by_media_type.get(media_type)
        if source_type is None:
            raise SourceDetectionError(
                f"Aucun type de source déclaré pour le media type {media_type!r}."
            )
        return DetectedSource(
            source=source,
            source_type=source_type,
            media_type=media_type,
            confidence=1.0,
        )


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
