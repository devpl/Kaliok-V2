from __future__ import annotations

from typing import Protocol

from kaliok.rag.types import ExtractedDocument


class ContentProvider(Protocol):
    """Provide normalized documentary content for an existing Kaliok reference."""

    def provide(self, reference: object) -> ExtractedDocument: ...
