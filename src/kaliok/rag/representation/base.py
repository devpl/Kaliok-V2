from __future__ import annotations

from typing import Protocol, Sequence

from kaliok.rag.types import ExtractedDocument, RetrievalUnit


class RepresentationBuilder(Protocol):
    def build(self, document: ExtractedDocument) -> Sequence[RetrievalUnit]: ...
