from __future__ import annotations

from typing import Protocol

from kaliok.rag.types import ExtractedDocument


class Extractor(Protocol):
    def extract(self, document: object) -> ExtractedDocument: ...
