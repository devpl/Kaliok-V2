from __future__ import annotations

from typing import Protocol, Sequence

from kaliok.rag.types import EmbeddingRecord


class IndexStore(Protocol):
    def write(self, records: Sequence[EmbeddingRecord]) -> None: ...
