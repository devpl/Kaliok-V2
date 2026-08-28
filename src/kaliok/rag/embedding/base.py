from __future__ import annotations

from typing import Protocol, Sequence

from kaliok.rag.types import EmbeddingRecord, RetrievalUnit


class Embedder(Protocol):
    def embed_units(
        self,
        units: Sequence[RetrievalUnit],
    ) -> Sequence[EmbeddingRecord]: ...

    def embed_query(self, question: str) -> Sequence[float]: ...
