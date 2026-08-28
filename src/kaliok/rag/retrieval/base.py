from __future__ import annotations

from typing import Protocol, Sequence

from kaliok.rag.types import Candidate


class Retriever(Protocol):
    def retrieve(
        self,
        query_embedding: Sequence[float],
        *,
        top_k: int,
    ) -> Sequence[Candidate]: ...
