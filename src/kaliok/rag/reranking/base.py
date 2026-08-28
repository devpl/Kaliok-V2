from __future__ import annotations

from typing import Protocol, Sequence

from kaliok.rag.types import RankedCandidate


class Reranker(Protocol):
    def rerank(
        self,
        question: str,
        candidates: Sequence[RankedCandidate],
    ) -> Sequence[RankedCandidate]: ...
