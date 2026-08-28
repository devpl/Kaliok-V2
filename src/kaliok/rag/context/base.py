from __future__ import annotations

from typing import Protocol, Sequence

from kaliok.rag.types import ContextBundle, RankedCandidate


class ContextBuilder(Protocol):
    def build(
        self,
        question: str,
        candidates: Sequence[RankedCandidate],
    ) -> ContextBundle: ...
