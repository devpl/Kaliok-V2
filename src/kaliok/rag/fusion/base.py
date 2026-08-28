from __future__ import annotations

from typing import Protocol, Sequence

from kaliok.rag.types import RankedCandidate


class FusionStrategy(Protocol):
    def fuse(
        self,
        candidates: Sequence[RankedCandidate],
    ) -> Sequence[RankedCandidate]: ...
