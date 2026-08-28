from __future__ import annotations

from typing import Protocol

from kaliok.rag.types import ContextBundle, RagAnswer


class Generator(Protocol):
    def generate(self, question: str, context: ContextBundle) -> RagAnswer: ...
