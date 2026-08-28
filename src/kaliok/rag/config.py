from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RagPipelineConfig:
    content_provider: str
    representation: str
    embedder: str
    index_store: str
    retriever: str
    context_builder: str
    generator: str
    fusion: str | None = None
    reranker: str | None = None
    retrieval_top_k: int = 10

    def __post_init__(self) -> None:
        if self.retrieval_top_k <= 0:
            raise ValueError("retrieval_top_k doit être strictement positif.")
