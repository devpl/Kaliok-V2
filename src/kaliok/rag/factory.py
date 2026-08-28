from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from kaliok.rag.config import RagPipelineConfig
from kaliok.rag.context.base import ContextBuilder
from kaliok.rag.embedding.base import Embedder
from kaliok.rag.fusion.base import FusionStrategy
from kaliok.rag.generation.base import Generator
from kaliok.rag.indexing.base import IndexStore
from kaliok.rag.representation.base import RepresentationBuilder
from kaliok.rag.reranking.base import Reranker
from kaliok.rag.retrieval.base import Retriever
from kaliok.rag.source.base import ContentProvider


@dataclass(frozen=True)
class RagComponents:
    content_provider: ContentProvider
    representation_builder: RepresentationBuilder
    embedder: Embedder
    index_store: IndexStore
    retriever: Retriever
    context_builder: ContextBuilder
    generator: Generator
    fusion: FusionStrategy | None = None
    reranker: Reranker | None = None


class RagComponentFactory(Protocol):
    def create(self, config: RagPipelineConfig) -> RagComponents: ...
