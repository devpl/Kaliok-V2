from __future__ import annotations

from typing import Sequence

from kaliok.rag.context.base import ContextBuilder
from kaliok.rag.embedding.base import Embedder
from kaliok.rag.extraction.base import Extractor
from kaliok.rag.fusion.base import FusionStrategy
from kaliok.rag.generation.base import Generator
from kaliok.rag.indexing.base import IndexStore
from kaliok.rag.representation.base import RepresentationBuilder
from kaliok.rag.reranking.base import Reranker
from kaliok.rag.retrieval.base import Retriever
from kaliok.rag.types import EmbeddingRecord, RagAnswer, RankedCandidate


class RagOrchestrator:
    def __init__(
        self,
        *,
        extractor: Extractor,
        representation_builder: RepresentationBuilder,
        embedder: Embedder,
        index_store: IndexStore,
        retriever: Retriever,
        context_builder: ContextBuilder,
        generator: Generator,
        fusion: FusionStrategy | None = None,
        reranker: Reranker | None = None,
        retrieval_top_k: int = 10,
    ) -> None:
        if retrieval_top_k <= 0:
            raise ValueError("retrieval_top_k doit être strictement positif.")
        self._extractor = extractor
        self._representation_builder = representation_builder
        self._embedder = embedder
        self._index_store = index_store
        self._retriever = retriever
        self._fusion = fusion
        self._reranker = reranker
        self._context_builder = context_builder
        self._generator = generator
        self._retrieval_top_k = retrieval_top_k

    def index(self, document: object) -> tuple[EmbeddingRecord, ...]:
        extracted = self._extractor.extract(document)
        units = tuple(self._representation_builder.build(extracted))
        records = tuple(self._embedder.embed_units(units))
        self._index_store.write(records)
        return records

    def answer(self, question: str) -> RagAnswer:
        query_embedding = self._embedder.embed_query(question)
        candidates = self._retriever.retrieve(
            query_embedding,
            top_k=self._retrieval_top_k,
        )
        ranked: Sequence[RankedCandidate] = tuple(
            RankedCandidate(
                candidate=candidate,
                rank=rank,
                score=candidate.score,
            )
            for rank, candidate in enumerate(candidates, start=1)
        )
        if self._fusion is not None:
            ranked = tuple(self._fusion.fuse(ranked))
        if self._reranker is not None:
            ranked = tuple(self._reranker.rerank(question, ranked))
        context = self._context_builder.build(question, ranked)
        return self._generator.generate(question, context)
