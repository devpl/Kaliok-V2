from __future__ import annotations

from typing import Sequence
from uuid import uuid4

from kaliok.observability.base import Observer
from kaliok.observability.events import ObservabilityEvent
from kaliok.observability.noop import NoOpObserver
from kaliok.observability.timing import Timer
from kaliok.rag.context.base import ContextBuilder
from kaliok.rag.embedding.base import Embedder
from kaliok.rag.fusion.base import FusionStrategy
from kaliok.rag.generation.base import Generator
from kaliok.rag.indexing.base import IndexStore
from kaliok.rag.representation.base import RepresentationBuilder
from kaliok.rag.reranking.base import Reranker
from kaliok.rag.retrieval.base import Retriever
from kaliok.rag.source.base import ContentProvider
from kaliok.rag.types import EmbeddingRecord, RagAnswer, RankedCandidate


class RagOrchestrator:
    def __init__(
        self,
        *,
        content_provider: ContentProvider,
        representation_builder: RepresentationBuilder,
        embedder: Embedder,
        index_store: IndexStore,
        retriever: Retriever,
        context_builder: ContextBuilder,
        generator: Generator,
        fusion: FusionStrategy | None = None,
        reranker: Reranker | None = None,
        retrieval_top_k: int = 10,
        observer: Observer | None = None,
    ) -> None:
        if retrieval_top_k <= 0:
            raise ValueError("retrieval_top_k doit être strictement positif.")
        self._content_provider = content_provider
        self._representation_builder = representation_builder
        self._embedder = embedder
        self._index_store = index_store
        self._retriever = retriever
        self._fusion = fusion
        self._reranker = reranker
        self._context_builder = context_builder
        self._generator = generator
        self._retrieval_top_k = retrieval_top_k
        self._observer = observer if observer is not None else NoOpObserver()

    def index(self, document: object) -> tuple[EmbeddingRecord, ...]:
        execution_id = str(uuid4())
        total_timer = Timer.start()
        operation = "index"
        implementation = type(self).__name__
        identity: dict[str, object | None] = {}
        self._emit(
            "rag.index.started",
            execution_id,
            component="rag",
            implementation=implementation,
            operation=operation,
            input_count=1,
        )
        try:
            operation = "source"
            implementation = type(self._content_provider).__name__
            timer = Timer.start()
            extracted = self._content_provider.provide(document)
            identity = self._identity(extracted.provenance)
            self._emit(
                "rag.source.completed",
                execution_id,
                component="source",
                implementation=implementation,
                operation=operation,
                duration_ms=timer.elapsed_ms(),
                input_count=1,
                output_count=1,
                success=True,
                **identity,
            )

            operation = "representation"
            implementation = type(self._representation_builder).__name__
            timer = Timer.start()
            units = tuple(self._representation_builder.build(extracted))
            self._emit(
                "rag.representation.completed",
                execution_id,
                component="representation",
                implementation=implementation,
                operation=operation,
                duration_ms=timer.elapsed_ms(),
                input_count=1,
                output_count=len(units),
                success=True,
                **identity,
            )

            operation = "embedding"
            implementation = type(self._embedder).__name__
            timer = Timer.start()
            records = tuple(self._embedder.embed_units(units))
            model = self._single_model(records)
            self._emit(
                "rag.embedding.completed",
                execution_id,
                component="embedding",
                implementation=implementation,
                operation=operation,
                model=model,
                duration_ms=timer.elapsed_ms(),
                input_count=len(units),
                output_count=len(records),
                success=True,
                **identity,
            )

            operation = "indexing"
            implementation = type(self._index_store).__name__
            timer = Timer.start()
            self._index_store.write(records)
            self._emit(
                "rag.indexing.completed",
                execution_id,
                component="indexing",
                implementation=implementation,
                operation=operation,
                model=model,
                duration_ms=timer.elapsed_ms(),
                input_count=len(records),
                output_count=len(records),
                success=True,
                **identity,
            )
            self._emit(
                "rag.index.completed",
                execution_id,
                component="rag",
                implementation=type(self).__name__,
                operation="index",
                model=model,
                duration_ms=total_timer.elapsed_ms(),
                input_count=1,
                output_count=len(records),
                success=True,
                **identity,
            )
            return records
        except Exception as error:
            self._emit_error(
                execution_id,
                total_timer,
                operation,
                implementation,
                error,
                identity,
            )
            raise

    def answer(self, question: str) -> RagAnswer:
        execution_id = str(uuid4())
        total_timer = Timer.start()
        operation = "answer"
        implementation = type(self).__name__
        identity: dict[str, object | None] = {}
        self._emit(
            "rag.answer.started",
            execution_id,
            component="rag",
            implementation=implementation,
            operation=operation,
            input_count=1,
            top_k=self._retrieval_top_k,
        )
        try:
            operation = "query_embedding"
            implementation = type(self._embedder).__name__
            query_embedding = self._embedder.embed_query(question)

            operation = "retrieval"
            implementation = type(self._retriever).__name__
            timer = Timer.start()
            candidates = tuple(
                self._retriever.retrieve(
                    query_embedding,
                    top_k=self._retrieval_top_k,
                )
            )
            if candidates:
                identity = self._identity(candidates[0].unit.provenance)
            self._emit(
                "rag.retrieval.completed",
                execution_id,
                component="retrieval",
                implementation=implementation,
                operation=operation,
                duration_ms=timer.elapsed_ms(),
                input_count=1,
                output_count=len(candidates),
                top_k=self._retrieval_top_k,
                success=True,
                **identity,
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
                operation = "fusion"
                implementation = type(self._fusion).__name__
                timer = Timer.start()
                input_count = len(ranked)
                ranked = tuple(self._fusion.fuse(ranked))
                self._emit(
                    "rag.fusion.completed",
                    execution_id,
                    component="fusion",
                    implementation=implementation,
                    operation=operation,
                    duration_ms=timer.elapsed_ms(),
                    input_count=input_count,
                    output_count=len(ranked),
                    success=True,
                    **identity,
                )

            if self._reranker is not None:
                operation = "reranking"
                implementation = type(self._reranker).__name__
                timer = Timer.start()
                input_count = len(ranked)
                ranked = tuple(self._reranker.rerank(question, ranked))
                self._emit(
                    "rag.reranking.completed",
                    execution_id,
                    component="reranking",
                    implementation=implementation,
                    operation=operation,
                    duration_ms=timer.elapsed_ms(),
                    input_count=input_count,
                    output_count=len(ranked),
                    success=True,
                    **identity,
                )

            operation = "context"
            implementation = type(self._context_builder).__name__
            timer = Timer.start()
            context = self._context_builder.build(question, ranked)
            self._emit(
                "rag.context.completed",
                execution_id,
                component="context",
                implementation=implementation,
                operation=operation,
                duration_ms=timer.elapsed_ms(),
                input_count=len(ranked),
                output_count=1,
                success=True,
                **identity,
            )

            operation = "generation"
            implementation = type(self._generator).__name__
            timer = Timer.start()
            answer = self._generator.generate(question, context)
            self._emit(
                "rag.generation.completed",
                execution_id,
                component="generation",
                implementation=implementation,
                operation=operation,
                duration_ms=timer.elapsed_ms(),
                input_count=1,
                output_count=1,
                success=True,
                **identity,
            )
            self._emit(
                "rag.answer.completed",
                execution_id,
                component="rag",
                implementation=type(self).__name__,
                operation="answer",
                duration_ms=total_timer.elapsed_ms(),
                input_count=1,
                output_count=1,
                top_k=self._retrieval_top_k,
                success=True,
                **identity,
            )
            return answer
        except Exception as error:
            self._emit_error(
                execution_id,
                total_timer,
                operation,
                implementation,
                error,
                identity,
            )
            raise

    def _emit(
        self,
        event_name: str,
        execution_id: str,
        **values: object,
    ) -> None:
        try:
            self._observer.emit(
                ObservabilityEvent(
                    event_name=event_name,
                    execution_id=execution_id,
                    correlation_id=execution_id,
                    **values,
                )
            )
        except Exception:
            # Un backend d'observabilité ne doit pas modifier le résultat RAG.
            return None

    def _emit_error(
        self,
        execution_id: str,
        total_timer: Timer,
        operation: str,
        implementation: str,
        error: Exception,
        identity: dict[str, object | None],
    ) -> None:
        self._emit(
            "rag.error",
            execution_id,
            component="rag",
            implementation=implementation,
            operation=operation,
            duration_ms=total_timer.elapsed_ms(),
            success=False,
            error_type=type(error).__name__,
            error_message=str(error),
            **identity,
        )

    @staticmethod
    def _identity(provenance: object) -> dict[str, object | None]:
        return {
            "document_id": getattr(provenance, "document_id", None),
            "document_version_id": getattr(
                provenance,
                "document_version_id",
                None,
            ),
            "processing_run_id": getattr(
                provenance,
                "processing_run_id",
                None,
            ),
        }

    @staticmethod
    def _single_model(records: Sequence[EmbeddingRecord]) -> str | None:
        models = {record.model for record in records}
        if len(models) == 1:
            return next(iter(models))
        return None
