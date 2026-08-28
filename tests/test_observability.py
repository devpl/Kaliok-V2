from __future__ import annotations

from datetime import timezone

import pytest

from kaliok.observability import (
    CompositeObserver,
    NoOpObserver,
    ObservabilityEvent,
    Observer,
    Timer,
)
from kaliok.rag import (
    Candidate,
    ContextBundle,
    EmbeddingRecord,
    ExtractedDocument,
    Provenance,
    RagAnswer,
    RagOrchestrator,
    RetrievalUnit,
)


class RecordingObserver:
    def __init__(self) -> None:
        self.events: list[ObservabilityEvent] = []

    def emit(self, event: ObservabilityEvent) -> None:
        self.events.append(event)


class FailingObserver:
    def emit(self, event: ObservabilityEvent) -> None:
        raise RuntimeError("backend indisponible")


class FakeContentProvider:
    def provide(self, document):
        return ExtractedDocument(
            content="contenu",
            provenance=Provenance(
                document_id="document-1",
                document_version_id="version-1",
                processing_run_id="run-1",
            ),
        )


class FakeRepresentationBuilder:
    def build(self, document):
        return [RetrievalUnit("unit-1", document.content, document.provenance)]


class FakeEmbedder:
    def embed_units(self, units):
        return [EmbeddingRecord(unit, (1.0, 0.0), "model-1") for unit in units]

    def embed_query(self, question):
        return (1.0, 0.0)


class FakeIndexStore:
    def write(self, records):
        return None


class FakeRetriever:
    def retrieve(self, query_embedding, *, top_k):
        unit = RetrievalUnit(
            "unit-1",
            "passage",
            Provenance(
                document_id="document-1",
                document_version_id="version-1",
                processing_run_id="run-1",
            ),
        )
        return [Candidate(unit, score=0.8)]


class FailingRetriever:
    def retrieve(self, query_embedding, *, top_k):
        raise ValueError("recherche impossible")


class FakeFusion:
    def fuse(self, candidates):
        return candidates


class FakeReranker:
    def rerank(self, question, candidates):
        return candidates


class FakeContextBuilder:
    def build(self, question, candidates):
        return ContextBundle(question, "contexte", tuple(candidates))


class FakeGenerator:
    def generate(self, question, context):
        return RagAnswer("réponse", context)


def make_orchestrator(
    observer=None,
    *,
    fusion=None,
    reranker=None,
    retriever=None,
):
    return RagOrchestrator(
        content_provider=FakeContentProvider(),
        representation_builder=FakeRepresentationBuilder(),
        embedder=FakeEmbedder(),
        index_store=FakeIndexStore(),
        retriever=retriever or FakeRetriever(),
        context_builder=FakeContextBuilder(),
        generator=FakeGenerator(),
        fusion=fusion,
        reranker=reranker,
        observer=observer,
        retrieval_top_k=5,
    )


def test_observability_public_api_and_noop_observer():
    event = ObservabilityEvent(event_name="test.event")

    NoOpObserver().emit(event)

    assert Observer is not None
    assert event.timestamp.tzinfo is timezone.utc
    assert event.document_id is None
    assert event.extra_data == {}


def test_composite_observer_fans_out_and_isolates_backend_failures():
    first = RecordingObserver()
    second = RecordingObserver()
    event = ObservabilityEvent(event_name="test.event")
    composite = CompositeObserver([first, FailingObserver(), second])

    composite.emit(event)

    assert first.events == [event]
    assert second.events == [event]


def test_timer_uses_perf_counter(monkeypatch):
    values = iter((10.0, 10.125))
    monkeypatch.setattr(
        "kaliok.observability.timing.perf_counter",
        lambda: next(values),
    )

    timer = Timer.start()

    assert timer.elapsed_ms() == pytest.approx(125.0)


def test_rag_index_event_order_and_shared_execution_identity():
    observer = RecordingObserver()

    records = make_orchestrator(observer).index("document")

    assert len(records) == 1
    assert [event.event_name for event in observer.events] == [
        "rag.index.started",
        "rag.source.completed",
        "rag.representation.completed",
        "rag.embedding.completed",
        "rag.indexing.completed",
        "rag.index.completed",
    ]
    execution_ids = {event.execution_id for event in observer.events}
    correlation_ids = {event.correlation_id for event in observer.events}
    assert len(execution_ids) == 1
    assert execution_ids == correlation_ids
    assert None not in execution_ids
    assert observer.events[-1].model == "model-1"
    assert observer.events[-1].document_version_id == "version-1"
    assert all(
        event.duration_ms is not None and event.duration_ms >= 0
        for event in observer.events[1:]
    )


def test_rag_answer_event_order_with_optional_steps():
    observer = RecordingObserver()

    answer = make_orchestrator(
        observer,
        fusion=FakeFusion(),
        reranker=FakeReranker(),
    ).answer("question")

    assert answer.text == "réponse"
    assert [event.event_name for event in observer.events] == [
        "rag.answer.started",
        "rag.retrieval.completed",
        "rag.fusion.completed",
        "rag.reranking.completed",
        "rag.context.completed",
        "rag.generation.completed",
        "rag.answer.completed",
    ]
    assert {event.execution_id for event in observer.events} == {
        observer.events[0].execution_id
    }
    assert all(
        event.correlation_id == event.execution_id for event in observer.events
    )


def test_rag_answer_omits_optional_step_events_when_not_executed():
    observer = RecordingObserver()

    make_orchestrator(observer).answer("question")

    names = [event.event_name for event in observer.events]
    assert "rag.fusion.completed" not in names
    assert "rag.reranking.completed" not in names
    assert names == [
        "rag.answer.started",
        "rag.retrieval.completed",
        "rag.context.completed",
        "rag.generation.completed",
        "rag.answer.completed",
    ]


def test_rag_emits_error_and_preserves_original_exception():
    observer = RecordingObserver()
    orchestrator = make_orchestrator(observer, retriever=FailingRetriever())

    with pytest.raises(ValueError, match="recherche impossible"):
        orchestrator.answer("question")

    error = observer.events[-1]
    assert [event.event_name for event in observer.events] == [
        "rag.answer.started",
        "rag.error",
    ]
    assert error.operation == "retrieval"
    assert error.success is False
    assert error.error_type == "ValueError"
    assert error.error_message == "recherche impossible"
    assert error.execution_id == observer.events[0].execution_id


def test_rag_functional_result_is_unchanged_with_noop_or_failing_observer():
    without_backend = make_orchestrator(NoOpObserver()).answer("question")
    failing_backend = make_orchestrator(FailingObserver()).answer("question")

    assert without_backend.text == "réponse"
    assert failing_backend.text == "réponse"
