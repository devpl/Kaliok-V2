from __future__ import annotations

from datetime import datetime, timedelta, timezone

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    SpanExportResult,
    SpanExporter,
)
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import StatusCode

from kaliok.observability import (
    NoOpObserver,
    ObservabilityEvent,
    OpenTelemetryObserver,
    Observer,
    create_opentelemetry_observer,
)


BASE_TIME = datetime(2026, 8, 28, 12, 0, tzinfo=timezone.utc)


def make_observer(**kwargs):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    observer = OpenTelemetryObserver(
        provider.get_tracer("kaliok.tests"),
        tracer_provider=provider,
        **kwargs,
    )
    return observer, exporter


def event(name, execution_id, *, offset=0, **kwargs):
    return ObservabilityEvent(
        event_name=name,
        timestamp=BASE_TIME + timedelta(seconds=offset),
        execution_id=execution_id,
        correlation_id=execution_id,
        **kwargs,
    )


def test_implements_observer_contract_and_correlates_rag_trace():
    observer, exporter = make_observer()

    observer.emit(
        event(
            "rag.answer.started",
            "rag-1",
            component="rag",
            operation="answer",
            top_k=5,
        )
    )
    observer.emit(
        event(
            "rag.retrieval.completed",
            "rag-1",
            offset=1,
            component="retrieval",
            operation="retrieval",
            implementation="VectorRetriever",
            document_id="document-1",
            document_version_id="version-1",
            model="bge-m3",
            duration_ms=125.0,
            input_count=1,
            output_count=5,
            top_k=5,
            success=True,
        )
    )
    observer.emit(
        event(
            "rag.answer.completed",
            "rag-1",
            offset=2,
            duration_ms=2000.0,
            success=True,
        )
    )

    spans = exporter.get_finished_spans()
    root = next(span for span in spans if span.name == "rag.answer")
    retrieval = next(
        span for span in spans if span.name == "rag.retrieval.completed"
    )
    assert Observer is not None
    assert callable(observer.emit)
    assert retrieval.context.trace_id == root.context.trace_id
    assert retrieval.parent.span_id == root.context.span_id
    assert retrieval.end_time - retrieval.start_time == 125_000_000
    assert retrieval.status.status_code is StatusCode.OK
    assert retrieval.attributes["kaliok.execution_id"] == "rag-1"
    assert retrieval.attributes["kaliok.document_id"] == "document-1"
    assert retrieval.attributes["kaliok.document_version_id"] == "version-1"
    assert retrieval.attributes["kaliok.model"] == "bge-m3"
    assert retrieval.attributes["kaliok.output_count"] == 5
    assert root.status.status_code is StatusCode.OK


def test_correlates_ingestion_and_intermediate_steps():
    observer, exporter = make_observer()

    observer.emit(event("ingestion.started", "ingestion-1"))
    for index, name in enumerate(
        (
            "ingestion.detection.completed",
            "ingestion.processing.completed",
            "ingestion.storage.completed",
        ),
        start=1,
    ):
        observer.emit(
            event(
                name,
                "ingestion-1",
                offset=index,
                duration_ms=10.0 * index,
                success=True,
            )
        )
    observer.emit(
        event(
            "ingestion.completed",
            "ingestion-1",
            offset=4,
            document_id="document-2",
            document_version_id="version-2",
            processing_run_id="run-2",
            duration_ms=4000.0,
            success=True,
        )
    )

    spans = exporter.get_finished_spans()
    root = next(span for span in spans if span.name == "ingestion")
    children = [span for span in spans if span.name != "ingestion"]
    assert len(children) == 3
    assert all(child.context.trace_id == root.context.trace_id for child in children)
    assert all(child.parent.span_id == root.context.span_id for child in children)
    assert root.attributes["kaliok.document_id"] == "document-2"
    assert root.attributes["kaliok.processing_run_id"] == "run-2"


def test_error_closes_root_without_exporting_raw_error_message():
    observer, exporter = make_observer(
        allowed_extra_data_keys=frozenset(
            {
                "safe_counter",
                "prompt",
                "api_token",
                "document_content",
                "callback_url",
            }
        )
    )
    observer.emit(event("rag.index.started", "error-1"))
    observer.emit(
        event(
            "rag.error",
            "error-1",
            offset=1,
            success=False,
            error_type="ValueError",
            error_message="secret-personnel-ne-pas-exporter",
            extra_data={
                "safe_counter": 3,
                "prompt": "question confidentielle",
                "api_token": "token-secret",
                "document_content": "contenu confidentiel",
                "callback_url": "https://example.invalid/callback?token=secret",
            },
        )
    )

    root = exporter.get_finished_spans()[0]
    exported = repr(dict(root.attributes))
    assert root.status.status_code is StatusCode.ERROR
    assert root.status.description == "ValueError"
    assert root.attributes["error.type"] == "ValueError"
    assert root.attributes["kaliok.extra.safe_counter"] == 3
    assert "secret-personnel-ne-pas-exporter" not in exported
    assert "question confidentielle" not in exported
    assert "token-secret" not in exported
    assert "contenu confidentiel" not in exported
    assert "example.invalid" not in exported


def test_concurrent_executions_keep_independent_trace_ids():
    observer, exporter = make_observer()
    observer.emit(event("rag.answer.started", "execution-a"))
    observer.emit(event("ingestion.started", "execution-b"))
    observer.emit(
        event(
            "rag.answer.completed",
            "execution-a",
            offset=1,
            success=True,
        )
    )
    observer.emit(
        event(
            "ingestion.completed",
            "execution-b",
            offset=1,
            success=True,
        )
    )

    spans = exporter.get_finished_spans()
    assert len(spans) == 2
    assert spans[0].context.trace_id != spans[1].context.trace_id


def test_capacity_limit_closes_abandoned_execution():
    observer, exporter = make_observer(max_open_executions=1)

    observer.emit(event("rag.answer.started", "first"))
    observer.emit(event("ingestion.started", "second"))

    abandoned = exporter.get_finished_spans()[0]
    assert abandoned.attributes["kaliok.abandoned"] is True
    assert abandoned.attributes["kaliok.abandon_reason"] == "capacity_limit"
    assert abandoned.status.status_code is StatusCode.ERROR
    observer.shutdown()


def test_execution_timeout_closes_expired_root_on_next_emit():
    clock_values = iter((0.0, 11.0))
    observer, exporter = make_observer(
        max_execution_age_seconds=10.0,
        clock=lambda: next(clock_values),
    )
    observer.emit(event("rag.answer.started", "expired"))

    observer.emit(event("ingestion.started", "next-execution"))

    expired = next(
        span for span in exporter.get_finished_spans() if span.name == "rag.answer"
    )
    assert expired.attributes["kaliok.abandoned"] is True
    assert expired.attributes["kaliok.abandon_reason"] == "execution_timeout"
    assert expired.status.status_code is StatusCode.ERROR
    observer.shutdown()


def test_shutdown_closes_open_execution_as_abandoned():
    observer, exporter = make_observer()
    observer.emit(event("ingestion.started", "open-on-shutdown"))

    observer.shutdown()

    span = exporter.get_finished_spans()[0]
    assert span.name == "ingestion"
    assert span.attributes["kaliok.abandoned"] is True
    assert span.attributes["kaliok.abandon_reason"] == "observer_shutdown"
    assert span.status.status_code is StatusCode.ERROR


class RaisingExporter(SpanExporter):
    def export(self, spans):
        raise RuntimeError("export impossible")

    def shutdown(self):
        return None


def test_exporter_failure_never_escapes_emit():
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(RaisingExporter()))
    observer = OpenTelemetryObserver(provider.get_tracer("failure-test"))

    observer.emit(event("rag.answer.started", "failure"))
    observer.emit(
        event(
            "rag.answer.completed",
            "failure",
            offset=1,
            success=True,
        )
    )


def test_factory_is_noop_when_disabled_or_unconfigured(monkeypatch):
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", raising=False)
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
    assert isinstance(create_opentelemetry_observer(), NoOpObserver)

    monkeypatch.setenv("OTEL_SDK_DISABLED", "false")
    assert isinstance(create_opentelemetry_observer(), NoOpObserver)


def test_factory_returns_opentelemetry_observer_when_configured(monkeypatch):
    monkeypatch.setenv("OTEL_SDK_DISABLED", "false")
    monkeypatch.setenv(
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "http://collector.invalid:4318",
    )
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf")
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL", raising=False)

    observer = create_opentelemetry_observer()

    assert isinstance(observer, OpenTelemetryObserver)
    observer.shutdown()


def test_factory_returns_noop_for_unsupported_protocol(monkeypatch):
    monkeypatch.setenv("OTEL_SDK_DISABLED", "false")
    monkeypatch.setenv(
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "http://collector.invalid:4317",
    )
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_PROTOCOL", "grpc")
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL", raising=False)

    assert isinstance(create_opentelemetry_observer(), NoOpObserver)
