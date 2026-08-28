from __future__ import annotations

import os
from dataclasses import dataclass
from threading import RLock
from time import monotonic
from typing import Any, Callable
from urllib.parse import urlparse

from opentelemetry.trace import (
    Span,
    SpanKind,
    Status,
    StatusCode,
    Tracer,
    set_span_in_context,
)

from kaliok.observability.base import Observer
from kaliok.observability.events import ObservabilityEvent
from kaliok.observability.noop import NoOpObserver


_TERMINAL_EVENTS = {
    "ingestion.completed",
    "rag.answer.completed",
    "rag.index.completed",
}
_SENSITIVE_KEY_PARTS = {
    "answer",
    "authorization",
    "content",
    "cookie",
    "password",
    "prompt",
    "secret",
    "token",
}
_ATTRIBUTE_FIELDS = {
    "correlation_id": "kaliok.correlation_id",
    "execution_id": "kaliok.execution_id",
    "component": "kaliok.component",
    "implementation": "kaliok.implementation",
    "operation": "kaliok.operation",
    "document_id": "kaliok.document_id",
    "document_version_id": "kaliok.document_version_id",
    "processing_run_id": "kaliok.processing_run_id",
    "model": "kaliok.model",
    "duration_ms": "kaliok.duration_ms",
    "input_count": "kaliok.input_count",
    "output_count": "kaliok.output_count",
    "top_k": "kaliok.top_k",
    "success": "kaliok.success",
    "error_type": "error.type",
}


@dataclass
class _OpenExecution:
    span: Span
    opened_at: float


class OpenTelemetryObserver:
    def __init__(
        self,
        tracer: Tracer,
        *,
        tracer_provider: Any | None = None,
        allowed_extra_data_keys: frozenset[str] = frozenset(),
        max_open_executions: int = 1000,
        max_execution_age_seconds: float = 3600.0,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        if max_open_executions <= 0:
            raise ValueError("max_open_executions doit être strictement positif.")
        if max_execution_age_seconds <= 0:
            raise ValueError(
                "max_execution_age_seconds doit être strictement positif."
            )
        self._tracer = tracer
        self._tracer_provider = tracer_provider
        self._allowed_extra_data_keys = allowed_extra_data_keys
        self._max_open_executions = max_open_executions
        self._max_execution_age_seconds = max_execution_age_seconds
        self._clock = clock
        self._executions: dict[str, _OpenExecution] = {}
        self._lock = RLock()

    def emit(self, event: ObservabilityEvent) -> None:
        try:
            with self._lock:
                now = self._clock()
                self._cleanup_expired(now)
                if event.event_name.endswith(".started"):
                    self._start_execution(event, now)
                elif self._is_terminal(event):
                    self._finish_execution(event)
                else:
                    self._record_step(event)
        except Exception:
            return None

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        try:
            if self._tracer_provider is None:
                return True
            result = self._tracer_provider.force_flush(timeout_millis)
            return result is not False
        except Exception:
            return False

    def shutdown(self) -> None:
        try:
            with self._lock:
                for execution_id in list(self._executions):
                    self._abandon(execution_id, "observer_shutdown")
            if self._tracer_provider is not None:
                self._tracer_provider.shutdown()
        except Exception:
            return None

    def _start_execution(
        self,
        event: ObservabilityEvent,
        opened_at: float,
    ) -> None:
        execution_id = self._execution_key(event)
        if execution_id in self._executions:
            self._abandon(execution_id, "duplicate_started_event")
        while len(self._executions) >= self._max_open_executions:
            oldest = min(
                self._executions,
                key=lambda key: self._executions[key].opened_at,
            )
            self._abandon(oldest, "capacity_limit")

        span = self._tracer.start_span(
            event.event_name.removesuffix(".started"),
            kind=SpanKind.INTERNAL,
            start_time=self._timestamp_ns(event),
            attributes=self._attributes(event),
        )
        self._executions[execution_id] = _OpenExecution(span, opened_at)

    def _record_step(self, event: ObservabilityEvent) -> None:
        execution = self._executions.get(self._execution_key(event))
        context = (
            set_span_in_context(execution.span)
            if execution is not None
            else None
        )
        end_time = self._timestamp_ns(event)
        duration_ms = max(event.duration_ms or 0.0, 0.0)
        start_time = end_time - int(duration_ms * 1_000_000)
        span = self._tracer.start_span(
            event.event_name,
            context=context,
            kind=SpanKind.INTERNAL,
            start_time=start_time,
            attributes=self._attributes(event),
        )
        self._apply_status(span, event)
        span.end(end_time=end_time)

    def _finish_execution(self, event: ObservabilityEvent) -> None:
        execution_id = self._execution_key(event)
        execution = self._executions.pop(execution_id, None)
        if execution is None:
            self._record_step(event)
            return
        for key, value in self._attributes(event).items():
            execution.span.set_attribute(key, value)
        self._apply_status(execution.span, event)
        execution.span.end(end_time=self._timestamp_ns(event))

    def _cleanup_expired(self, now: float) -> None:
        expired = [
            execution_id
            for execution_id, execution in self._executions.items()
            if now - execution.opened_at >= self._max_execution_age_seconds
        ]
        for execution_id in expired:
            self._abandon(execution_id, "execution_timeout")

    def _abandon(self, execution_id: str, reason: str) -> None:
        execution = self._executions.pop(execution_id, None)
        if execution is None:
            return
        execution.span.set_attribute("kaliok.abandoned", True)
        execution.span.set_attribute("kaliok.abandon_reason", reason)
        execution.span.set_status(Status(StatusCode.ERROR, reason))
        execution.span.end()

    def _attributes(self, event: ObservabilityEvent) -> dict[str, Any]:
        attributes: dict[str, Any] = {"kaliok.event_name": event.event_name}
        for field_name, attribute_name in _ATTRIBUTE_FIELDS.items():
            value = getattr(event, field_name)
            if value is not None:
                attributes[attribute_name] = self._attribute_value(value)
        for key in self._allowed_extra_data_keys:
            if self._is_sensitive_key(key):
                continue
            value = event.extra_data.get(key)
            if self._is_safe_extra_value(value):
                attributes[f"kaliok.extra.{key}"] = value
        return attributes

    @staticmethod
    def _apply_status(span: Span, event: ObservabilityEvent) -> None:
        if event.success is True:
            span.set_status(Status(StatusCode.OK))
        elif event.success is False or event.event_name.endswith(".error"):
            description = event.error_type or "observed_error"
            span.set_status(Status(StatusCode.ERROR, description))

    @staticmethod
    def _execution_key(event: ObservabilityEvent) -> str:
        identifier = event.execution_id or event.correlation_id
        if identifier is None:
            return f"uncorrelated:{id(event)}"
        return str(identifier)

    @staticmethod
    def _is_terminal(event: ObservabilityEvent) -> bool:
        return event.event_name in _TERMINAL_EVENTS or event.event_name.endswith(
            ".error"
        )

    @staticmethod
    def _timestamp_ns(event: ObservabilityEvent) -> int:
        return int(event.timestamp.timestamp() * 1_000_000_000)

    @staticmethod
    def _attribute_value(value: Any) -> Any:
        if isinstance(value, (bool, float, int, str)):
            return value
        return str(value)

    @staticmethod
    def _is_safe_extra_value(value: Any) -> bool:
        if isinstance(value, (bool, float, int)):
            return True
        if not isinstance(value, str) or len(value) > 256:
            return False
        parsed = urlparse(value)
        if parsed.scheme and (
            parsed.query
            or parsed.username is not None
            or parsed.password is not None
        ):
            return False
        return True

    @staticmethod
    def _is_sensitive_key(key: str) -> bool:
        normalized = key.casefold()
        return any(part in normalized for part in _SENSITIVE_KEY_PARTS)


def create_opentelemetry_observer() -> Observer:
    if os.getenv("OTEL_SDK_DISABLED", "false").casefold() == "true":
        return NoOpObserver()
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT") or os.getenv(
        "OTEL_EXPORTER_OTLP_ENDPOINT"
    )
    protocol = os.getenv("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL") or os.getenv(
        "OTEL_EXPORTER_OTLP_PROTOCOL",
        "http/protobuf",
    )
    if not endpoint or protocol != "http/protobuf":
        return NoOpObserver()

    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.semconv.resource import ResourceAttributes

        service_name = os.getenv("OTEL_SERVICE_NAME", "kaliok")
        provider = TracerProvider(
            resource=Resource.create(
                {ResourceAttributes.SERVICE_NAME: service_name}
            )
        )
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        tracer = provider.get_tracer("kaliok.observability")
        return OpenTelemetryObserver(tracer, tracer_provider=provider)
    except Exception:
        return NoOpObserver()
