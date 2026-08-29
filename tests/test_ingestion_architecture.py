from __future__ import annotations

from pathlib import Path

import pytest

from kaliok.ingestion import (
    DeclaredMediaTypeDetector,
    DetectedSource,
    DocumentStore,
    IngestionOrchestrator,
    IngestionRequest,
    IngestionResult,
    NoSourceIngestorError,
    NormalizedContentUnit,
    NormalizedDocument,
    SourceDetector,
    SourceDetectionError,
    SourceIngestor,
    SourceIngestorSelector,
    SourceReference,
)
from kaliok.observability import NoOpObserver, ObservabilityEvent


class FakeDetector:
    def __init__(self, calls, source_type="generic"):
        self.calls = calls
        self.source_type = source_type

    def detect(self, source):
        self.calls.append(("detect", source))
        return DetectedSource(
            source=source,
            source_type=self.source_type,
            media_type=source.media_type,
            confidence=0.9,
        )


class FakeIngestor:
    def __init__(self, calls, supported_type="generic"):
        self.calls = calls
        self.supported_type = supported_type

    def supports(self, source):
        self.calls.append(("supports", self.supported_type, source.source_type))
        return source.source_type == self.supported_type

    def ingest(self, request, source):
        self.calls.append(("ingest", request, source))
        return NormalizedDocument(
            source=source,
            units=(
                NormalizedContentUnit(
                    order=0,
                    content_type="text",
                    content="contenu normalisé",
                ),
            ),
            filename=request.source.name,
            storage_uri=request.source.uri or "memory://normalized",
            content_hash="normalized-hash",
            file_size=request.source.size,
            mime_type=source.media_type,
            title=request.source.name,
        )


class FailingIngestor(FakeIngestor):
    def ingest(self, request, source):
        raise RuntimeError("normalisation impossible")


class FakeStore:
    def __init__(self, calls):
        self.calls = calls

    def store(self, request, document):
        self.calls.append(("store", request, document))
        return IngestionResult(
            document_id="document-1",
            document_version_id="version-1",
            processing_run_id="run-1",
            detected_source=document.source,
            status="completed",
        )


class RecordingObserver:
    def __init__(self):
        self.events: list[ObservabilityEvent] = []

    def emit(self, event):
        self.events.append(event)


def make_request():
    parent = SourceReference(
        name="message source",
        uri="source://messages/42",
        media_type="message/example",
        external_id="message-42",
    )
    return IngestionRequest(
        source=SourceReference(
            name="élément joint",
            uri="source://messages/42/items/1",
            media_type="application/example",
            size=1234,
            external_id="item-1",
            parent=parent,
        ),
        request_id="request-1",
    )


def make_orchestrator(calls, *, observer=None, ingestors=None):
    return IngestionOrchestrator(
        detector=FakeDetector(calls),
        ingestor_selector=SourceIngestorSelector(
            ingestors or [FakeIngestor(calls)]
        ),
        document_store=FakeStore(calls),
        observer=observer,
    )


def test_public_protocols_and_types_are_available():
    assert all(
        contract is not None
        for contract in (SourceDetector, SourceIngestor, DocumentStore)
    )
    request = make_request()
    assert request.source.parent is not None
    assert request.source.parent.external_id == "message-42"
    assert request.source.size == 1234


def test_selector_chooses_first_compatible_ingestor():
    calls = []
    first = FakeIngestor(calls, supported_type="other")
    second = FakeIngestor(calls, supported_type="generic")
    source = FakeDetector(calls).detect(make_request().source)

    selected = SourceIngestorSelector([first, second]).select(source)

    assert selected is second


def test_declared_media_type_detector_uses_injected_mapping():
    source = SourceReference(name="notes", media_type="text/example")

    detected = DeclaredMediaTypeDetector(
        {"text/example": "declared-text"}
    ).detect(source)

    assert detected.source is source
    assert detected.source_type == "declared-text"
    assert detected.media_type == "text/example"
    assert detected.confidence == 1.0


def test_declared_media_type_detector_rejects_unknown_media_type():
    source = SourceReference(name="notes", media_type="text/unknown")

    with pytest.raises(SourceDetectionError, match="text/unknown"):
        DeclaredMediaTypeDetector({"text/example": "declared-text"}).detect(
            source
        )


def test_selector_rejects_unsupported_source():
    source = DetectedSource(make_request().source, source_type="unknown")

    with pytest.raises(NoSourceIngestorError, match="unknown"):
        SourceIngestorSelector([]).select(source)


def test_orchestrator_is_format_independent_and_store_creates_result():
    calls = []

    result = make_orchestrator(calls, observer=NoOpObserver()).ingest(
        make_request()
    )

    assert result == IngestionResult(
        document_id="document-1",
        document_version_id="version-1",
        processing_run_id="run-1",
        detected_source=result.detected_source,
        status="completed",
    )
    assert [call[0] for call in calls] == [
        "detect",
        "supports",
        "ingest",
        "store",
    ]
    assert result.detected_source.source.parent == make_request().source.parent


def test_observability_event_order_and_correlation():
    calls = []
    observer = RecordingObserver()

    result = make_orchestrator(calls, observer=observer).ingest(make_request())

    assert result.status == "completed"
    assert [event.event_name for event in observer.events] == [
        "ingestion.started",
        "ingestion.detection.completed",
        "ingestion.processing.completed",
        "ingestion.storage.completed",
        "ingestion.completed",
    ]
    assert len({event.execution_id for event in observer.events}) == 1
    assert all(
        event.correlation_id == event.execution_id for event in observer.events
    )
    assert observer.events[-1].document_id == "document-1"
    assert observer.events[-1].document_version_id == "version-1"
    assert all(
        event.duration_ms is not None and event.duration_ms >= 0
        for event in observer.events[1:]
    )


def test_ingestion_error_is_emitted_and_original_exception_is_raised():
    calls = []
    observer = RecordingObserver()
    orchestrator = make_orchestrator(
        calls,
        observer=observer,
        ingestors=[FailingIngestor(calls)],
    )

    with pytest.raises(RuntimeError, match="normalisation impossible"):
        orchestrator.ingest(make_request())

    error = observer.events[-1]
    assert error.event_name == "ingestion.error"
    assert error.operation == "processing"
    assert error.success is False
    assert error.error_type == "RuntimeError"
    assert error.error_message == "normalisation impossible"
    assert error.execution_id == observer.events[0].execution_id


def test_ingestion_package_has_no_rag_or_concrete_format_dependencies():
    package = Path(__file__).resolve().parents[1] / "src" / "kaliok" / "ingestion"
    core_files = [
        path
        for path in package.glob("*.py")
        if path.name != "__init__.py"
    ]
    source = "\n".join(
        path.read_text(encoding="utf-8") for path in core_files
    )

    assert "kaliok.rag" not in source
    for forbidden in ("pdf", "ocr", "docling", "docx", "html", "mail"):
        assert forbidden not in source.lower()
    assert "sqlalchemy" not in source.lower()
    assert "sqlmodel" not in source.lower()
