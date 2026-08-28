from __future__ import annotations

from uuid import uuid4

from kaliok.ingestion.base import DocumentStore, SourceDetector
from kaliok.ingestion.detection import SourceIngestorSelector
from kaliok.ingestion.types import IngestionRequest, IngestionResult
from kaliok.observability.base import Observer
from kaliok.observability.events import ObservabilityEvent
from kaliok.observability.noop import NoOpObserver
from kaliok.observability.timing import Timer


class IngestionOrchestrator:
    def __init__(
        self,
        *,
        detector: SourceDetector,
        ingestor_selector: SourceIngestorSelector,
        document_store: DocumentStore,
        observer: Observer | None = None,
    ) -> None:
        self._detector = detector
        self._ingestor_selector = ingestor_selector
        self._document_store = document_store
        self._observer = observer if observer is not None else NoOpObserver()

    def ingest(self, request: IngestionRequest) -> IngestionResult:
        execution_id = str(uuid4())
        total_timer = Timer.start()
        operation = "ingestion"
        implementation = type(self).__name__
        self._emit(
            "ingestion.started",
            execution_id,
            component="ingestion",
            implementation=implementation,
            operation=operation,
            input_count=1,
        )
        try:
            operation = "detection"
            implementation = type(self._detector).__name__
            timer = Timer.start()
            detected = self._detector.detect(request.source)
            self._emit(
                "ingestion.detection.completed",
                execution_id,
                component="ingestion.detection",
                implementation=implementation,
                operation=operation,
                duration_ms=timer.elapsed_ms(),
                input_count=1,
                output_count=1,
                success=True,
            )

            operation = "selection"
            implementation = type(self._ingestor_selector).__name__
            ingestor = self._ingestor_selector.select(detected)

            operation = "processing"
            implementation = type(ingestor).__name__
            timer = Timer.start()
            normalized = ingestor.ingest(request, detected)
            self._emit(
                "ingestion.processing.completed",
                execution_id,
                component="ingestion.processing",
                implementation=implementation,
                operation=operation,
                duration_ms=timer.elapsed_ms(),
                input_count=1,
                output_count=1,
                success=True,
            )

            operation = "storage"
            implementation = type(self._document_store).__name__
            timer = Timer.start()
            result = self._document_store.store(request, normalized)
            identity = self._result_identity(result)
            self._emit(
                "ingestion.storage.completed",
                execution_id,
                component="ingestion.storage",
                implementation=implementation,
                operation=operation,
                duration_ms=timer.elapsed_ms(),
                input_count=1,
                output_count=1,
                success=True,
                **identity,
            )
            self._emit(
                "ingestion.completed",
                execution_id,
                component="ingestion",
                implementation=type(self).__name__,
                operation="ingestion",
                duration_ms=total_timer.elapsed_ms(),
                input_count=1,
                output_count=1,
                success=True,
                **identity,
            )
            return result
        except Exception as error:
            self._emit(
                "ingestion.error",
                execution_id,
                component="ingestion",
                implementation=implementation,
                operation=operation,
                duration_ms=total_timer.elapsed_ms(),
                success=False,
                error_type=type(error).__name__,
                error_message=str(error),
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
            return None

    @staticmethod
    def _result_identity(result: IngestionResult) -> dict[str, object | None]:
        return {
            "document_id": result.document_id,
            "document_version_id": result.document_version_id,
            "processing_run_id": result.processing_run_id,
        }
