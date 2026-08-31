from uuid import uuid4

from fastapi.testclient import TestClient

from kaliok.api.dependencies import get_session
from kaliok.api.main import app
from kaliok.ingestion.detection import SourceDetectionError
from kaliok.ingestion.types import (
    DetectedSource,
    IngestionResult,
)


class FakeSession:
    def __init__(self):
        self.commit_count = 0
        self.rollback_count = 0

    def commit(self):
        self.commit_count += 1

    def rollback(self):
        self.rollback_count += 1


class SuccessfulOrchestrator:
    def __init__(self, result):
        self.result = result
        self.request = None

    def ingest(self, request):
        self.request = request
        return self.result


class FailingOrchestrator:
    def __init__(self, error):
        self.error = error

    def ingest(self, request):
        raise self.error


def make_result():
    from kaliok.ingestion.types import SourceReference

    source_reference = SourceReference(
        name="document.txt",
        uri="file:///tmp/document.txt",
        media_type="text/plain",
    )

    source = DetectedSource(
        source=source_reference,
        source_type="plain_text",
        media_type="text/plain",
        confidence=1.0,
    )

    return IngestionResult(
        document_id=uuid4(),
        document_version_id=uuid4(),
        detected_source=source,
        status="created",
    )


def test_ingest_document_commits_transaction(monkeypatch):
    from kaliok.api import ingestion as ingestion_api

    session = FakeSession()
    result = make_result()
    orchestrator = SuccessfulOrchestrator(result)

    app.dependency_overrides[get_session] = lambda: session
    monkeypatch.setattr(
        ingestion_api,
        "create_ingestion_orchestrator",
        lambda current_session: orchestrator,
    )

    try:
        client = TestClient(app)

        response = client.post(
            "/ingestion",
            json={
                "source": {
                    "name": "document.txt",
                    "uri": "file:///tmp/document.txt",
                    "media_type": "text/plain",
                    "size": 123,
                    "external_id": "source-123",
                }
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200

    payload = response.json()

    assert payload == {
        "document_id": str(result.document_id),
        "document_version_id": str(result.document_version_id),
        "status": "created",
        "source_type": "plain_text",
        "media_type": "text/plain",
    }

    assert session.commit_count == 1
    assert session.rollback_count == 0

    assert orchestrator.request.source.name == "document.txt"
    assert orchestrator.request.source.uri == "file:///tmp/document.txt"
    assert orchestrator.request.source.media_type == "text/plain"
    assert orchestrator.request.source.size == 123
    assert orchestrator.request.source.external_id == "source-123"


def test_ingest_document_rolls_back_expected_error(monkeypatch):
    from kaliok.api import ingestion as ingestion_api

    session = FakeSession()
    orchestrator = FailingOrchestrator(
        SourceDetectionError("Type de source inconnu.")
    )

    app.dependency_overrides[get_session] = lambda: session
    monkeypatch.setattr(
        ingestion_api,
        "create_ingestion_orchestrator",
        lambda current_session: orchestrator,
    )

    try:
        client = TestClient(app)

        response = client.post(
            "/ingestion",
            json={
                "source": {
                    "name": "document.bin",
                    "uri": "file:///tmp/document.bin",
                    "media_type": "application/octet-stream",
                }
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 400
    assert response.json() == {
        "detail": "Type de source inconnu.",
    }

    assert session.commit_count == 0
    assert session.rollback_count == 1


def test_ingest_document_rolls_back_unexpected_error(monkeypatch):
    from kaliok.api import ingestion as ingestion_api

    session = FakeSession()
    orchestrator = FailingOrchestrator(
        RuntimeError("Erreur inattendue")
    )

    app.dependency_overrides[get_session] = lambda: session
    monkeypatch.setattr(
        ingestion_api,
        "create_ingestion_orchestrator",
        lambda current_session: orchestrator,
    )

    try:
        client = TestClient(app, raise_server_exceptions=False)

        response = client.post(
            "/ingestion",
            json={
                "source": {
                    "name": "document.txt",
                    "uri": "file:///tmp/document.txt",
                    "media_type": "text/plain",
                }
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 500

    assert session.commit_count == 0
    assert session.rollback_count == 1
