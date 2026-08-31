from datetime import datetime, timezone
from uuid import uuid4

from fastapi.testclient import TestClient

from kaliok.api.dependencies import get_session
from kaliok.api.main import app


class FakeResult:
    def __init__(self, values):
        self.values = values

    def all(self):
        return self.values


class FakeSession:
    def __init__(self, *, document=None, versions=None):
        self.document = document
        self.versions = list(versions or [])

    def get(self, model, object_id):
        return self.document

    def exec(self, statement):
        return FakeResult(self.versions)


class FakeDocument:
    def __init__(self):
        now = datetime.now(timezone.utc)

        self.id = uuid4()
        self.source_id = None
        self.external_id = "external-test"
        self.title = "Document de test"
        self.document_family = "test"
        self.status = "active"
        self.language = "fr"
        self.created_at = now
        self.updated_at = now


class FakeVersion:
    def __init__(self, document_id, number, *, is_current):
        now = datetime.now(timezone.utc)

        self.id = uuid4()
        self.document_id = document_id
        self.version_number = number

        self.filename = f"document-v{number}.txt"
        self.mime_type = "text/plain"
        self.file_size = 100 + number
        self.storage_uri = f"file:///document-v{number}.txt"

        self.page_count = None

        self.document_type = "text"
        self.document_subtype = None

        self.version_status = "active"
        self.processing_status = "pending"
        self.readability_status = "unknown"
        self.readability_score = None

        self.is_current = is_current

        self.created_at = now
        self.processed_at = None


client = TestClient(app)


def test_get_document():
    document = FakeDocument()

    version_2 = FakeVersion(
        document.id,
        2,
        is_current=True,
    )
    version_1 = FakeVersion(
        document.id,
        1,
        is_current=False,
    )

    fake_session = FakeSession(
        document=document,
        versions=[
            version_2,
            version_1,
        ],
    )

    app.dependency_overrides[get_session] = lambda: fake_session

    try:
        response = client.get(
            f"/documents/{document.id}"
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200

    payload = response.json()

    assert payload["id"] == str(document.id)
    assert payload["title"] == "Document de test"

    assert payload["current_version"]["version_number"] == 2
    assert payload["current_version"]["is_current"] is True

    assert [
        version["version_number"]
        for version in payload["versions"]
    ] == [2, 1]


def test_get_unknown_document_returns_404():
    fake_session = FakeSession(document=None)

    app.dependency_overrides[get_session] = lambda: fake_session

    try:
        response = client.get(
            f"/documents/{uuid4()}"
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 404
    assert response.json() == {
        "detail": "Document introuvable",
    }