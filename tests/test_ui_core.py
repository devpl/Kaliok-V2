import os
from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

os.environ.setdefault(
    "DJANGO_SETTINGS_MODULE",
    "kaliok.ui.config.settings",
)

import django

django.setup()

from django.test import Client, override_settings
from django.urls import reverse

from kaliok.ui.core_ui import views


class FakeResult:
    def __init__(self, values):
        self.values = values

    def all(self):
        return self.values


class FakeSession:
    def __init__(self, *, document=None, query_results=None):
        self.document = document
        self.query_results = list(query_results or [])

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def get(self, model, object_id):
        return self.document

    def exec(self, statement):
        values = self.query_results.pop(0)
        return FakeResult(values)


def make_document():
    return SimpleNamespace(
        id=uuid4(),
        title="Document de test",
        status="active",
        document_family=None,
        language="fr",
        created_at=datetime.now(timezone.utc),
    )


def make_version(number, *, is_current):
    return SimpleNamespace(
        id=uuid4(),
        version_number=number,
        filename=f"document-v{number}.txt",
        mime_type="text/plain",
        file_size=100 + number,
        page_count=None,
        processing_status="pending",
        readability_status="unknown",
        storage_uri=f"file:///document-v{number}.txt",
        is_current=is_current,
    )


@override_settings(ALLOWED_HOSTS=["testserver"])
def test_home_page_responds(monkeypatch):
    fake_session = FakeSession(query_results=[[]])

    monkeypatch.setattr(
        views,
        "Session",
        lambda engine: fake_session,
    )

    client = Client()

    response = client.get(reverse("home"))

    assert response.status_code == 200
    assert b"Documents" in response.content


@override_settings(ALLOWED_HOSTS=["testserver"])
def test_unknown_document_returns_404(monkeypatch):
    fake_session = FakeSession(document=None)

    monkeypatch.setattr(
        views,
        "Session",
        lambda engine: fake_session,
    )

    client = Client()

    response = client.get(
        reverse(
            "document_detail",
            kwargs={"document_id": uuid4()},
        )
    )

    assert response.status_code == 404


@override_settings(ALLOWED_HOSTS=["testserver"])
def test_document_detail_displays_version_history(monkeypatch):
    document = make_document()

    version_2 = make_version(2, is_current=True)
    version_1 = make_version(1, is_current=False)

    fake_session = FakeSession(
        document=document,
        query_results=[
            [version_2, version_1],
        ],
    )

    monkeypatch.setattr(
        views,
        "Session",
        lambda engine: fake_session,
    )

    client = Client()

    response = client.get(
        reverse(
            "document_detail",
            kwargs={"document_id": document.id},
        )
    )

    content = response.content.decode()

    assert response.status_code == 200
    assert "Historique des versions" in content
    assert "document-v2.txt" in content
    assert "document-v1.txt" in content
    assert content.index("document-v2.txt") < content.index("document-v1.txt")