import os
from uuid import uuid4

os.environ.setdefault(
    "DJANGO_SETTINGS_MODULE",
    "kaliok.ui.config.settings",
)

import django

django.setup()

from django.http import Http404
from django.test import Client, override_settings
from django.urls import reverse

from kaliok.ui.core_ui import views


def make_api_document(document_id):
    return {
        "id": str(document_id),
        "source_id": None,
        "external_id": None,
        "title": "Document de test",
        "document_family": None,
        "status": "active",
        "language": "fr",
        "created_at": "2026-08-31T09:00:00+00:00",
        "updated_at": "2026-08-31T09:00:00+00:00",
        "current_version": {
            "id": str(uuid4()),
            "version_number": 2,
            "filename": "document-v2.txt",
            "mime_type": "text/plain",
            "file_size": 102,
            "storage_uri": "file:///document-v2.txt",
            "page_count": None,
            "document_type": "text",
            "document_subtype": None,
            "version_status": "active",
            "processing_status": "pending",
            "readability_status": "unknown",
            "readability_score": None,
            "is_current": True,
            "created_at": "2026-08-31T09:10:00+00:00",
            "processed_at": None,
        },
        "versions": [
            {
                "id": str(uuid4()),
                "version_number": 2,
                "filename": "document-v2.txt",
                "mime_type": "text/plain",
                "file_size": 102,
                "storage_uri": "file:///document-v2.txt",
                "page_count": None,
                "document_type": "text",
                "document_subtype": None,
                "version_status": "active",
                "processing_status": "pending",
                "readability_status": "unknown",
                "readability_score": None,
                "is_current": True,
                "created_at": "2026-08-31T09:10:00+00:00",
                "processed_at": None,
            },
            {
                "id": str(uuid4()),
                "version_number": 1,
                "filename": "document-v1.txt",
                "mime_type": "text/plain",
                "file_size": 101,
                "storage_uri": "file:///document-v1.txt",
                "page_count": None,
                "document_type": "text",
                "document_subtype": None,
                "version_status": "active",
                "processing_status": "pending",
                "readability_status": "unknown",
                "readability_score": None,
                "is_current": False,
                "created_at": "2026-08-31T09:05:00+00:00",
                "processed_at": None,
            },
        ],
    }


@override_settings(ALLOWED_HOSTS=["testserver"])
def test_home_page_responds(monkeypatch):
    monkeypatch.setattr(
        views,
        "get_api_documents",
        lambda: [
            {
                "id": str(uuid4()),
                "title": "Document liste test",
                "status": "active",
                "document_family": None,
                "language": "fr",
                "created_at": "2026-08-31T09:00:00+00:00",
                "updated_at": "2026-08-31T09:00:00+00:00",
            }
        ],
    )

    monkeypatch.setattr(
        views,
        "get_api_status",
        lambda: {
            "status": "ok",
            "service": "kaliok-api",
        },
    )

    client = Client()

    response = client.get(
        reverse("home")
    )

    assert response.status_code == 200
    assert b"Documents" in response.content
    assert b"Document liste test" in response.content
    assert b"API technique" in response.content
    assert b"disponible" in response.content


@override_settings(ALLOWED_HOSTS=["testserver"])
def test_home_page_handles_unavailable_api(monkeypatch):
    monkeypatch.setattr(
        views,
        "get_api_documents",
        lambda: None,
    )

    monkeypatch.setattr(
        views,
        "get_api_status",
        lambda: {
            "status": "error",
            "service": "kaliok-api",
        },
    )

    client = Client()

    response = client.get(
        reverse("home")
    )

    assert response.status_code == 200
    assert b"indisponible" in response.content


@override_settings(ALLOWED_HOSTS=["testserver"])
def test_unknown_document_returns_404(monkeypatch):
    def fake_get_api_document(document_id):
        raise Http404("Document introuvable")

    monkeypatch.setattr(
        views,
        "get_api_document",
        fake_get_api_document,
    )

    client = Client()

    response = client.get(
        reverse(
            "document_detail",
            kwargs={
                "document_id": uuid4(),
            },
        )
    )

    assert response.status_code == 404


@override_settings(ALLOWED_HOSTS=["testserver"])
def test_document_detail_displays_version_history(
    monkeypatch,
):
    document_id = uuid4()

    monkeypatch.setattr(
        views,
        "get_api_document",
        lambda requested_id: make_api_document(
            requested_id
        ),
    )

    client = Client()

    response = client.get(
        reverse(
            "document_detail",
            kwargs={
                "document_id": document_id,
            },
        )
    )

    content = response.content.decode()

    assert response.status_code == 200
    assert "Document de test" in content
    assert "Historique des versions" in content
    assert "document-v2.txt" in content
    assert "document-v1.txt" in content

    assert (
        content.index("document-v2.txt")
        < content.index("document-v1.txt")
    )


@override_settings(ALLOWED_HOSTS=["testserver"])
def test_document_detail_returns_503_when_api_is_unavailable(
    monkeypatch,
):
    monkeypatch.setattr(
        views,
        "get_api_document",
        lambda document_id: None,
    )

    client = Client()

    response = client.get(
        reverse(
            "document_detail",
            kwargs={
                "document_id": uuid4(),
            },
        )
    )

    assert response.status_code == 503
    assert (
        response.content.decode()
        == "API technique indisponible"
    )