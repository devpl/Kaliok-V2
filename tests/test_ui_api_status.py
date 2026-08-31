import os
from uuid import uuid4

os.environ.setdefault(
    "DJANGO_SETTINGS_MODULE",
    "kaliok.ui.config.settings",
)

import django

django.setup()

import httpx
import pytest
from django.http import Http404
from django.test import override_settings

from kaliok.ui.core_ui import views


class FakeResponse:
    def __init__(
        self,
        payload=None,
        *,
        status_code=200,
        json_error=None,
    ):
        self.payload = payload
        self.status_code = status_code
        self.json_error = json_error

    def raise_for_status(self):
        if self.status_code >= 400:
            request = httpx.Request(
                "GET",
                "http://api-test",
            )
            response = httpx.Response(
                self.status_code,
                request=request,
            )
            raise httpx.HTTPStatusError(
                "Erreur HTTP",
                request=request,
                response=response,
            )

    def json(self):
        if self.json_error is not None:
            raise self.json_error

        return self.payload


@override_settings(
    KALIOK_API_BASE_URL="http://api-test:9123",
)
def test_get_api_status_available(monkeypatch):
    def fake_get(url, timeout):
        assert url == "http://api-test:9123/health"
        assert timeout == 2.0

        return FakeResponse(
            {
                "status": "ok",
                "service": "kaliok-api",
            }
        )

    monkeypatch.setattr(
        views.httpx,
        "get",
        fake_get,
    )

    result = views.get_api_status()

    assert result == {
        "status": "ok",
        "service": "kaliok-api",
    }


@override_settings(
    KALIOK_API_BASE_URL="http://api-test:9123",
)
def test_get_api_status_unavailable(monkeypatch):
    def fake_get(url, timeout):
        raise httpx.ConnectError(
            "API indisponible"
        )

    monkeypatch.setattr(
        views.httpx,
        "get",
        fake_get,
    )

    result = views.get_api_status()

    assert result == {
        "status": "error",
        "service": "kaliok-api",
    }


@override_settings(
    KALIOK_API_BASE_URL="http://api-test:9123",
)
def test_get_api_document_available(monkeypatch):
    document_id = uuid4()

    payload = {
        "id": str(document_id),
        "title": "Document de test",
        "current_version": None,
        "versions": [],
    }

    def fake_get(url, timeout):
        assert url == (
            "http://api-test:9123"
            f"/documents/{document_id}"
        )
        assert timeout == 5.0

        return FakeResponse(payload)

    monkeypatch.setattr(
        views.httpx,
        "get",
        fake_get,
    )

    result = views.get_api_document(
        document_id
    )

    assert result == payload


@override_settings(
    KALIOK_API_BASE_URL="http://api-test:9123",
)
def test_get_api_document_404(monkeypatch):
    monkeypatch.setattr(
        views.httpx,
        "get",
        lambda url, timeout: FakeResponse(
            status_code=404
        ),
    )

    with pytest.raises(Http404):
        views.get_api_document(
            uuid4()
        )


@override_settings(
    KALIOK_API_BASE_URL="http://api-test:9123",
)
def test_get_api_document_connection_error(
    monkeypatch,
):
    def fake_get(url, timeout):
        raise httpx.ConnectError(
            "API indisponible"
        )

    monkeypatch.setattr(
        views.httpx,
        "get",
        fake_get,
    )

    assert (
        views.get_api_document(uuid4())
        is None
    )


@override_settings(
    KALIOK_API_BASE_URL="http://api-test:9123",
)
def test_get_api_document_invalid_json(
    monkeypatch,
):
    monkeypatch.setattr(
        views.httpx,
        "get",
        lambda url, timeout: FakeResponse(
            json_error=ValueError(
                "JSON invalide"
            )
        ),
    )

    assert (
        views.get_api_document(uuid4())
        is None
    )

@override_settings(
    KALIOK_API_BASE_URL="http://kaliok-api-test.invalid",
)
def test_get_api_documents_available(monkeypatch):
    payload = [
        {
            "id": str(uuid4()),
            "title": "Document 1",
        }
    ]

    def fake_get(url, timeout):
        assert url == (
            "http://kaliok-api-test.invalid/documents"
        )
        assert timeout == 5.0

        return FakeResponse(payload)

    monkeypatch.setattr(
        views.httpx,
        "get",
        fake_get,
    )

    result = views.get_api_documents()

    assert result == payload


@override_settings(
    KALIOK_API_BASE_URL="http://kaliok-api-test.invalid",
)
def test_get_api_documents_unavailable(monkeypatch):
    def fake_get(url, timeout):
        raise httpx.ConnectError(
            "API indisponible"
        )

    monkeypatch.setattr(
        views.httpx,
        "get",
        fake_get,
    )

    assert views.get_api_documents() is None

@override_settings(
    KALIOK_API_BASE_URL="http://kaliok-api-test.invalid",
)
def test_ingest_api_document_available(
    monkeypatch,
    tmp_path,
):
    path = tmp_path / "document.txt"
    path.write_text(
        "Document de test.",
        encoding="utf-8",
    )

    document_id = uuid4()
    version_id = uuid4()

    payload = {
        "document_id": str(document_id),
        "document_version_id": str(version_id),
        "status": "created",
        "source_type": "plain_text",
        "media_type": "text/plain",
    }

    def fake_post(url, json, timeout):
        assert url == (
            "http://kaliok-api-test.invalid/ingestion"
        )
        assert timeout == 30.0
        assert json == {
            "source": {
                "name": "document.txt",
                "uri": path.resolve().as_uri(),
                "media_type": "text/plain",
                "size": path.stat().st_size,
            }
        }

        return FakeResponse(payload)

    monkeypatch.setattr(
        views.httpx,
        "post",
        fake_post,
    )

    result = views.ingest_api_document(
        path=path,
        original_name="document.txt",
        size=path.stat().st_size,
    )

    assert result == payload


@override_settings(
    KALIOK_API_BASE_URL="http://kaliok-api-test.invalid",
)
def test_ingest_api_document_unavailable(
    monkeypatch,
    tmp_path,
):
    path = tmp_path / "document.txt"
    path.write_text(
        "Document de test.",
        encoding="utf-8",
    )

    def fake_post(url, json, timeout):
        raise httpx.ConnectError(
            "API indisponible"
        )

    monkeypatch.setattr(
        views.httpx,
        "post",
        fake_post,
    )

    result = views.ingest_api_document(
        path=path,
        original_name="document.txt",
        size=path.stat().st_size,
    )

    assert result is None
