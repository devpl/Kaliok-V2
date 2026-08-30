import os

os.environ.setdefault(
    "DJANGO_SETTINGS_MODULE",
    "kaliok.ui.config.settings",
)

import django

django.setup()

import httpx

from kaliok.ui.core_ui import views


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def test_get_api_status_available(monkeypatch):
    def fake_get(url, timeout):
        assert url == "http://127.0.0.1:8010/health"
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


def test_get_api_status_unavailable(monkeypatch):
    def fake_get(url, timeout):
        raise httpx.ConnectError("API indisponible")

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