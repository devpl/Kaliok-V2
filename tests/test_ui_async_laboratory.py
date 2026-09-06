from urllib.parse import parse_qs, urlparse
from uuid import uuid4

from django.test import Client, override_settings
from django.urls import reverse

from kaliok.ui.core_ui import views


@override_settings(ALLOWED_HOSTS=["testserver"])
def test_discovery_async_slice_reuses_api_and_returns_only_candidates(
    monkeypatch,
):
    run_id = str(uuid4())
    candidate_id = str(uuid4())
    calls = []

    def request(method, path, **kwargs):
        calls.append((method, path))
        return (
            {
                "items": [
                    {
                        "candidate_id": candidate_id,
                        "unit_content": "Nouméa",
                    }
                ],
                "total": 1,
            },
            None,
        )

    monkeypatch.setattr(
        views,
        "evaluation_api_request",
        request,
    )

    response = Client().get(
        reverse("rag_laboratory_data"),
        {
            "kind": "discovery",
            "run": run_id,
            "value": "Nouméa",
            "limit": 25,
            "offset": 0,
        },
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["total"] == 1
    assert len(payload["items"]) == 1
    assert payload["items"][0]["candidate_id"] == candidate_id
    assert payload["items"][0]["unit_content"] == "Nouméa"

    assert len(calls) == 1

    method, path = calls[0]

    assert method == "GET"

    parsed = urlparse(path)
    query = parse_qs(parsed.query)

    assert parsed.path == (
        f"/discovery/runs/{run_id}/candidates"
    )

    assert query["value"] == ["Nouméa"]
    assert query["limit"] == ["25"]
    assert query["offset"] == ["0"]


@override_settings(ALLOWED_HOSTS=["testserver"])
def test_entity_resolution_async_slice_reuses_api(monkeypatch):
    run_id = str(uuid4())
    entity_id = str(uuid4())
    calls = []

    def request(method, path, **kwargs):
        calls.append((method, path))

        if path.startswith(
            f"/entity-resolution/runs/{run_id}/entities"
        ):
            return (
                {
                    "items": [
                        {
                            "id": entity_id,
                            "canonical_label": "Nouméa",
                            "entity_type": "place_name",
                            "membership_count": 3,
                            "document_count": 1,
                            "page_count": 3,
                        }
                    ],
                    "total": 1,
                },
                None,
            )

        if path == (
            f"/entity-resolution/entities/{entity_id}"
        ):
            return (
                {
                    "entity": {
                        "id": entity_id,
                        "canonical_label": "Nouméa",
                        "entity_type": "place_name",
                    },
                    "memberships": [],
                },
                None,
            )

        raise AssertionError(
            f"Unexpected API request: {method} {path}"
        )

    monkeypatch.setattr(
        views,
        "evaluation_api_request",
        request,
    )

    response = Client().get(
        reverse("rag_laboratory_data"),
        {
            "kind": "entity_resolution",
            "run": run_id,
            "limit": 25,
            "offset": 0,
        },
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["total"] == 1
    assert len(payload["items"]) == 1
    assert payload["items"][0]["id"] == entity_id
    assert payload["items"][0]["canonical_label"] == "Nouméa"

    assert len(calls) >= 1

    method, path = calls[0]

    assert method == "GET"

    parsed = urlparse(path)
    query = parse_qs(parsed.query)

    assert parsed.path == (
        f"/entity-resolution/runs/{run_id}/entities"
    )

    assert query["limit"] == ["25"]
    assert query["offset"] == ["0"]