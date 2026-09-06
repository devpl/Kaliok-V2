from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from fastapi.testclient import TestClient

import kaliok.api.evaluation as api_module
from kaliok.api.dependencies import get_session
from kaliok.api.main import app


class ReadService:
    run_id = uuid4()
    candidate_id = uuid4()
    version_id = uuid4()
    calls = []

    def __init__(self, session):
        self.session = session

    def list_runs(self, *, limit, offset):
        self.calls.append(("runs", limit, offset))
        return {"items": [{
            "run_id": self.run_id,
            "document_version_id": self.version_id,
            "filename": "rapport-Nouméa.pdf",
            "status": "completed",
            "engine": "kaliok",
            "engine_version": "candidate-discovery-v1",
            "completed_at": datetime(2026, 9, 4, tzinfo=timezone.utc),
            "normalization_run_id": uuid4(),
            "metrics": {"candidate_count": 18},
        }], "limit": limit, "offset": offset, "total": 1}

    def get_run(self, run_id):
        self.calls.append(("run", run_id))
        if run_id != self.run_id:
            return None
        return {"run_id": run_id, "groups": [{
            "normalized_value": "Nouméa",
            "candidate_type": "place_name",
            "occurrence_count": 18,
            "pages": [1, 3],
            "page_count": 2,
        }]}

    def list_candidates(self, run_id, **filters):
        self.calls.append(("candidates", run_id, filters))
        if run_id != self.run_id:
            return None
        return {"items": [{
            "candidate_id": self.candidate_id,
            "raw_value": "Nouméa",
            "normalized_value": "Nouméa",
            "candidate_type": "place_name",
            "page_number": 3,
            "exact_text": "Nouméa",
        }], "limit": filters["limit"], "offset": filters["offset"], "total": 1}

    def get_candidate(self, candidate_id):
        self.calls.append(("candidate", candidate_id))
        if candidate_id != self.candidate_id:
            return None
        return {
            "candidate": {"id": candidate_id, "raw_value": "Nouméa"},
            "document": {"filename": "rapport.pdf"},
            "provenance": [{
                "occurrence": {"exact_text": "Nouméa"},
                "normalized_content_unit": {"sources": [{
                    "content_block": {"fragments": [{"page_number": 3}]}
                }]},
            }],
        }

    def compare_runs(self, run_a, run_b):
        self.calls.append(("compare", run_a, run_b))
        return {"summary": {"total_occurrences_a": 67, "total_occurrences_b": 72, "delta": 5}, "differences": []}


def _client(monkeypatch):
    ReadService.calls = []
    monkeypatch.setattr(api_module, "CandidateDiscoveryReadService", ReadService)
    app.dependency_overrides[get_session] = lambda: object()
    return TestClient(app)


def test_discovery_run_list_and_detail_preserve_unicode(monkeypatch):
    client = _client(monkeypatch)
    try:
        response = client.get("/rag/evaluation/discovery/runs?limit=20&offset=2")
        assert response.status_code == 200
        assert response.json()["items"][0]["filename"] == "rapport-Nouméa.pdf"
        assert response.json()["total"] == 1
        assert ("runs", 20, 2) in ReadService.calls

        detail = client.get(f"/rag/evaluation/discovery/runs/{ReadService.run_id}")
        assert detail.status_code == 200
        assert detail.json()["groups"][0]["normalized_value"] == "Nouméa"
        assert client.get(f"/rag/evaluation/discovery/runs/{uuid4()}").status_code == 404
    finally:
        app.dependency_overrides.clear()


def test_discovery_candidate_filters_and_detail_provenance(monkeypatch):
    client = _client(monkeypatch)
    try:
        response = client.get(
            f"/rag/evaluation/discovery/runs/{ReadService.run_id}/candidates",
            params={"candidate_type": "place_name", "value": "Nouméa", "page": 3, "limit": 25},
        )
        assert response.status_code == 200
        assert response.json()["items"][0]["exact_text"] == "Nouméa"
        assert response.json()["total"] == 1
        call = next(item for item in ReadService.calls if item[0] == "candidates")
        assert call[2] == {
            "candidate_type": "place_name",
            "value": "Nouméa",
            "page": 3,
            "limit": 25,
            "offset": 0,
        }
        detail = client.get(
            f"/rag/evaluation/discovery/candidates/{ReadService.candidate_id}"
        )
        assert detail.status_code == 200
        assert detail.json()["provenance"][0]["normalized_content_unit"]["sources"][0]["content_block"]["fragments"][0]["page_number"] == 3
        assert client.get(
            f"/rag/evaluation/discovery/candidates/{uuid4()}"
        ).status_code == 404
    finally:
        app.dependency_overrides.clear()


def test_discovery_compare_endpoint(monkeypatch):
    client = _client(monkeypatch)
    other = uuid4()
    try:
        response = client.get("/rag/evaluation/discovery/compare", params={"run_a": ReadService.run_id, "run_b": other})
        assert response.status_code == 200
        assert response.json()["summary"]["delta"] == 5
        assert ("compare", ReadService.run_id, other) in ReadService.calls
    finally:
        app.dependency_overrides.clear()
