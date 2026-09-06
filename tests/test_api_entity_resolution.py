from uuid import uuid4

from fastapi.testclient import TestClient

import kaliok.api.evaluation as api_module
from kaliok.api.dependencies import get_session
from kaliok.api.main import app


class ReadService:
    run_id, entity_id, version_id = uuid4(), uuid4(), uuid4()
    calls = []

    def __init__(self, session): self.session = session
    def list_runs(self, **kwargs):
        self.calls.append(("runs", kwargs))
        return {"items": [{"run_id": self.run_id, "status": "completed", "entity_count": 1}], **{key: kwargs[key] for key in ("limit", "offset")}, "total": 1}
    def get_run(self, run_id):
        self.calls.append(("run", run_id))
        return {"run": {"run_id": run_id}, "documents": [{"filename": "Nouméa.pdf"}], "entity_types": ["place_name"], "metrics": {}, "summary": {}} if run_id == self.run_id else None
    def list_entities(self, run_id, **kwargs):
        self.calls.append(("entities", run_id, kwargs))
        return {"items": [{"id": self.entity_id, "canonical_label": "Nouméa", "membership_count": 2}], "limit": kwargs["limit"], "offset": kwargs["offset"], "total": 1} if run_id == self.run_id else None
    def get_entity(self, entity_id):
        self.calls.append(("entity", entity_id))
        return {"entity": {"id": entity_id, "canonical_label": "Nouméa"}, "memberships": [{"evidences": [{"signal_key": "normalized_value_exact"}], "provenance": [{"exact_text": "Nouméa"}]}]} if entity_id == self.entity_id else None


def _client(monkeypatch):
    ReadService.calls = []
    monkeypatch.setattr(api_module, "EntityResolutionReadService", ReadService)
    app.dependency_overrides[get_session] = lambda: object()
    return TestClient(app)


def test_entity_resolution_run_routes_filters_pagination_and_404(monkeypatch):
    client = _client(monkeypatch)
    try:
        response = client.get("/rag/evaluation/entity-resolution/runs", params={"status": "completed", "document_version_id": ReadService.version_id, "limit": 20, "offset": 2})
        assert response.status_code == 200 and response.json()["total"] == 1
        assert ReadService.calls[0][1] == {"status": "completed", "document_version_id": ReadService.version_id, "limit": 20, "offset": 2}
        assert client.get(f"/rag/evaluation/entity-resolution/runs/{ReadService.run_id}").json()["documents"][0]["filename"] == "Nouméa.pdf"
        assert client.get(f"/rag/evaluation/entity-resolution/runs/{uuid4()}").status_code == 404
    finally: app.dependency_overrides.clear()


def test_entity_resolution_entity_routes_filters_detail_unicode_and_404(monkeypatch):
    client = _client(monkeypatch)
    try:
        response = client.get(f"/rag/evaluation/entity-resolution/runs/{ReadService.run_id}/entities", params={"entity_type": "place_name", "canonical_label": "nouméa", "status": "proposed", "grouped_only": True, "limit": 25})
        assert response.status_code == 200 and response.json()["items"][0]["canonical_label"] == "Nouméa"
        call = next(item for item in ReadService.calls if item[0] == "entities")
        assert call[2]["entity_type"] == "place_name" and call[2]["canonical_label"] == "nouméa" and call[2]["grouped_only"] is True
        detail = client.get(f"/rag/evaluation/entity-resolution/entities/{ReadService.entity_id}")
        assert detail.json()["memberships"][0]["provenance"][0]["exact_text"] == "Nouméa"
        assert client.get(f"/rag/evaluation/entity-resolution/entities/{uuid4()}").status_code == 404
    finally: app.dependency_overrides.clear()
