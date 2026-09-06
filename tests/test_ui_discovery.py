from __future__ import annotations

import os
from datetime import datetime, timezone
from uuid import uuid4

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "kaliok.ui.config.settings")

import django

django.setup()

from django.test import Client, override_settings
from django.urls import reverse

from kaliok.ui.core_ui import views


NOW = datetime(2026, 9, 4, 14, 30, tzinfo=timezone.utc).isoformat()


def _discovery_data(*, status="completed", candidate_count=1):
    run_id = str(uuid4())
    normalization_id = str(uuid4())
    version_id = str(uuid4())
    candidate_id = str(uuid4())
    unit_id = str(uuid4())
    block_id = str(uuid4())
    fragment_id = str(uuid4())
    content = '<script>alert("x")</script> Adresse à Nouméa.'
    start = content.index("Nouméa")
    run = {
        "run_id": run_id,
        "document_version_id": version_id,
        "filename": "rapport-Nouméa.pdf",
        "status": status,
        "engine": "kaliok",
        "engine_version": "candidate-discovery-v1",
        "started_at": NOW,
        "completed_at": NOW if status != "running" else None,
        "normalization_run_id": normalization_id,
        "metrics": {
            "source_unit_count": 4,
            "candidate_count": candidate_count,
            "candidate_type_counts": {"place_name": candidate_count} if candidate_count else {},
            "skipped_unit_count": 3,
        },
        "configuration": {},
        "error_message": "Dictionnaire indisponible" if status == "failed" else None,
    }
    detail = {
        **run,
        "document": {"filename": run["filename"], "document_version_id": version_id},
        "normalization_run": {"run_id": normalization_id, "status": "completed"},
        "detectors": [{"key": "lexical_dictionary", "version": "1"}],
        "candidate_types": ["place_name"],
        "groups": ([{
            "normalized_value": "Nouméa",
            "candidate_type": "place_name",
            "occurrence_count": candidate_count,
            "pages": [1, 3] if candidate_count else [],
            "page_count": 2 if candidate_count else 0,
        }] if candidate_count else []),
    }
    candidate = {
        "candidate_id": candidate_id,
        "candidate_type": "place_name",
        "raw_value": "Nouméa",
        "normalized_value": "Nouméa",
        "confidence": 0.95,
        "detector_key": "lexical_dictionary",
        "detector_version": "1",
        "exact_text": "Nouméa",
        "start_offset": start,
        "end_offset": start + len("Nouméa"),
        "unit_index": 2,
        "unit_content": content,
        "page_number": 3,
        "filename": run["filename"],
        "processing_run_id": run_id,
        "normalized_content_unit_id": unit_id,
        "content_block_id": block_id,
        "content_block_fragment_id": fragment_id,
        "bbox": None,
        "coordinate_system": None,
    }
    return run, detail, [candidate]


def _fake_api(run, detail, candidates, calls, *, total=None):
    def request(method, path, *, body=None, timeout=10.0):
        calls.append((method, path))
        if path.startswith("/discovery/runs?"):
            return {"items": [run], "limit": 100, "offset": 0, "total": 1}, None
        if path == f"/discovery/runs/{run['run_id']}":
            return detail, None
        if path.startswith(f"/discovery/runs/{run['run_id']}/candidates"):
            return {
                "items": candidates,
                "limit": 25,
                "offset": 0,
                "total": len(candidates) if total is None else total,
            }, None
        return [], None
    return request


@override_settings(ALLOWED_HOSTS=["testserver"])
def test_discovery_ui_is_public_facing_unicode_safe_and_escaped(monkeypatch):
    run, detail, candidates = _discovery_data()
    calls = []
    monkeypatch.setattr(views, "evaluation_api_request", _fake_api(run, detail, candidates, calls))

    response = Client().get(reverse("rag_laboratory"), {"tab": "discovery"})
    content = response.content.decode("utf-8")

    assert response.status_code == 200
    assert "Découverte documentaire" in content
    assert "Voir les noms, lieux, expressions" in content
    assert "rapport-Nouméa.pdf" in content
    assert "Nouméa" in content and "Noum?" not in content
    assert "Éléments trouvés" in content and ">1<" in content
    assert "2 pages" in content and "Page 3" in content
    assert "<mark>Nouméa</mark>" in content
    assert "<script>" not in content
    assert "&lt;script&gt;" in content
    assert "<summary>Détails techniques" in content
    assert "normalized_content_unit" not in content
    assert any("value=Noum%C3%A9a" in path for _, path in calls)


@override_settings(ALLOWED_HOSTS=["testserver"])
def test_discovery_ui_shows_zero_result_as_valid_history(monkeypatch):
    run, detail, candidates = _discovery_data(candidate_count=0)
    monkeypatch.setattr(views, "evaluation_api_request", _fake_api(run, detail, candidates, []))

    content = Client().get(reverse("rag_laboratory"), {"tab": "discovery"}).content.decode()

    assert "Terminée" in content
    assert "Aucun élément repéré" in content
    assert "s’est terminée correctement" in content


@override_settings(ALLOWED_HOSTS=["testserver"])
def test_discovery_ui_shows_failed_run_without_hiding_history(monkeypatch):
    run, detail, candidates = _discovery_data(status="failed", candidate_count=0)
    monkeypatch.setattr(views, "evaluation_api_request", _fake_api(run, detail, candidates, []))

    content = Client().get(reverse("rag_laboratory"), {"tab": "discovery"}).content.decode()

    assert "Échec" in content
    assert "Cette analyse n’a pas pu se terminer" in content
    assert "Dictionnaire indisponible" in content


@override_settings(ALLOWED_HOSTS=["testserver"])
def test_discovery_ui_uses_available_types_and_progressive_loading(monkeypatch):
    run, detail, candidates = _discovery_data()
    monkeypatch.setattr(
        views,
        "evaluation_api_request",
        _fake_api(run, detail, candidates, [], total=40),
    )

    content = Client().get(reverse("rag_laboratory"), {
        "tab": "discovery",
        "candidate_type": "place_name",
    }).content.decode()

    assert '<select name="candidate_type">' in content
    assert '<option value="place_name" selected>' in content
    assert "1 occurrence affichée sur 40" in content
    assert "Afficher plus" in content
    assert "discovery_limit=50" in content
