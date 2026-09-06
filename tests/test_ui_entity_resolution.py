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


def _data(status="completed", count=1, grouped=True):
    run_id, entity_id = str(uuid4()), str(uuid4())
    run = {"run_id": run_id, "status": status, "engine": "kaliok", "engine_version": "declared-normalized-exact-v1", "configuration": {}, "metrics": {}, "entity_count": count, "membership_count": 2 if grouped and count else count, "grouped_entity_count": 1 if grouped and count else 0, "singleton_entity_count": 0 if grouped else count, "document_version_count": 2 if grouped and count else count, "started_at": datetime.now(timezone.utc).isoformat(), "completed_at": datetime.now(timezone.utc).isoformat() if status != "running" else None, "error_message": "échec technique" if status == "failed" else None}
    detail = {"run": run, "metrics": run["metrics"], "documents": [], "entity_types": ["organization_name"], "summary": {}}
    summary = {"id": entity_id, "entity_type": "organization_name", "canonical_label": '<script>alert(1)</script> Cour des comptes', "membership_count": 2 if grouped else 1, "document_count": 2 if grouped else 1, "page_count": 1, "confidence": 1.0 if grouped else None}
    signal = "normalized_value_exact" if grouped else "singleton_normalized_value"
    membership = {"membership_id": str(uuid4()), "document": {"filename": "rapport-Nouméa.pdf"}, "candidate": {"candidate_id": str(uuid4()), "candidate_discovery_run_id": str(uuid4()), "normalization_run_id": str(uuid4()), "raw_value": "COUR DES COMPTES", "normalized_value": "Cour des comptes"}, "evidences": [{"signal_key": signal, "method": "declared-normalized-exact-v1", "score": 1.0 if grouped else None, "explanation": "Même valeur"}], "provenance": [{"start_offset": 7, "end_offset": 23, "exact_text": "Cour des comptes", "normalized_content_unit": {"id": str(uuid4()), "content": "Source Cour des comptes ici", "sources": [{"content_block": {"id": str(uuid4()), "fragments": [{"page_id": str(uuid4()), "page_number": 2, "document_version_id": str(uuid4()), "bbox": None, "coordinate_system": None}]}}]}}]}
    entity = {"entity": {"id": entity_id, "canonical_label": summary["canonical_label"], "entity_type": "organization_name", "confidence": summary["confidence"]}, "run": run, "memberships": [membership, membership] if grouped else [membership], "evidences": membership["evidences"]}
    return run, detail, summary, entity


def _fake(run, detail, summary, entity, total=None):
    def request(method, path, **kwargs):
        if path.startswith("/entity-resolution/runs?limit="): return {"items": [run], "total": 1}, None
        if path == f"/entity-resolution/runs/{run['run_id']}": return detail, None
        if path.startswith(f"/entity-resolution/runs/{run['run_id']}/entities?"): return {"items": [summary] if run["entity_count"] else [], "total": total if total is not None else run["entity_count"]}, None
        if path == f"/entity-resolution/entities/{summary['id']}": return entity, None
        return [], None
    return request


@override_settings(ALLOWED_HOSTS=["testserver"])
def test_entity_resolution_ui_grouped_highlight_filters_details_and_xss(monkeypatch):
    run, detail, summary, entity = _data()
    monkeypatch.setattr(views, "evaluation_api_request", _fake(run, detail, summary, entity, total=40))
    content = Client().get(reverse("rag_laboratory"), {"tab": "entity_resolution", "entity_type": "organization_name", "canonical_label": "Cour"}).content.decode()
    assert "Résolution d’entités" in content and "Découverte documentaire" in content
    assert "Entité proposée" in content and "Organisation" in content
    assert "2 occurrences" in content and "2 documents" in content
    assert "Pourquoi ce rapprochement ?" in content and "declared-normalized-exact-v1" in content
    assert "rapport-Nouméa.pdf" in content and "Page 2" in content
    assert "<mark>Cour des comptes</mark>" in content
    assert "<script>" not in content and "&lt;script&gt;" in content
    assert "Détails techniques" in content and "Afficher plus" in content
    assert "entity_resolution_limit=50" in content


@override_settings(ALLOWED_HOSTS=["testserver"])
def test_entity_resolution_ui_singleton_and_run_states(monkeypatch):
    for status, count, expected in (("completed", 0, "aucune entité proposée"), ("failed", 0, "Analyse échouée"), ("running", 0, "Analyse en cours")):
        run, detail, summary, entity = _data(status=status, count=count, grouped=False)
        monkeypatch.setattr(views, "evaluation_api_request", _fake(run, detail, summary, entity))
        content = Client().get(reverse("rag_laboratory"), {"tab": "entity_resolution"}).content.decode()
        assert expected in content
    run, detail, summary, entity = _data(grouped=False)
    monkeypatch.setattr(views, "evaluation_api_request", _fake(run, detail, summary, entity))
    content = Client().get(reverse("rag_laboratory"), {"tab": "entity_resolution"}).content.decode()
    assert "aucun rapprochement avec une autre occurrence" in content
    assert "Une valeur normalisée est disponible" in content
    entity["memberships"][0]["evidences"][0]["signal_key"] = "singleton_no_normalized_value"
    content = Client().get(reverse("rag_laboratory"), {"tab": "entity_resolution"}).content.decode()
    assert "Aucune valeur normalisée n’est déclarée" in content
