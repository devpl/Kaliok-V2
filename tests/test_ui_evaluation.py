from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime, timezone
from uuid import uuid4

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "kaliok.ui.config.settings")

import django

django.setup()

from django.test import Client, override_settings
from django.urls import reverse

from kaliok.ui.core_ui import views


NOW = datetime(2026, 9, 2, 10, 0, tzinfo=timezone.utc).isoformat()


def laboratory_data():
    document_id, version_id = str(uuid4()), str(uuid4())
    suite_id, question_id = str(uuid4()), str(uuid4())
    revision_1, revision_2 = str(uuid4()), str(uuid4())
    campaign_id, attempt_id = str(uuid4()), str(uuid4())
    question = {
        "id": question_id,
        "origin": "evaluation",
        "question_text": "Quel est le délai ?",
        "expected_answer": "Trente jours.",
        "document_id": document_id,
        "document_version_id": version_id,
        "status": "pending",
        "created_at": NOW,
    }
    suite = {
        "id": suite_id,
        "name": "Suite contrats",
        "description": "Questions contractuelles",
        "status": "active",
        "created_at": NOW,
        "updated_at": NOW,
    }
    campaign = {
        "id": campaign_id,
        "name": "Comparaison septembre",
        "suite_id": suite_id,
        "status": "completed",
        "configuration": {
            "configuration_revision_ids": [revision_1, revision_2],
            "repetitions": 2,
        },
        "created_at": NOW,
        "started_at": NOW,
        "completed_at": NOW,
    }
    attempt = {
        "id": attempt_id,
        "question_id": question_id,
        "attempt_number": 1,
        "configuration_revision_id": revision_1,
        "evaluation_campaign_id": campaign_id,
        "strategy": "normalized",
        "status": "completed",
        "answer_text": "Le délai est de trente jours.",
        "failure_reason": None,
        "configuration": {
            "generation": {"provider": "ollama", "model": "llama", "temperature": 0.2},
            "retrieval": {"top_k": 5},
            "context": {"builder": "standard"},
            "embedding": {"model": "bge-m3"},
        },
        "metrics": {"duration_ms": 1420},
        "created_at": NOW,
        "completed_at": NOW,
    }
    return {
        "documents": [{
            "document_id": document_id,
            "title": "Contrat fournisseur",
            "document_version_id": version_id,
            "filename": "contrat.pdf",
            "version_number": 3,
            "processing_status": "completed",
            "page_count": 12,
        }],
        "suites": [suite],
        "suite": {**suite, "questions": [question]},
        "configurations": [
            {"profile_id": str(uuid4()), "profile_key": "production", "label": "Production", "is_active": True, "revisions": [{"revision_id": revision_1, "revision_number": 2, "status": "active", "created_at": NOW, "change_reason": "Réglage stable"}]},
            {"profile_id": str(uuid4()), "profile_key": "candidate", "label": "Candidate", "is_active": False, "revisions": [{"revision_id": revision_2, "revision_number": 7, "status": "retired", "created_at": NOW, "change_reason": None}]},
        ],
        "campaigns": [campaign],
        "campaign": campaign,
        "attempts": [attempt],
        "attempt": attempt,
        "question": question,
        "revisions": [revision_1, revision_2],
    }


def fake_api(data, calls):
    def request(method, path, *, body=None, timeout=10.0):
        calls.append((method, path, body, timeout))
        fixed = {
            "/documents": data["documents"],
            "/suites": data["suites"],
            "/configurations": data["configurations"],
            "/campaigns": data["campaigns"],
            f"/suites/{data['suite']['id']}": data["suite"],
            f"/campaigns/{data['campaign']['id']}/attempts": data["attempts"],
            f"/attempts/{data['attempt']['id']}": {
                "attempt": data["attempt"],
                "question": data["question"],
                "evidence": [{"id": str(uuid4()), "document_version_id": data["documents"][0]["document_version_id"], "rank": 1, "score": 0.91, "evidence_text": "Passage contractuel.", "extra_data": {"page": 4}}],
                "feedback": [],
            },
        }
        if method == "POST" and path == "/campaigns":
            return data["campaign"], None
        if method == "POST" and path.endswith("/run"):
            return {
                "campaign_id": data["campaign"]["id"], "suite_id": data["suite"]["id"], "status": "completed",
                "total_requested_runs": 4, "completed_runs": 3, "failed_runs": 1,
                "technical_success_rate": 75.0, "average_duration_ms": 1420.0,
                "average_retrieved_passage_count": 5.0, "started_at": NOW, "completed_at": NOW,
                "results": [{"question_id": data["question"]["id"], "question_attempt_id": data["attempt"]["id"], "document_id": data["documents"][0]["document_id"], "document_version_id": data["documents"][0]["document_version_id"], "answer": "Trente jours.", "configuration_revision_id": data["revisions"][0], "configuration": {}, "metrics": {"duration_ms": 1420}, "sources": [{"rank": 1, "score": 0.91, "document_version_id": data["documents"][0]["document_version_id"], "text": "Passage contractuel.", "extra_data": {"page": 4}}]}],
                "errors": [{"question_id": data["question"]["id"], "configuration_revision_id": data["revisions"][1], "repetition": 2, "error_type": "RuntimeError", "message": "Échec contrôlé"}],
                "by_configuration": [{"configuration_revision_id": data["revisions"][0], "total_requested_runs": 2, "completed_runs": 2, "failed_runs": 0, "technical_success_rate": 100.0, "average_duration_ms": 1420.0, "average_retrieved_passage_count": 5.0}],
            }, None
        if method == "POST" and path.endswith("/feedback"):
            return {"id": str(uuid4())}, None
        return fixed.get(path, []), None
    return request


@override_settings(ALLOWED_HOSTS=["testserver"])
def test_laboratory_page_navigation_loads_documents_suites_and_configurations(monkeypatch):
    data, calls = laboratory_data(), []
    monkeypatch.setattr(views, "evaluation_api_request", fake_api(data, calls))
    response = Client().get(reverse("rag_laboratory"), {"suite": data["suite"]["id"]})
    content = response.content.decode()
    assert response.status_code == 200
    assert "Laboratoire RAG" in content
    assert "Contrat fournisseur" in content and "contrat.pdf" in content
    assert "Suite contrats" in content and "Quel est le délai ?" in content
    assert "Production" in content and "Candidate" in content
    assert content.count('name="configuration_revision_ids"') == 2
    assert f'href="{reverse("rag_laboratory")}"' in content


@override_settings(ALLOWED_HOSTS=["testserver"])
def test_laboratory_calculates_runs_and_launches_campaign(monkeypatch):
    data, calls = laboratory_data(), []
    monkeypatch.setattr(views, "evaluation_api_request", fake_api(data, calls))
    response = Client().post(reverse("rag_laboratory"), {
        "action": "run_campaign", "campaign_name": "Comparatif", "suite_id": data["suite"]["id"],
        "configuration_revision_ids": data["revisions"], "repetitions": "2",
    })
    content = response.content.decode()
    create = next(call for call in calls if call[0:2] == ("POST", "/campaigns"))
    assert create[2]["configuration_revision_ids"] == data["revisions"]
    assert create[2]["repetitions"] == 2
    assert any(call[0] == "POST" and call[1].endswith("/run") for call in calls)
    assert "Demandés" in content and "75,0 %" in content
    assert "Échec contrôlé" in content and "Passage contractuel." in content
    assert "data-total-runs" in content
    script = Path(
        "src/kaliok/ui/core_ui/static/core_ui/rag_laboratory.js"
    ).read_text(encoding="utf-8")
    assert "configurations * questions * repeats" in script


@override_settings(ALLOWED_HOSTS=["testserver"])
def test_laboratory_creates_and_removes_suite_question(monkeypatch):
    data, calls = laboratory_data(), []

    def request(method, path, *, body=None, timeout=10.0):
        calls.append((method, path, body, timeout))
        if method == "POST" and path == "/questions":
            return data["question"], None
        if method == "POST" and path.endswith("/questions"):
            return {"id": str(uuid4())}, None
        if method == "DELETE":
            return {}, None
        return fake_api(data, [])(method, path, body=body, timeout=timeout)

    monkeypatch.setattr(views, "evaluation_api_request", request)
    client = Client()
    client.post(reverse("rag_laboratory"), {
        "action": "create_question", "suite_id": data["suite"]["id"],
        "question_text": "Question ajoutée", "expected_answer": "Réponse",
        "document_id": data["documents"][0]["document_id"],
        "document_version_id": data["documents"][0]["document_version_id"],
    })
    assert any(method == "POST" and path == "/questions" for method, path, _, _ in calls)
    assert any(method == "POST" and path.endswith("/questions") for method, path, _, _ in calls)
    client.post(reverse("rag_laboratory"), {
        "action": "remove_question", "suite_id": data["suite"]["id"],
        "question_id": data["question"]["id"],
    })
    assert any(method == "DELETE" for method, _, _, _ in calls)


@override_settings(ALLOWED_HOSTS=["testserver"])
def test_laboratory_history_detail_never_runs_rag(monkeypatch):
    data, calls = laboratory_data(), []
    monkeypatch.setattr(views, "evaluation_api_request", fake_api(data, calls))
    response = Client().get(reverse("rag_laboratory"), {
        "tab": "history", "campaign": data["campaign"]["id"], "attempt": data["attempt"]["id"],
    })
    content = response.content.decode()
    assert response.status_code == 200
    assert "Comparaison septembre" in content
    assert "Le délai est de trente jours." in content
    assert "Passage contractuel." in content
    assert "bge-m3" in content and "top_k" in content
    assert not any(method == "POST" and path.endswith("/run") for method, path, _, _ in calls)


@override_settings(ALLOWED_HOSTS=["testserver"])
def test_laboratory_feedback_form_posts_to_existing_api(monkeypatch):
    data, calls = laboratory_data(), []
    monkeypatch.setattr(views, "evaluation_api_request", fake_api(data, calls))
    response = Client().post(reverse("rag_laboratory"), {
        "action": "feedback", "attempt_id": data["attempt"]["id"], "is_correct": "false",
        "rating": "2", "corrected_answer": "Réponse corrigée", "comment": "À revoir",
    })
    feedback = next(call for call in calls if call[0] == "POST" and call[1].endswith("/feedback"))
    assert feedback[2] == {"is_correct": False, "rating": 2, "corrected_answer": "Réponse corrigée", "comment": "À revoir"}
    assert "Feedback enregistré." in response.content.decode()


@override_settings(ALLOWED_HOSTS=["testserver"])
def test_laboratory_uses_plain_language_and_preserves_zero_values(monkeypatch):
    data, calls = laboratory_data(), []

    revision = data["configurations"][0]["revisions"][0]
    revision["values"] = {
        "model": "mistral",
        "model_label": "Mistral",
        "temperature": 0.0,
        "top_k": 5,
    }

    data["attempt"]["configuration"]["generation"]["temperature"] = 0.0

    monkeypatch.setattr(
        views,
        "evaluation_api_request",
        fake_api(data, calls),
    )

    response = Client().get(
        reverse("rag_laboratory"),
        {
            "suite": data["suite"]["id"],
            "attempt": data["attempt"]["id"],
        },
    )

    assert response.status_code == 200

    content = response.content.decode()

    # Navigation principale du nouveau Lab.
    for label in (
        "Composer",
        "Exécuter",
        "Inspecter",
        "Comparer",
        "Catalogue",
    ):
        assert label in content

    # Les anciennes fonctions restent accessibles dans la zone héritée.
    assert "Réglages de l’IA" in content

    # Les réglages historiques doivent toujours être rendus correctement.
    assert content.count('class="setting-item"') >= 3
    assert "Modèle IA<strong>Mistral</strong>" in content
    assert "Liberté de réponse<strong>Très faible (0,0)</strong>" in content
    assert "Passages consultés<strong>5</strong>" in content

    # Une valeur numérique nulle ne doit jamais disparaître à cause
    # d'un test de vérité implicite.
    assert "0.0" in content

    assert "Détails techniques" in content
    assert "Cette réponse n’a pas encore été évaluée." in content
    assert "data-feedback-form hidden" in content


def test_laboratory_tab_css_has_readable_interaction_states():
    css = Path("src/kaliok/ui/core_ui/static/core_ui/app.css").read_text(
        encoding="utf-8"
    )
    assert ".tab-button { color:#344054" in css
    assert ".tab-button:hover" in css
    assert ".tab-button:focus-visible" in css
    assert '.tab-button[aria-selected="true"]' in css
    assert "border-bottom-color:var(--brand)" in css


@override_settings(ALLOWED_HOSTS=["testserver"])
def test_ai_settings_show_accessible_advanced_details(monkeypatch):
    data, calls = laboratory_data(), []
    revision = data["configurations"][1]["revisions"][0]
    revision.update({
        "revision_number": 2,
        "status": "draft",
        "change_reason": "Comparaison Qwen avec davantage de contexte",
        "values": {
            "model": "qwen3:8b",
            "model_label": "Qwen 3 8B",
            "temperature": 0.2,
            "top_k": 8,
            "provider": "ollama",
            "provider_label": "Ollama",
            "builder": "ranked",
            "builder_label": "Contexte classé",
            "embedding_model": "bge-m3",
            "embedding_model_label": "BGE M3",
        },
    })
    monkeypatch.setattr(views, "evaluation_api_request", fake_api(data, calls))

    response = Client().get(reverse("rag_laboratory"), {"tab": "settings"})
    content = response.content.decode()

    assert response.status_code == 200
    assert "<summary>Réglages avancés</summary>" in content
    assert "<details open" not in content
    assert "Fournisseur IA" in content and "Ollama" in content
    assert "Préparation du contexte" in content and "Contexte classé" in content
    assert "Modèle de recherche documentaire" in content and "BGE M3" in content
    assert "lecture seule" in content.lower()
    assert "Comparaison Qwen avec davantage de contexte" in content
    assert "Qwen 3 8B" in content
    assert "Très faible (0,2)" in content
    assert "Utiliser ce réglage" in content
    assert "Les expériences déjà enregistrées conserveront leurs anciens réglages." in content


@override_settings(ALLOWED_HOSTS=["testserver"])
def test_ai_settings_group_versions_and_keep_details_compact(monkeypatch):
    data, calls = laboratory_data(), []
    active = data["configurations"][0]["revisions"][0]
    active["values"] = {"model": "mistral", "model_label": "Mistral", "temperature": 0.0, "top_k": 5}
    draft = data["configurations"][1]["revisions"][0]
    draft.update({"status": "draft", "values": {"model": "qwen3:8b", "model_label": "Qwen 3 8B", "temperature": 0.2, "top_k": 8}})
    retired = {**draft, "revision_id": str(uuid4()), "revision_number": 1, "status": "retired"}
    data["configurations"][1]["revisions"].append(retired)
    monkeypatch.setattr(views, "evaluation_api_request", fake_api(data, calls))

    content = Client().get(reverse("rag_laboratory"), {"tab": "settings"}).content.decode()

    assert "Réglage utilisé actuellement" in content
    assert "Versions en préparation" in content
    assert "Anciennes versions (1)" in content
    assert "<details class=\"retired-group\">" in content
    assert content.count("data-version-detail hidden") == 3
    assert content.count('aria-expanded="false"') == 3
    assert "Voir et modifier" in content and "Consulter" in content
    assert "Créer une nouvelle version à partir de celle-ci" in content
    assert "<details class=\"create-version\">" in content
    script = Path("src/kaliok/ui/core_ui/static/core_ui/rag_laboratory.js").read_text(encoding="utf-8")
    assert "versionRows.forEach" in script
    assert "otherDetail.hidden = true" in script


@override_settings(ALLOWED_HOSTS=["testserver"])
def test_identical_revision_requires_a_change_or_reason(monkeypatch):
    data, calls = laboratory_data(), []
    source = data["configurations"][0]["revisions"][0]
    source["values"] = {"model": "mistral", "temperature": 0.0, "top_k": 5}
    monkeypatch.setattr(views, "evaluation_api_request", fake_api(data, calls))

    response = Client().post(reverse("rag_laboratory"), {
        "action": "create_configuration_revision",
        "source_revision_id": source["revision_id"],
        "model": "mistral", "temperature": "0.0", "top_k": "5",
        "change_reason": "",
    })

    assert "Ces réglages sont identiques à la version source." in response.content.decode()
    assert not any(call[:2] == ("POST", "/configurations/revisions") for call in calls)


@override_settings(ALLOWED_HOSTS=["testserver"])
def test_identical_revision_with_reason_remains_allowed(monkeypatch):
    data, calls = laboratory_data(), []
    source = data["configurations"][0]["revisions"][0]
    source["values"] = {"model": "mistral", "temperature": 0.0, "top_k": 5}
    monkeypatch.setattr(views, "evaluation_api_request", fake_api(data, calls))

    Client().post(reverse("rag_laboratory"), {
        "action": "create_configuration_revision",
        "source_revision_id": source["revision_id"],
        "model": "mistral", "temperature": "0.0", "top_k": "5",
        "change_reason": "Nouveau jalon documenté",
    })

    assert any(call[:2] == ("POST", "/configurations/revisions") for call in calls)


def test_technical_rate_accepts_fraction_or_percentage():
    fraction = views._decorate_run_result(
        {"technical_success_rate": 1.0, "results": [], "by_configuration": []},
        [], [],
    )
    percentage = views._decorate_run_result(
        {"technical_success_rate": 100.0, "results": [], "by_configuration": []},
        [], [],
    )
    assert fraction["technical_success_rate_percent"] == 100
    assert percentage["technical_success_rate_percent"] == 100
