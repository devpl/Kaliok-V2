from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode
from uuid import UUID, uuid4

import httpx
from django.conf import settings
from django.http import Http404, HttpResponse, JsonResponse
from django.shortcuts import redirect, render
from django.urls import reverse

from .forms import DocumentUploadForm, RagQuestionForm


def evaluation_api_request(
    method: str,
    path: str,
    *,
    body: dict | None = None,
    timeout: float = 10.0,
) -> tuple[dict | list | None, str | None]:
    try:
        response = httpx.request(
            method,
            f"{settings.KALIOK_API_BASE_URL}/rag/evaluation{path}",
            json=body,
            timeout=timeout,
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as error:
        try:
            detail = error.response.json().get("detail")
        except (ValueError, AttributeError):
            detail = None
        return None, detail or "L’API a refusé la demande."
    except httpx.HTTPError:
        return None, "Le service d’évaluation RAG est indisponible."

    if response.status_code == 204:
        return {}, None
    try:
        return response.json(), None
    except ValueError:
        return None, "La réponse de l’API est invalide."


def _evaluation_lists() -> tuple[list, list, list, list[str]]:
    values = []
    errors = []
    for path, label in (
        ("/documents", "documents"),
        ("/suites", "suites"),
        ("/configurations", "configurations"),
        ("/campaigns", "historique"),
    ):
        payload, error = evaluation_api_request("GET", path)
        values.append(payload if isinstance(payload, list) else [])
        if error:
            errors.append(f"Impossible de charger {label} : {error}")
    return (*values, errors)


def _safe_int(value, default=1):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _decorate_run_result(result, suites, configurations):
    if not isinstance(result, dict):
        return result

    question_labels = {}
    for suite in suites:
        for question in suite.get("questions", []):
            question_labels[question.get("id")] = question

    revision_labels = {}
    for profile in configurations:
        profile_label = profile.get("label") or profile.get("profile_key") or "Profil"
        for revision in profile.get("revisions", []):
            revision_labels[revision.get("revision_id")] = (
                f"{profile_label} — Version {revision.get('revision_number')}"
            )

    success_rate = result.get("technical_success_rate")
    if isinstance(success_rate, (int, float)):
        result["technical_success_rate_percent"] = (
            success_rate * 100 if success_rate <= 1 else success_rate
        )

    for item in result.get("results", []):
        item["question"] = question_labels.get(item.get("question_id"), {})
        item["configuration_label"] = revision_labels.get(
            item.get("configuration_revision_id"),
            "Révision sélectionnée",
        )

    for item in result.get("by_configuration", []):
        item["configuration_label"] = revision_labels.get(
            item.get("configuration_revision_id"),
            "Révision sélectionnée",
        )
        rate = item.get("technical_success_rate")
        if isinstance(rate, (int, float)):
            item["technical_success_rate_percent"] = rate * 100 if rate <= 1 else rate

    return result


def _decorate_configurations(configurations):
    for profile in configurations:
        for revision in profile.get("revisions", []):
            values = revision.setdefault("values", {})
            for key in (
                "model", "model_label", "temperature", "top_k",
                "provider", "provider_label", "builder", "builder_label",
                "embedding_model", "embedding_model_label",
            ):
                values.setdefault(key, None)
            temperature = values["temperature"]
            if isinstance(temperature, (int, float)):
                # UX scale documented here while the exact stored value remains visible.
                if temperature <= 0.2:
                    freedom = "Très faible"
                elif temperature <= 0.5:
                    freedom = "Faible"
                elif temperature <= 0.8:
                    freedom = "Équilibrée"
                else:
                    freedom = "Libre"
                exact = format(temperature, ".1f").replace(".", ",")
                revision["temperature_display"] = f"{freedom} ({exact})"
            else:
                revision["temperature_display"] = "—"
            options = revision.setdefault("options", {})
            for key in ("model", "provider", "builder"):
                options.setdefault(key, [])
        profile["active_revisions"] = [
            item for item in profile.get("revisions", []) if item.get("status") == "active"
        ]
        profile["draft_revisions"] = [
            item for item in profile.get("revisions", []) if item.get("status") == "draft"
        ]
        profile["retired_revisions"] = [
            item for item in profile.get("revisions", []) if item.get("status") == "retired"
        ]
    return configurations


def _is_unjustified_identical_revision(configurations, source_id, submitted, reason):
    if reason.strip():
        return False
    source = next(
        (
            revision
            for profile in configurations
            for revision in profile.get("revisions", [])
            if str(revision.get("revision_id")) == str(source_id)
        ),
        None,
    )
    if source is None:
        return False
    current = source.get("values", {})

    def normalized(key, value):
        try:
            if key == "temperature":
                return float(value)
            if key == "top_k":
                return int(value)
        except (TypeError, ValueError):
            return value
        return value

    return all(
        normalized(key, value) == normalized(key, current.get(key))
        for key, value in submitted.items()
    )


def rag_laboratory(request):
    documents, suites, configurations, campaigns, load_errors = _evaluation_lists()
    configurations = _decorate_configurations(configurations)
    selected_suite_id = request.POST.get("suite_id") or request.GET.get("suite")
    selected_attempt_id = request.GET.get("attempt")
    selected_campaign_id = request.GET.get("campaign")
    active_tab = request.GET.get("tab", "experiment")
    notice = None
    action_error = None
    run_result = None
    attempt_detail = None
    campaign_attempts = []
    discovery_runs = []
    discovery_documents = []
    discovery_detail = None
    discovery_candidates = []
    selected_discovery_run_id = request.GET.get("discovery_run")
    selected_discovery_document_id = request.GET.get("discovery_document")
    discovery_limit = min(max(_safe_int(request.GET.get("discovery_limit"), 25), 1), 200)
    discovery_total = 0
    discovery_comparison = None
    selected_compare_a = request.GET.get("compare_a")
    selected_compare_b = request.GET.get("compare_b")
    entity_resolution_runs = []
    entity_resolution_detail = None
    resolved_entities = []
    entity_resolution_summaries = []
    selected_entity_resolution_run_id = request.GET.get("entity_resolution_run")
    entity_resolution_limit = min(max(_safe_int(request.GET.get("entity_resolution_limit"), 25), 1), 100)
    entity_resolution_total = 0

    if request.method == "POST":
        action = request.POST.get("action")
        if action == "create_suite":
            payload, action_error = evaluation_api_request(
                "POST", "/suites", body={
                    "name": request.POST.get("name", "").strip(),
                    "description": request.POST.get("description") or None,
                    "status": "active",
                }
            )
            if payload:
                selected_suite_id = payload["id"]
                return redirect(f"{reverse('rag_laboratory')}?suite={selected_suite_id}")
        elif action == "create_question":
            payload, action_error = evaluation_api_request(
                "POST", "/questions", body={
                    "question_text": request.POST.get("question_text", "").strip(),
                    "expected_answer": request.POST.get("expected_answer") or None,
                    "document_id": request.POST.get("document_id") or None,
                    "document_version_id": request.POST.get("document_version_id") or None,
                }
            )
            if payload and selected_suite_id:
                _, action_error = evaluation_api_request(
                    "POST", f"/suites/{selected_suite_id}/questions",
                    body={"question_id": payload["id"]},
                )
            if payload and not action_error:
                notice = "Question créée et ajoutée à la suite."
        elif action == "remove_question":
            _, action_error = evaluation_api_request(
                "DELETE",
                f"/suites/{selected_suite_id}/questions/{request.POST.get('question_id')}",
            )
            if not action_error:
                notice = "Question retirée de la suite."
        elif action == "run_campaign":
            revision_ids = request.POST.getlist("configuration_revision_ids")
            repetitions = _safe_int(request.POST.get("repetitions"), 1)
            campaign, action_error = evaluation_api_request(
                "POST", "/campaigns", body={
                    "name": request.POST.get("campaign_name", "").strip(),
                    "suite_id": selected_suite_id,
                    "configuration_revision_ids": revision_ids,
                    "repetitions": repetitions,
                }
            )
            if campaign and not action_error:
                run_result, action_error = evaluation_api_request(
                    "POST", f"/campaigns/{campaign['id']}/run", timeout=3600.0
                )
                active_tab = "results"
        elif action == "feedback":
            selected_attempt_id = request.POST.get("attempt_id")
            correctness = request.POST.get("is_correct")
            assessment = request.POST.get("assessment") or None
            feedback_id = request.POST.get("feedback_id")
            method = "PATCH" if feedback_id else "POST"
            path = f"/feedback/{feedback_id}" if feedback_id else f"/attempts/{selected_attempt_id}/feedback"
            feedback_body = {
                    "is_correct": {"correct": True, "incorrect": False}.get(assessment, {"true": True, "false": False}.get(correctness)),
                    "rating": _safe_int(request.POST.get("rating"), None),
                    "corrected_answer": request.POST.get("corrected_answer") or None,
                    "comment": request.POST.get("comment") or None,
            }
            if assessment:
                feedback_body["assessment"] = assessment
            _, action_error = evaluation_api_request(method, path, body=feedback_body)
            if not action_error:
                notice = "Feedback enregistré. Votre évaluation est à jour."
            active_tab = "history"
        elif action == "create_configuration_revision":
            values = {
                key: request.POST.get(key)
                for key in ("model", "temperature", "top_k", "provider", "builder")
                if request.POST.get(key) not in (None, "")
            }
            change_reason = request.POST.get("change_reason") or ""
            if _is_unjustified_identical_revision(
                configurations,
                request.POST.get("source_revision_id"),
                values,
                change_reason,
            ):
                action_error = (
                    "Ces réglages sont identiques à la version source. "
                    "Modifiez au moins un réglage ou indiquez un motif."
                )
            else:
                _, action_error = evaluation_api_request(
                    "POST", "/configurations/revisions", body={
                        "source_revision_id": request.POST.get("source_revision_id"),
                        "values": values,
                        "change_reason": change_reason or None,
                    }
                )
                if not action_error:
                    notice = "Nouvelle version créée. Elle reste en préparation."
            active_tab = "settings"
        elif action == "activate_configuration_revision":
            _, action_error = evaluation_api_request(
                "POST", f"/configurations/revisions/{request.POST.get('revision_id')}/activate"
            )
            if not action_error:
                notice = "Cette version est maintenant utilisée."
            active_tab = "settings"

        documents, suites, configurations, campaigns, refreshed_errors = _evaluation_lists()
        configurations = _decorate_configurations(configurations)
        load_errors.extend(refreshed_errors)

    selected_suite = None
    if selected_suite_id:
        selected_suite, suite_error = evaluation_api_request(
            "GET", f"/suites/{selected_suite_id}"
        )
        if suite_error:
            action_error = action_error or suite_error
    if selected_campaign_id:
        campaign_attempts, history_error = evaluation_api_request(
            "GET", f"/campaigns/{selected_campaign_id}/attempts"
        )
        if history_error:
            action_error = action_error or history_error
        active_tab = "history"
    if selected_attempt_id:
        attempt_detail, detail_error = evaluation_api_request(
            "GET", f"/attempts/{selected_attempt_id}"
        )
        if detail_error:
            action_error = action_error or detail_error
        active_tab = "history"

    if active_tab:
        runs_payload, discovery_error = evaluation_api_request(
            "GET", "/discovery/runs?limit=100"
        )
        discovery_runs = (
            runs_payload.get("items", []) if isinstance(runs_payload, dict) else []
        )
        for run in discovery_runs:
            for key in ("started_at", "completed_at"):
                if isinstance(run.get(key), str):
                    try:
                        run[key] = datetime.fromisoformat(
                            run[key].replace("Z", "+00:00")
                        )
                    except ValueError:
                        pass
        if discovery_error:
            action_error = action_error or discovery_error
        documents_by_version = {}
        for run in discovery_runs:
            version_id = str(run.get("document_version_id", ""))
            document = documents_by_version.setdefault(
                version_id,
                {
                    "document_version_id": version_id,
                    "filename": run.get("filename"),
                    "status": run.get("status"),
                    "generation_count": 0,
                    "run_id": str(run.get("run_id")),
                },
            )
            document["generation_count"] += 1
        discovery_documents = list(documents_by_version.values())
        if not selected_discovery_document_id and discovery_documents:
            selected_discovery_document_id = discovery_documents[0]["document_version_id"]
        document_runs = [
            run
            for run in discovery_runs
            if str(run.get("document_version_id")) == selected_discovery_document_id
        ]
        if selected_compare_a and selected_compare_b:
            discovery_comparison, comparison_error = evaluation_api_request(
                "GET", f"/discovery/compare?{urlencode({'run_a': selected_compare_a, 'run_b': selected_compare_b})}"
            )
            if comparison_error:
                action_error = action_error or comparison_error
        if not selected_discovery_run_id:
            selected = next(
                (run for run in document_runs if run.get("status") == "completed"),
                document_runs[0] if document_runs else None,
            )
            selected_discovery_run_id = str(selected["run_id"]) if selected else None
        if selected_discovery_run_id:
            discovery_detail, detail_error = evaluation_api_request(
                "GET", f"/discovery/runs/{selected_discovery_run_id}"
            )
            if detail_error:
                action_error = action_error or detail_error
            if isinstance(discovery_detail, dict):
                for key in ("started_at", "completed_at"):
                    if isinstance(discovery_detail.get(key), str):
                        try:
                            discovery_detail[key] = datetime.fromisoformat(
                                discovery_detail[key].replace("Z", "+00:00")
                            )
                        except ValueError:
                            pass
                discovery_detail["page_count"] = len(
                    {
                        page
                        for group in discovery_detail.get("groups", [])
                        for page in group.get("pages", [])
                    }
                )
                filters = {
                    "candidate_type": request.GET.get("candidate_type"),
                    "value": request.GET.get("value"),
                    "page": request.GET.get("page"),
                }
                if not any(filters.values()) and discovery_detail.get("groups"):
                    first_group = discovery_detail["groups"][0]
                    filters["candidate_type"] = first_group.get("candidate_type")
                    filters["value"] = first_group.get("normalized_value")
                query_values = {key: value for key, value in filters.items() if value}
                query_values["limit"] = discovery_limit
                query = urlencode(query_values)
                path = f"/discovery/runs/{selected_discovery_run_id}/candidates"
                candidates_payload, candidates_error = evaluation_api_request(
                    "GET", f"{path}?{query}" if query else path
                )
                discovery_candidates = (
                    candidates_payload.get("items", [])
                    if isinstance(candidates_payload, dict)
                    else []
                )
                discovery_total = (
                    int(candidates_payload.get("total", 0))
                    if isinstance(candidates_payload, dict)
                    else 0
                )
                if candidates_error:
                    action_error = action_error or candidates_error
                for candidate in discovery_candidates:
                    content = candidate.get("unit_content") or ""
                    start = candidate.get("start_offset")
                    end = candidate.get("end_offset")
                    if (
                        isinstance(start, int)
                        and isinstance(end, int)
                        and 0 <= start <= end <= len(content)
                    ):
                        candidate["highlight_before"] = content[:start]
                        candidate["highlight_text"] = content[start:end]
                        candidate["highlight_after"] = content[end:]
                    else:
                        candidate["highlight_before"] = content
                        candidate["highlight_text"] = ""
                        candidate["highlight_after"] = ""
                discovery_detail["selected_filters"] = filters
                detector = next(iter(discovery_detail.get("detectors", [])), {})
                discovery_detail["detector_label"] = (
                    f"{detector.get('key', 'Méthode inconnue')} v{detector.get('version', '—')}"
                )
                identity = discovery_detail.get("dictionary")
                discovery_detail["dictionary_label"] = (
                    f"{identity.get('name')} v{identity.get('version')}"
                    if isinstance(identity, dict) else "Dictionnaire historique / identité non enregistrée"
                )

    if active_tab:
        runs_payload, resolution_error = evaluation_api_request(
            "GET", "/entity-resolution/runs?limit=100"
        )
        entity_resolution_runs = runs_payload.get("items", []) if isinstance(runs_payload, dict) else []
        if resolution_error:
            action_error = action_error or resolution_error
        for run in entity_resolution_runs:
            run["strategy_label"] = run.get("engine_version") or (run.get("configuration") or {}).get("strategy") or "Stratégie non renseignée"
            for key in ("started_at", "completed_at"):
                if isinstance(run.get(key), str):
                    try:
                        run[key] = datetime.fromisoformat(run[key].replace("Z", "+00:00"))
                    except ValueError:
                        pass
        if not selected_entity_resolution_run_id and entity_resolution_runs:
            selected = next((item for item in entity_resolution_runs if item.get("status") == "completed"), entity_resolution_runs[0])
            selected_entity_resolution_run_id = str(selected["run_id"])
        if selected_entity_resolution_run_id:
            entity_resolution_detail, detail_error = evaluation_api_request(
                "GET", f"/entity-resolution/runs/{selected_entity_resolution_run_id}"
            )
            if detail_error:
                action_error = action_error or detail_error
            if isinstance(entity_resolution_detail, dict) and isinstance(entity_resolution_detail.get("run"), dict):
                selected_run = entity_resolution_detail["run"]
                selected_run["strategy_label"] = selected_run.get("engine_version") or (selected_run.get("configuration") or {}).get("strategy") or "Stratégie non renseignée"
            filters = {
                "entity_type": request.GET.get("entity_type"),
                "canonical_label": request.GET.get("canonical_label"),
                "limit": entity_resolution_limit,
            }
            path = f"/entity-resolution/runs/{selected_entity_resolution_run_id}/entities?{urlencode({key: value for key, value in filters.items() if value not in (None, '')})}"
            entities_payload, entities_error = evaluation_api_request("GET", path)
            if entities_error:
                action_error = action_error or entities_error
            summaries = entities_payload.get("items", []) if isinstance(entities_payload, dict) else []
            entity_resolution_summaries = summaries
            entity_resolution_total = int(entities_payload.get("total", 0)) if isinstance(entities_payload, dict) else 0
            selected_entity_id = request.GET.get("entity") or (str(summaries[0]["id"]) if summaries else None)
            for summary in [item for item in summaries if str(item["id"]) == selected_entity_id][:1]:
                detail, item_error = evaluation_api_request("GET", f"/entity-resolution/entities/{summary['id']}")
                if item_error:
                    action_error = action_error or item_error
                    continue
                if not isinstance(detail, dict):
                    continue
                detail["summary"] = summary
                detail["type_label"] = {
                    "organization_name": "Organisation",
                    "person_name": "Personne",
                    "place_name": "Lieu",
                    "organization": "Organisation",
                    "person": "Personne",
                    "place": "Lieu",
                }.get(summary.get("entity_type"), summary.get("entity_type"))
                for membership in detail.get("memberships", []):
                    for provenance in membership.get("provenance", []):
                        content = provenance.get("normalized_content_unit", {}).get("content") or ""
                        start, end = provenance.get("start_offset"), provenance.get("end_offset")
                        if isinstance(start, int) and isinstance(end, int) and 0 <= start <= end <= len(content):
                            provenance["highlight_before"] = content[:start]
                            provenance["highlight_text"] = content[start:end]
                            provenance["highlight_after"] = content[end:]
                        else:
                            provenance["highlight_before"] = content
                            provenance["highlight_text"] = ""
                            provenance["highlight_after"] = ""
                resolved_entities.append(detail)

    for collection in (documents, suites, campaigns):
        for item in collection:
            for key in ("created_at", "started_at", "completed_at"):
                if isinstance(item.get(key), str):
                    try:
                        item[key] = datetime.fromisoformat(item[key].replace("Z", "+00:00"))
                    except ValueError:
                        pass

    run_result = _decorate_run_result(
        run_result, [selected_suite] if selected_suite else [], configurations
    )
    suite_names = {item.get("id"): item.get("name") for item in suites}
    for campaign in campaigns:
        campaign["suite_name"] = suite_names.get(
            campaign.get("suite_id"), "Suite non disponible"
        )
        campaign["question_count"] = next(
            (len(item.get("questions", [])) for item in suites if item.get("id") == campaign.get("suite_id")),
            None,
        )
    campaigns.sort(key=lambda item: item.get("created_at", ""), reverse=True)
    return render(request, "core_ui/rag_laboratory.html", {
        "documents": documents,
        "suites": suites,
        "configurations": configurations,
        "campaigns": campaigns,
        "selected_suite": selected_suite,
        "selected_suite_id": selected_suite_id,
        "selected_campaign_id": selected_campaign_id,
        "campaign_attempts": campaign_attempts or [],
        "attempt_detail": attempt_detail,
        "run_result": run_result,
        "active_tab": active_tab,
        "notice": notice,
        "api_errors": load_errors,
        "action_error": action_error,
        "discovery_runs": discovery_runs,
        "discovery_documents": discovery_documents,
        "discovery_detail": discovery_detail,
        "discovery_candidates": discovery_candidates,
        "selected_discovery_run_id": selected_discovery_run_id,
        "selected_discovery_document_id": selected_discovery_document_id,
        "discovery_limit": discovery_limit,
        "discovery_total": discovery_total,
        "discovery_has_more": len(discovery_candidates) < discovery_total,
        "discovery_next_limit": min(discovery_limit + 25, 200),
        "discovery_comparison": discovery_comparison,
        "selected_compare_a": selected_compare_a,
        "selected_compare_b": selected_compare_b,
        "discovery_document_runs": document_runs,
        "entity_resolution_runs": entity_resolution_runs,
        "entity_resolution_detail": entity_resolution_detail,
        "resolved_entities": resolved_entities,
        "entity_resolution_summaries": entity_resolution_summaries,
        "selected_entity_resolution_run_id": selected_entity_resolution_run_id,
        "entity_resolution_total": entity_resolution_total,
        "entity_resolution_has_more": len(resolved_entities) < entity_resolution_total,
        "entity_resolution_next_limit": min(entity_resolution_limit + 25, 100),
    })


def rag_laboratory_data(request):
    """Return one asynchronous laboratory slice without changing API contracts."""
    kind = request.GET.get("kind")
    if kind == "discovery":
        run_id = request.GET.get("run")
        if not run_id:
            return JsonResponse({"error": "Analyse de découverte manquante."}, status=400)
        values = {
            "candidate_type": request.GET.get("candidate_type"),
            "value": request.GET.get("value"),
            "page": request.GET.get("page"),
            "limit": min(max(_safe_int(request.GET.get("limit"), 25), 1), 200),
            "offset": max(_safe_int(request.GET.get("offset"), 0), 0),
        }
        query = urlencode({key: value for key, value in values.items() if value not in (None, "")})
        payload, error = evaluation_api_request(
            "GET", f"/discovery/runs/{run_id}/candidates?{query}"
        )
        if error:
            return JsonResponse({"error": error}, status=502)
        candidates = payload if isinstance(payload, dict) else {}
        for candidate in candidates.get("items", []):
            content = candidate.get("unit_content") or ""
            start, end = candidate.get("start_offset"), candidate.get("end_offset")
            if isinstance(start, int) and isinstance(end, int) and 0 <= start <= end <= len(content):
                candidate["highlight_before"] = content[:start]
                candidate["highlight_text"] = content[start:end]
                candidate["highlight_after"] = content[end:]
            else:
                candidate["highlight_before"], candidate["highlight_text"], candidate["highlight_after"] = content, "", ""
        return JsonResponse({"items": candidates.get("items", []), "total": candidates.get("total", 0)})

    if kind == "entity_resolution":
        run_id = request.GET.get("run")
        if not run_id:
            return JsonResponse({"error": "Génération de résolution manquante."}, status=400)
        params = {
            "entity_type": request.GET.get("entity_type"),
            "canonical_label": request.GET.get("canonical_label"),
            "limit": min(max(_safe_int(request.GET.get("limit"), 25), 1), 100),
            "offset": max(_safe_int(request.GET.get("offset"), 0), 0),
        }
        query = urlencode({key: value for key, value in params.items() if value not in (None, "")})
        summaries, error = evaluation_api_request(
            "GET", f"/entity-resolution/runs/{run_id}/entities?{query}"
        )
        if error:
            return JsonResponse({"error": error}, status=502)
        result = summaries if isinstance(summaries, dict) else {}
        selected_entity = request.GET.get("entity")
        detail = None
        if selected_entity:
            detail, error = evaluation_api_request(
                "GET", f"/entity-resolution/entities/{selected_entity}"
            )
            if error:
                return JsonResponse({"error": error}, status=502)
            if isinstance(detail, dict):
                for membership in detail.get("memberships", []):
                    for provenance in membership.get("provenance", []):
                        content = provenance.get("normalized_content_unit", {}).get("content") or ""
                        start, end = provenance.get("start_offset"), provenance.get("end_offset")
                        if isinstance(start, int) and isinstance(end, int) and 0 <= start <= end <= len(content):
                            provenance["highlight_before"] = content[:start]
                            provenance["highlight_text"] = content[start:end]
                            provenance["highlight_after"] = content[end:]
                        else:
                            provenance["highlight_before"], provenance["highlight_text"], provenance["highlight_after"] = content, "", ""
        return JsonResponse({"items": result.get("items", []), "total": result.get("total", 0), "detail": detail})

    return JsonResponse({"error": "Type de données inconnu."}, status=400)


def get_api_status() -> dict[str, str]:
    try:
        response = httpx.get(
            f"{settings.KALIOK_API_BASE_URL}/health",
            timeout=2.0,
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError:
        return {
            "status": "error",
            "service": "kaliok-api",
        }


def get_api_documents() -> list[dict] | None:
    try:
        response = httpx.get(
            f"{settings.KALIOK_API_BASE_URL}/documents",
            timeout=5.0,
        )
        response.raise_for_status()
    except httpx.HTTPError:
        return None

    try:
        payload = response.json()
    except ValueError:
        return None

    if not isinstance(payload, list):
        return None

    return payload


def get_api_document(document_id: UUID) -> dict | None:
    try:
        response = httpx.get(
            f"{settings.KALIOK_API_BASE_URL}/documents/{document_id}",
            timeout=5.0,
        )
    except httpx.RequestError:
        return None

    if response.status_code == 404:
        raise Http404("Document introuvable")

    try:
        response.raise_for_status()
    except httpx.HTTPError:
        return None

    try:
        payload = response.json()
    except ValueError:
        return None

    if not isinstance(payload, dict):
        return None

    return payload


def ingest_api_document(
    *,
    path: Path,
    original_name: str,
    size: int,
) -> dict | None:
    body = {
        "source": {
            "name": original_name,
            "uri": path.resolve().as_uri(),
            "media_type": "text/plain",
            "size": size,
        }
    }

    try:
        response = httpx.post(
            f"{settings.KALIOK_API_BASE_URL}/ingestion",
            json=body,
            timeout=30.0,
        )
        response.raise_for_status()
    except httpx.HTTPError:
        return None

    try:
        payload = response.json()
    except ValueError:
        return None

    if not isinstance(payload, dict):
        return None

    return payload


def ask_api_rag(
    *,
    document_id: UUID,
    question: str,
) -> dict | None:
    body = {
        "document_id": str(document_id),
        "question": question,
        "top_k": 5,
    }

    try:
        response = httpx.post(
            f"{settings.KALIOK_API_BASE_URL}/rag/answer",
            json=body,
            timeout=300.0,
        )
        response.raise_for_status()
    except httpx.HTTPError:
        return None

    try:
        payload = response.json()
    except ValueError:
        return None

    if not isinstance(payload, dict):
        return None

    return payload


def home(request):
    documents = get_api_documents()

    if documents is None:
        documents = []

    return render(
        request,
        "core_ui/home.html",
        {
            "documents": documents,
            "api_status": get_api_status(),
            "upload_form": DocumentUploadForm(),
        },
    )


def upload_document(request):
    if request.method != "POST":
        return redirect("home")

    form = DocumentUploadForm(request.POST, request.FILES)

    if not form.is_valid():
        documents = get_api_documents()

        if documents is None:
            documents = []

        return render(
            request,
            "core_ui/home.html",
            {
                "documents": documents,
                "api_status": get_api_status(),
                "upload_form": form,
            },
            status=400,
        )

    uploaded_file = form.cleaned_data["file"]

    upload_dir = Path(settings.KALIOK_UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)

    original_name = Path(uploaded_file.name).name
    stored_name = f"{uuid4()}_{original_name}"
    stored_path = upload_dir / stored_name

    try:
        with stored_path.open("wb") as destination:
            for chunk in uploaded_file.chunks():
                destination.write(chunk)

        result = ingest_api_document(
            path=stored_path,
            original_name=original_name,
            size=uploaded_file.size,
        )

        if result is None:
            stored_path.unlink(missing_ok=True)
            return HttpResponse(
                "L'ingestion par l'API technique a échoué.",
                status=503,
                content_type="text/plain; charset=utf-8",
            )

        document_id = result.get("document_id")
        if document_id is None:
            stored_path.unlink(missing_ok=True)
            return HttpResponse(
                "Réponse d'ingestion invalide.",
                status=502,
                content_type="text/plain; charset=utf-8",
            )

        return redirect(
            "document_detail",
            document_id=document_id,
        )
    except OSError:
        stored_path.unlink(missing_ok=True)
        return HttpResponse(
            "Impossible d'enregistrer le fichier envoyé.",
            status=500,
            content_type="text/plain; charset=utf-8",
        )


def document_detail(request, document_id: UUID):
    document = get_api_document(document_id)

    if document is None:
        return HttpResponse(
            "API technique indisponible",
            status=503,
            content_type="text/plain; charset=utf-8",
        )

    rag_answer = None
    rag_error = None

    if request.method == "POST":
        rag_form = RagQuestionForm(request.POST)

        if rag_form.is_valid():
            rag_answer = ask_api_rag(
                document_id=document_id,
                question=rag_form.cleaned_data["question"],
            )

            if rag_answer is None:
                rag_error = (
                    "Le service RAG n'a pas pu répondre à la question."
                )
    else:
        rag_form = RagQuestionForm()

    return render(
        request,
        "core_ui/document_detail.html",
        {
            "document": document,
            "current_version": document.get("current_version"),
            "versions": document.get("versions", []),
            "rag_form": rag_form,
            "rag_answer": rag_answer,
            "rag_error": rag_error,
        },
    )
