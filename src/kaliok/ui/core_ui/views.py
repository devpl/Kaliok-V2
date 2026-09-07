from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from uuid import UUID, uuid4

import httpx
from django.conf import settings
from django.http import Http404, HttpResponse, JsonResponse
from django.shortcuts import redirect, render
from django.urls import reverse
from sqlalchemy import func
from sqlmodel import Session, select

from kaliok.discovery import CandidateDiscoveryReadService
from kaliok.execution import ExecutionContext
from kaliok.normalization.comparison import ContentNormalizationComparisonService
from kaliok.pipeline import (
    ComponentBinding,
    ManifestExecutionService,
    PipelineManifest,
    build_current_production_manifest,
    build_kaliok_component_registry,
    build_kaliok_runtime_registry,
)
from kaliok.storage.database import create_database_engine
from kaliok.storage.models import (
    ContentBlock,
    DiscoveredCandidate,
    Document,
    DocumentVersion,
    NormalizedContentUnit,
    NormalizedContentUnitSource,
    Page,
    ProcessingRun,
)

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


def _pipeline_a_manifest(selection=None, registry=None) -> PipelineManifest:
    """Build Pipeline_A from the current Web selection.

    The default deliberately remains the current production subset. A Web
    selection is grouped by component identity so one multi-capability
    component becomes one manifest binding instead of several fake ones.
    """
    production = build_current_production_manifest()
    if selection in (None, ""):
        return PipelineManifest(
            pipeline_key="pipeline-a",
            revision="experiment-v1",
            bindings=production.bindings,
        )
    if isinstance(selection, dict):
        selection = selection.get("bindings", [])
    if not isinstance(selection, (list, tuple)):
        raise ValueError("La sélection Pipeline_A doit être une liste de bindings.")

    grouped = {}
    order = []
    for item in selection:
        if not isinstance(item, dict):
            raise ValueError("Chaque binding Pipeline_A doit être un objet JSON.")
        component_key = str(item.get("component_key", "")).strip()
        component_version = str(item.get("component_version", "")).strip()
        if not component_key or not component_version:
            raise ValueError("Un binding Pipeline_A doit préciser son composant et sa version.")
        identity = (component_key, component_version)
        definition = registry.get(*identity) if registry else None
        capabilities = item.get("capabilities") or (definition.provides if definition else ())
        if not isinstance(capabilities, (list, tuple)):
            raise ValueError("Les capabilities d'un binding doivent être une liste.")
        if identity not in grouped:
            grouped[identity] = {
                "binding_key": str(item.get("binding_key") or "").strip(),
                "component_key": component_key,
                "component_version": component_version,
                "capabilities": [],
                "configuration": item.get("configuration") or {},
                "dependencies": list(item.get("dependencies") or []),
                "enabled": bool(item.get("enabled", True)),
            }
            order.append(identity)
        target = grouped[identity]
        for capability in capabilities:
            capability = str(capability).strip()
            if capability and capability not in target["capabilities"]:
                target["capabilities"].append(capability)
        target["enabled"] = target["enabled"] and bool(item.get("enabled", True))
        if item.get("configuration") is not None:
            target["configuration"] = item["configuration"]

    bindings = []
    used_keys = set()
    binding_keys = {}
    for index, identity in enumerate(order, start=1):
        candidate = grouped[identity]["binding_key"] or f"binding-{index}"
        if candidate in used_keys:
            candidate = f"{candidate}-{index}"
        used_keys.add(candidate)
        binding_keys[identity] = candidate
    for index, identity in enumerate(order, start=1):
        item = grouped[identity]
        binding_key = binding_keys[identity]
        dependencies = list(item["dependencies"])
        if "normalization" in item["capabilities"] and not dependencies:
            for previous in order[:index - 1]:
                if "document_extraction" in grouped[previous]["capabilities"]:
                    dependency = binding_keys[previous]
                    if dependency != binding_key:
                        dependencies.append(dependency)
                    break
        bindings.append(ComponentBinding(binding_key=binding_key, **{
            key: item[key]
            for key in ("component_key", "component_version", "capabilities", "configuration", "enabled")
        }, dependencies=dependencies))

    manifest = PipelineManifest(
        pipeline_key="pipeline-a",
        revision="experiment-v1",
        bindings=tuple(bindings),
    )
    if registry is not None:
        manifest.validate(registry)
    return manifest


def _iso(value):
    return value.isoformat() if isinstance(value, datetime) else value


def _duration_ms(started_at, completed_at):
    if not started_at or not completed_at:
        return None
    duration = (completed_at - started_at).total_seconds() * 1000
    if duration <= 0:
        return None
    return round(duration, 3)


def _pipeline_manifest_payload(manifest, registry):
    bindings = []
    for binding in manifest.bindings:
        definition = registry.get(binding.component_key, binding.component_version)
        bindings.append({
            **binding.to_dict(),
            "definition": definition.to_dict() if definition else None,
        })
    return {
        **manifest.to_dict(),
        "manifest_hash": manifest.manifest_hash,
        "bindings": bindings,
        "description": "Pipeline de production — description actuellement partielle"
        if manifest.pipeline_key == "pipeline-p"
        else "Variante expérimentale construite depuis le sous-ensemble réel de Pipeline_P.",
    }


def _pipeline_components_payload(registry, runtime_registry):
    payload = []
    for definition in registry.definitions:
        item = definition.to_dict()
        executable = runtime_registry.has(*definition.identity)
        item["runtime_status"] = "EXECUTABLE" if executable else "CONNU — NON RACCORDÉ"
        item["runtime_executable"] = executable
        payload.append(item)
    return payload


def _pipeline_capabilities_payload(registry, runtime_registry, manifest):
    selected_by_capability = {}
    for binding in manifest.bindings:
        for capability in binding.capabilities:
            selected_by_capability[capability] = binding
    capabilities = []
    for capability in registry.capabilities:
        definitions = registry.for_capability(capability)
        selected = selected_by_capability.get(capability)
        if selected is None:
            executable = any(
                runtime_registry.has(*definition.identity)
                for definition in definitions
            )
            status = (
                "EXÉCUTABLE — NON SÉLECTIONNÉE"
                if executable
                else "DISPONIBLE — NON SÉLECTIONNÉE"
            )
        elif runtime_registry.has(selected.component_key, selected.component_version):
            status = "EXÉCUTABLE"
        else:
            status = "CONNU — NON RACCORDÉ"
        capabilities.append({
            "key": capability,
            "status": status,
            "selected_binding_key": selected.binding_key if selected else None,
            "selected_component": selected.to_dict() if selected else None,
            "components": [
                {
                    **definition.to_dict(),
                    "runtime_status": "EXECUTABLE" if runtime_registry.has(*definition.identity) else "CONNU — NON RACCORDÉ",
                    "runtime_executable": runtime_registry.has(*definition.identity),
                }
                for definition in definitions
            ],
        })
    return capabilities


def _lab_document_payload(version, document):
    return {
        "document_id": str(document.id),
        "document_version_id": str(version.id),
        "title": document.title or version.filename,
        "filename": version.filename,
        "version_number": version.version_number,
        "processing_status": version.processing_status,
        "version_status": version.version_status,
        "page_count": version.page_count,
        "file_hash_short": (version.file_hash or "")[:12],
        # This is the same precondition enforced by store_document_perception.
        "executable": version.page_count is not None,
    }


def _run_pipeline_metadata(run):
    configuration = run.configuration or {}
    return configuration.get("pipeline") if isinstance(configuration, dict) else None


def _count_run_artifacts(session, run):
    if run.process_type == "document_extraction":
        return int(session.exec(
            select(func.count(ContentBlock.id))
            .join(Page, ContentBlock.page_id == Page.id)
            .where(ContentBlock.processing_run_id == run.id)
        ).one())
    if run.process_type == "content_normalization":
        return int(session.exec(
            select(func.count(NormalizedContentUnit.id)).where(
                NormalizedContentUnit.processing_run_id == run.id
            )
        ).one())
    if run.process_type == "candidate_discovery":
        return int(session.exec(
            select(func.count(DiscoveredCandidate.id)).where(
                DiscoveredCandidate.processing_run_id == run.id
            )
        ).one())
    return None


def _run_payload(session, run, *, include_configuration=True):
    return {
        "id": str(run.id),
        "process_type": run.process_type,
        "status": run.status,
        "engine": run.engine,
        "engine_version": run.engine_version,
        "execution_environment": run.execution_environment,
        "execution_group_id": str(run.execution_group_id) if run.execution_group_id else None,
        "started_at": _iso(run.started_at),
        "completed_at": _iso(run.completed_at),
        "duration_ms": _duration_ms(run.started_at, run.completed_at),
        "metrics": run.metrics or {},
        "artifact_count": _count_run_artifacts(session, run),
        "configuration": run.configuration or {} if include_configuration else None,
        "error": run.error_message,
        "pipeline_metadata": _run_pipeline_metadata(run),
    }


def _pipeline_history(session, version_id, limit=20):
    runs = list(session.exec(
        select(ProcessingRun)
        .where(
            ProcessingRun.document_version_id == version_id,
            ProcessingRun.execution_environment == "experiment",
            ProcessingRun.execution_group_id.is_not(None),
        )
        .order_by(ProcessingRun.started_at.desc())
    ).all())
    grouped = {}
    for run in runs:
        group_id = str(run.execution_group_id)
        group = grouped.setdefault(group_id, {"execution_group_id": group_id, "runs": []})
        group["runs"].append(run)
    history = []
    for group in list(grouped.values())[:limit]:
        group["runs"].sort(key=lambda item: item.started_at or datetime.min.replace(tzinfo=timezone.utc))
        payloads = [_run_payload(session, run, include_configuration=False) for run in group["runs"]]
        started = [run.started_at for run in group["runs"] if run.started_at]
        completed = [run.completed_at for run in group["runs"] if run.completed_at]
        metadata = next((item.get("pipeline_metadata") for item in payloads if item.get("pipeline_metadata")), {}) or {}
        history.append({
            "execution_group_id": group["execution_group_id"],
            "pipeline_key": metadata.get("pipeline_key", "pipeline-a"),
            "revision": metadata.get("revision", "experiment-v1"),
            "status": "failed" if any(item["status"] == "failed" for item in payloads) else "completed",
            "started_at": _iso(min(started)) if started else None,
            "completed_at": _iso(max(completed)) if completed else None,
            "duration_ms": _duration_ms(min(started), max(completed)) if started and completed else None,
            "perception": next((item for item in payloads if item["process_type"] == "document_extraction"), None),
            "normalization": next((item for item in payloads if item["process_type"] == "content_normalization"), None),
            "discovery": next((item for item in payloads if item["process_type"] == "candidate_discovery"), None),
        })
    return history


def _pipeline_inspection(session, version_id, run, kind, offset, limit):
    if kind == "perception":
        rows = session.exec(
            select(ContentBlock, Page)
            .join(Page, ContentBlock.page_id == Page.id)
            .where(
                Page.document_version_id == version_id,
                ContentBlock.processing_run_id == run.id,
            )
            .order_by(Page.page_number, ContentBlock.reading_order, ContentBlock.block_index)
            .offset(offset)
            .limit(limit)
        ).all()
        total = int(session.exec(
            select(func.count(ContentBlock.id))
            .join(Page, ContentBlock.page_id == Page.id)
            .where(Page.document_version_id == version_id, ContentBlock.processing_run_id == run.id)
        ).one())
        return {
            "kind": kind,
            "items": [{
                "id": str(block.id),
                "page": page.page_number,
                "type": block.block_type,
                "method": block.extraction_method,
                "engine": block.extraction_engine,
                "engine_version": block.extraction_engine_version,
                "content": block.content,
                "confidence": block.confidence,
                "bbox": block.bbox,
            } for block, page in rows],
            "total": total,
            "offset": offset,
            "limit": limit,
        }
    if kind == "normalization":
        units = list(session.exec(
            select(NormalizedContentUnit)
            .where(
                NormalizedContentUnit.document_version_id == version_id,
                NormalizedContentUnit.processing_run_id == run.id,
            )
            .order_by(NormalizedContentUnit.unit_index)
            .offset(offset)
            .limit(limit)
        ).all())
        total = int(session.exec(
            select(func.count(NormalizedContentUnit.id)).where(
                NormalizedContentUnit.document_version_id == version_id,
                NormalizedContentUnit.processing_run_id == run.id,
            )
        ).one())
        unit_ids = [unit.id for unit in units]
        sources = list(session.exec(
            select(NormalizedContentUnitSource, ContentBlock, Page)
            .join(ContentBlock, NormalizedContentUnitSource.content_block_id == ContentBlock.id)
            .join(Page, ContentBlock.page_id == Page.id)
            .where(NormalizedContentUnitSource.normalized_content_unit_id.in_(unit_ids))
            .order_by(NormalizedContentUnitSource.source_order)
        ).all()) if unit_ids else []
        sources_by_unit = {}
        for source, block, page in sources:
            sources_by_unit.setdefault(str(source.normalized_content_unit_id), []).append({
                "block_id": str(block.id),
                "page": page.page_number,
                "content": block.content,
                "source_order": source.source_order,
            })
        return {
            "kind": kind,
            "items": [{
                "id": str(unit.id),
                "order": unit.unit_index,
                "content_type": unit.content_type,
                "content": unit.content,
                "source_unit_id": unit.source_unit_id,
                "sources": sources_by_unit.get(str(unit.id), []),
            } for unit in units],
            "total": total,
            "offset": offset,
            "limit": limit,
        }
    if kind == "discovery":
        payload = CandidateDiscoveryReadService(session).list_candidates(
            run.id,
            limit=limit,
            offset=offset,
        )
        if payload is None:
            return None
        items = []
        for item in payload["items"]:
            item = {
                key: str(value) if isinstance(value, UUID) else value
                for key, value in item.items()
            }
            content = item.get("unit_content") or ""
            start, end = item.get("start_offset"), item.get("end_offset")
            if isinstance(start, int) and isinstance(end, int) and 0 <= start <= end <= len(content):
                item["highlight_before"] = content[:start]
                item["highlight_text"] = content[start:end]
                item["highlight_after"] = content[end:]
            items.append(item)
        return {
            "kind": kind,
            "items": items,
            "total": payload["total"],
            "offset": payload["offset"],
            "limit": payload["limit"],
        }
    return None


def _pipeline_lab_state(session, *, version_id=None, group_id=None, inspect=None, offset=0, limit=25, selection=None):
    rows = session.exec(
        select(DocumentVersion, Document)
        .join(Document, Document.id == DocumentVersion.document_id)
        .where(Document.status != "deleted")
        .order_by(DocumentVersion.created_at.desc(), DocumentVersion.version_number.desc())
    ).all()
    documents = [_lab_document_payload(version, document) for version, document in rows]
    if version_id is None:
        selected = next((item for item in documents if item["executable"]), None)
    else:
        selected = next((item for item in documents if item["document_version_id"] == str(version_id)), None)
    registry = build_kaliok_component_registry()
    runtime_registry = build_kaliok_runtime_registry()
    production = build_current_production_manifest()
    experimental = _pipeline_a_manifest(selection, registry)
    result = None
    if selected:
        version_id = UUID(selected["document_version_id"])
        runs = list(session.exec(
            select(ProcessingRun)
            .where(ProcessingRun.document_version_id == version_id)
            .order_by(ProcessingRun.started_at.desc())
        ).all())
        latest = {}
        for run in runs:
            latest.setdefault(run.process_type, run)
        stages = []
        process_types = {
            "document_extraction": "document_extraction",
            "normalization": "content_normalization",
            "entity_discovery": "candidate_discovery",
        }
        selected_by_capability = {
            capability: binding
            for binding in experimental.bindings
            for capability in binding.capabilities
        }
        for capability in registry.capabilities:
            binding = selected_by_capability.get(capability)
            component = binding.to_dict() if binding else None
            if binding is None:
                status = (
                    "EXÉCUTABLE — NON SÉLECTIONNÉE"
                    if any(runtime_registry.has(*definition.identity) for definition in registry.for_capability(capability))
                    else "DISPONIBLE — NON SÉLECTIONNÉE"
                )
            elif runtime_registry.has(binding.component_key, binding.component_version):
                status = "EXÉCUTABLE"
            else:
                status = "CONNU — NON RACCORDÉ"
            process_type = process_types.get(capability)
            run = latest.get(process_type) if process_type else None
            stages.append({
                "key": capability,
                "capability": capability,
                "component": component,
                "status": status,
                "last_run": _run_payload(session, run, include_configuration=False) if run else None,
            })
        if group_id:
            group_uuid = UUID(str(group_id))
            group_runs = list(session.exec(
                select(ProcessingRun).where(
                    ProcessingRun.document_version_id == version_id,
                    ProcessingRun.execution_environment == "experiment",
                    ProcessingRun.execution_group_id == group_uuid,
                ).order_by(ProcessingRun.started_at)
            ).all())
            if group_runs:
                run_payloads = [_run_payload(session, run) for run in group_runs]
                started = [run.started_at for run in group_runs if run.started_at]
                completed = [run.completed_at for run in group_runs if run.completed_at]
                result = {
                    "execution_group_id": str(group_uuid),
                    "status": "failed" if any(item["status"] == "failed" for item in run_payloads) else "completed",
                    "document": selected,
                    "duration_ms": _duration_ms(min(started), max(completed)) if started and completed else None,
                    "perception": next((item for item in run_payloads if item["process_type"] == "document_extraction"), None),
                    "normalization": next((item for item in run_payloads if item["process_type"] == "content_normalization"), None),
                    "discovery": next((item for item in run_payloads if item["process_type"] == "candidate_discovery"), None),
                }
                normalization_run = next((run for run in group_runs if run.process_type == "content_normalization" and run.status == "completed"), None)
                production_run = session.exec(
                    select(ProcessingRun)
                    .where(
                        ProcessingRun.document_version_id == version_id,
                        ProcessingRun.process_type == "content_normalization",
                        ProcessingRun.execution_environment == "production",
                        ProcessingRun.status == "completed",
                    )
                    .order_by(ProcessingRun.started_at.desc())
                ).first()
                if normalization_run and production_run:
                    comparison = ContentNormalizationComparisonService(session).compare(
                        production_run.id,
                        normalization_run.id,
                    )
                    result["comparison"] = {
                        "available": True,
                        "run_p": comparison["run_p"]["metrics"],
                        "run_a": comparison["run_a"]["metrics"],
                        "delta": comparison["delta"],
                    }
                else:
                    result["comparison"] = {
                        "available": False,
                        "message": "Aucune référence P comparable disponible.",
                    }
                if inspect in {"perception", "normalization", "discovery"}:
                    target = result.get(inspect)
                    if target:
                        target["inspection"] = _pipeline_inspection(
                            session,
                            version_id,
                            session.get(ProcessingRun, UUID(target["id"])),
                            inspect,
                            offset,
                            limit,
                        )
        history = _pipeline_history(session, version_id)
    else:
        selected_by_capability = {
            capability: binding
            for binding in experimental.bindings
            for capability in binding.capabilities
        }
        stages = []
        for capability in registry.capabilities:
            binding = selected_by_capability.get(capability)
            if binding is None:
                status = (
                    "EXÉCUTABLE — NON SÉLECTIONNÉE"
                    if any(runtime_registry.has(*definition.identity) for definition in registry.for_capability(capability))
                    else "DISPONIBLE — NON SÉLECTIONNÉE"
                )
            elif runtime_registry.has(binding.component_key, binding.component_version):
                status = "EXÉCUTABLE"
            else:
                status = "CONNU — NON RACCORDÉ"
            stages.append({
                "key": capability,
                "capability": capability,
                "component": binding.to_dict() if binding else None,
                "status": status,
                "last_run": None,
            })
        history = []
    return {
        "documents": documents,
        "selected_document": selected,
        "pipeline_reference": _pipeline_manifest_payload(production, registry),
        "pipeline_experiment": _pipeline_manifest_payload(experimental, registry),
        "pipeline_selection": [binding.to_dict() for binding in experimental.bindings],
        "components": _pipeline_components_payload(registry, runtime_registry),
        "capabilities": _pipeline_capabilities_payload(registry, runtime_registry, experimental),
        "stages": stages,
        "history": history,
        "result": result,
    }


def _pipeline_error_payload(message):
    return {"error": "Le Lab ne peut pas charger les données de pipeline.", "technical_error": str(message)}


def _pipeline_request_payload(request):
    try:
        import json
        payload = json.loads(request.body or b"{}")
    except (TypeError, ValueError):
        payload = request.POST
    return payload


def _pipeline_selection_from_payload(payload):
    selection = payload.get("selection", payload.get("bindings"))
    if isinstance(selection, str):
        import json
        try:
            selection = json.loads(selection)
        except ValueError as error:
            raise ValueError("La sélection Pipeline_A n'est pas un JSON valide.") from error
    return selection


def rag_laboratory_pipeline(request):
    """Read or execute the real manifest-driven experimental pipeline."""
    if request.method == "GET":
        payload = request.GET
    elif request.method == "POST":
        payload = _pipeline_request_payload(request)
    else:
        return JsonResponse({"error": "Méthode non autorisée."}, status=405)
    try:
        version_id = UUID(str(payload.get("document_version_id"))) if payload.get("document_version_id") else None
        group_id = payload.get("execution_group_id")
        inspect = payload.get("inspect")
        selection = _pipeline_selection_from_payload(payload)
        offset = max(_safe_int(payload.get("offset"), 0), 0)
        limit = min(max(_safe_int(payload.get("limit"), 25), 1), 100)
    except (ValueError, TypeError):
        return JsonResponse({"error": "Document ou exécution invalide."}, status=400)
    if request.method == "GET":
        try:
            with Session(create_database_engine()) as session:
                return JsonResponse(_pipeline_lab_state(session, version_id=version_id, group_id=group_id, inspect=inspect, offset=offset, limit=limit, selection=selection))
        except Exception as error:
            return JsonResponse(_pipeline_error_payload(error), status=503)

    if payload.get("action") == "configure":
        try:
            with Session(create_database_engine()) as session:
                state = _pipeline_lab_state(
                    session,
                    version_id=version_id,
                    selection=selection,
                )
                return JsonResponse(state)
        except ValueError as error:
            return JsonResponse({"error": str(error)}, status=422)
        except Exception as error:
            return JsonResponse(_pipeline_error_payload(error), status=503)

    if version_id is None:
        return JsonResponse({"error": "Une DocumentVersion réelle est requise."}, status=400)
    try:
        with Session(create_database_engine()) as session:
            version = session.get(DocumentVersion, version_id)
            if version is None:
                return JsonResponse({"error": "DocumentVersion introuvable."}, status=404)
            if version.page_count is None:
                return JsonResponse(
                    {"error": "Cette DocumentVersion est visible mais non exécutable pour le runtime."},
                    status=422,
                )
            registry = build_kaliok_component_registry()
            manifest = _pipeline_a_manifest(selection, registry)
            execution_context = ExecutionContext(environment="experiment")
            runtime_registry = build_kaliok_runtime_registry()
            runner = ManifestExecutionService(
                registry,
                runtime_registry,
            )
            try:
                if selection in (None, ""):
                    # Keep the established real path for the default A subset.
                    runner.execute_document_pipeline(
                        session,
                        manifest=manifest,
                        document_version_id=version.id,
                        execution_context=execution_context,
                    )
                    execution_messages = []
                else:
                    execution_messages = []
                    selected = {
                        capability: next(
                            (binding for binding in manifest.bindings
                             if binding.enabled and capability in binding.capabilities),
                            None,
                        )
                        for capability in ("document_extraction", "normalization", "entity_discovery")
                    }
                    wired = {
                        capability: binding is not None
                        and runtime_registry.has(binding.component_key, binding.component_version)
                        for capability, binding in selected.items()
                    }
                    for binding in manifest.bindings:
                        if binding.enabled and not runtime_registry.has(binding.component_key, binding.component_version):
                            execution_messages.append(
                                f"{', '.join(binding.capabilities)} sélectionnée(s) avec {binding.component_key}@{binding.component_version} : CONNU — NON RACCORDÉ."
                            )
                    if (
                        wired["document_extraction"]
                        and wired["normalization"]
                        and (selected["entity_discovery"] is None or wired["entity_discovery"])
                    ):
                        runner.execute_document_pipeline(
                            session,
                            manifest=manifest,
                            document_version_id=version.id,
                            execution_context=execution_context,
                        )
                    else:
                        perception_result = None
                        if wired["document_extraction"]:
                            perception_result = runner.execute_document_extraction(
                                session,
                                manifest=manifest,
                                document_version_id=version.id,
                                execution_context=execution_context,
                            )
                        if wired["normalization"]:
                            if perception_result is None:
                                execution_messages.append(
                                    "normalization ne peut pas démarrer : document_extraction n'est pas exécutable dans cette sélection."
                                )
                            else:
                                runner.execute_normalization(
                                    session,
                                    manifest=manifest,
                                    document_version_id=version.id,
                                    perception_processing_run_id=perception_result.processing_run_id,
                                    execution_context=execution_context,
                                )
                        if wired["entity_discovery"]:
                            if perception_result is None or not wired["normalization"]:
                                execution_messages.append(
                                    "entity_discovery ne peut pas démarrer : document_extraction et normalization doivent être exécutables."
                                )
                            else:
                                normalization_run = session.exec(
                                    select(ProcessingRun)
                                    .where(
                                        ProcessingRun.document_version_id == version.id,
                                        ProcessingRun.process_type == "content_normalization",
                                        ProcessingRun.execution_group_id == execution_context.execution_group_id,
                                    )
                                    .order_by(ProcessingRun.started_at.desc())
                                ).first()
                                if normalization_run is not None:
                                    runner.execute_entity_discovery(
                                        session,
                                        manifest=manifest,
                                        document_version_id=version.id,
                                        normalization_processing_run_id=normalization_run.id,
                                        execution_context=execution_context,
                                    )
                    if not any(selected.values()):
                        execution_messages.append("Aucune étape exécutable sélectionnée dans Pipeline_A.")
                session.commit()
                status = 200
            except Exception as error:
                session.commit()
                status = 200
                state = _pipeline_lab_state(
                    session,
                    version_id=version.id,
                    group_id=execution_context.execution_group_id,
                    selection=selection,
                )
                state["execution_error"] = "L’exécution a échoué. Consultez les détails techniques du run."
                state["technical_error"] = str(error)
                return JsonResponse(state, status=status)
            state = _pipeline_lab_state(
                session,
                version_id=version.id,
                group_id=execution_context.execution_group_id,
                selection=selection,
            )
            if execution_messages:
                state["execution_error"] = " ".join(execution_messages)
            return JsonResponse(state, status=status)
    except Exception as error:
        return JsonResponse(_pipeline_error_payload(error), status=503)


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
    pipeline_lab = None
    pipeline_error = None

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
    try:
        selected_pipeline_version = request.GET.get("document_version_id")
        with Session(create_database_engine()) as session:
            pipeline_lab = _pipeline_lab_state(
                session,
                version_id=UUID(selected_pipeline_version) if selected_pipeline_version else None,
            )
    except Exception as error:
        pipeline_error = str(error)
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
        "pipeline_lab": pipeline_lab,
        "pipeline_error": pipeline_error,
        "pipeline_lab_url": reverse("rag_laboratory_pipeline"),
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
