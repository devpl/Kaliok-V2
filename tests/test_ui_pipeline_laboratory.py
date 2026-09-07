from __future__ import annotations

import os
from uuid import uuid4

import pytest

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "kaliok.ui.config.settings")

import django

django.setup()

from django.test import Client, override_settings
from django.urls import reverse

from kaliok.pipeline import (
    ComponentDefinition,
    ComponentRegistry,
    build_current_production_manifest,
    build_kaliok_component_registry,
    build_kaliok_runtime_registry,
)
from kaliok.ui.core_ui import views


class FakeSession:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def get(self, model, identifier):
        return type("Version", (), {"id": identifier, "page_count": 1})()

    def commit(self):
        return None


def pipeline_state(version_id, group_id=None):
    return {
        "documents": [{
            "document_version_id": str(version_id),
            "document_id": str(uuid4()),
            "filename": "réel-東京.txt",
            "version_number": 1,
            "processing_status": "completed",
            "page_count": 1,
            "executable": True,
        }],
        "selected_document": {"document_version_id": str(version_id), "filename": "réel-東京.txt"},
        "pipeline_reference": {"pipeline_key": "pipeline-p", "revision": "partial-v1", "bindings": []},
        "pipeline_experiment": {"pipeline_key": "pipeline-a", "revision": "experiment-v1", "bindings": []},
        "stages": [],
        "history": [],
        "result": {"execution_group_id": str(group_id)} if group_id else None,
    }


@override_settings(ALLOWED_HOSTS=["testserver"])
def test_pipeline_get_is_read_only_and_returns_real_state(monkeypatch):
    version_id = uuid4()
    captured = {}

    monkeypatch.setattr(views, "create_database_engine", lambda: object())
    monkeypatch.setattr(views, "Session", lambda engine: FakeSession())

    def fake_state(session, **kwargs):
        captured.update(kwargs)
        return pipeline_state(version_id)

    monkeypatch.setattr(views, "_pipeline_lab_state", fake_state)

    response = Client().get(
        reverse("rag_laboratory_pipeline"),
        {"document_version_id": str(version_id)},
    )

    assert response.status_code == 200
    assert response.json()["selected_document"]["filename"] == "réel-東京.txt"
    assert captured["version_id"] == version_id


@override_settings(ALLOWED_HOSTS=["testserver"])
def test_pipeline_document_change_keeps_all_options_and_selects_requested_version(monkeypatch):
    version_a = uuid4()
    version_b = uuid4()
    documents = [
        {"document_version_id": str(version_a), "filename": "A.pdf", "executable": True},
        {"document_version_id": str(version_b), "filename": "B.pdf", "executable": True},
    ]
    monkeypatch.setattr(views, "create_database_engine", lambda: object())
    monkeypatch.setattr(views, "Session", lambda engine: FakeSession())
    monkeypatch.setattr(
        views,
        "_pipeline_lab_state",
        lambda session, **kwargs: {
            **pipeline_state(kwargs["version_id"]),
            "documents": documents,
            "selected_document": next(item for item in documents if item["document_version_id"] == str(kwargs["version_id"])),
        },
    )

    client = Client()
    for requested in (version_b, version_a):
        response = client.get(reverse("rag_laboratory_pipeline"), {"document_version_id": str(requested)})
        payload = response.json()
        assert [item["filename"] for item in payload["documents"]] == ["A.pdf", "B.pdf"]
        assert payload["selected_document"]["document_version_id"] == str(requested)


@override_settings(ALLOWED_HOSTS=["testserver"])
def test_pipeline_post_delegates_to_manifest_runtime_and_returns_group(monkeypatch):
    version_id = uuid4()
    group_id = uuid4()
    calls = {}

    class FakeVersion:
        id = version_id

    class FakeRunner:
        def __init__(self, components, runtimes):
            calls["constructed"] = True

        def execute_document_pipeline(self, session, **kwargs):
            calls["context"] = kwargs["execution_context"]
            calls["manifest"] = kwargs["manifest"]
            calls["version_id"] = kwargs["document_version_id"]
            return None

    class FakeExecutionContext:
        def __init__(self, *, environment):
            self.environment = environment
            self.execution_group_id = group_id

    monkeypatch.setattr(views, "create_database_engine", lambda: object())
    monkeypatch.setattr(views, "Session", lambda engine: FakeSession())
    monkeypatch.setattr(views, "ExecutionContext", FakeExecutionContext)
    monkeypatch.setattr(views, "ManifestExecutionService", FakeRunner)
    monkeypatch.setattr(views, "build_kaliok_component_registry", lambda: object())
    monkeypatch.setattr(views, "build_kaliok_runtime_registry", lambda: object())
    monkeypatch.setattr(views, "_pipeline_lab_state", lambda session, **kwargs: pipeline_state(version_id, group_id))

    response = Client().post(
        reverse("rag_laboratory_pipeline"),
        data={"document_version_id": str(version_id)},
        content_type="application/json",
    )

    assert response.status_code == 200
    assert calls["constructed"] is True
    assert calls["context"].environment == "experiment"
    assert calls["context"].execution_group_id == group_id
    assert calls["version_id"] == version_id
    assert calls["manifest"].pipeline_key == "pipeline-a"
    assert response.json()["result"]["execution_group_id"] == str(group_id)


def test_pipeline_a_is_derived_from_the_real_production_bindings():
    production = build_current_production_manifest()
    experimental = views._pipeline_a_manifest()

    assert experimental.pipeline_key == "pipeline-a"
    assert experimental.revision == "experiment-v1"
    assert experimental.bindings == production.bindings


def test_real_registry_exposes_components_by_capability_and_runtime_state():
    registry = build_kaliok_component_registry()
    capabilities = views._pipeline_capabilities_payload(
        registry,
        build_kaliok_runtime_registry(),
        views._pipeline_a_manifest(),
    )

    by_key = {item["key"]: item for item in capabilities}
    assert {item["component_key"] for item in by_key["normalization"]["components"]} == {"kaliok-normalizer"}
    assert by_key["normalization"]["status"] == "EXÉCUTABLE"
    assert by_key["entity_discovery"]["components"][0]["runtime_status"] == "EXECUTABLE"


def test_pipeline_a_groups_one_web_component_across_capabilities_and_validates_scope():
    registry = ComponentRegistry([
        ComponentDefinition(
            component_key="multi-tool",
            version="1",
            provides=("document_extraction", "ocr", "layout_analysis"),
        ),
    ])
    manifest = views._pipeline_a_manifest([
        {"component_key": "multi-tool", "component_version": "1", "capabilities": ["document_extraction"]},
        {"component_key": "multi-tool", "component_version": "1", "capabilities": ["ocr", "layout_analysis"]},
    ], registry)

    assert len(manifest.bindings) == 1
    assert manifest.bindings[0].capabilities == ("document_extraction", "ocr", "layout_analysis")

    with pytest.raises(ValueError, match="ne fournit pas"):
        views._pipeline_a_manifest([
            {"component_key": "multi-tool", "component_version": "1", "capabilities": ["embedding"]},
        ], registry)


def test_pipeline_p_remains_unchanged_and_zero_timestamp_duration_is_unmeasured():
    production = build_current_production_manifest()
    assert production == build_current_production_manifest()
    assert views._duration_ms(None, None) is None


def test_pending_version_is_visible_but_not_executable():
    version = type("Version", (), {
        "id": uuid4(),
        "filename": "pending.pdf",
        "version_number": 1,
        "processing_status": "pending",
        "version_status": "active",
        "page_count": None,
        "file_hash": "hash",
    })()
    document = type("Document", (), {"id": uuid4(), "title": "Pending"})()

    payload = views._lab_document_payload(version, document)

    assert payload["processing_status"] == "pending"
    assert payload["executable"] is False


@override_settings(ALLOWED_HOSTS=["testserver"])
def test_pipeline_post_rejects_pending_version_before_runtime(monkeypatch):
    version_id = uuid4()

    class PendingSession(FakeSession):
        def get(self, model, identifier):
            return type("Version", (), {"id": identifier, "page_count": None})()

    monkeypatch.setattr(views, "create_database_engine", lambda: object())
    monkeypatch.setattr(views, "Session", lambda engine: PendingSession())
    monkeypatch.setattr(views, "ManifestExecutionService", lambda *args: (_ for _ in ()).throw(AssertionError("runtime interdit")))

    response = Client().post(
        reverse("rag_laboratory_pipeline"),
        data={"document_version_id": str(version_id)},
        content_type="application/json",
    )

    assert response.status_code == 422
    assert "non exécutable" in response.json()["error"]


@override_settings(ALLOWED_HOSTS=["testserver"])
def test_laboratory_renders_pipeline_section_without_a_full_reload(monkeypatch):
    version_id = uuid4()
    monkeypatch.setattr(views, "_evaluation_lists", lambda: ([], [], [], [], []))
    monkeypatch.setattr(views, "_pipeline_lab_state", lambda session, **kwargs: pipeline_state(version_id))

    response = Client().get(reverse("rag_laboratory"))

    assert response.status_code == 200
    content = response.content.decode()
    assert "Pipeline en gestation" in content
    assert "Lancer perception + normalisation" in content
    assert "pipeline-laboratory" in content


def test_pipeline_js_does_not_shadow_dom_document():
    js = open(
        os.path.join(
            os.path.dirname(views.__file__),
            "static",
            "core_ui",
            "rag_laboratory.js",
        ),
        encoding="utf-8",
    ).read()

    assert "forEach((document)" not in js
    assert "forEach((documentItem)" in js
    assert "document.createElement(\"option\")" in js


@override_settings(ALLOWED_HOSTS=["testserver"])
def test_pipeline_post_respects_django_csrf(monkeypatch):
    monkeypatch.setattr(views, "create_database_engine", lambda: object())
    client = Client(enforce_csrf_checks=True)

    response = client.post(
        reverse("rag_laboratory_pipeline"),
        data={"document_version_id": str(uuid4())},
        content_type="application/json",
    )

    assert response.status_code == 403
