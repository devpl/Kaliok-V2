from __future__ import annotations

from collections.abc import Iterator
import os

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "kaliok.ui.config.settings")

import django

django.setup()

import httpx
import pytest
from django.test import Client, override_settings
from django.urls import reverse
from fastapi.testclient import TestClient
from sqlalchemy import func, inspect
from sqlmodel import Session, select

from kaliok.api.dependencies import get_session
from kaliok.api.main import app
from kaliok.pipeline.composer import RagComposerReadService
from kaliok.storage.database import create_database_engine
from kaliok.storage.models import (
    ArtifactType,
    Capability,
    CapabilityArtifactContract,
    ComponentCapability,
    PipelineBinding,
    PipelineBindingNode,
    PipelineDefinition,
    PipelineRevision,
    RagTemplateEdge,
    RagTemplateNode,
    ResourceInstance,
    ResourceInstanceCapability,
)
from kaliok.ui.core_ui import rag_prototype as composer_ui


@pytest.fixture
def composer_session() -> Iterator[Session]:
    engine = create_database_engine()
    with engine.connect() as connection:
        transaction = connection.begin()
        try:
            if "pipeline_binding_nodes" not in inspect(connection).get_table_names():
                pytest.skip("Les migrations de composition ne sont pas appliquées.")
            with Session(bind=connection) as session:
                yield session
        finally:
            if transaction.is_active:
                transaction.rollback()
    engine.dispose()


def _draft(session: Session) -> PipelineRevision:
    return session.exec(
        select(PipelineRevision).where(PipelineRevision.status == "draft")
    ).one()


def _revision_projection(payload, revision_id):
    return next(
        revision
        for pipeline in payload["pipelines"]
        for revision in pipeline["revisions"]
        if revision["id"] == str(revision_id)
    )


def test_real_pipeline_projects_template_nodes_edges_and_selected_bindings_without_mutation(
    composer_session: Session,
):
    counts_before = {
        model.__tablename__: composer_session.exec(select(func.count()).select_from(model)).one()
        for model in (
            PipelineDefinition,
            PipelineRevision,
            RagTemplateNode,
            RagTemplateEdge,
            PipelineBinding,
            PipelineBindingNode,
            ResourceInstance,
        )
    }

    payload = RagComposerReadService(composer_session).project()
    draft = _draft(composer_session)
    projected = _revision_projection(payload, draft.id)

    persisted_nodes = composer_session.exec(
        select(RagTemplateNode).where(
            RagTemplateNode.rag_template_revision_id == draft.rag_template_revision_id
        )
    ).all()
    persisted_edges = composer_session.exec(
        select(RagTemplateEdge).where(
            RagTemplateEdge.rag_template_revision_id == draft.rag_template_revision_id
        )
    ).all()
    assert {item["id"] for item in projected["nodes"]} == {
        str(item.id) for item in persisted_nodes
    }
    assert {item["id"] for item in projected["edges"]} == {
        str(item.id) for item in persisted_edges
    }
    assert any(item["selected_binding"] for item in projected["nodes"])
    assert any(item["selected_binding"] is None for item in projected["nodes"])

    counts_after = {
        model.__tablename__: composer_session.exec(select(func.count()).select_from(model)).one()
        for model in (
            PipelineDefinition,
            PipelineRevision,
            RagTemplateNode,
            RagTemplateEdge,
            PipelineBinding,
            PipelineBindingNode,
            ResourceInstance,
        )
    }
    assert counts_after == counts_before


def test_capability_contracts_are_relational_with_no_legacy_jsonb_fallback(
    composer_session: Session,
):
    draft = _draft(composer_session)
    node = composer_session.exec(
        select(RagTemplateNode)
        .where(RagTemplateNode.rag_template_revision_id == draft.rag_template_revision_id)
        .order_by(RagTemplateNode.position)
    ).first()
    assert node is not None
    capability = composer_session.get(Capability, node.capability_id)
    assert capability is not None
    contracts = composer_session.exec(
        select(CapabilityArtifactContract)
        .where(CapabilityArtifactContract.capability_id == capability.id)
        .order_by(
            CapabilityArtifactContract.direction,
            CapabilityArtifactContract.position,
            CapabilityArtifactContract.port_key,
        )
    ).all()
    assert contracts
    artifact_types = {
        item.id: item for item in composer_session.exec(select(ArtifactType)).all()
    }

    capability.input_artifact_types = ["legacy-input-only"]
    capability.output_artifact_types = ["legacy-output-only"]
    composer_session.add(capability)
    composer_session.flush()

    projected = _revision_projection(
        RagComposerReadService(composer_session).project(), draft.id
    )
    projected_capability = next(
        item["capability"] for item in projected["nodes"] if item["id"] == str(node.id)
    )
    for direction, field in (("input", "input_contracts"), ("output", "output_contracts")):
        expected = [
            {
                "artifact_type": {
                    "id": str(artifact_types[contract.artifact_type_id].id),
                    "key": artifact_types[contract.artifact_type_id].artifact_type_key,
                    "display_name": artifact_types[contract.artifact_type_id].display_name,
                    "version": artifact_types[contract.artifact_type_id].version,
                },
                "port_key": contract.port_key,
                "required": contract.required,
                "cardinality": contract.cardinality,
                "position": contract.position,
            }
            for contract in contracts
            if contract.direction == direction
        ]
        assert projected_capability[field] == expected
    assert "input_artifact_types" not in projected_capability
    assert "output_artifact_types" not in projected_capability
    assert "legacy-input-only" not in str(projected_capability)
    assert "legacy-output-only" not in str(projected_capability)

    for contract in contracts:
        composer_session.delete(contract)
    composer_session.flush()
    without_relational_contracts = _revision_projection(
        RagComposerReadService(composer_session).project(), draft.id
    )
    projected_capability = next(
        item["capability"]
        for item in without_relational_contracts["nodes"]
        if item["id"] == str(node.id)
    )
    assert projected_capability["input_contracts"] == []
    assert projected_capability["output_contracts"] == []
    assert "legacy-input-only" not in str(projected_capability)
    assert "legacy-output-only" not in str(projected_capability)


def test_alternative_is_preserved_and_never_selected_as_fallback(composer_session: Session):
    draft = _draft(composer_session)
    linked_node_ids = {
        row.rag_template_node_id
        for row in composer_session.exec(
            select(PipelineBindingNode).where(PipelineBindingNode.pipeline_revision_id == draft.id)
        ).all()
    }
    node = next(
        row
        for row in composer_session.exec(
            select(RagTemplateNode).where(
                RagTemplateNode.rag_template_revision_id == draft.rag_template_revision_id
            )
        ).all()
        if row.id not in linked_node_ids
    )
    component_capability = composer_session.exec(
        select(ComponentCapability).where(ComponentCapability.capability_id == node.capability_id)
    ).one()
    position = composer_session.exec(
        select(func.max(PipelineBinding.position)).where(
            PipelineBinding.pipeline_revision_id == draft.id
        )
    ).one() or 0
    binding = PipelineBinding(
        pipeline_revision_id=draft.id,
        binding_key="composer-test-alternative",
        component_version_id=component_capability.component_version_id,
        position=position + 1,
    )
    composer_session.add(binding)
    composer_session.flush()
    composer_session.add(
        PipelineBindingNode(
            pipeline_binding_id=binding.id,
            rag_template_node_id=node.id,
            pipeline_revision_id=draft.id,
            rag_template_revision_id=draft.rag_template_revision_id,
            enabled=True,
            is_selected=False,
            priority=4,
        )
    )
    composer_session.flush()

    projected = _revision_projection(RagComposerReadService(composer_session).project(), draft.id)
    projected_node = next(item for item in projected["nodes"] if item["id"] == str(node.id))
    assert projected_node["selected_binding"] is None
    assert [item["binding"]["key"] for item in projected_node["alternatives"]] == [
        "composer-test-alternative"
    ]
    assert "Aucun binding sélectionné." in projected_node["issues"]


def test_resource_and_disabled_states_are_projected_without_exposing_values(
    composer_session: Session,
):
    draft = _draft(composer_session)
    link = composer_session.exec(
        select(PipelineBindingNode).where(
            PipelineBindingNode.pipeline_revision_id == draft.id,
            PipelineBindingNode.is_selected.is_(True),
        )
    ).first()
    assert link is not None
    node = composer_session.get(RagTemplateNode, link.rag_template_node_id)
    binding = composer_session.get(PipelineBinding, link.pipeline_binding_id)
    assert node is not None and binding is not None
    component_capability = composer_session.exec(
        select(ComponentCapability).where(
            ComponentCapability.component_version_id == binding.component_version_id,
            ComponentCapability.capability_id == node.capability_id,
        )
    ).one()
    resource = ResourceInstance(
        component_version_id=binding.component_version_id,
        instance_key="composer-test-resource",
        display_name="Ressource Composer test",
        runtime_kind="test",
    )
    composer_session.add(resource)
    composer_session.flush()
    composer_session.add(
        ResourceInstanceCapability(
            resource_instance_id=resource.id,
            component_capability_id=component_capability.id,
            availability_status="available",
        )
    )
    binding.resource_instance_id = resource.id
    binding.configuration = {"api_token": "must-not-leak"}
    binding.enabled = False
    node.enabled = False
    composer_session.add(binding)
    composer_session.add(node)
    composer_session.flush()

    payload = RagComposerReadService(composer_session).project()
    projected = _revision_projection(payload, draft.id)
    projected_node = next(item for item in projected["nodes"] if item["id"] == str(node.id))
    selected = projected_node["selected_binding"]
    assert selected["resource_instance"]["display_name"] == "Ressource Composer test"
    assert selected["resource_instance"]["capability_availability"] == "available"
    assert selected["binding"]["enabled"] is False
    assert projected_node["enabled"] is False
    assert "api_token" in selected["binding"]["configuration"]["keys"]
    assert "must-not-leak" not in str(payload)


def test_binding_without_resource_is_supported(composer_session: Session):
    draft = _draft(composer_session)
    projected = _revision_projection(RagComposerReadService(composer_session).project(), draft.id)
    selected = next(item["selected_binding"] for item in projected["nodes"] if item["selected_binding"])
    assert selected["resource_instance"] is None


def test_composer_endpoint_returns_projection(composer_session: Session):
    def override_session():
        yield composer_session

    app.dependency_overrides[get_session] = override_session
    try:
        response = TestClient(app).get("/rag/composer")
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 200
    assert response.json()["pipelines"]


@override_settings(KALIOK_API_BASE_URL="http://api-test:9123")
def test_django_composer_handles_available_api(monkeypatch):
    payload = {"pipelines": [{"id": "p", "key": "pipeline-p", "display_name": "Réel", "description": None, "revisions": []}]}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return payload

    monkeypatch.setattr(composer_ui.httpx, "get", lambda url, timeout: Response())
    response = Client(HTTP_HOST="localhost").get(reverse("rag_prototype"))
    assert response.status_code == 200
    assert "pipeline-p" in response.content.decode()
    assert "service de composition RAG est indisponible" not in response.content.decode()


@override_settings(KALIOK_API_BASE_URL="http://api-test:9123")
def test_django_composer_handles_unavailable_api(monkeypatch):
    def unavailable(url, timeout):
        raise httpx.ConnectError("indisponible")

    monkeypatch.setattr(composer_ui.httpx, "get", unavailable)
    response = Client(HTTP_HOST="localhost").get(reverse("rag_prototype"))
    assert response.status_code == 200
    assert "service de composition RAG est indisponible" in response.content.decode()


@override_settings(KALIOK_API_BASE_URL="http://api-test:9123")
def test_django_composer_step_test_relays_to_existing_api(monkeypatch):
    observed = {}

    class Response:
        status_code = 200

        def json(self):
            return {"status": "refused", "missing_inputs": [{"artifact_type_key": "discovered_candidates"}]}

    def post(url, content, headers, timeout):
        observed.update(url=url, content=content, headers=headers, timeout=timeout)
        return Response()

    monkeypatch.setattr(composer_ui.httpx, "post", post)
    response = Client(HTTP_HOST="localhost").post(
        reverse("rag_prototype_step_test"),
        data='{"pipeline_revision_id":"00000000-0000-0000-0000-000000000001"}',
        content_type="application/json",
    )
    assert response.status_code == 200
    assert response.json()["status"] == "refused"
    assert observed["url"] == "http://api-test:9123/rag/composer/execute-step"
    assert observed["headers"] == {"Content-Type": "application/json"}


@override_settings(KALIOK_API_BASE_URL="http://api-test:9123")
def test_django_composer_step_test_reports_unavailable_api(monkeypatch):
    monkeypatch.setattr(
        composer_ui.httpx,
        "post",
        lambda *args, **kwargs: (_ for _ in ()).throw(httpx.ConnectError("indisponible")),
    )
    response = Client(HTTP_HOST="localhost").post(
        reverse("rag_prototype_step_test"), data="{}", content_type="application/json"
    )
    assert response.status_code == 503
    assert response.json() == {
        "status": "failed",
        "error": "Le service d’exécution est indisponible.",
    }
