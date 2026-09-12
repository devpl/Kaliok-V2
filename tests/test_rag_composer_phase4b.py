from __future__ import annotations

from collections.abc import Iterator
import os

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "kaliok.ui.config.settings")

import django

django.setup()

import pytest
from django.test import Client, override_settings
from django.urls import reverse
from fastapi.testclient import TestClient
from sqlalchemy import func, inspect
from sqlmodel import Session, select

from kaliok.api.dependencies import get_session
from kaliok.api.main import app
from kaliok.pipeline.composer import RagComposerReadService
from kaliok.pipeline.composer_mutations import RagComposerMutationService
from kaliok.storage.database import create_database_engine
from kaliok.storage.models import (
    AuditEvent,
    Capability,
    ComponentCapability,
    PipelineBinding,
    PipelineBindingCapability,
    PipelineBindingDependency,
    PipelineBindingNode,
    PipelineRevision,
    RagTemplateEdge,
    RagTemplateNode,
    RagTemplateRevision,
    ResourceInstance,
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


def _source(session: Session) -> PipelineRevision:
    return session.exec(
        select(PipelineRevision)
        .where(PipelineRevision.rag_template_revision_id.is_not(None))
        .order_by(PipelineRevision.revision_number)
    ).first()


def test_fork_clones_and_remaps_the_complete_composition(composer_session: Session):
    source = _source(composer_session)
    assert source is not None and source.rag_template_revision_id is not None
    source_nodes = composer_session.exec(
        select(RagTemplateNode).where(
            RagTemplateNode.rag_template_revision_id == source.rag_template_revision_id
        )
    ).all()
    source_edges = composer_session.exec(
        select(RagTemplateEdge).where(
            RagTemplateEdge.rag_template_revision_id == source.rag_template_revision_id
        )
    ).all()
    source_bindings = composer_session.exec(
        select(PipelineBinding).where(PipelineBinding.pipeline_revision_id == source.id)
    ).all()
    source_links = composer_session.exec(
        select(PipelineBindingNode).where(PipelineBindingNode.pipeline_revision_id == source.id)
    ).all()
    source_binding_ids = [row.id for row in source_bindings]
    source_capability_links = composer_session.exec(
        select(PipelineBindingCapability).where(
            PipelineBindingCapability.pipeline_binding_id.in_(source_binding_ids)
        )
    ).all() if source_binding_ids else []
    source_dependencies = composer_session.exec(
        select(PipelineBindingDependency).where(
            PipelineBindingDependency.pipeline_binding_id.in_(source_binding_ids)
        )
    ).all() if source_binding_ids else []

    fork = RagComposerMutationService(composer_session).fork_pipeline_revision_with_graph(source.id)

    assert fork.id != source.id
    assert fork.status == "draft"
    assert fork.rag_template_revision_id != source.rag_template_revision_id
    clone_nodes = composer_session.exec(
        select(RagTemplateNode).where(
            RagTemplateNode.rag_template_revision_id == fork.rag_template_revision_id
        )
    ).all()
    clone_edges = composer_session.exec(
        select(RagTemplateEdge).where(
            RagTemplateEdge.rag_template_revision_id == fork.rag_template_revision_id
        )
    ).all()
    clone_bindings = composer_session.exec(
        select(PipelineBinding).where(PipelineBinding.pipeline_revision_id == fork.id)
    ).all()
    clone_links = composer_session.exec(
        select(PipelineBindingNode).where(PipelineBindingNode.pipeline_revision_id == fork.id)
    ).all()
    clone_binding_ids = [row.id for row in clone_bindings]
    assert len(clone_nodes) == len(source_nodes)
    assert len(clone_edges) == len(source_edges)
    assert len(clone_bindings) == len(source_bindings)
    assert len(clone_links) == len(source_links)
    assert len(composer_session.exec(
        select(PipelineBindingCapability).where(
            PipelineBindingCapability.pipeline_binding_id.in_(clone_binding_ids)
        )
    ).all() if clone_binding_ids else []) == len(source_capability_links)
    assert len(composer_session.exec(
        select(PipelineBindingDependency).where(
            PipelineBindingDependency.pipeline_binding_id.in_(clone_binding_ids)
        )
    ).all() if clone_binding_ids else []) == len(source_dependencies)
    source_node_by_id = {row.id: row for row in source_nodes}
    source_binding_by_id = {row.id: row for row in source_bindings}
    clone_link_signatures = {
        (
            next(row.binding_key for row in clone_bindings if row.id == link.pipeline_binding_id),
            next(row.node_key for row in clone_nodes if row.id == link.rag_template_node_id),
            link.enabled,
            link.is_selected,
            link.priority,
        )
        for link in clone_links
    }
    source_link_signatures = {
        (
            source_binding_by_id[link.pipeline_binding_id].binding_key,
            source_node_by_id[link.rag_template_node_id].node_key,
            link.enabled,
            link.is_selected,
            link.priority,
        )
        for link in source_links
    }
    assert clone_link_signatures == source_link_signatures
    assert all(link.rag_template_revision_id == fork.rag_template_revision_id for link in clone_links)
    assert {row.id for row in clone_nodes}.isdisjoint({row.id for row in source_nodes})
    assert {row.id for row in clone_bindings}.isdisjoint({row.id for row in source_bindings})
    composer_session.refresh(source)
    assert source.status != "draft" or source.id != fork.id
    event = composer_session.exec(
        select(AuditEvent).where(
            AuditEvent.action == "rag_composer_revision_forked",
            AuditEvent.object_id == fork.id,
        )
    ).one()
    assert event.after_state["source_pipeline_revision_id"] == str(source.id)


def test_fork_savepoint_rolls_back_every_partial_clone(composer_session: Session, monkeypatch):
    source = _source(composer_session)
    before = composer_session.exec(select(func.count()).select_from(PipelineRevision)).one()
    original_flush = composer_session.flush
    calls = 0

    def fail_during_clone(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("forced clone failure")
        return original_flush(*args, **kwargs)

    monkeypatch.setattr(composer_session, "flush", fail_during_clone)
    with pytest.raises(RuntimeError, match="forced clone failure"):
        RagComposerMutationService(composer_session).fork_pipeline_revision_with_graph(source.id)
    monkeypatch.setattr(composer_session, "flush", original_flush)
    assert composer_session.exec(select(func.count()).select_from(PipelineRevision)).one() == before


def test_create_node_persists_real_unconfigured_capability_only(composer_session: Session):
    fork = RagComposerMutationService(composer_session).fork_pipeline_revision_with_graph(
        _source(composer_session).id
    )
    capability = composer_session.exec(
        select(Capability).where(Capability.is_active.is_(True)).order_by(Capability.display_order)
    ).first()
    counts = {
        model: composer_session.exec(select(func.count()).select_from(model)).one()
        for model in (RagTemplateEdge, PipelineBinding, ResourceInstance)
    }
    first = RagComposerMutationService(composer_session).create_node(fork.id, capability.id)
    second = RagComposerMutationService(composer_session).create_node(fork.id, capability.id)

    assert first.capability_id == capability.id
    assert first.rag_template_revision_id == fork.rag_template_revision_id
    assert first.display_name == capability.display_name
    assert first.enabled is True and first.configuration == {}
    assert first.node_key != second.node_key
    assert all(
        composer_session.exec(select(func.count()).select_from(model)).one() == count
        for model, count in counts.items()
    )
    assert not composer_session.exec(
        select(PipelineBindingNode).where(PipelineBindingNode.rag_template_node_id == first.id)
    ).all()
    assert composer_session.exec(
        select(AuditEvent).where(
            AuditEvent.action == "rag_composer_node_created",
            AuditEvent.object_id == first.id,
        )
    ).one()


def test_empty_work_revision_can_select_real_tools_and_connect_compatible_nodes(composer_session: Session):
    empty = RagComposerMutationService(composer_session).create_empty_revision(_source(composer_session).id)
    assert empty.status == "draft"
    assert composer_session.exec(select(RagTemplateNode).where(
        RagTemplateNode.rag_template_revision_id == empty.rag_template_revision_id)).all() == []
    extraction = composer_session.exec(select(Capability).where(
        Capability.capability_key == "document_extraction")).one()
    normalization = composer_session.exec(select(Capability).where(
        Capability.capability_key == "normalization")).one()
    source_node = RagComposerMutationService(composer_session).create_node(empty.id, extraction.id)
    target_node = RagComposerMutationService(composer_session).create_node(empty.id, normalization.id)
    for node in (source_node, target_node):
        tool = composer_session.exec(select(ComponentCapability).where(
            ComponentCapability.capability_id == node.capability_id)).first()
        assert tool is not None
        link = RagComposerMutationService(composer_session).select_node_tool(
            empty.id, node.id, tool.component_version_id
        )
        assert link.is_selected is True
    edge = RagComposerMutationService(composer_session).create_edge(
        empty.id, source_node.id, target_node.id
    )
    assert edge.source_port_key == "out_content_blocks"
    assert edge.target_port_key == "in_content_blocks"


def test_create_node_refuses_active_or_shared_graph(composer_session: Session):
    source = _source(composer_session)
    capability = composer_session.exec(select(Capability).where(Capability.is_active.is_(True))).first()
    if source.status == "active":
        with pytest.raises(ValueError, match="Modifier le RAG"):
            RagComposerMutationService(composer_session).create_node(source.id, capability.id)
    else:
        shared = PipelineRevision(
            pipeline_definition_id=source.pipeline_definition_id,
            rag_template_revision_id=source.rag_template_revision_id,
            revision_number=max(
                composer_session.exec(
                    select(PipelineRevision.revision_number).where(
                        PipelineRevision.pipeline_definition_id == source.pipeline_definition_id
                    )
                ).all()
            ) + 1,
            status="draft",
            manifest_hash=source.manifest_hash,
        )
        composer_session.add(shared)
        composer_session.flush()
        with pytest.raises(ValueError, match="n’est pas isolée"):
            RagComposerMutationService(composer_session).create_node(source.id, capability.id)


def test_api_fork_and_create_node_are_real(composer_session: Session):
    source = _source(composer_session)
    capability = composer_session.exec(select(Capability).where(Capability.is_active.is_(True))).first()

    def override_session():
        yield composer_session

    app.dependency_overrides[get_session] = override_session
    try:
        fork_response = TestClient(app).post(
            "/rag/composer/revisions/fork", json={"pipeline_revision_id": str(source.id)}
        )
        assert fork_response.status_code == 201
        fork_id = fork_response.json()["pipeline_revision_id"]
        node_response = TestClient(app).post(
            "/rag/composer/nodes",
            json={"pipeline_revision_id": fork_id, "capability_id": str(capability.id)},
        )
        assert node_response.status_code == 201
        projected = RagComposerReadService(composer_session).project()
        revision = next(
            row
            for pipeline in projected["pipelines"]
            for row in pipeline["revisions"]
            if row["id"] == fork_id
        )
        assert node_response.json()["rag_template_node_id"] in {
            row["id"] for row in revision["nodes"]
        }
    finally:
        app.dependency_overrides.clear()


def test_api_mutations_return_explicit_errors(composer_session: Session):
    def override_session():
        yield composer_session

    app.dependency_overrides[get_session] = override_session
    try:
        response = TestClient(app).post(
            "/rag/composer/revisions/fork",
            json={"pipeline_revision_id": "00000000-0000-0000-0000-000000000000"},
        )
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 409
    assert "introuvable" in response.json()["detail"]


@override_settings(KALIOK_API_BASE_URL="http://api-test:9123")
def test_django_same_origin_mutation_relays(monkeypatch):
    observed = []

    class Response:
        status_code = 201

        def json(self):
            return {"status": "created"}

    def post(url, content, headers, timeout):
        observed.append((url, content, headers, timeout))
        return Response()

    monkeypatch.setattr(composer_ui.httpx, "post", post)
    client = Client(HTTP_HOST="localhost")
    assert client.post(reverse("rag_prototype_fork"), data="{}", content_type="application/json").status_code == 201
    assert client.post(reverse("rag_prototype_create_node"), data="{}", content_type="application/json").status_code == 201
    assert [item[0] for item in observed] == [
        "http://api-test:9123/rag/composer/revisions/fork",
        "http://api-test:9123/rag/composer/nodes",
    ]


def test_ui_exposes_real_mutations_and_projection_refresh():
    from pathlib import Path

    root = Path(__file__).parents[1]
    script = (root / "src/kaliok/ui/core_ui/static/core_ui/rag_prototype.js").read_text("utf-8")
    styles = (root / "src/kaliok/ui/core_ui/static/core_ui/rag_prototype.css").read_text("utf-8")
    assert "Créer depuis Production" in script
    assert "+ Ajouter une fonction" in script
    assert "payload.capabilities" in script
    assert 'fetch("data/"' in script
    assert 'postMutation("revisions/fork/"' in script
    assert 'postMutation("nodes/"' in script
    assert "background: #1d3034" not in styles
    assert "background: #f8f9f8" in styles
