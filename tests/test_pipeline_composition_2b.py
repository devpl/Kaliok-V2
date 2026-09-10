from __future__ import annotations

from collections.abc import Iterator

import pytest
from sqlalchemy import inspect
from sqlmodel import Session, select

from kaliok.pipeline.composition import PipelineCompositionService
from kaliok.storage.database import create_database_engine
from kaliok.storage.models import (
    AuditEvent,
    ComponentVersion,
    PipelineBinding,
    PipelineBindingCapability,
    PipelineBindingNode,
    PipelineRevision,
    RagTemplateNode,
    ResourceInstance,
)


@pytest.fixture
def composition_session() -> Iterator[Session]:
    engine = create_database_engine()
    with engine.connect() as connection:
        transaction = connection.begin()
        try:
            if "pipeline_binding_nodes" not in inspect(connection).get_table_names():
                pytest.skip("La migration 2B n'est pas appliquée dans cette base.")
            with Session(bind=connection) as session:
                yield session
        finally:
            if transaction.is_active:
                transaction.rollback()
    engine.dispose()


def _revision(session: Session, status: str) -> PipelineRevision:
    return session.exec(
        select(PipelineRevision).where(PipelineRevision.status == status)
    ).one()


def test_2b_backfill_is_complete_and_runtime_stays_legacy(composition_session: Session):
    active = _revision(composition_session, "active")
    draft = _revision(composition_session, "draft")
    links = composition_session.exec(select(PipelineBindingNode)).all()

    assert len(links) == 6
    assert sum(link.pipeline_revision_id == active.id for link in links) == 2
    assert sum(link.pipeline_revision_id == draft.id for link in links) == 4
    assert all(link.enabled and link.is_selected and link.priority == 0 for link in links)
    assert composition_session.exec(select(AuditEvent)).all() == []

    result = PipelineCompositionService(composition_session).validate_pipeline_composition(active.id)
    assert result.structure_valid is True
    assert result.activatable is True
    assert result.runtime_ready is False
    assert result.runtime_errors


def test_projection_divergence_is_reported_without_repair(composition_session: Session):
    draft = _revision(composition_session, "draft")
    service = PipelineCompositionService(composition_session)
    link = service.load_binding_nodes(draft.id)[0]
    legacy = composition_session.exec(
        select(PipelineBindingCapability).where(
            PipelineBindingCapability.pipeline_binding_id == link.pipeline_binding_id
        )
    ).first()
    assert legacy is not None
    composition_session.delete(legacy)
    composition_session.flush()

    comparison = service.compare_legacy_projection(draft.id)
    assert comparison["is_coherent"] is False
    assert comparison["missing"]
    assert composition_session.exec(select(PipelineBindingCapability)).all()


def test_graph_managed_mutations_are_audited_and_active_is_immutable(composition_session: Session):
    active = _revision(composition_session, "active")
    draft = _revision(composition_session, "draft")
    service = PipelineCompositionService(composition_session)
    draft_link = service.load_binding_nodes(draft.id)[0]
    node = composition_session.get(RagTemplateNode, draft_link.rag_template_node_id)
    assert node is not None

    with pytest.raises(ValueError, match="active est immuable"):
        service.update_configuration(
            active.id,
            service.load_binding_nodes(active.id)[0].pipeline_binding_id,
            service.load_binding_nodes(active.id)[0].rag_template_node_id,
            {},
        )

    with pytest.raises(ValueError, match="sensible"):
        service.update_configuration(
            draft.id,
            draft_link.pipeline_binding_id,
            draft_link.rag_template_node_id,
            {"token": "do-not-store"},
        )

    service.set_priority(
        draft.id,
        draft_link.pipeline_binding_id,
        draft_link.rag_template_node_id,
        3,
    )
    events = composition_session.exec(
        select(AuditEvent).where(AuditEvent.object_type == "pipeline_binding_nodes")
    ).all()
    assert {event.action for event in events} == {"binding_node_priority_changed"}


def test_unreachable_resource_keeps_structure_valid_but_not_runtime_ready(
    composition_session: Session,
):
    draft = _revision(composition_session, "draft")
    service = PipelineCompositionService(composition_session)
    link = service.load_binding_nodes(draft.id)[0]
    binding = composition_session.get(PipelineBinding, link.pipeline_binding_id)
    assert binding is not None
    instance = ResourceInstance(
        component_version_id=binding.component_version_id,
        instance_key="test-unreachable-2b",
        display_name="Test unreachable",
        runtime_kind="test",
        status="unknown",
        health_status="unreachable",
    )
    composition_session.add(instance)
    composition_session.flush()
    service.set_resource_instance(draft.id, binding.id, instance.id)

    result = service.validate_pipeline_composition(draft.id)
    assert result.structure_valid is True
    assert result.activatable is True
    assert result.runtime_ready is False
    assert any("non joignable" in error for error in result.runtime_errors)


def test_resource_instance_component_version_mismatch_is_rejected(composition_session: Session):
    draft = _revision(composition_session, "draft")
    service = PipelineCompositionService(composition_session)
    binding = composition_session.exec(
        select(PipelineBinding).where(PipelineBinding.pipeline_revision_id == draft.id)
    ).first()
    assert binding is not None
    versions = composition_session.exec(select(ComponentVersion)).all()
    other = next(version for version in versions if version.id != binding.component_version_id)
    instance = ResourceInstance(
        component_version_id=other.id,
        instance_key="test-mismatch-2b",
        display_name="Test mismatch",
        runtime_kind="test",
    )
    composition_session.add(instance)
    composition_session.flush()
    with pytest.raises(ValueError, match="ne correspond pas"):
        service.set_resource_instance(draft.id, binding.id, instance.id)
