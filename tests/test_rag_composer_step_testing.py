from __future__ import annotations

from collections.abc import Iterator
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, inspect
from sqlmodel import Session, select

from kaliok.api.dependencies import get_session
from kaliok.api.main import app
from kaliok.entity_resolution import EntityResolutionService
from kaliok.execution import ExecutionProvenanceService
from kaliok.pipeline.step_testing import RagComposerStepTestService, StepInput, StepTestRefused
from kaliok.storage.database import create_database_engine
from kaliok.storage.models import (
    ArtifactType, Capability, CapabilityArtifactContract, Component,
    ComponentCapability, ComponentVersion, DiscoveredCandidate, Document,
    DocumentVersion, Entity, Execution, ExecutionArtifact, ExecutionStep,
    PipelineBinding, PipelineBindingNode, PipelineDefinition, PipelineRevision,
    ProcessingRun, RagTemplate, RagTemplateNode, RagTemplateRevision,
    ResourceInstance,
)


@pytest.fixture
def step_test_session() -> Iterator[Session]:
    """Use the migrated PostgreSQL catalogue inside an outer rollback."""
    engine = create_database_engine()
    with engine.connect() as connection:
        transaction = connection.begin()
        try:
            tables = set(inspect(connection).get_table_names())
            if not {"executions", "execution_steps", "execution_artifacts"} <= tables:
                pytest.skip("Migration M3 non appliquée dans cette base.")
            with Session(bind=connection) as session:
                yield session
        finally:
            if transaction.is_active:
                transaction.rollback()
    engine.dispose()


def _selection(session: Session):
    """Create a complete relational fixture; the outer DB transaction rolls it back."""
    suffix = uuid4().hex
    input_type = ArtifactType(artifact_type_key="discovered_candidates", version=suffix,
        display_name="Candidates", storage_kind="relational")
    output_type = ArtifactType(artifact_type_key="entities", version=suffix,
        display_name="Entities", storage_kind="relational")
    existing = session.exec(select(Capability).where(Capability.capability_key == "entity_resolution")).first()
    new_capability = existing is None
    if existing is not None:
        capability = existing
    else:
        capability = Capability(capability_key="entity_resolution", display_name="Resolution")
        session.add(capability)
    component = Component(component_key=f"step-test-{suffix}", display_name="Step test")
    template = RagTemplate(template_key=f"step-test-{suffix}", display_name="Step test")
    document = Document(title="Step test")
    session.add_all([input_type, output_type, component, template, document])
    session.flush()
    version = ComponentVersion(component_id=component.id, version="1")
    template_revision = RagTemplateRevision(rag_template_id=template.id, revision_number=1)
    document_version = DocumentVersion(document_id=document.id, version_number=1,
        filename="step-test.txt", file_hash=suffix, storage_uri="test://step-test")
    session.add_all([version, template_revision, document_version])
    session.flush()
    component_capability = ComponentCapability(component_version_id=version.id, capability_id=capability.id)
    node = RagTemplateNode(rag_template_revision_id=template_revision.id, node_key="resolution",
        capability_id=capability.id, display_name="Resolution", requirement_mode="required",
        configuration={"fixture": "node"})
    definition = PipelineDefinition(pipeline_key=f"step-test-{suffix}", display_name="Step test")
    run = ProcessingRun(document_version_id=document_version.id, process_type="entity_discovery", status="completed")
    session.add_all([component_capability, node, definition, run])
    session.flush()
    pipeline = PipelineRevision(pipeline_definition_id=definition.id,
        rag_template_revision_id=template_revision.id, revision_number=1, manifest_hash=suffix)
    candidate = DiscoveredCandidate(document_version_id=document_version.id, processing_run_id=run.id,
        candidate_type="person", raw_value="Ada Lovelace", normalized_value="ada lovelace",
        detector_key="fixture")
    session.add_all([pipeline, candidate])
    session.flush()
    binding = PipelineBinding(pipeline_revision_id=pipeline.id, binding_key="resolution",
        component_version_id=version.id, position=0, configuration={"fixture": "binding"})
    session.add(binding)
    session.flush()
    link = PipelineBindingNode(pipeline_binding_id=binding.id, rag_template_node_id=node.id,
        pipeline_revision_id=pipeline.id, rag_template_revision_id=template_revision.id,
        is_selected=True, configuration={"fixture": "link"})
    session.add(link)
    if new_capability:
        session.add_all([
            CapabilityArtifactContract(capability_id=capability.id, artifact_type_id=input_type.id,
                direction="input", port_key="candidates", required=True),
            CapabilityArtifactContract(capability_id=capability.id, artifact_type_id=output_type.id,
                direction="output", port_key="entities", required=True),
        ])
    session.flush()
    return pipeline, node, link, candidate


def _input(candidate: DiscoveredCandidate) -> StepInput:
    return StepInput("discovered_candidates", candidate.id)


def _refusal(session: Session, **changes):
    pipeline, node, link, candidate = _selection(session)
    return RagComposerStepTestService(session), pipeline, node, link, candidate


@pytest.mark.parametrize("kind", ["missing", "unknown-node", "disabled-node"])
def test_rejects_invalid_or_disabled_node_without_provenance(step_test_session: Session, kind: str):
    service, pipeline, node, _, candidate = _refusal(step_test_session)
    before = step_test_session.exec(select(func.count()).select_from(Execution)).one()
    if kind == "missing":
        pipeline_id, node_id = uuid4(), node.id
    elif kind == "unknown-node":
        pipeline_id, node_id = pipeline.id, uuid4()
    else:
        node.enabled = False
        step_test_session.flush()
        pipeline_id, node_id = pipeline.id, node.id
    with pytest.raises(StepTestRefused):
        service.execute(pipeline_revision_id=pipeline_id, node_id=node_id, inputs=[_input(candidate)])
    assert step_test_session.exec(select(func.count()).select_from(Execution)).one() == before


def test_rejects_missing_selected_binding_and_missing_or_wrong_input(step_test_session: Session):
    service, pipeline, node, link, candidate = _refusal(step_test_session)
    link.is_selected = False
    step_test_session.flush()
    with pytest.raises(StepTestRefused, match="Aucun binding"):
        service.execute(pipeline_revision_id=pipeline.id, node_id=node.id, inputs=[_input(candidate)])
    link.is_selected = True
    step_test_session.flush()
    with pytest.raises(StepTestRefused) as missing:
        service.execute(pipeline_revision_id=pipeline.id, node_id=node.id, inputs=[])
    assert missing.value.kind == "missing_inputs"
    with pytest.raises(StepTestRefused, match="type ou port"):
        service.execute(pipeline_revision_id=pipeline.id, node_id=node.id,
                        inputs=[StepInput("entities", candidate.id)])
    with pytest.raises(StepTestRefused, match="introuvable"):
        service.execute(pipeline_revision_id=pipeline.id, node_id=node.id,
                        inputs=[StepInput("discovered_candidates", uuid4())])


def test_rejects_disabled_pipeline_binding_without_provenance(step_test_session: Session):
    service, pipeline, node, link, candidate = _refusal(step_test_session)
    binding = step_test_session.get(PipelineBinding, link.pipeline_binding_id)
    assert binding is not None
    binding.enabled = False
    step_test_session.flush()
    with pytest.raises(StepTestRefused, match="désactivé"):
        service.execute(pipeline_revision_id=pipeline.id, node_id=node.id, inputs=[_input(candidate)])


def test_success_persists_only_explicit_step_provenance(step_test_session: Session):
    service, pipeline, node, link, candidate = _refusal(step_test_session)
    resource_count = step_test_session.exec(select(func.count()).select_from(ResourceInstance)).one()
    result = service.execute(pipeline_revision_id=pipeline.id, node_id=node.id, inputs=[_input(candidate)])
    execution = step_test_session.get(Execution, result["execution_id"])
    step = step_test_session.get(ExecutionStep, result["execution_step_id"])
    run = step_test_session.get(ProcessingRun, result["processing_runs"][0])
    artifacts = step_test_session.exec(select(ExecutionArtifact).where(
        ExecutionArtifact.execution_step_id == step.id
    )).all()
    assert result["status"] == "completed"
    assert execution.scope == "lab" and execution.execution_mode == "step"
    assert execution.requested_rag_template_node_id == node.id
    assert step.rag_template_node_id == node.id and step.pipeline_binding_id == link.pipeline_binding_id
    assert step.resource_instance_id is None
    assert step.configuration["node"] == node.configuration
    assert run.execution_step_id == step.id and run.execution_environment == "experiment"
    assert {artifact.role for artifact in artifacts} == {"input", "output"}
    assert len([a for a in artifacts if a.role == "input"]) == 1
    assert len([a for a in artifacts if a.role == "output"]) == len(result["outputs"])
    assert result["inputs"][0] not in result["outputs"]
    assert execution.status == step.status == "completed"
    assert step_test_session.exec(select(func.count()).select_from(ResourceInstance)).one() == resource_count
    assert step_test_session.exec(select(ExecutionStep).where(ExecutionStep.execution_id == execution.id)).all() == [step]


def test_runtime_failure_keeps_failed_trace_without_outputs(step_test_session: Session, monkeypatch):
    service, pipeline, node, _, candidate = _refusal(step_test_session)
    monkeypatch.setattr(EntityResolutionService, "resolve", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
    result = service.execute(pipeline_revision_id=pipeline.id, node_id=node.id, inputs=[_input(candidate)])
    execution = step_test_session.get(Execution, result["execution_id"])
    step = step_test_session.get(ExecutionStep, result["execution_step_id"])
    artifacts = step_test_session.exec(select(ExecutionArtifact).where(
        ExecutionArtifact.execution_step_id == step.id
    )).all()
    assert result["status"] == "failed" and result["outputs"] == []
    assert execution.status == step.status == "failed"
    assert [artifact.role for artifact in artifacts] == ["input"]


def test_unsupported_node_refuses_without_any_execution(step_test_session: Session, monkeypatch):
    pipeline, _, _, _ = _selection(step_test_session)
    unsupported_capability = Capability(capability_key=f"normalization-{uuid4().hex}", display_name="Unsupported")
    step_test_session.add(unsupported_capability)
    step_test_session.flush()
    node = RagTemplateNode(rag_template_revision_id=pipeline.rag_template_revision_id,
        node_key="unsupported", capability_id=unsupported_capability.id,
        display_name="Unsupported", requirement_mode="optional")
    step_test_session.add(node)
    step_test_session.flush()
    before = step_test_session.exec(select(func.count()).select_from(Execution)).one()
    service = RagComposerStepTestService(step_test_session)
    monkeypatch.setattr(service, "_selected_binding", lambda *_: (object(), object(), object(), None))
    monkeypatch.setattr(service, "_validate_inputs", lambda *_: [])
    with pytest.raises(StepTestRefused) as refused:
        service.execute(pipeline_revision_id=pipeline.id, node_id=node.id, inputs=[])
    assert refused.value.kind == "unsupported"
    assert step_test_session.exec(select(func.count()).select_from(Execution)).one() == before


def test_execute_step_endpoint_returns_coherent_completed_result(step_test_session: Session):
    _, pipeline, node, _, candidate = _refusal(step_test_session)
    def override_session():
        yield step_test_session
    app.dependency_overrides[get_session] = override_session
    try:
        response = TestClient(app).post("/rag/composer/execute-step", json={
            "pipeline_revision_id": str(pipeline.id), "rag_template_node_id": str(node.id),
            "inputs": [{"artifact_type_key": "discovered_candidates", "artifact_id": str(candidate.id)}],
        })
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed" and body["inputs"] == [str(candidate.id)]
    assert body["outputs"] and body["processing_runs"]
