from __future__ import annotations

from collections.abc import Iterator
from uuid import uuid4

import pytest
from sqlalchemy import inspect
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select

from kaliok.execution import ExecutionProvenanceService
from kaliok.storage.database import create_database_engine
from kaliok.storage.models import (
    DiscoveredCandidate,
    Entity,
    Execution,
    ExecutionArtifact,
    ExecutionArtifactDiscoveredCandidate,
    ExecutionArtifactEntity,
    ExecutionArtifactNormalizedContentUnit,
    ExecutionStep,
    NormalizedContentUnit,
    PipelineRevision,
    ProcessingRun,
    RagTemplateNode,
    RagTemplateRevision,
)


@pytest.fixture
def execution_session() -> Iterator[Session]:
    engine = create_database_engine()
    with engine.connect() as connection:
        transaction = connection.begin()
        try:
            if "executions" not in inspect(connection).get_table_names():
                pytest.skip("Migration 3 non appliquée dans cette base.")
            with Session(bind=connection) as session:
                yield session
        finally:
            if transaction.is_active:
                transaction.rollback()
    engine.dispose()


def _pipeline_and_node(session: Session) -> tuple[PipelineRevision, RagTemplateNode]:
    pipeline = session.exec(
        select(PipelineRevision).where(PipelineRevision.rag_template_revision_id.is_not(None))
    ).first()
    assert pipeline is not None and pipeline.rag_template_revision_id is not None
    node = session.exec(
        select(RagTemplateNode).where(
            RagTemplateNode.rag_template_revision_id == pipeline.rag_template_revision_id
        ).order_by(RagTemplateNode.position)
    ).first()
    assert node is not None
    return pipeline, node


def _execution_and_step(session: Session, *, sequence_no: int = 0) -> tuple[Execution, ExecutionStep]:
    pipeline, node = _pipeline_and_node(session)
    service = ExecutionProvenanceService(session)
    execution = service.create_execution(
        scope="lab",
        execution_mode="step",
        actor_type="user",
        pipeline_revision_id=pipeline.id,
        requested_rag_template_node_id=node.id,
    )
    step = service.create_step(
        execution.id,
        sequence_no=sequence_no,
        rag_template_node_id=node.id,
        configuration={"effective": "snapshot"},
    )
    return execution, step


def test_m3_schema_is_additive_and_legacy_rows_are_unbackfilled(execution_session: Session):
    inspector = inspect(execution_session.connection())
    assert {
        "executions",
        "execution_steps",
        "execution_artifacts",
        "execution_artifact_content_blocks",
        "execution_artifact_normalized_content_units",
        "execution_artifact_discovered_candidates",
        "execution_artifact_entities",
    } <= set(inspector.get_table_names())
    assert "execution_step_id" in {
        column["name"] for column in inspector.get_columns("processing_runs")
    }
    assert execution_session.exec(select(Execution)).all() == []
    assert execution_session.exec(select(ExecutionStep)).all() == []
    assert execution_session.exec(select(ExecutionArtifact)).all() == []
    assert execution_session.exec(
        select(ProcessingRun).where(ProcessingRun.execution_step_id.is_not(None))
    ).all() == []


def test_create_execution_step_and_status_timestamps(execution_session: Session):
    execution, step = _execution_and_step(execution_session)
    service = ExecutionProvenanceService(execution_session)

    assert execution.status == "pending"
    assert step.status == "pending"
    assert step.pipeline_revision_id == execution.pipeline_revision_id
    assert step.rag_template_revision_id == execution.rag_template_revision_id
    assert step.configuration == {"effective": "snapshot"}

    service.transition_execution(execution.id, "running")
    service.transition_step(step.id, "running")
    service.transition_step(step.id, "completed")
    service.transition_execution(execution.id, "completed")
    assert execution.started_at is not None and execution.completed_at is not None
    assert step.started_at is not None and step.completed_at is not None


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (("scope", "invalid", "Scope"), ("execution_mode", "invalid", "Mode")),
)
def test_invalid_execution_identity_is_rejected(
    execution_session: Session, field: str, value: str, message: str
):
    pipeline, _ = _pipeline_and_node(execution_session)
    arguments = {
        "scope": "lab",
        "execution_mode": "step",
        "actor_type": "user",
        "pipeline_revision_id": pipeline.id,
        field: value,
    }
    with pytest.raises(ValueError, match=message):
        ExecutionProvenanceService(execution_session).create_execution(**arguments)


def test_database_rejects_invalid_status_and_node_template_mismatch(execution_session: Session):
    pipeline, node = _pipeline_and_node(execution_session)
    execution_session.add(Execution(
        scope="lab",
        execution_mode="step",
        status="invented",
        actor_type="user",
        pipeline_revision_id=pipeline.id,
        rag_template_revision_id=pipeline.rag_template_revision_id,
    ))
    with pytest.raises(IntegrityError):
        execution_session.flush()
    execution_session.rollback()

    other_revision = RagTemplateRevision(
        rag_template_id=execution_session.get(
            RagTemplateRevision, pipeline.rag_template_revision_id
        ).rag_template_id,
        revision_number=99,
        status="draft",
    )
    execution_session.add(other_revision)
    execution_session.flush()
    other_node = RagTemplateNode(
        rag_template_revision_id=other_revision.id,
        node_key=f"m3-mismatch-{uuid4().hex}",
        capability_id=node.capability_id,
        display_name="M3 mismatch",
        requirement_mode="optional",
    )
    execution_session.add(other_node)
    execution_session.flush()
    with pytest.raises(ValueError, match="node"):
        ExecutionProvenanceService(execution_session).create_execution(
            scope="lab",
            execution_mode="step",
            actor_type="user",
            pipeline_revision_id=pipeline.id,
            requested_rag_template_node_id=other_node.id,
        )


def test_processing_run_link_is_optional_and_one_step_accepts_many_runs(execution_session: Session):
    _, step = _execution_and_step(execution_session)
    runs = execution_session.exec(
        select(ProcessingRun).where(ProcessingRun.execution_step_id.is_(None)).limit(3)
    ).all()
    assert len(runs) == 3
    service = ExecutionProvenanceService(execution_session)
    service.attach_processing_run(step.id, runs[0].id)
    service.attach_processing_run(step.id, runs[1].id)
    assert runs[0].execution_step_id == step.id
    assert runs[1].execution_step_id == step.id
    assert runs[2].execution_step_id is None


def test_artifact_roles_support_n_to_d_to_r_without_same_run_invariant(execution_session: Session):
    unit = execution_session.exec(
        select(NormalizedContentUnit).where(
            NormalizedContentUnit.processing_run_id.is_not(None)
        )
    ).first()
    candidate = execution_session.exec(select(DiscoveredCandidate)).first()
    entity = execution_session.exec(select(Entity)).first()
    assert unit is not None and candidate is not None and entity is not None

    first_execution, first_step = _execution_and_step(execution_session)
    pipeline, node = _pipeline_and_node(execution_session)
    service = ExecutionProvenanceService(execution_session)
    second_execution = service.create_execution(
        scope="lab", execution_mode="step", actor_type="user",
        pipeline_revision_id=pipeline.id,
    )
    second_step = service.create_step(
        second_execution.id, sequence_no=0, rag_template_node_id=node.id
    )

    unit_output = service.record_output(first_step.id, unit)
    unit_input = service.record_input(second_step.id, unit)
    candidate_output = service.record_output(second_step.id, candidate)
    entity_output = service.record_output(second_step.id, entity)

    assert unit_output.id != unit_input.id
    assert unit_output.role == "output" and unit_input.role == "input"
    assert candidate.processing_run_id != unit.processing_run_id
    assert entity.processing_run_id != candidate.processing_run_id
    assert execution_session.get(ExecutionArtifactNormalizedContentUnit, unit_input.id) is not None
    assert execution_session.get(ExecutionArtifactDiscoveredCandidate, candidate_output.id) is not None
    assert execution_session.get(ExecutionArtifactEntity, entity_output.id) is not None
    assert first_execution.scope == second_execution.scope == "lab"


def test_invalid_status_transition_is_rejected(execution_session: Session):
    execution, step = _execution_and_step(execution_session)
    service = ExecutionProvenanceService(execution_session)
    with pytest.raises(ValueError, match="pending -> completed"):
        service.transition_execution(execution.id, "completed")
    with pytest.raises(ValueError, match="pending -> skipped"):
        service.transition_step(step.id, "skipped")


def test_migration_declares_no_data_backfill():
    migration = (
        __import__("pathlib").Path(__file__).parents[1]
        / "migrations/versions/d3e4f5a6b7c8_add_execution_provenance.py"
    ).read_text(encoding="utf-8")
    assert "op.execute(" not in migration
    assert "INSERT INTO executions" not in migration
    assert "UPDATE processing_runs" not in migration
