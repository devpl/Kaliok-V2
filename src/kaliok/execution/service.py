from __future__ import annotations

from typing import Any, Literal, TypeVar
from uuid import UUID

from sqlmodel import Session, SQLModel, select

from kaliok.storage.models import (
    ArtifactType,
    ContentBlock,
    DiscoveredCandidate,
    Entity,
    Execution,
    ExecutionArtifact,
    ExecutionArtifactContentBlock,
    ExecutionArtifactDiscoveredCandidate,
    ExecutionArtifactEntity,
    ExecutionArtifactNormalizedContentUnit,
    ExecutionStep,
    NormalizedContentUnit,
    PipelineBindingNode,
    PipelineRevision,
    ProcessingRun,
    RagTemplateNode,
    ResourceInstance,
    utc_now,
)


ExecutionScope = Literal["production", "lab", "evaluation"]
ExecutionMode = Literal["step", "prerequisites", "zone", "pipeline"]
ArtifactRole = Literal["input", "output"]

_EXECUTION_TRANSITIONS = {
    "pending": {"running"},
    "running": {"completed", "failed", "cancelled"},
}
_STEP_TRANSITIONS = {
    "pending": {"running"},
    "running": {"completed", "failed", "skipped", "cancelled"},
}

ArtifactModel = TypeVar("ArtifactModel", bound=SQLModel)


class ExecutionProvenanceService:
    """Persistence primitives for orchestration; this is intentionally not a planner."""

    def __init__(self, session: Session):
        self.session = session

    def create_execution(
        self,
        *,
        scope: ExecutionScope,
        execution_mode: ExecutionMode,
        actor_type: str,
        pipeline_revision_id: UUID,
        requested_rag_template_node_id: UUID | None = None,
        actor_user_id: UUID | None = None,
        configuration_revision_id: UUID | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Execution:
        if scope not in {"production", "lab", "evaluation"}:
            raise ValueError(f"Scope d'exécution invalide : {scope}.")
        if execution_mode not in {"step", "prerequisites", "zone", "pipeline"}:
            raise ValueError(f"Mode d'exécution invalide : {execution_mode}.")
        if not actor_type.strip():
            raise ValueError("actor_type ne peut pas être vide.")
        pipeline = self.session.get(PipelineRevision, pipeline_revision_id)
        if pipeline is None or pipeline.rag_template_revision_id is None:
            raise ValueError("La révision de pipeline doit référencer une révision de template.")
        if requested_rag_template_node_id is not None:
            self._require_node(requested_rag_template_node_id, pipeline.rag_template_revision_id)
        execution = Execution(
            scope=scope,
            execution_mode=execution_mode,
            actor_type=actor_type,
            actor_user_id=actor_user_id,
            configuration_revision_id=configuration_revision_id,
            pipeline_revision_id=pipeline.id,
            rag_template_revision_id=pipeline.rag_template_revision_id,
            requested_rag_template_node_id=requested_rag_template_node_id,
            extra_data=dict(metadata or {}),
        )
        self.session.add(execution)
        self.session.flush()
        return execution

    def create_step(
        self,
        execution_id: UUID,
        *,
        sequence_no: int,
        rag_template_node_id: UUID,
        pipeline_binding_id: UUID | None = None,
        resource_instance_id: UUID | None = None,
        configuration: dict[str, Any] | None = None,
    ) -> ExecutionStep:
        execution = self._require(Execution, execution_id, "Execution")
        if sequence_no < 0:
            raise ValueError("sequence_no doit être positif ou nul.")
        self._require_node(rag_template_node_id, execution.rag_template_revision_id)
        if pipeline_binding_id is not None:
            binding_node = self.session.exec(
                select(PipelineBindingNode).where(
                    PipelineBindingNode.pipeline_binding_id == pipeline_binding_id,
                    PipelineBindingNode.pipeline_revision_id == execution.pipeline_revision_id,
                    PipelineBindingNode.rag_template_node_id == rag_template_node_id,
                    PipelineBindingNode.rag_template_revision_id == execution.rag_template_revision_id,
                )
            ).first()
            if binding_node is None:
                raise ValueError("Le binding n'est pas associé à ce node dans cette révision.")
        if resource_instance_id is not None:
            self._require(ResourceInstance, resource_instance_id, "ResourceInstance")
        step = ExecutionStep(
            execution_id=execution.id,
            sequence_no=sequence_no,
            pipeline_revision_id=execution.pipeline_revision_id,
            rag_template_revision_id=execution.rag_template_revision_id,
            rag_template_node_id=rag_template_node_id,
            pipeline_binding_id=pipeline_binding_id,
            resource_instance_id=resource_instance_id,
            configuration=dict(configuration or {}),
        )
        self.session.add(step)
        self.session.flush()
        return step

    def transition_execution(
        self,
        execution_id: UUID,
        status: str,
        *,
        error_message: str | None = None,
    ) -> Execution:
        execution = self._require(Execution, execution_id, "Execution")
        self._transition(execution, status, _EXECUTION_TRANSITIONS, error_message)
        return execution

    def transition_step(
        self,
        execution_step_id: UUID,
        status: str,
        *,
        error_message: str | None = None,
    ) -> ExecutionStep:
        step = self._require(ExecutionStep, execution_step_id, "ExecutionStep")
        self._transition(step, status, _STEP_TRANSITIONS, error_message)
        return step

    def attach_processing_run(self, execution_step_id: UUID, processing_run_id: UUID) -> ProcessingRun:
        self._require(ExecutionStep, execution_step_id, "ExecutionStep")
        run = self._require(ProcessingRun, processing_run_id, "ProcessingRun")
        if run.execution_step_id not in {None, execution_step_id}:
            raise ValueError("Ce ProcessingRun appartient déjà à un autre ExecutionStep.")
        run.execution_step_id = execution_step_id
        self.session.add(run)
        self.session.flush()
        return run

    def record_input(self, execution_step_id: UUID, artifact: SQLModel, *, metadata: dict[str, Any] | None = None) -> ExecutionArtifact:
        return self._record_artifact(execution_step_id, "input", artifact, metadata=metadata)

    def record_output(self, execution_step_id: UUID, artifact: SQLModel, *, metadata: dict[str, Any] | None = None) -> ExecutionArtifact:
        return self._record_artifact(execution_step_id, "output", artifact, metadata=metadata)

    def _record_artifact(
        self,
        execution_step_id: UUID,
        role: ArtifactRole,
        artifact: SQLModel,
        *,
        metadata: dict[str, Any] | None,
    ) -> ExecutionArtifact:
        self._require(ExecutionStep, execution_step_id, "ExecutionStep")
        mapping = {
            ContentBlock: ("content_blocks", ExecutionArtifactContentBlock, "content_block_id"),
            NormalizedContentUnit: ("normalized_content_units", ExecutionArtifactNormalizedContentUnit, "normalized_content_unit_id"),
            DiscoveredCandidate: ("discovered_candidates", ExecutionArtifactDiscoveredCandidate, "discovered_candidate_id"),
            Entity: ("entities", ExecutionArtifactEntity, "entity_id"),
        }
        spec = mapping.get(type(artifact))
        if spec is None:
            raise TypeError(f"Type d'artefact non supporté en M3 : {type(artifact).__name__}.")
        artifact_id = getattr(artifact, "id", None)
        if artifact_id is None or self.session.get(type(artifact), artifact_id) is None:
            raise ValueError("L'artefact doit être persisté avant d'enregistrer sa provenance.")
        artifact_key, link_model, target_field = spec
        artifact_type = self.session.exec(
            select(ArtifactType).where(
                ArtifactType.artifact_type_key == artifact_key,
                ArtifactType.is_active.is_(True),
            ).order_by(ArtifactType.version.desc())
        ).first()
        if artifact_type is None:
            raise ValueError(f"ArtifactType actif introuvable : {artifact_key}.")
        envelope = ExecutionArtifact(
            execution_step_id=execution_step_id,
            role=role,
            artifact_type_id=artifact_type.id,
            extra_data=dict(metadata or {}),
        )
        self.session.add(envelope)
        self.session.flush()
        self.session.add(link_model(execution_artifact_id=envelope.id, **{target_field: artifact_id}))
        self.session.flush()
        return envelope

    def _require_node(self, node_id: UUID, revision_id: UUID) -> RagTemplateNode:
        node = self.session.get(RagTemplateNode, node_id)
        if node is None or node.rag_template_revision_id != revision_id:
            raise ValueError("Le node n'appartient pas à la révision de template attendue.")
        return node

    def _require(self, model: type[ArtifactModel], object_id: UUID, label: str) -> ArtifactModel:
        value = self.session.get(model, object_id)
        if value is None:
            raise ValueError(f"{label} introuvable : {object_id}.")
        return value

    def _transition(self, value: Any, status: str, transitions: dict[str, set[str]], error_message: str | None) -> None:
        if status not in transitions.get(value.status, set()):
            raise ValueError(f"Transition de statut invalide : {value.status} -> {status}.")
        now = utc_now()
        value.status = status
        value.updated_at = now
        if status == "running":
            value.started_at = now
            value.completed_at = None
            value.error_message = None
        else:
            value.completed_at = now
            value.error_message = error_message if status == "failed" else None
        self.session.add(value)
        self.session.flush()


__all__ = [
    "ArtifactRole",
    "ExecutionMode",
    "ExecutionProvenanceService",
    "ExecutionScope",
]
