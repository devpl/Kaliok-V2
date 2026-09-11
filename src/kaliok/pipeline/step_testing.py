"""Single-node Lab execution for the RAG Composer.

This module is intentionally an invocation service, not a graph planner.
Only runtimes that can consume the explicitly selected relational artefacts
are admitted here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID

from sqlmodel import Session, select

from kaliok.entity_resolution import EntityResolutionService
from kaliok.execution import ExecutionContext, ExecutionProvenanceService
from kaliok.storage.models import (
    ArtifactType, CapabilityArtifactContract, ComponentCapability,
    ComponentVersion, DiscoveredCandidate, Entity, PipelineBinding,
    PipelineBindingNode, PipelineRevision, RagTemplateNode,
    ResourceInstance, ResourceInstanceCapability, ProcessingRun,
)


_ARTIFACT_MODELS = {"discovered_candidates": DiscoveredCandidate}


@dataclass(frozen=True)
class StepInput:
    artifact_type_key: str
    artifact_id: UUID
    port_key: str | None = None


class StepTestRefused(ValueError):
    """Expected pre-launch refusal: no Execution is persisted."""

    def __init__(self, message: str, *, kind: str, details: list[dict[str, str]] | None = None):
        super().__init__(message)
        self.kind = kind
        self.details = details or []


class RagComposerStepTestService:
    def __init__(self, session: Session):
        self.session = session

    def execute(self, *, pipeline_revision_id: UUID, node_id: UUID, inputs: list[StepInput]) -> dict[str, Any]:
        pipeline = self.session.get(PipelineRevision, pipeline_revision_id)
        node = self.session.get(RagTemplateNode, node_id)
        if pipeline is None or pipeline.rag_template_revision_id is None:
            raise StepTestRefused("Révision de pipeline inconnue ou sans template RAG.", kind="invalid_request")
        if node is None or node.rag_template_revision_id != pipeline.rag_template_revision_id:
            raise StepTestRefused("Le node n'existe pas dans cette révision de template.", kind="invalid_request")
        if not node.enabled:
            raise StepTestRefused("Ce node est désactivé.", kind="invalid_request")

        binding, link, version, resource = self._selected_binding(pipeline, node)
        contracts = list(self.session.exec(select(CapabilityArtifactContract).where(
            CapabilityArtifactContract.capability_id == node.capability_id,
        ).order_by(CapabilityArtifactContract.direction, CapabilityArtifactContract.position)).all())
        resolved_inputs = self._validate_inputs(contracts, inputs)

        # The only currently direct artefact-to-service runtime is resolution.
        # Other nodes require a ProcessingRun/current selection or production
        # index state, so are deliberately not made to look executable.
        if self._capability_key(node.capability_id) != "entity_resolution":
            raise StepTestRefused(
                "Cette capability n'est pas encore testable dans le Lab : son runtime ne consomme pas encore les artefacts sélectionnés explicitement.",
                kind="unsupported",
            )

        provenance = ExecutionProvenanceService(self.session)
        snapshot = {
            "node": dict(node.configuration or {}),
            "binding_link": dict(link.configuration or {}),
            "binding": dict(binding.configuration or {}),
        }
        execution = provenance.create_execution(
            scope="lab", execution_mode="step", actor_type="system",
            pipeline_revision_id=pipeline.id, requested_rag_template_node_id=node.id,
            metadata={"source": "rag-composer", "capability": "entity_resolution"},
        )
        step = provenance.create_step(execution.id, sequence_no=0, rag_template_node_id=node.id,
            pipeline_binding_id=binding.id, resource_instance_id=resource.id if resource else None,
            configuration=snapshot)
        provenance.transition_execution(execution.id, "running")
        provenance.transition_step(step.id, "running")
        for artifact in resolved_inputs:
            provenance.record_input(step.id, artifact)
        # Persist the trace before crossing the runtime boundary.  A failed
        # runtime must not erase its Execution/Step or validated INPUTs.
        self.session.commit()
        try:
            # Keep every runtime-side write inside a savepoint.  In particular,
            # this preserves the committed technical trace when callers use an
            # outer transaction (as the PostgreSQL test fixtures do).
            with self.session.begin_nested():
                result = EntityResolutionService(self.session).resolve(
                    [item.id for item in resolved_inputs],
                    execution_context=ExecutionContext(environment="experiment", pipeline_revision_id=pipeline.id, execution_step_id=step.id),
                )
                run = self.session.get(ProcessingRun, result.processing_run_id)
                outputs = list(self.session.exec(select(Entity).where(Entity.processing_run_id == result.processing_run_id)).all())
                for artifact in outputs:
                    provenance.record_output(step.id, artifact)
            provenance.transition_step(step.id, "completed")
            provenance.transition_execution(execution.id, "completed")
            self.session.commit()
            return self._response(execution.id, step.id, "completed", binding.id, resource.id if resource else None, result.processing_run_id, resolved_inputs, outputs)
        except Exception as error:
            # Keep the M3 Execution/Step trace and validated INPUT provenance.
            # Runtime-side writes inside the savepoint have already been rolled back,
            # so no partial ProcessingRun or OUTPUT is claimed after a failed invocation.
            provenance = ExecutionProvenanceService(self.session)
            provenance.transition_step(step.id, "failed", error_message=str(error))
            provenance.transition_execution(execution.id, "failed", error_message=str(error))
            self.session.commit()
            return self._response(execution.id, step.id, "failed", binding.id, resource.id if resource else None, None, resolved_inputs, [], error=str(error))

    def _selected_binding(self, pipeline, node):
        links = list(self.session.exec(select(PipelineBindingNode).where(
            PipelineBindingNode.pipeline_revision_id == pipeline.id,
            PipelineBindingNode.rag_template_node_id == node.id,
            PipelineBindingNode.is_selected.is_(True),
        )).all())
        if not links:
            raise StepTestRefused("Aucun binding sélectionné pour ce node.", kind="invalid_request")
        if len(links) != 1:
            raise StepTestRefused("Plusieurs bindings sont sélectionnés pour ce node.", kind="invalid_request")
        link = links[0]
        binding = self.session.get(PipelineBinding, link.pipeline_binding_id)
        if not link.enabled or binding is None or not binding.enabled:
            raise StepTestRefused("Le binding sélectionné est désactivé ou introuvable.", kind="invalid_request")
        version = self.session.get(ComponentVersion, binding.component_version_id)
        if version is None or version.status != "available":
            raise StepTestRefused("La version du composant n'est pas disponible.", kind="invalid_request")
        matches = list(self.session.exec(select(ComponentCapability).where(
            ComponentCapability.component_version_id == version.id,
            ComponentCapability.capability_id == node.capability_id,
        )).all())
        if len(matches) != 1:
            raise StepTestRefused("Le composant ne fournit pas exactement la capability du node.", kind="invalid_request")
        resource = self.session.get(ResourceInstance, binding.resource_instance_id) if binding.resource_instance_id else None
        if binding.resource_instance_id and resource is None:
            raise StepTestRefused("La ResourceInstance du binding est introuvable.", kind="invalid_request")
        if resource:
            availability = self.session.exec(select(ResourceInstanceCapability).where(
                ResourceInstanceCapability.resource_instance_id == resource.id,
                ResourceInstanceCapability.component_capability_id == matches[0].id,
            )).first()
            if availability is not None and availability.availability_status != "available":
                raise StepTestRefused("La capability n'est pas disponible sur cette ResourceInstance.", kind="invalid_request")
        return binding, link, version, resource

    def _validate_inputs(self, contracts, inputs):
        required = [c for c in contracts if c.direction == "input" and c.required is not False]
        contract_by_key = {self.session.get(ArtifactType, c.artifact_type_id).artifact_type_key: c for c in required}
        values = []
        for raw in inputs:
            contract = contract_by_key.get(raw.artifact_type_key)
            if contract is None or (raw.port_key and raw.port_key != contract.port_key):
                raise StepTestRefused("Le type ou port d'artefact fourni ne correspond pas au contrat.", kind="invalid_request")
            model = _ARTIFACT_MODELS.get(raw.artifact_type_key)
            if model is None:
                raise StepTestRefused("Type d'artefact non supporté pour test unitaire.", kind="unsupported")
            artifact = self.session.get(model, raw.artifact_id)
            if artifact is None:
                raise StepTestRefused("Artefact d'entrée introuvable.", kind="invalid_request")
            values.append(artifact)
        missing = [key for key in contract_by_key if not any(item.artifact_type_key == key for item in inputs)]
        if missing:
            raise StepTestRefused("Entrée requise manquante : " + ", ".join(missing) + ".", kind="missing_inputs", details=[{"artifact_type_key": key} for key in missing])
        return values

    def _capability_key(self, capability_id):
        from kaliok.storage.models import Capability
        return self.session.get(Capability, capability_id).capability_key

    @staticmethod
    def _response(execution_id, step_id, status, binding_id, resource_id, run_id, inputs, outputs, error=None):
        return {"execution_id": str(execution_id), "execution_step_id": str(step_id), "status": status,
                "binding_id": str(binding_id), "resource_instance_id": str(resource_id) if resource_id else None,
                "processing_runs": [str(run_id)] if run_id else [],
                "inputs": [str(item.id) for item in inputs], "outputs": [str(item.id) for item in outputs], "error": error}
