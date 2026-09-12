"""Safe single-node adapters for the experimental RAG Lab."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol
from uuid import UUID

from sqlmodel import Session, select

from kaliok.documents.reader import read_document
from kaliok.entity_resolution import EntityResolutionService
from kaliok.execution import ExecutionContext, ExecutionProvenanceService
from kaliok.indexing.service import store_document_perception
from kaliok.normalization import ContentNormalizationService
from kaliok.pipeline.components import ComponentBinding
from kaliok.pipeline.runtime import KaliokDiscoveryAdapter
from kaliok.storage.models import (
    ArtifactType, Capability, CapabilityArtifactContract, Component,
    ComponentCapability, ComponentVersion, ContentBlock, DiscoveredCandidate,
    DocumentVersion, Entity, Execution, NormalizedContentUnit, Page,
    PipelineBinding, PipelineBindingNode, PipelineRevision, ProcessingRun,
    RagTemplateNode, ResourceInstance, ResourceInstanceCapability,
)


_ARTIFACT_MODELS = {
    "content_blocks": ContentBlock,
    "normalized_content_units": NormalizedContentUnit,
    "discovered_candidates": DiscoveredCandidate,
    "entities": Entity,
}
_OUTPUTS = {
    "document_extraction": ("content_blocks", ContentBlock, "block_index"),
    "normalization": ("normalized_content_units", NormalizedContentUnit, "unit_index"),
    "entity_discovery": ("discovered_candidates", DiscoveredCandidate, "created_at"),
    "entity_resolution": ("entities", Entity, "entity_index"),
}
_UNSUPPORTED = {
    "chunking": "Le découpage existant est encore couplé à index_document(); le Lab ne l'exécute pas afin de ne pas modifier l'état current/production.",
    "indexing": "L'indexation existante écrit dans l'index documentaire de production; aucun index expérimental isolé n'est encore disponible.",
}


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


@dataclass(frozen=True)
class LabAdapterResult:
    run: ProcessingRun
    outputs: list[Any]


@dataclass(frozen=True)
class LabAdapterContext:
    session: Session
    document: DocumentVersion
    binding: PipelineBinding
    component: Component
    version: ComponentVersion
    execution_context: ExecutionContext
    inputs: list[Any]
    capability_key: str

    @property
    def pipeline_metadata(self) -> dict[str, object]:
        return {
            "source": "rag-composer-lab",
            "capability": self.capability_key,
            "component_key": self.component.component_key,
            "component_version": self.version.version,
            "binding_key": self.binding.binding_key,
        }

    @property
    def runtime_binding(self) -> ComponentBinding:
        return ComponentBinding(
            binding_key=self.binding.binding_key,
            component_key=self.component.component_key,
            component_version=self.version.version,
            capabilities=(self.capability_key,),
            configuration=dict(self.binding.configuration or {}),
        )


class LabStepAdapter(Protocol):
    def execute(self, context: LabAdapterContext) -> LabAdapterResult: ...


class DocumentExtractionLabAdapter:
    def execute(self, context: LabAdapterContext) -> LabAdapterResult:
        if context.binding.configuration:
            raise ValueError("La Lecture Kaliok ne prend pas de configuration de binding.")
        run = store_document_perception(
            context.session, context.document, read_document(context.document.storage_uri),
            execution_context=context.execution_context, activate_as_current=False,
            pipeline_metadata=context.pipeline_metadata,
        )
        return LabAdapterResult(run, _run_outputs(context.session, "document_extraction", run.id))


class NormalizationLabAdapter:
    def execute(self, context: LabAdapterContext) -> LabAdapterResult:
        run_id = _one_input_run(context.inputs, ContentBlock, "ContentBlock")
        result = ContentNormalizationService(context.session).normalize(
            context.document.id, perception_processing_run_id=run_id,
            execution_context=context.execution_context,
            pipeline_metadata=context.pipeline_metadata,
        )
        run = context.session.get(ProcessingRun, result.processing_run_id)
        return LabAdapterResult(run, _run_outputs(context.session, "normalization", run.id))


class DiscoveryLabAdapter:
    def execute(self, context: LabAdapterContext) -> LabAdapterResult:
        run_id = _one_input_run(context.inputs, NormalizedContentUnit, "NormalizedContentUnit")
        result = KaliokDiscoveryAdapter().execute(
            context.session, binding=context.runtime_binding,
            document_version_id=context.document.id,
            normalization_processing_run_id=run_id,
            execution_context=context.execution_context,
            pipeline_metadata=context.pipeline_metadata,
        )
        run = context.session.get(ProcessingRun, result.processing_run_id)
        return LabAdapterResult(run, _run_outputs(context.session, "entity_discovery", run.id))


class EntityResolutionLabAdapter:
    def execute(self, context: LabAdapterContext) -> LabAdapterResult:
        result = EntityResolutionService(context.session).resolve(
            [item.id for item in context.inputs], execution_context=context.execution_context,
        )
        run = context.session.get(ProcessingRun, result.processing_run_id)
        return LabAdapterResult(run, _run_outputs(context.session, "entity_resolution", run.id))


_ADAPTERS: dict[str, LabStepAdapter] = {
    "document_extraction": DocumentExtractionLabAdapter(),
    "normalization": NormalizationLabAdapter(),
    "entity_discovery": DiscoveryLabAdapter(),
    "entity_resolution": EntityResolutionLabAdapter(),
}


def capability_execution_support(capability_key: str) -> tuple[bool, str | None]:
    if capability_key in _ADAPTERS:
        return True, None
    return False, _UNSUPPORTED.get(capability_key, "Aucun adaptateur d'exécution Lab n'est raccordé à cette capability.")


def _run_outputs(session: Session, capability_key: str, run_id: UUID) -> list[Any]:
    _, model, order = _OUTPUTS[capability_key]
    return list(session.exec(select(model).where(model.processing_run_id == run_id).order_by(getattr(model, order), model.id)).all())


def _one_input_run(inputs: list[Any], model: type[Any], label: str) -> UUID:
    if not inputs or any(not isinstance(item, model) for item in inputs):
        raise ValueError(f"Des artefacts {label} explicites sont requis.")
    run_ids = {item.processing_run_id for item in inputs}
    if None in run_ids or len(run_ids) != 1:
        raise ValueError(f"Les artefacts {label} doivent provenir d'une seule génération persistée.")
    return next(iter(run_ids))


class RagComposerStepTestService:
    def __init__(self, session: Session):
        self.session = session

    def execute(self, *, pipeline_revision_id: UUID, node_id: UUID,
                document_version_id: UUID | None = None, inputs: list[StepInput]) -> dict[str, Any]:
        pipeline = self.session.get(PipelineRevision, pipeline_revision_id)
        node = self.session.get(RagTemplateNode, node_id)
        if pipeline is None or pipeline.rag_template_revision_id is None:
            raise StepTestRefused("Révision de pipeline inconnue ou sans template RAG.", kind="invalid_request")
        if pipeline.status != "draft":
            raise StepTestRefused("Le RAG de production est une référence en lecture seule; créez ou ouvrez un RAG de travail.", kind="production_read_only")
        if node is None or node.rag_template_revision_id != pipeline.rag_template_revision_id:
            raise StepTestRefused("Le node n'existe pas dans cette révision de template.", kind="invalid_request")
        if not node.enabled:
            raise StepTestRefused("Ce node est désactivé.", kind="invalid_request")
        capability = self.session.get(Capability, node.capability_id)
        capability_key = capability.capability_key if capability else ""
        supported, reason = capability_execution_support(capability_key)
        if not supported:
            raise StepTestRefused(reason or "Capability non testable.", kind="unsupported")

        binding, link, version, component, resource = self._selected_binding(pipeline, node)
        contracts = list(self.session.exec(select(CapabilityArtifactContract).where(
            CapabilityArtifactContract.capability_id == node.capability_id,
        ).order_by(CapabilityArtifactContract.direction, CapabilityArtifactContract.position)).all())
        declared_inputs = {}
        for contract in contracts:
            if contract.direction == "input":
                artifact_type = self.session.get(ArtifactType, contract.artifact_type_id)
                if artifact_type:
                    declared_inputs[artifact_type.artifact_type_key] = contract
        for raw in inputs:
            contract = declared_inputs.get(raw.artifact_type_key)
            if contract is None or (raw.port_key and raw.port_key != contract.port_key):
                raise StepTestRefused("Le type ou port d'artefact fourni ne correspond pas au contrat.", kind="invalid_request")
            model = _ARTIFACT_MODELS.get(raw.artifact_type_key)
            if model is None or self.session.get(model, raw.artifact_id) is None:
                raise StepTestRefused("Artefact d'entrée introuvable ou non supporté.", kind="invalid_request")
        if not inputs and capability_key != "document_extraction":
            self._validate_inputs(contracts, inputs, None, capability_key)
        document = self.session.get(DocumentVersion, document_version_id) if document_version_id else None
        if document is None and inputs:
            model = _ARTIFACT_MODELS.get(inputs[0].artifact_type_key)
            first = self.session.get(model, inputs[0].artifact_id) if model else None
            inferred_id = self._artifact_document_id(first) if first else None
            document = self.session.get(DocumentVersion, inferred_id) if inferred_id else None
        if document is None:
            raise StepTestRefused("Sélectionnez une DocumentVersion réelle pour ce test.", kind="missing_document")
        resolved_inputs = self._validate_inputs(contracts, inputs, document, capability_key)
        provenance = ExecutionProvenanceService(self.session)
        execution = provenance.create_execution(
            scope="lab", execution_mode="step", actor_type="system",
            pipeline_revision_id=pipeline.id, requested_rag_template_node_id=node.id,
            metadata={"source": "rag-composer", "capability": capability_key,
                      "document_version_id": str(document.id)},
        )
        step = provenance.create_step(
            execution.id, sequence_no=0, rag_template_node_id=node.id,
            pipeline_binding_id=binding.id, resource_instance_id=resource.id if resource else None,
            configuration={"node": dict(node.configuration or {}),
                           "binding_link": dict(link.configuration or {}),
                           "binding": dict(binding.configuration or {}),
                           "document_version_id": str(document.id)},
        )
        provenance.transition_execution(execution.id, "running")
        provenance.transition_step(step.id, "running")
        for artifact in resolved_inputs:
            provenance.record_input(step.id, artifact)
        self.session.commit()
        try:
            with self.session.begin_nested():
                result = _ADAPTERS[capability_key].execute(LabAdapterContext(
                    self.session, document, binding, component, version,
                    ExecutionContext(environment="experiment", pipeline_revision_id=pipeline.id,
                                     execution_step_id=step.id),
                    resolved_inputs, capability_key,
                ))
                if result.run is None:
                    raise RuntimeError("Le service n'a pas persisté son ProcessingRun.")
                for artifact in result.outputs:
                    provenance.record_output(step.id, artifact)
            provenance.transition_step(step.id, "completed")
            provenance.transition_execution(execution.id, "completed")
            self.session.commit()
            return self._response(execution.id, step.id, "completed", binding.id,
                                  resource.id if resource else None, result.run,
                                  resolved_inputs, result.outputs, capability_key, document.id)
        except Exception as error:
            provenance = ExecutionProvenanceService(self.session)
            provenance.transition_step(step.id, "failed", error_message=str(error))
            provenance.transition_execution(execution.id, "failed", error_message=str(error))
            self.session.commit()
            return self._response(execution.id, step.id, "failed", binding.id,
                                  resource.id if resource else None, None,
                                  resolved_inputs, [], capability_key, document.id, str(error))

    def _selected_binding(self, pipeline, node):
        links = list(self.session.exec(select(PipelineBindingNode).where(
            PipelineBindingNode.pipeline_revision_id == pipeline.id,
            PipelineBindingNode.rag_template_node_id == node.id,
            PipelineBindingNode.is_selected.is_(True),
        )).all())
        if len(links) != 1:
            message = "Aucun binding / outil sélectionné pour cette fonction." if not links else "Plusieurs outils sont sélectionnés pour cette fonction."
            raise StepTestRefused(message, kind="invalid_request")
        link = links[0]
        binding = self.session.get(PipelineBinding, link.pipeline_binding_id)
        if not link.enabled or binding is None or not binding.enabled:
            raise StepTestRefused("L'outil sélectionné est désactivé ou introuvable.", kind="invalid_request")
        version = self.session.get(ComponentVersion, binding.component_version_id)
        if version is None or version.status != "available":
            raise StepTestRefused("La version du composant n'est pas disponible.", kind="invalid_request")
        component = self.session.get(Component, version.component_id)
        matches = list(self.session.exec(select(ComponentCapability).where(
            ComponentCapability.component_version_id == version.id,
            ComponentCapability.capability_id == node.capability_id,
        )).all())
        if component is None or len(matches) != 1:
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
        return binding, link, version, component, resource

    def _validate_inputs(self, contracts, inputs, document, capability_key):
        by_key = {}
        for contract in contracts:
            if contract.direction != "input":
                continue
            artifact_type = self.session.get(ArtifactType, contract.artifact_type_id)
            if artifact_type:
                by_key[artifact_type.artifact_type_key] = contract
        if capability_key == "document_extraction":
            if inputs:
                raise StepTestRefused("La Lecture utilise uniquement la DocumentVersion sélectionnée.", kind="invalid_request")
            return []
        values, seen = [], set()
        for raw in inputs:
            contract = by_key.get(raw.artifact_type_key)
            if contract is None or (raw.port_key and raw.port_key != contract.port_key):
                raise StepTestRefused("Le type ou port d'artefact fourni ne correspond pas au contrat.", kind="invalid_request")
            model = _ARTIFACT_MODELS.get(raw.artifact_type_key)
            artifact = self.session.get(model, raw.artifact_id) if model else None
            if artifact is None:
                raise StepTestRefused("Artefact d'entrée introuvable ou non supporté.", kind="invalid_request")
            if artifact.id in seen:
                continue
            if self._artifact_document_id(artifact) != document.id:
                raise StepTestRefused("Un artefact d'entrée appartient à une autre DocumentVersion.", kind="invalid_request")
            seen.add(artifact.id)
            values.append(artifact)
        missing = [key for key, contract in by_key.items()
                   if contract.required is not False and not any(item.artifact_type_key == key for item in inputs)]
        if missing:
            raise StepTestRefused("Entrée requise manquante : " + ", ".join(missing) + ".",
                                  kind="missing_inputs",
                                  details=[{"artifact_type_key": key} for key in missing])
        return values

    def _artifact_document_id(self, artifact):
        if isinstance(artifact, NormalizedContentUnit):
            return artifact.document_version_id
        run = self.session.get(ProcessingRun, getattr(artifact, "processing_run_id", None))
        if run:
            return run.document_version_id
        if isinstance(artifact, ContentBlock):
            page = self.session.get(Page, artifact.page_id)
            return page.document_version_id if page else None
        return None

    @staticmethod
    def _duration_ms(started_at: datetime | None, completed_at: datetime | None):
        return max(0, round((completed_at - started_at).total_seconds() * 1000)) if started_at and completed_at else None

    def _response(self, execution_id, step_id, status, binding_id, resource_id, run,
                  inputs, outputs, capability_key, document_version_id, error=None):
        execution = self.session.get(Execution, execution_id)
        output_type = _OUTPUTS.get(capability_key, (None, None, None))[0]
        metrics = dict(run.metrics or {}) if run else {}
        metrics.update({"input_count": 1 if capability_key == "document_extraction" else len(inputs),
                        "output_count": len(outputs), "output_artifact_type": output_type})
        return {
            "execution_id": str(execution_id), "execution_step_id": str(step_id),
            "status": status, "capability_key": capability_key,
            "document_version_id": str(document_version_id), "binding_id": str(binding_id),
            "resource_instance_id": str(resource_id) if resource_id else None,
            "processing_runs": [str(run.id)] if run else [],
            "duration_ms": self._duration_ms(execution.started_at, execution.completed_at) if execution else None,
            "inputs": [str(item.id) for item in inputs], "outputs": [str(item.id) for item in outputs],
            "input_artifacts": [{"id": str(item.id), "artifact_type_key": _artifact_key(item)} for item in inputs],
            "output_artifacts": [{"id": str(item.id), "artifact_type_key": output_type} for item in outputs],
            "metrics": metrics, "error": error,
        }


def _artifact_key(value: Any) -> str:
    return next((key for key, model in _ARTIFACT_MODELS.items() if isinstance(value, model)), type(value).__name__)


__all__ = ["LabStepAdapter", "RagComposerStepTestService", "StepInput", "StepTestRefused", "capability_execution_support"]
