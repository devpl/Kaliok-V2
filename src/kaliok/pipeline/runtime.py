from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol
from uuid import UUID

from sqlmodel import Session

from kaliok.documents.reader import read_document
from kaliok.execution import ExecutionContext
from kaliok.indexing.service import (
    PERCEPTION_ENGINE,
    PERCEPTION_VERSION,
    store_document_perception,
)
from kaliok.normalization import ContentNormalizationResult, ContentNormalizationService
from kaliok.normalization.service import ENGINE_VERSION
from kaliok.pipeline.components import ComponentBinding, ComponentRegistry
from kaliok.pipeline.manifest import PipelineManifest
from kaliok.storage.models import DocumentVersion, ProcessingRun


class NormalizationRuntimeAdapter(Protocol):
    def execute(
        self,
        session: Session,
        *,
        binding: ComponentBinding,
        document_version_id: UUID,
        perception_processing_run_id: UUID,
        execution_context: ExecutionContext,
        pipeline_metadata: Mapping[str, object],
    ) -> ContentNormalizationResult: ...


class PerceptionRuntimeAdapter(Protocol):
    def execute(
        self,
        session: Session,
        *,
        binding: ComponentBinding,
        document_version_id: UUID,
        execution_context: ExecutionContext,
        pipeline_metadata: Mapping[str, object],
    ) -> object: ...


class ComponentRuntimeRegistry:
    """Runtime-only mapping from component identity to targeted adapters."""

    def __init__(self) -> None:
        self._adapters: dict[tuple[str, str], object] = {}

    def register(
        self,
        component_key: str,
        version: str,
        adapter: object,
    ) -> None:
        identity = (component_key, version)
        if identity in self._adapters:
            raise ValueError(f"Runtime adapter déjà enregistré : {component_key}@{version}.")
        self._adapters[identity] = adapter

    def resolve(
        self,
        component_key: str,
        version: str,
    ) -> object:
        adapter = self._adapters.get((component_key, version))
        if adapter is not None:
            return adapter
        if any(key == component_key for key, _ in self._adapters):
            raise ValueError(
                "Version runtime incompatible pour le component "
                f"{component_key}@{version}."
            )
        raise ValueError(
            "Runtime adapter absent pour le component "
            f"{component_key}@{version}."
        )


class KaliokNormalizationAdapter:
    """Adapter real for the existing ContentNormalizationService."""

    def execute(
        self,
        session: Session,
        *,
        binding: ComponentBinding,
        document_version_id: UUID,
        perception_processing_run_id: UUID,
        execution_context: ExecutionContext,
        pipeline_metadata: Mapping[str, object],
    ) -> ContentNormalizationResult:
        if binding.configuration:
            raise ValueError(
                "Configuration non supportée par le normalizer Kaliok : "
                f"{', '.join(sorted(binding.configuration))}."
            )
        return ContentNormalizationService(session).normalize(
            document_version_id,
            perception_processing_run_id=perception_processing_run_id,
            execution_context=execution_context,
            pipeline_metadata=pipeline_metadata,
        )


class KaliokPerceptionAdapter:
    """Adapter for the existing Kaliok reader/storage perception path."""

    def execute(
        self,
        session: Session,
        *,
        binding: ComponentBinding,
        document_version_id: UUID,
        execution_context: ExecutionContext,
        pipeline_metadata: Mapping[str, object],
    ) -> object:
        if binding.configuration:
            raise ValueError(
                "Configuration non supportée par kaliok-reader : "
                f"{', '.join(sorted(binding.configuration))}."
            )
        version = session.get(DocumentVersion, document_version_id)
        if version is None:
            raise ValueError(f"DocumentVersion inconnue : {document_version_id}.")
        document_content = read_document(version.storage_uri)
        return store_document_perception(
            session,
            version,
            document_content,
            execution_context=execution_context,
            activate_as_current=False,
            pipeline_metadata=pipeline_metadata,
        )


@dataclass(frozen=True)
class ManifestExecutionResult:
    result: ContentNormalizationResult | ProcessingRun
    binding_key: str
    component_key: str
    component_version: str
    capability: str
    manifest_hash: str
    document_version_id: UUID | None = None
    artifact_metadata: Mapping[str, object] = field(default_factory=dict)

    @property
    def processing_run_id(self) -> UUID:
        if isinstance(self.result, ProcessingRun):
            return self.result.id
        return self.result.processing_run_id


class ManifestExecutionService:
    """Execute exactly one selected capability through a manifest binding."""

    def __init__(
        self,
        component_registry: ComponentRegistry,
        runtime_registry: ComponentRuntimeRegistry,
    ) -> None:
        self._component_registry = component_registry
        self._runtime_registry = runtime_registry

    def execute_normalization(
        self,
        session: Session,
        *,
        manifest: PipelineManifest,
        document_version_id: UUID,
        perception_processing_run_id: UUID,
        execution_context: ExecutionContext,
    ) -> ManifestExecutionResult:
        return self.execute_capability(
            session,
            manifest=manifest,
            capability="normalization",
            document_version_id=document_version_id,
            perception_processing_run_id=perception_processing_run_id,
            execution_context=execution_context,
        )

    def execute_capability(
        self,
        session: Session,
        *,
        manifest: PipelineManifest,
        capability: str,
        document_version_id: UUID,
        perception_processing_run_id: UUID,
        execution_context: ExecutionContext,
    ) -> ManifestExecutionResult:
        if capability != "normalization":
            raise ValueError(
                "La capability manifest-driven exécutable dans ce lot est "
                "normalization."
            )
        manifest.validate(self._component_registry)
        binding = self._select_binding(manifest, capability)
        adapter = self._runtime_registry.resolve(
            binding.component_key,
            binding.component_version,
        )
        if not hasattr(adapter, "execute"):
            raise ValueError("Runtime adapter absent ou invalide pour le binding.")
        pipeline_metadata = self._pipeline_metadata(manifest, binding, capability)
        result = adapter.execute(
            session,
            binding=binding,
            document_version_id=document_version_id,
            perception_processing_run_id=perception_processing_run_id,
            execution_context=execution_context,
            pipeline_metadata=pipeline_metadata,
        )
        return ManifestExecutionResult(
            result=result,
            binding_key=binding.binding_key,
            component_key=binding.component_key,
            component_version=binding.component_version,
            capability=capability,
            manifest_hash=manifest.manifest_hash,
            document_version_id=result.document_version_id,
            artifact_metadata={
                "processing_run_id": result.processing_run_id,
            },
        )

    def execute_document_extraction(
        self,
        session: Session,
        *,
        manifest: PipelineManifest,
        document_version_id: UUID,
        execution_context: ExecutionContext,
    ) -> ManifestExecutionResult:
        capability = "document_extraction"
        manifest.validate(self._component_registry)
        binding = self._select_binding(manifest, capability)
        adapter = self._runtime_registry.resolve(
            binding.component_key,
            binding.component_version,
        )
        if not hasattr(adapter, "execute"):
            raise ValueError("Runtime adapter absent ou invalide pour le binding.")
        result = adapter.execute(
            session,
            binding=binding,
            document_version_id=document_version_id,
            execution_context=execution_context,
            pipeline_metadata=self._pipeline_metadata(manifest, binding, capability),
        )
        if not isinstance(result, ProcessingRun):
            raise ValueError(
                "Le résultat de document_extraction ne référence pas un ProcessingRun."
            )
        return ManifestExecutionResult(
            result=result,
            binding_key=binding.binding_key,
            component_key=binding.component_key,
            component_version=binding.component_version,
            capability=capability,
            manifest_hash=manifest.manifest_hash,
            document_version_id=result.document_version_id,
            artifact_metadata={"processing_run_id": result.id},
        )

    def execute_document_pipeline(
        self,
        session: Session,
        *,
        manifest: PipelineManifest,
        document_version_id: UUID,
        execution_context: ExecutionContext,
    ) -> "DocumentPipelineExecutionResult":
        perception = self.execute_document_extraction(
            session,
            manifest=manifest,
            document_version_id=document_version_id,
            execution_context=execution_context,
        )
        normalization = self.execute_normalization(
            session,
            manifest=manifest,
            document_version_id=document_version_id,
            perception_processing_run_id=perception.processing_run_id,
            execution_context=execution_context,
        )
        return DocumentPipelineExecutionResult(
            perception=perception,
            normalization=normalization,
        )

    @staticmethod
    def _pipeline_metadata(
        manifest: PipelineManifest,
        binding: ComponentBinding,
        capability: str,
    ) -> dict[str, object]:
        return {
            "pipeline_key": manifest.pipeline_key,
            "revision": manifest.revision,
            "manifest_hash": manifest.manifest_hash,
            "binding_key": binding.binding_key,
            "component_key": binding.component_key,
            "component_version": binding.component_version,
            "capability": capability,
            "binding_configuration": binding.to_dict()["configuration"],
        }

    @staticmethod
    def _select_binding(
        manifest: PipelineManifest,
        capability: str,
    ) -> ComponentBinding:
        candidates = [
            binding
            for binding in manifest.bindings
            if binding.enabled and capability in binding.capabilities
        ]
        if not candidates:
            raise ValueError(
                f"Aucun binding actif ne fournit la capability {capability}."
            )
        if len(candidates) > 1:
            keys = ", ".join(binding.binding_key for binding in candidates)
            raise ValueError(
                f"Plusieurs bindings fournissent {capability} sans stratégie "
                f"de sélection explicite : {keys}."
            )
        return candidates[0]


@dataclass(frozen=True)
class DocumentPipelineExecutionResult:
    perception: ManifestExecutionResult
    normalization: ManifestExecutionResult


def build_kaliok_runtime_registry() -> ComponentRuntimeRegistry:
    registry = ComponentRuntimeRegistry()
    registry.register(
        PERCEPTION_ENGINE,
        PERCEPTION_VERSION,
        KaliokPerceptionAdapter(),
    )
    registry.register(
        "kaliok-normalizer",
        ENGINE_VERSION,
        KaliokNormalizationAdapter(),
    )
    return registry


__all__ = [
    "ComponentRuntimeRegistry",
    "DocumentPipelineExecutionResult",
    "KaliokPerceptionAdapter",
    "KaliokNormalizationAdapter",
    "ManifestExecutionResult",
    "ManifestExecutionService",
    "NormalizationRuntimeAdapter",
    "build_kaliok_runtime_registry",
]
