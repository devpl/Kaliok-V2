from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol
from uuid import UUID

from sqlmodel import Session

from kaliok.documents.reader import read_document
from kaliok.discovery import (
    CandidateDiscoveryResult,
    CandidateDiscoveryService,
    LexicalCandidateDetector,
    LexicalTerm,
)
from kaliok.execution import ExecutionContext
from kaliok.hashing import canonical_json_hash
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

    def has(self, component_key: str, version: str) -> bool:
        return (component_key, version) in self._adapters


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


class KaliokDiscoveryAdapter:
    """Adapter for candidate discovery with an explicit manifest configuration.

    The Pipeline Lab must provide detector terms in the binding configuration;
    this adapter deliberately never falls back to a development dictionary.
    """

    def execute(
        self,
        session: Session,
        *,
        binding: ComponentBinding,
        document_version_id: UUID,
        normalization_processing_run_id: UUID,
        execution_context: ExecutionContext,
        pipeline_metadata: Mapping[str, object],
    ) -> CandidateDiscoveryResult:
        detectors = self._detectors(binding.configuration)
        result = CandidateDiscoveryService(session).discover(
            document_version_id,
            normalization_processing_run_id,
            detectors,
            execution_context=execution_context,
        )
        run = session.get(ProcessingRun, result.processing_run_id)
        if run is not None:
            configuration = dict(run.configuration or {})
            configuration["pipeline"] = dict(pipeline_metadata)
            run.configuration = configuration
            run.configuration_hash = canonical_json_hash(configuration)
            session.add(run)
            session.flush()
        return result

    @staticmethod
    def _detectors(configuration: Mapping[str, object]) -> tuple[LexicalCandidateDetector, ...]:
        if not isinstance(configuration, Mapping):
            raise ValueError("La configuration de entity_discovery doit être un objet JSON.")
        raw_detectors = configuration.get("detectors")
        if not isinstance(raw_detectors, list) or not raw_detectors:
            raise ValueError(
                "entity_discovery exige une configuration explicite 'detectors' "
                "avec au moins un détecteur lexical; aucun dictionnaire implicite n'est utilisé."
            )
        detectors: list[LexicalCandidateDetector] = []
        for index, raw_detector in enumerate(raw_detectors, start=1):
            if not isinstance(raw_detector, Mapping):
                raise ValueError(f"Le détecteur {index} doit être un objet JSON.")
            key = raw_detector.get("key", "lexical_dictionary")
            version = raw_detector.get("version", "1")
            if key != "lexical_dictionary" or version != "1":
                raise ValueError(
                    f"Détecteur non raccordé : {key}@{version}. "
                    "Seul lexical_dictionary@1 est disponible dans ce runtime."
                )
            raw_terms = raw_detector.get("terms")
            if not isinstance(raw_terms, list) or not raw_terms:
                raise ValueError(f"Le détecteur {index} doit préciser une liste 'terms' non vide.")
            terms: list[LexicalTerm] = []
            for term_index, raw_term in enumerate(raw_terms, start=1):
                if not isinstance(raw_term, Mapping):
                    raise ValueError(f"Le terme {index}.{term_index} doit être un objet JSON.")
                value = raw_term.get("value")
                candidate_type = raw_term.get("candidate_type")
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"Le terme {index}.{term_index} doit préciser 'value'.")
                if not isinstance(candidate_type, str) or not candidate_type.strip():
                    raise ValueError(
                        f"Le terme {index}.{term_index} doit préciser 'candidate_type'."
                    )
                payload = raw_term.get("payload", {})
                if not isinstance(payload, Mapping):
                    raise ValueError(f"Le payload du terme {index}.{term_index} doit être un objet JSON.")
                confidence = raw_term.get("confidence")
                if confidence is not None and not isinstance(confidence, (int, float)):
                    raise ValueError(
                        f"La confiance du terme {index}.{term_index} doit être numérique ou nulle."
                    )
                terms.append(
                    LexicalTerm(
                        value=value,
                        candidate_type=candidate_type,
                        normalized_value=raw_term.get("normalized_value"),
                        payload=dict(payload),
                        confidence=confidence,
                    )
                )
            detectors.append(
                LexicalCandidateDetector(
                    terms,
                    case_sensitive=bool(raw_detector.get("case_sensitive", False)),
                    boundary_policy=raw_detector.get("boundary_policy", "unicode_word"),
                )
            )
        return tuple(detectors)


@dataclass(frozen=True)
class ManifestExecutionResult:
    result: ContentNormalizationResult | CandidateDiscoveryResult | ProcessingRun
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

    def execute_entity_discovery(
        self,
        session: Session,
        *,
        manifest: PipelineManifest,
        document_version_id: UUID,
        normalization_processing_run_id: UUID,
        execution_context: ExecutionContext,
    ) -> ManifestExecutionResult:
        capability = "entity_discovery"
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
            normalization_processing_run_id=normalization_processing_run_id,
            execution_context=execution_context,
            pipeline_metadata=self._pipeline_metadata(manifest, binding, capability),
        )
        return ManifestExecutionResult(
            result=result,
            binding_key=binding.binding_key,
            component_key=binding.component_key,
            component_version=binding.component_version,
            capability=capability,
            manifest_hash=manifest.manifest_hash,
            document_version_id=result.document_version_id,
            artifact_metadata={"processing_run_id": result.processing_run_id},
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
        discovery = None
        if any(
            binding.enabled and "entity_discovery" in binding.capabilities
            for binding in manifest.bindings
        ):
            discovery = self.execute_entity_discovery(
                session,
                manifest=manifest,
                document_version_id=document_version_id,
                normalization_processing_run_id=normalization.processing_run_id,
                execution_context=execution_context,
            )
        return DocumentPipelineExecutionResult(
            perception=perception,
            normalization=normalization,
            discovery=discovery,
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
    discovery: ManifestExecutionResult | None = None


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
    registry.register(
        "kaliok-candidate-discovery",
        "candidate-discovery-v1",
        KaliokDiscoveryAdapter(),
    )
    return registry


__all__ = [
    "ComponentRuntimeRegistry",
    "DocumentPipelineExecutionResult",
    "KaliokPerceptionAdapter",
    "KaliokNormalizationAdapter",
    "KaliokDiscoveryAdapter",
    "ManifestExecutionResult",
    "ManifestExecutionService",
    "NormalizationRuntimeAdapter",
    "build_kaliok_runtime_registry",
]
