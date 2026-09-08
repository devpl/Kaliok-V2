from kaliok.pipeline.components import (
    Capability,
    ComponentBinding,
    ComponentDefinition,
)
from kaliok.pipeline.comparison import (
    PipelineManifestComparison,
    PipelineManifestComparator,
    compare_manifests,
)
from kaliok.pipeline.manifest import PipelineManifest
from kaliok.pipeline.registry import ComponentRegistry
from kaliok.pipeline.production import build_current_production_manifest
from kaliok.pipeline.real_registry import (
    build_kaliok_component_registry,
    build_static_kaliok_component_registry,
)
from kaliok.pipeline.persistence import (
    PipelinePersistenceService,
    bootstrap_catalog,
    build_component_registry_from_db,
    persist_manifest,
)
from kaliok.pipeline.runtime import (
    ComponentRuntimeRegistry,
    DocumentPipelineExecutionResult,
    KaliokDiscoveryAdapter,
    KaliokEntityResolutionAdapter,
    KaliokPerceptionAdapter,
    KaliokNormalizationAdapter,
    ManifestExecutionResult,
    ManifestExecutionService,
    build_kaliok_runtime_registry,
)

__all__ = [
    "Capability",
    "ComponentBinding",
    "ComponentDefinition",
    "ComponentRegistry",
    "PipelineManifest",
    "PipelineManifestComparison",
    "PipelineManifestComparator",
    "compare_manifests",
    "ComponentRuntimeRegistry",
    "DocumentPipelineExecutionResult",
    "KaliokDiscoveryAdapter",
    "KaliokEntityResolutionAdapter",
    "KaliokPerceptionAdapter",
    "KaliokNormalizationAdapter",
    "ManifestExecutionResult",
    "ManifestExecutionService",
    "build_current_production_manifest",
    "build_kaliok_component_registry",
    "build_static_kaliok_component_registry",
    "build_kaliok_runtime_registry",
    "PipelinePersistenceService",
    "bootstrap_catalog",
    "build_component_registry_from_db",
    "persist_manifest",
]
