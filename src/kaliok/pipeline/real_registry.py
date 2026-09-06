"""Fact-based registry of the replaceable Kaliok components currently present."""

from kaliok.indexing.service import PERCEPTION_ENGINE, PERCEPTION_VERSION
from kaliok.normalization.service import ENGINE_VERSION
from kaliok.pipeline.components import ComponentDefinition, ComponentRegistry


def build_kaliok_component_registry() -> ComponentRegistry:
    """Build definitions from existing engine/strategy constants only.

    This registry deliberately describes a partial inventory. Components whose
    runtime version is supplied dynamically (for example Docling and Ollama)
    are not assigned an invented product version here.
    """
    return ComponentRegistry(
        [
            ComponentDefinition(
                component_key=PERCEPTION_ENGINE,
                version=PERCEPTION_VERSION,
                provides=("document_extraction",),
                metadata={
                    "source": "kaliok.indexing.service.PERCEPTION_ENGINE",
                    "scope": "stored page/content-block perception",
                },
            ),
            ComponentDefinition(
                component_key="kaliok-normalizer",
                version=ENGINE_VERSION,
                provides=("normalization",),
                metadata={
                    "implementation": "kaliok.normalization.ContentNormalizationService",
                },
            ),
            ComponentDefinition(
                component_key="kaliok-candidate-discovery",
                version="candidate-discovery-v1",
                provides=("entity_discovery",),
                metadata={
                    "implementation": "kaliok.discovery.CandidateDiscoveryService",
                },
            ),
            ComponentDefinition(
                component_key="kaliok-entity-resolution",
                version="declared-normalized-exact-v1",
                provides=("entity_resolution",),
                metadata={
                    "implementation": "kaliok.entity_resolution.EntityResolutionService",
                },
            ),
            ComponentDefinition(
                component_key="kaliok-semantic-chunker",
                version="llamaindex-semantic-cleaning-v1",
                provides=("chunking",),
                metadata={
                    "source": "kaliok.indexing.service.CHUNKING_STRATEGY",
                    "scope": "semantic chunking used by index_document",
                },
            ),
            ComponentDefinition(
                component_key="postgres-normalized-index",
                version="normalized-content-unit-v1",
                provides=("indexing",),
                metadata={
                    "implementation": "kaliok.rag_runtime.postgres.PostgresVectorIndexStore",
                },
            ),
        ]
    )


__all__ = ["build_kaliok_component_registry"]
