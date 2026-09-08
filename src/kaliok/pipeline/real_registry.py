"""Fact-based registry of the replaceable Kaliok components currently present."""

from kaliok.indexing.service import (
    CHUNKING_STRATEGY,
    CHUNKING_VERSION,
    PERCEPTION_ENGINE,
    PERCEPTION_VERSION,
)
from kaliok.normalization.service import ENGINE_VERSION
from kaliok.pipeline.components import ComponentDefinition, ComponentRegistry
from kaliok.rag_runtime.postgres import (
    NORMALIZED_CHUNKING_STRATEGY,
    NORMALIZED_CHUNKING_VERSION,
)


def build_static_kaliok_component_registry() -> ComponentRegistry:
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
                configuration_schema={
                    "type": "object",
                    "required": ["detectors"],
                    "properties": {
                        "detectors": {
                            "type": "array",
                            "minItems": 1,
                            "items": {
                                "type": "object",
                                "required": ["terms"],
                                "properties": {
                                    "key": {"const": "lexical_dictionary"},
                                    "version": {"const": "1"},
                                    "terms": {
                                        "type": "array",
                                        "minItems": 1,
                                        "items": {
                                            "type": "object",
                                            "required": ["value", "candidate_type"],
                                        },
                                    },
                                    "case_sensitive": {"type": "boolean"},
                                    "boundary_policy": {
                                        "enum": ["unicode_word", "substring"],
                                    },
                                },
                            },
                        },
                    },
                },
                metadata={
                    "implementation": "kaliok.discovery.CandidateDiscoveryService",
                    "configuration_note": (
                        "Détecteurs et termes explicites requis; aucun dictionnaire implicite."
                    ),
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
                version=f"{CHUNKING_STRATEGY}@{CHUNKING_VERSION}",
                provides=("chunking",),
                metadata={
                    "source": "kaliok.indexing.service.CHUNKING_STRATEGY",
                    "scope": "semantic chunking used by index_document",
                },
            ),
            ComponentDefinition(
                component_key="postgres-normalized-index",
                version=(
                    f"{NORMALIZED_CHUNKING_STRATEGY}@"
                    f"{NORMALIZED_CHUNKING_VERSION}"
                ),
                provides=("indexing",),
                metadata={
                    "implementation": "kaliok.rag_runtime.postgres.PostgresVectorIndexStore",
                },
            ),
        ]
    )


def build_kaliok_component_registry(session=None) -> ComponentRegistry:
    """Return the DB projection when a session is supplied.

    The no-argument form remains a compatibility fallback for legacy callers
    and for environments before the catalogue migration/bootstrap is applied.
    """
    if session is None:
        return build_static_kaliok_component_registry()
    from kaliok.pipeline.persistence import build_component_registry_from_db

    return build_component_registry_from_db(session)


__all__ = [
    "build_kaliok_component_registry",
    "build_static_kaliok_component_registry",
]
