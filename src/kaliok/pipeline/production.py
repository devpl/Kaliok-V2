from kaliok.indexing.service import PERCEPTION_ENGINE, PERCEPTION_VERSION
from kaliok.normalization.service import ENGINE_VERSION
from kaliok.pipeline.components import ComponentBinding
from kaliok.pipeline.manifest import PipelineManifest


def build_current_production_manifest() -> PipelineManifest:
    """Describe only the identifiable perception/normalization P subset."""
    return PipelineManifest(
        pipeline_key="pipeline-p",
        revision="partial-v1",
        bindings=(
            ComponentBinding(
                binding_key="perception",
                component_key=PERCEPTION_ENGINE,
                component_version=PERCEPTION_VERSION,
                capabilities=("document_extraction",),
            ),
            ComponentBinding(
                binding_key="normalization",
                component_key="kaliok-normalizer",
                component_version=ENGINE_VERSION,
                capabilities=("normalization",),
                dependencies=("perception",),
            ),
        ),
    )


__all__ = ["build_current_production_manifest"]
