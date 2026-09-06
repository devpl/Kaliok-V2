from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from kaliok.configuration import bootstrap_rag_configuration
from kaliok.execution import ExecutionContext, apply_execution_context
from kaliok.pipeline import (
    ComponentBinding,
    ComponentDefinition,
    ComponentRegistry,
    PipelineManifest,
    compare_manifests,
)
from kaliok.storage.models import ProcessingRun


def _registry() -> ComponentRegistry:
    return ComponentRegistry(
        [
            ComponentDefinition(
                component_key="extraction_tool",
                version="1.0",
                provides=("document_extraction",),
            ),
            ComponentDefinition(
                component_key="normalizer",
                version="2.0",
                provides=("normalization",),
            ),
            ComponentDefinition(
                component_key="multimodal_tool",
                version="3.0",
                provides=(
                    "document_extraction",
                    "layout_analysis",
                    "normalization",
                ),
            ),
        ]
    )


def test_component_definition_supports_one_and_many_capabilities_and_is_immutable():
    simple = ComponentDefinition(
        component_key="docling",
        version="2.5",
        provides=("document_extraction",),
    )
    multi = ComponentDefinition(
        component_key="future-tool",
        version="0.1",
        provides=("document_extraction", "ocr", "layout_analysis"),
        requires=("document_extraction",),
        metadata={"vendor": "test"},
    )

    assert simple.provides == ("document_extraction",)
    assert multi.provides == (
        "document_extraction",
        "ocr",
        "layout_analysis",
    )
    assert multi.to_dict()["metadata"] == {"vendor": "test"}
    with pytest.raises(FrozenInstanceError):
        simple.version = "3.0"  # type: ignore[misc]


def test_registry_rejects_unknown_components_and_unsupported_capabilities():
    registry = _registry()
    unknown = ComponentBinding(
        binding_key="unknown",
        component_key="missing",
        component_version="1.0",
        capabilities=("ocr",),
    )
    unsupported = ComponentBinding(
        binding_key="extraction",
        component_key="extraction_tool",
        component_version="1.0",
        capabilities=("ocr",),
    )

    with pytest.raises(ValueError, match="Component inconnu"):
        registry.validate_binding(unknown)
    with pytest.raises(ValueError, match="ne fournit pas"):
        registry.validate_binding(unsupported)


def test_manifest_minimal_ordered_and_multi_component_capability_validates():
    registry = _registry()
    manifest = PipelineManifest(
        pipeline_key="pipeline-p",
        revision="1",
        bindings=(
            ComponentBinding(
                binding_key="extract",
                component_key="extraction_tool",
                component_version="1.0",
                capabilities=("document_extraction",),
            ),
            ComponentBinding(
                binding_key="normalize",
                component_key="normalizer",
                component_version="2.0",
                capabilities=("normalization",),
                dependencies=("extract",),
            ),
        ),
    )
    multimodal = PipelineManifest(
        pipeline_key="pipeline-a",
        revision="1",
        bindings=(
            ComponentBinding(
                binding_key="multimodal",
                component_key="multimodal_tool",
                component_version="3.0",
                capabilities=(
                    "document_extraction",
                    "layout_analysis",
                    "normalization",
                ),
            ),
        ),
    )

    manifest.validate(registry)
    multimodal.validate(registry)
    assert [binding.binding_key for binding in manifest.bindings] == [
        "extract",
        "normalize",
    ]
    assert multimodal.bindings[0].capabilities == (
        "document_extraction",
        "layout_analysis",
        "normalization",
    )


@pytest.mark.parametrize(
    "factory",
    [
        lambda: ComponentBinding(
            binding_key="",
            component_key="tool",
            component_version="1",
            capabilities=("ocr",),
        ),
        lambda: ComponentBinding(
            binding_key="tool",
            component_key="tool",
            component_version="",
            capabilities=("ocr",),
        ),
        lambda: ComponentBinding(
            binding_key="tool",
            component_key="tool",
            component_version="1",
            capabilities=(),
        ),
    ],
)
def test_binding_required_fields_are_validated(factory):
    with pytest.raises(ValueError):
        factory()


def test_manifest_rejects_duplicate_or_invalid_dependencies():
    binding = ComponentBinding(
        binding_key="extract",
        component_key="extraction_tool",
        component_version="1.0",
        capabilities=("document_extraction",),
    )
    with pytest.raises(ValueError, match="dupliqués"):
        PipelineManifest(
            pipeline_key="p",
            revision="1",
            bindings=(binding, binding),
        )
    with pytest.raises(ValueError, match="inconnue"):
        PipelineManifest(
            pipeline_key="p",
            revision="1",
            bindings=(
                ComponentBinding(
                    binding_key="extract",
                    component_key="extraction_tool",
                    component_version="1.0",
                    capabilities=("document_extraction",),
                    dependencies=("missing",),
                ),
            ),
        )
    with pytest.raises(ValueError, match="lui-même"):
        PipelineManifest(
            pipeline_key="p",
            revision="1",
            bindings=(
                ComponentBinding(
                    binding_key="extract",
                    component_key="extraction_tool",
                    component_version="1.0",
                    capabilities=("document_extraction",),
                    dependencies=("extract",),
                ),
            ),
        )


def test_configuration_must_be_json_serializable():
    with pytest.raises(ValueError, match="sérialisable en JSON"):
        ComponentBinding(
            binding_key="extract",
            component_key="extraction_tool",
            component_version="1.0",
            capabilities=("document_extraction",),
            configuration={"invalid": object()},
        )


def test_manifest_hash_is_deterministic_order_sensitive_and_runtime_metadata_free():
    first = ComponentBinding(
        binding_key="first",
        component_key="tool-a",
        component_version="1",
        capabilities=("extraction",),
        configuration={"threshold": 1},
    )
    second = ComponentBinding(
        binding_key="second",
        component_key="tool-b",
        component_version="1",
        capabilities=("normalization",),
    )
    manifest = PipelineManifest(
        pipeline_key="p",
        revision="1",
        bindings=(first, second),
        runtime_metadata={"created_at": "one"},
    )
    same = PipelineManifest(
        pipeline_key="p",
        revision="1",
        bindings=(first, second),
        runtime_metadata={"created_at": "two", "metrics": {"x": 1}},
    )
    reversed_manifest = PipelineManifest(
        pipeline_key="p",
        revision="1",
        bindings=(second, first),
    )
    changed_version = PipelineManifest(
        pipeline_key="p",
        revision="1",
        bindings=(
            ComponentBinding(
                binding_key="first",
                component_key="tool-a",
                component_version="2",
                capabilities=("extraction",),
                configuration={"threshold": 1},
            ),
            second,
        ),
    )

    assert manifest.canonical_json() == same.canonical_json()
    assert manifest.manifest_hash == same.manifest_hash
    assert manifest.manifest_hash != reversed_manifest.manifest_hash
    assert manifest.manifest_hash != changed_version.manifest_hash
    assert "created_at" not in manifest.canonical_json()


def test_manifest_snapshot_can_be_embedded_in_existing_run_configuration():
    manifest = PipelineManifest(
        pipeline_key="pipeline-p",
        revision="7",
        bindings=(
            ComponentBinding(
                binding_key="extract",
                component_key="docling",
                component_version="2",
                capabilities=("document_extraction",),
            ),
        ),
    )
    snapshot = manifest.configuration_snapshot()
    run = ProcessingRun(
        process_type="document_extraction",
        status="completed",
        configuration=snapshot,
    )

    assert run.configuration["pipeline_manifest"] == manifest.to_dict()
    assert run.configuration["pipeline_manifest_hash"] == manifest.manifest_hash


def test_manifest_does_not_replace_execution_context_or_processing_run_identity():
    context = ExecutionContext(environment="experiment")
    run = ProcessingRun(
        process_type="document_extraction",
        status="completed",
        configuration={"pipeline_manifest_hash": "hash"},
    )

    assert context.environment == "experiment"
    assert run.execution_environment is None
    assert run.execution_group_id is None
    assert "pipeline_manifest_hash" in run.configuration


def test_manifest_uses_existing_configuration_revision_as_execution_reference(
    configuration_session,
):
    bootstrapped = bootstrap_rag_configuration(configuration_session)
    manifest = PipelineManifest(
        pipeline_key="pipeline-p",
        revision="1",
        bindings=(
            ComponentBinding(
                binding_key="extract",
                component_key="docling",
                component_version="2",
                capabilities=("document_extraction",),
            ),
        ),
    )
    run = ProcessingRun(
        process_type="document_extraction",
        status="completed",
        configuration=manifest.configuration_snapshot(),
    )

    apply_execution_context(
        configuration_session,
        run,
        ExecutionContext(
            environment="production",
            configuration_revision_id=bootstrapped.revision_id,
        ),
    )

    assert run.configuration_revision_id == bootstrapped.revision_id
    assert run.configuration["pipeline_manifest_hash"] == manifest.manifest_hash


def test_p_a_diff_reports_replacement_by_one_multi_capability_component():
    registry = _registry()
    pipeline_p = PipelineManifest(
        pipeline_key="pipeline-p",
        revision="1",
        bindings=(
            ComponentBinding(
                binding_key="extract",
                component_key="extraction_tool",
                component_version="1.0",
                capabilities=("document_extraction",),
            ),
            ComponentBinding(
                binding_key="normalize",
                component_key="normalizer",
                component_version="2.0",
                capabilities=("normalization",),
            ),
        ),
    )
    pipeline_a = PipelineManifest(
        pipeline_key="pipeline-a",
        revision="1",
        bindings=(
            ComponentBinding(
                binding_key="multimodal",
                component_key="multimodal_tool",
                component_version="3.0",
                capabilities=(
                    "document_extraction",
                    "layout_analysis",
                    "normalization",
                ),
            ),
        ),
    )
    pipeline_p.validate(registry)
    pipeline_a.validate(registry)

    diff = compare_manifests(pipeline_p, pipeline_a).to_dict()

    assert [item["component_key"] for item in diff["components"]["added"]] == [
        "multimodal_tool"
    ]
    assert {
        item["component_key"] for item in diff["components"]["removed"]
    } == {"extraction_tool", "normalizer"}
    assert {
        item["capability"] for item in diff["capability_grouping_changes"]
    } == {"document_extraction", "normalization", "layout_analysis"}
    assert "better" not in str(diff).lower()


def test_p_a_diff_reports_version_configuration_and_capability_changes():
    pipeline_p = PipelineManifest(
        pipeline_key="pipeline-p",
        revision="1",
        bindings=(
            ComponentBinding(
                binding_key="retriever",
                component_key="retriever",
                component_version="1",
                capabilities=("retrieval",),
                configuration={"top_k": 5},
            ),
        ),
    )
    pipeline_a = PipelineManifest(
        pipeline_key="pipeline-a",
        revision="2",
        bindings=(
            ComponentBinding(
                binding_key="retriever",
                component_key="retriever",
                component_version="2",
                capabilities=("retrieval", "fusion"),
                configuration={"top_k": 10},
            ),
        ),
    )

    changes = compare_manifests(pipeline_p, pipeline_a).binding_changes["retriever"]

    assert set(changes) == {"component_version", "configuration", "capabilities"}
    assert changes["component_version"] == {
        "pipeline_p": "1",
        "pipeline_a": "2",
    }


def test_disabled_binding_may_be_partial_and_multiple_bindings_can_share_capability():
    manifest = PipelineManifest(
        pipeline_key="partial",
        revision="1",
        bindings=(
            ComponentBinding(
                binding_key="vector",
                component_key="vector",
                component_version="1",
                capabilities=("retrieval",),
            ),
            ComponentBinding(
                binding_key="lexical",
                component_key="lexical",
                component_version="1",
                capabilities=("retrieval",),
            ),
            ComponentBinding(
                binding_key="future",
                component_key="future",
                component_version="0",
                capabilities=(),
                enabled=False,
            ),
        ),
    )

    assert len(manifest.bindings) == 3
    assert manifest.bindings[2].enabled is False
