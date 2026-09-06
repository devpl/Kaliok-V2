from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from kaliok.pipeline.manifest import PipelineManifest


@dataclass(frozen=True)
class PipelineManifestComparison:
    """Fact-based structural comparison; it deliberately has no quality score."""

    pipeline_p: dict[str, Any]
    pipeline_a: dict[str, Any]
    components: dict[str, list[dict[str, Any]]]
    binding_changes: dict[str, dict[str, Any]]
    capability_grouping_changes: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "pipeline_p": self.pipeline_p,
            "pipeline_a": self.pipeline_a,
            "components": self.components,
            "binding_changes": self.binding_changes,
            "capability_grouping_changes": self.capability_grouping_changes,
        }


class PipelineManifestComparator:
    """Stateless facade for callers that prefer an object-based comparator."""

    @staticmethod
    def compare(
        pipeline_p: PipelineManifest,
        pipeline_a: PipelineManifest,
    ) -> PipelineManifestComparison:
        return compare_manifests(pipeline_p, pipeline_a)


def compare_manifests(
    pipeline_p: PipelineManifest,
    pipeline_a: PipelineManifest,
) -> PipelineManifestComparison:
    """Return only factual changes between two ordered manifests."""
    bindings_p = {binding.binding_key: binding for binding in pipeline_p.bindings}
    bindings_a = {binding.binding_key: binding for binding in pipeline_a.bindings}
    added_keys = sorted(set(bindings_a) - set(bindings_p))
    removed_keys = sorted(set(bindings_p) - set(bindings_a))

    added = [
        _component_payload(bindings_a[key])
        for key in added_keys
    ]
    removed = [
        _component_payload(bindings_p[key])
        for key in removed_keys
    ]
    changes: dict[str, dict[str, Any]] = {}
    for key in sorted(set(bindings_p) & set(bindings_a)):
        first = bindings_p[key]
        second = bindings_a[key]
        binding_changes: dict[str, Any] = {}
        for field_name in (
            "component_key",
            "component_version",
            "configuration",
            "capabilities",
            "dependencies",
            "enabled",
        ):
            first_value = getattr(first, field_name)
            second_value = getattr(second, field_name)
            if first_value != second_value:
                binding_changes[field_name] = {
                    "pipeline_p": _json_value(first_value),
                    "pipeline_a": _json_value(second_value),
                }
        if binding_changes:
            changes[key] = binding_changes

    grouping_changes = _grouping_changes(pipeline_p, pipeline_a)
    return PipelineManifestComparison(
        pipeline_p=_manifest_payload(pipeline_p),
        pipeline_a=_manifest_payload(pipeline_a),
        components={"added": added, "removed": removed},
        binding_changes=changes,
        capability_grouping_changes=grouping_changes,
    )


def _manifest_payload(manifest: PipelineManifest) -> dict[str, Any]:
    return {
        "pipeline_key": manifest.pipeline_key,
        "revision": manifest.revision,
        "manifest_hash": manifest.manifest_hash,
    }


def _component_payload(binding) -> dict[str, Any]:
    return {
        "binding_key": binding.binding_key,
        "component_key": binding.component_key,
        "component_version": binding.component_version,
        "capabilities": list(binding.capabilities),
    }


def _grouping_changes(
    pipeline_p: PipelineManifest,
    pipeline_a: PipelineManifest,
) -> list[dict[str, Any]]:
    groups_p = _capability_groups(pipeline_p)
    groups_a = _capability_groups(pipeline_a)
    changes = []
    for capability in sorted(set(groups_p) | set(groups_a)):
        if groups_p.get(capability, []) != groups_a.get(capability, []):
            changes.append(
                {
                    "capability": capability,
                    "pipeline_p_bindings": groups_p.get(capability, []),
                    "pipeline_a_bindings": groups_a.get(capability, []),
                }
            )
    return changes


def _capability_groups(manifest: PipelineManifest) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for binding in manifest.bindings:
        if not binding.enabled:
            continue
        for capability in binding.capabilities:
            groups.setdefault(capability, []).append(binding.binding_key)
    return groups


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    return value


__all__ = [
    "PipelineManifestComparison",
    "PipelineManifestComparator",
    "compare_manifests",
]
