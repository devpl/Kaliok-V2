from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

from kaliok.hashing import canonical_json_bytes, canonical_json_hash
from kaliok.pipeline.components import ComponentBinding, ComponentRegistry


@dataclass(frozen=True)
class PipelineManifest:
    """Immutable, ordered and hashable description of a pipeline composition."""

    pipeline_key: str
    revision: str
    bindings: tuple[ComponentBinding, ...] = ()
    runtime_metadata: Mapping[str, Any] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.pipeline_key, str) or not self.pipeline_key.strip():
            raise ValueError("pipeline_key doit être non vide.")
        if not isinstance(self.revision, str) or not self.revision.strip():
            raise ValueError("revision doit être non vide.")
        bindings = tuple(self.bindings)
        binding_keys = [binding.binding_key for binding in bindings]
        if len(set(binding_keys)) != len(binding_keys):
            raise ValueError("Les binding_key du manifest ne doivent pas être dupliqués.")
        known_keys = set(binding_keys)
        for binding in bindings:
            unknown = set(binding.dependencies) - known_keys
            if unknown:
                raise ValueError(
                    "Dépendance de binding inconnue : "
                    f"{', '.join(sorted(unknown))}."
                )
            if binding.binding_key in binding.dependencies:
                raise ValueError(
                    f"Le binding {binding.binding_key} ne peut pas dépendre de lui-même."
                )
        object.__setattr__(self, "pipeline_key", self.pipeline_key.strip())
        object.__setattr__(self, "revision", self.revision.strip())
        object.__setattr__(self, "bindings", bindings)
        object.__setattr__(
            self,
            "runtime_metadata",
            MappingProxyType(deepcopy(dict(self.runtime_metadata))),
        )
        try:
            canonical_json_bytes(self.to_dict())
        except (TypeError, ValueError) as error:
            raise ValueError("Le manifest doit être sérialisable en JSON.") from error

    def validate(self, registry: ComponentRegistry | None = None) -> None:
        """Validate structural rules and, optionally, component capabilities."""
        if registry is not None:
            registry.validate_manifest(self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pipeline_key": self.pipeline_key,
            "revision": self.revision,
            "bindings": [binding.to_dict() for binding in self.bindings],
        }

    def canonical_json(self) -> str:
        return canonical_json_bytes(self.to_dict()).decode("utf-8")

    @property
    def manifest_hash(self) -> str:
        return canonical_json_hash(self.to_dict())

    def configuration_snapshot(self) -> dict[str, Any]:
        """Return a JSON payload suitable for a ProcessingRun snapshot."""
        return {
            "pipeline_manifest": self.to_dict(),
            "pipeline_manifest_hash": self.manifest_hash,
        }


__all__ = ["PipelineManifest"]
