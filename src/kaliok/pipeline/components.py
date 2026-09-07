from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, TypeAlias

from kaliok.hashing import canonical_json_bytes


Capability: TypeAlias = str
JSONMapping: TypeAlias = Mapping[str, Any]


def _identifier(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} doit être non vide.")
    return value.strip()


def _identifiers(values: tuple[str, ...] | list[str], label: str) -> tuple[str, ...]:
    result = tuple(_identifier(value, label) for value in values)
    if len(set(result)) != len(result):
        raise ValueError(f"{label} ne doit pas contenir de doublons.")
    return result


def _configuration(value: Mapping[str, Any], label: str) -> MappingProxyType:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} doit être un objet JSON.")
    copied = deepcopy(dict(value))
    try:
        canonical_json_bytes(copied)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} doit être sérialisable en JSON.") from error
    return MappingProxyType(copied)


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    return deepcopy(value)


@dataclass(frozen=True)
class ComponentDefinition:
    """Immutable description of what a component can provide in general."""

    component_key: str
    version: str
    provides: tuple[Capability, ...]
    requires: tuple[Capability, ...] = ()
    configuration_schema: JSONMapping = field(default_factory=dict)
    metadata: JSONMapping = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "component_key",
            _identifier(self.component_key, "component_key"),
        )
        object.__setattr__(self, "version", _identifier(self.version, "version"))
        object.__setattr__(
            self,
            "provides",
            _identifiers(tuple(self.provides), "provides"),
        )
        object.__setattr__(
            self,
            "requires",
            _identifiers(tuple(self.requires), "requires"),
        )
        object.__setattr__(
            self,
            "configuration_schema",
            _configuration(self.configuration_schema, "configuration_schema"),
        )
        object.__setattr__(self, "metadata", _configuration(self.metadata, "metadata"))

    @property
    def identity(self) -> tuple[str, str]:
        return self.component_key, self.version

    def to_dict(self) -> dict[str, Any]:
        return {
            "component_key": self.component_key,
            "version": self.version,
            "provides": list(self.provides),
            "requires": list(self.requires),
            "configuration_schema": _json_value(self.configuration_schema),
            "metadata": _json_value(self.metadata),
        }


@dataclass(frozen=True)
class ComponentBinding:
    """Immutable use of one component definition inside one manifest."""

    binding_key: str
    component_key: str
    component_version: str
    capabilities: tuple[Capability, ...]
    configuration: JSONMapping = field(default_factory=dict)
    dependencies: tuple[str, ...] = ()
    enabled: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "binding_key",
            _identifier(self.binding_key, "binding_key"),
        )
        object.__setattr__(
            self,
            "component_key",
            _identifier(self.component_key, "component_key"),
        )
        object.__setattr__(
            self,
            "component_version",
            _identifier(self.component_version, "component_version"),
        )
        normalized_capabilities = _identifiers(
            tuple(self.capabilities),
            "capabilities",
        )
        if self.enabled and not normalized_capabilities:
            raise ValueError("Un binding actif doit fournir au moins une capability.")
        object.__setattr__(self, "capabilities", normalized_capabilities)
        object.__setattr__(
            self,
            "dependencies",
            _identifiers(tuple(self.dependencies), "dependencies"),
        )
        object.__setattr__(
            self,
            "configuration",
            _configuration(self.configuration, "configuration"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "binding_key": self.binding_key,
            "component_key": self.component_key,
            "component_version": self.component_version,
            "capabilities": list(self.capabilities),
            "configuration": _json_value(self.configuration),
            "dependencies": list(self.dependencies),
            "enabled": self.enabled,
        }


class ComponentRegistry:
    """Small in-memory registry for component definitions."""

    def __init__(
        self,
        definitions: tuple[ComponentDefinition, ...] | list[ComponentDefinition] = (),
    ) -> None:
        self._definitions: dict[tuple[str, str], ComponentDefinition] = {}
        for definition in definitions:
            self.register(definition)

    def register(self, definition: ComponentDefinition) -> None:
        if definition.identity in self._definitions:
            key, version = definition.identity
            raise ValueError(
                f"Component déjà enregistré : {key}@{version}."
            )
        self._definitions[definition.identity] = definition

    def get(self, component_key: str, version: str) -> ComponentDefinition | None:
        return self._definitions.get((component_key, version))

    @property
    def definitions(self) -> tuple[ComponentDefinition, ...]:
        """Return the registered definitions in registry order."""
        return tuple(self._definitions.values())

    @property
    def capabilities(self) -> tuple[Capability, ...]:
        """Return known capabilities without introducing a second taxonomy."""
        values: list[Capability] = []
        for definition in self.definitions:
            for capability in definition.provides:
                if capability not in values:
                    values.append(capability)
        return tuple(values)

    def for_capability(self, capability: Capability) -> tuple[ComponentDefinition, ...]:
        return tuple(
            definition
            for definition in self.definitions
            if capability in definition.provides
        )

    def require(self, component_key: str, version: str) -> ComponentDefinition:
        definition = self.get(component_key, version)
        if definition is None:
            raise ValueError(
                "Component inconnu : "
                f"{component_key}@{version}."
            )
        return definition

    def validate_binding(self, binding: ComponentBinding) -> None:
        definition = self.require(binding.component_key, binding.component_version)
        unsupported = set(binding.capabilities) - set(definition.provides)
        if unsupported:
            raise ValueError(
                f"Le component {binding.component_key}@{binding.component_version} "
                "ne fournit pas les capabilities demandées : "
                f"{', '.join(sorted(unsupported))}."
            )

    def validate_manifest(self, manifest: Any) -> None:
        for binding in manifest.bindings:
            self.validate_binding(binding)


__all__ = [
    "Capability",
    "ComponentBinding",
    "ComponentDefinition",
    "ComponentRegistry",
]
