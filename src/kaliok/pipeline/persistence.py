"""Transactional persistence for the descriptive pipeline catalogue.

The SQLModel records in this module are the descriptive authority.  The
existing dataclasses remain the runtime projection consumed by execution.
"""

from __future__ import annotations

from collections import defaultdict
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from sqlmodel import Session, select

from kaliok.hashing import canonical_json_hash
from kaliok.pipeline.composition import (
    CompositionValidationResult,
    PipelineCompositionService,
)
from kaliok.pipeline.components import ComponentBinding, ComponentDefinition, ComponentRegistry
from kaliok.pipeline.manifest import PipelineManifest
from kaliok.storage.models import (
    ArtifactType,
    Capability,
    CapabilityArtifactContract,
    Component,
    ComponentCapability,
    ComponentVersion,
    PipelineBinding,
    PipelineBindingCapability,
    PipelineBindingDependency,
    PipelineBindingNode,
    PipelineDefinition,
    PipelineRevision,
    RagTemplate,
    RagTemplateEdge,
    RagTemplateCapability,
    RagTemplateDependency,
    RagTemplateNode,
    RagTemplateRevision,
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


CAPABILITY_CATALOG: tuple[dict[str, Any], ...] = (
    {
        "key": "document_extraction",
        "display_name": "Lecture du document",
        "phase_key": "extraction",
        "input": ["raw_document"],
        "output": ["content_blocks"],
    },
    {
        "key": "normalization",
        "display_name": "Structuration du contenu",
        "phase_key": "normalization",
        "input": ["content_blocks"],
        "output": ["normalized_content_units"],
    },
    {
        "key": "entity_discovery",
        "display_name": "Découverte d’entités",
        "phase_key": "discovery",
        "input": ["normalized_content_units"],
        "output": ["discovered_candidates"],
    },
    {
        "key": "entity_resolution",
        "display_name": "Résolution d’entités",
        "phase_key": "resolution",
        "input": ["discovered_candidates"],
        "output": ["entities"],
    },
    {
        "key": "chunking",
        "display_name": "Découpage sémantique",
        "phase_key": "chunking",
        "input": ["normalized_content_units"],
        "output": ["document_chunks"],
    },
    {
        "key": "indexing",
        "display_name": "Indexation",
        "phase_key": "indexing",
        "input": ["document_chunks"],
        "output": ["search_index"],
    },
)

COMPONENT_LABELS = {
    "kaliok-reader": "Kaliok Reader",
    "kaliok-normalizer": "Kaliok Normalizer",
    "kaliok-candidate-discovery": "Kaliok Candidate Discovery",
    "kaliok-entity-resolution": "Kaliok Entity Resolution",
    "kaliok-semantic-chunker": "Kaliok Semantic Chunker",
    "postgres-normalized-index": "PostgreSQL Normalized Index",
}

EDGE_TYPES_KNOWN = frozenset(
    {"normal", "optional", "conditional", "fallback", "parallel", "merge", "loop"}
)
ARTIFACT_STORAGE_KINDS_KNOWN = frozenset(
    {"document", "relational_table", "index", "external", "transient", "composite"}
)
CARDINALITIES_KNOWN = frozenset({"one", "optional_one", "many", "one_or_many"})


@dataclass(frozen=True)
class GraphBackfillValidation:
    """Read-only result for checking the legacy-to-graph projection."""

    revision_id: UUID | None
    legacy_capability_count: int
    graph_node_count: int
    legacy_dependency_count: int
    graph_edge_count: int
    contract_count: int
    artifact_type_count: int
    issues: tuple[str, ...] = ()

    @property
    def is_valid(self) -> bool:
        return not self.issues


def _filter_revision(rows: list[Any], revision_id: UUID | None) -> list[Any]:
    if revision_id is None:
        return rows
    return [row for row in rows if row.rag_template_revision_id == revision_id]


def _catalog_capability(session: Session, key: str) -> Capability:
    item = session.exec(select(Capability).where(Capability.capability_key == key)).first()
    if item is None:
        raise ValueError(f"Capability inconnue dans le catalogue : {key}.")
    return item


class PipelinePersistenceService:
    """Read, validate and write catalogue and versioned pipeline records."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def composition_service(self) -> PipelineCompositionService:
        """Return the additive composition service without changing runtime reads."""
        return PipelineCompositionService(self.session)

    def load_binding_nodes(self, revision_id: UUID | None = None) -> list[PipelineBindingNode]:
        return self.composition_service().load_binding_nodes(revision_id)

    def validate_pipeline_composition(
        self,
        revision_id: UUID,
        *,
        raise_on_error: bool = False,
    ) -> CompositionValidationResult:
        return self.composition_service().validate_pipeline_composition(
            revision_id,
            raise_on_error=raise_on_error,
        )

    def load_artifact_types(self) -> list[ArtifactType]:
        """Load the descriptive artifact catalogue without affecting runtime."""
        return self.session.exec(
            select(ArtifactType).order_by(ArtifactType.artifact_type_key, ArtifactType.version)
        ).all()

    def load_capability_artifact_contracts(
        self,
        capability_id: UUID | None = None,
    ) -> list[CapabilityArtifactContract]:
        statement = select(CapabilityArtifactContract).order_by(
            CapabilityArtifactContract.capability_id,
            CapabilityArtifactContract.direction,
            CapabilityArtifactContract.position,
            CapabilityArtifactContract.id,
        )
        if capability_id is not None:
            statement = statement.where(CapabilityArtifactContract.capability_id == capability_id)
        return self.session.exec(statement).all()

    def load_template_nodes(self, revision_id: UUID | None = None) -> list[RagTemplateNode]:
        statement = select(RagTemplateNode).order_by(
            RagTemplateNode.rag_template_revision_id,
            RagTemplateNode.position,
            RagTemplateNode.node_key,
        )
        if revision_id is not None:
            statement = statement.where(RagTemplateNode.rag_template_revision_id == revision_id)
        return self.session.exec(statement).all()

    def load_template_edges(self, revision_id: UUID | None = None) -> list[RagTemplateEdge]:
        statement = select(RagTemplateEdge).order_by(
            RagTemplateEdge.rag_template_revision_id,
            RagTemplateEdge.priority,
            RagTemplateEdge.edge_key,
        )
        if revision_id is not None:
            statement = statement.where(RagTemplateEdge.rag_template_revision_id == revision_id)
        return self.session.exec(statement).all()

    def compare_legacy_vs_graph(self, revision_id: UUID | None = None) -> dict[str, Any]:
        """Return deterministic differences between legacy rows and graph rows."""
        legacy_capabilities = _filter_revision(
            self.session.exec(select(RagTemplateCapability)).all(), revision_id
        )
        graph_nodes = self.load_template_nodes(revision_id)
        legacy_dependencies = _filter_revision(
            self.session.exec(select(RagTemplateDependency)).all(), revision_id
        )
        graph_edges = self.load_template_edges(revision_id)

        legacy_node_keys = {
            (row.rag_template_revision_id, row.capability_id)
            for row in legacy_capabilities
        }
        graph_node_keys = {
            (row.rag_template_revision_id, row.capability_id)
            for row in graph_nodes
        }
        legacy_edge_keys = {
            (
                row.rag_template_revision_id,
                row.source_capability_id,
                row.target_capability_id,
            )
            for row in legacy_dependencies
        }
        capability_by_node_id = {row.id: row.capability_id for row in graph_nodes}
        graph_edge_keys = {
            (
                row.rag_template_revision_id,
                capability_by_node_id.get(row.source_node_id),
                capability_by_node_id.get(row.target_node_id),
            )
            for row in graph_edges
        }
        return {
            "revision_id": revision_id,
            "legacy_capability_count": len(legacy_capabilities),
            "graph_node_count": len(graph_nodes),
            "legacy_dependency_count": len(legacy_dependencies),
            "graph_edge_count": len(graph_edges),
            "missing_nodes": sorted(legacy_node_keys - graph_node_keys, key=str),
            "extra_nodes": sorted(graph_node_keys - legacy_node_keys, key=str),
            "missing_edges": sorted(legacy_edge_keys - graph_edge_keys, key=str),
            "extra_edges": sorted(graph_edge_keys - legacy_edge_keys, key=str),
        }

    def validate_graph_backfill(
        self,
        revision_id: UUID | None = None,
        *,
        raise_on_error: bool = False,
    ) -> GraphBackfillValidation:
        """Validate the additive projection and optionally raise on anomalies."""
        legacy_capabilities = _filter_revision(
            self.session.exec(select(RagTemplateCapability)).all(), revision_id
        )
        graph_nodes = self.load_template_nodes(revision_id)
        legacy_dependencies = _filter_revision(
            self.session.exec(select(RagTemplateDependency)).all(), revision_id
        )
        graph_edges = self.load_template_edges(revision_id)
        contracts = self.load_capability_artifact_contracts()
        artifact_types = self.load_artifact_types()
        capabilities = self.session.exec(select(Capability)).all()
        capability_by_id = {row.id: row for row in capabilities}
        artifact_by_id = {row.id: row for row in artifact_types}
        issues: list[str] = []

        node_counts = Counter((row.rag_template_revision_id, row.capability_id) for row in graph_nodes)
        legacy_counts = Counter((row.rag_template_revision_id, row.capability_id) for row in legacy_capabilities)
        for key, count in legacy_counts.items():
            if node_counts[key] != count:
                issues.append(f"legacy capability {key} maps to {node_counts[key]} graph nodes, expected {count}")
        for key in node_counts:
            if key not in legacy_counts:
                issues.append(f"graph node has no legacy capability {key}")

        node_by_revision_id = {(row.rag_template_revision_id, row.id): row for row in graph_nodes}
        edge_keys = Counter((row.rag_template_revision_id, row.edge_key) for row in graph_edges)
        if any(count > 1 for count in edge_keys.values()):
            issues.append("duplicate edge_key")
        node_keys = Counter((row.rag_template_revision_id, row.node_key) for row in graph_nodes)
        if any(count > 1 for count in node_keys.values()):
            issues.append("duplicate node_key")

        dependency_projection = Counter(
            (
                row.rag_template_revision_id,
                row.source_capability_id,
                row.target_capability_id,
            )
            for row in legacy_dependencies
        )
        edge_projection = Counter()
        for edge in graph_edges:
            source = node_by_revision_id.get((edge.rag_template_revision_id, edge.source_node_id))
            target = node_by_revision_id.get((edge.rag_template_revision_id, edge.target_node_id))
            if source is None or target is None:
                issues.append(f"edge {edge.edge_key} has a missing or cross-revision node")
                continue
            if source.rag_template_revision_id != target.rag_template_revision_id:
                issues.append(f"edge {edge.edge_key} crosses template revisions")
            edge_projection[(edge.rag_template_revision_id, source.capability_id, target.capability_id)] += 1
        if dependency_projection != edge_projection:
            issues.append("legacy dependencies and graph edges differ")

        for node in graph_nodes:
            if node.capability_id not in capability_by_id:
                issues.append(f"node {node.node_key} references an unknown capability")
        for contract in contracts:
            if contract.capability_id not in capability_by_id:
                issues.append(f"contract {contract.id} references an unknown capability")
            if contract.artifact_type_id not in artifact_by_id:
                issues.append(f"contract {contract.id} references an undefined artifact type")
            elif contract.cardinality is not None and contract.cardinality not in CARDINALITIES_KNOWN:
                # Extensible vocabulary: unknown values are reportable, not rejected.
                pass
        if any(
            artifact.storage_kind not in ARTIFACT_STORAGE_KINDS_KNOWN
            for artifact in artifact_types
        ):
            # Storage kinds intentionally remain extensible as varchar.
            pass
        if any(edge.edge_type not in EDGE_TYPES_KNOWN for edge in graph_edges):
            # Edge types intentionally remain extensible as varchar.
            pass

        historical_contracts: Counter[tuple[UUID, str, str]] = Counter()
        for capability in capabilities:
            for direction, values in (
                ("input", capability.input_artifact_types),
                ("output", capability.output_artifact_types),
            ):
                if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
                    issues.append(f"invalid historical artifact list for {capability.capability_key}.{direction}")
                    continue
                for value in values:
                    if value.strip():
                        historical_contracts[(capability.id, direction, value)] += 1
        actual_contracts = Counter(
            (row.capability_id, row.direction, artifact_by_id[row.artifact_type_id].artifact_type_key)
            for row in contracts
            if row.artifact_type_id in artifact_by_id
        )
        if historical_contracts != actual_contracts:
            issues.append("historical artifact declarations and contracts differ")

        report = GraphBackfillValidation(
            revision_id=revision_id,
            legacy_capability_count=len(legacy_capabilities),
            graph_node_count=len(graph_nodes),
            legacy_dependency_count=len(legacy_dependencies),
            graph_edge_count=len(graph_edges),
            contract_count=len(contracts),
            artifact_type_count=len(artifact_types),
            issues=tuple(dict.fromkeys(issues)),
        )
        if raise_on_error and not report.is_valid:
            raise ValueError("Backfill RAG graph incohérent : " + "; ".join(report.issues))
        return report

    def component_registry(self) -> ComponentRegistry:
        rows = self.session.exec(
            select(ComponentVersion, Component)
            .join(Component, Component.id == ComponentVersion.component_id)
            .order_by(ComponentVersion.created_at, ComponentVersion.id)
        ).all()
        definitions: list[ComponentDefinition] = []
        for version, component in rows:
            capabilities = self.session.exec(
                select(Capability.capability_key)
                .join(ComponentCapability, ComponentCapability.capability_id == Capability.id)
                .where(ComponentCapability.component_version_id == version.id)
                .order_by(Capability.display_order, Capability.capability_key)
            ).all()
            definitions.append(
                ComponentDefinition(
                    component_key=component.component_key,
                    version=version.version,
                    provides=tuple(capabilities),
                    configuration_schema=version.configuration_schema or {},
                    metadata={
                        **(version.extra_data or {}),
                        "display_name": component.display_name,
                        "vendor": component.vendor,
                    },
                )
            )
        return ComponentRegistry(definitions)

    def active_revision(self, pipeline_key: str = "pipeline-p") -> PipelineRevision | None:
        return self.session.exec(
            select(PipelineRevision)
            .join(PipelineDefinition, PipelineDefinition.id == PipelineRevision.pipeline_definition_id)
            .where(
                PipelineDefinition.pipeline_key == pipeline_key,
                PipelineRevision.status == "active",
            )
        ).first()

    def draft_revision(self, pipeline_key: str = "pipeline-p") -> PipelineRevision | None:
        return self.session.exec(
            select(PipelineRevision)
            .join(PipelineDefinition, PipelineDefinition.id == PipelineRevision.pipeline_definition_id)
            .where(
                PipelineDefinition.pipeline_key == pipeline_key,
                PipelineRevision.status == "draft",
            )
            .order_by(PipelineRevision.revision_number.desc())
        ).first()

    def load_manifest(self, revision_id: UUID) -> PipelineManifest:
        revision = self.session.get(PipelineRevision, revision_id)
        if revision is None:
            raise ValueError(f"PipelineRevision introuvable : {revision_id}.")
        definition = self.session.get(PipelineDefinition, revision.pipeline_definition_id)
        if definition is None:
            raise ValueError("PipelineDefinition de la révision introuvable.")
        binding_rows = self.session.exec(
            select(PipelineBinding)
            .where(PipelineBinding.pipeline_revision_id == revision.id)
            .order_by(PipelineBinding.position, PipelineBinding.id)
        ).all()
        bindings: list[ComponentBinding] = []
        for row in binding_rows:
            version = self.session.get(ComponentVersion, row.component_version_id)
            if version is None:
                raise ValueError(f"ComponentVersion introuvable pour le binding {row.binding_key}.")
            component = self.session.get(Component, version.component_id)
            if component is None:
                raise ValueError(f"Component du binding {row.binding_key} introuvable.")
            capability_rows = self.session.exec(
                select(PipelineBindingCapability, ComponentCapability, Capability)
                .join(ComponentCapability, ComponentCapability.id == PipelineBindingCapability.component_capability_id)
                .join(Capability, Capability.id == ComponentCapability.capability_id)
                .where(
                    PipelineBindingCapability.pipeline_binding_id == row.id,
                    PipelineBindingCapability.enabled.is_(True),
                )
                .order_by(Capability.display_order, Capability.capability_key)
            ).all()
            dependencies = self.session.exec(
                select(PipelineBindingDependency.depends_on_binding_id)
                .where(PipelineBindingDependency.pipeline_binding_id == row.id)
            ).all()
            dependency_keys = []
            for dependency_id in dependencies:
                dependency = self.session.get(PipelineBinding, dependency_id)
                if dependency is not None:
                    dependency_keys.append(dependency.binding_key)
            bindings.append(
                ComponentBinding(
                    binding_key=row.binding_key,
                    component_key=component.component_key,
                    component_version=version.version,
                    capabilities=tuple(capability.capability_key for _, _, capability in capability_rows),
                    configuration=row.configuration or {},
                    dependencies=tuple(dependency_keys),
                    enabled=row.enabled,
                )
            )
        manifest = PipelineManifest(
            pipeline_key=definition.pipeline_key,
            revision=str(revision.revision_number),
            bindings=tuple(bindings),
        )
        if revision.manifest_hash != manifest.manifest_hash:
            raise ValueError(
                f"Le manifest_hash de la PipelineRevision {revision.id} ne correspond pas à sa composition."
            )
        return manifest

    def validate_revision(self, revision_id: UUID) -> PipelineManifest:
        revision = self.session.get(PipelineRevision, revision_id)
        if revision is None:
            raise ValueError(f"PipelineRevision introuvable : {revision_id}.")
        manifest = self.load_manifest(revision_id)
        errors: list[str] = []
        binding_rows = self.session.exec(
            select(PipelineBinding).where(PipelineBinding.pipeline_revision_id == revision_id)
        ).all()
        binding_by_id = {row.id: row for row in binding_rows}
        binding_by_key = {row.binding_key: row for row in binding_rows}
        for binding in manifest.bindings:
            row = binding_by_key[binding.binding_key]
            version = self.session.get(ComponentVersion, row.component_version_id)
            if version is None:
                errors.append(f"{binding.binding_key}: ComponentVersion introuvable.")
                continue
            selected = self.session.exec(
                select(ComponentCapability, Capability)
                .join(Capability, Capability.id == ComponentCapability.capability_id)
                .join(
                    PipelineBindingCapability,
                    PipelineBindingCapability.component_capability_id == ComponentCapability.id,
                )
                .where(
                    PipelineBindingCapability.pipeline_binding_id == row.id,
                    PipelineBindingCapability.enabled.is_(True),
                )
            ).all()
            all_capabilities = self.session.exec(
                select(ComponentCapability, Capability)
                .join(Capability, Capability.id == ComponentCapability.capability_id)
                .where(ComponentCapability.component_version_id == version.id)
            ).all()
            all_by_bundle: dict[str, list[tuple[ComponentCapability, Capability]]] = defaultdict(list)
            selected_ids = {component_capability.id for component_capability, _ in selected}
            for component_capability, capability in all_capabilities:
                if component_capability.execution_bundle_key:
                    all_by_bundle[component_capability.execution_bundle_key].append(
                        (component_capability, capability)
                    )
                if component_capability.id in selected_ids and component_capability.component_version_id != version.id:
                    errors.append(f"{binding.binding_key}: capability incompatible avec la version du composant.")
            provided_ids = {component_capability.id for component_capability, _ in all_capabilities}
            for component_capability, capability in selected:
                if component_capability.id not in provided_ids:
                    errors.append(f"{binding.binding_key}: {capability.capability_key} n'appartient pas à cette version.")
            for bundle_key, members in all_by_bundle.items():
                modes = {member.invocation_mode for member, _ in members}
                selected_members = [member for member, _ in members if member.id in selected_ids]
                if "all_or_none" in modes and selected_members and len(selected_members) != len(members):
                    errors.append(
                        f"{binding.binding_key}: le bundle {bundle_key} all_or_none doit être sélectionné entièrement."
                    )
                if any(member.invocation_mode == "produced_with_bundle" for member in selected_members):
                    if not any(
                        member.id in selected_ids and member.invocation_mode != "produced_with_bundle"
                        for member, _ in members
                    ):
                        errors.append(
                            f"{binding.binding_key}: une capability produced_with_bundle ne peut pas être invoquée seule ({bundle_key})."
                        )
            try:
                canonical_json_hash(row.configuration or {})
            except (TypeError, ValueError) as error:
                errors.append(f"{binding.binding_key}: configuration JSON invalide ({error}).")
            for dependency in self.session.exec(
                select(PipelineBindingDependency).where(
                    PipelineBindingDependency.pipeline_binding_id == row.id
                )
            ).all():
                if dependency.depends_on_binding_id not in binding_by_id:
                    errors.append(f"{binding.binding_key}: dépendance vers un autre pipeline.")
        if revision.rag_template_revision_id is not None:
            template_caps = self.session.exec(
                select(RagTemplateCapability, Capability)
                .join(Capability, Capability.id == RagTemplateCapability.capability_id)
                .where(RagTemplateCapability.rag_template_revision_id == revision.rag_template_revision_id)
            ).all()
            selected_keys = {
                capability
                for binding in manifest.bindings
                if binding.enabled
                for capability in binding.capabilities
            }
            template_keys = {capability.capability_key for _, capability in template_caps}
            for template_capability, capability in template_caps:
                if template_capability.requirement_mode == "required" and capability.capability_key not in selected_keys:
                    errors.append(f"La capability requise est absente : {capability.capability_key}.")
            errors.extend(
                f"La capability {key} n'est pas autorisée par le template."
                for key in sorted(selected_keys - template_keys)
            )
            for dependency in self.session.exec(
                select(RagTemplateDependency).where(
                    RagTemplateDependency.rag_template_revision_id == revision.rag_template_revision_id
                )
            ).all():
                source = self.session.get(Capability, dependency.source_capability_id)
                target = self.session.get(Capability, dependency.target_capability_id)
                if target and target.capability_key in selected_keys and source and source.capability_key not in selected_keys:
                    errors.append(
                        f"Dépendance de template non satisfaite : {source.capability_key} → {target.capability_key}."
                    )
        if errors:
            raise ValueError("Validation de pipeline échouée : " + " ".join(errors))
        return manifest

    def _next_revision_number(self, definition_id: UUID) -> int:
        current = self.session.exec(
            select(PipelineRevision.revision_number)
            .where(PipelineRevision.pipeline_definition_id == definition_id)
            .order_by(PipelineRevision.revision_number.desc())
        ).first()
        return (current or 0) + 1

    def save_revision(
        self,
        manifest: PipelineManifest,
        *,
        status: str = "draft",
        revision_number: int | None = None,
        rag_template_revision_id: UUID | None = None,
        change_reason: str | None = None,
        created_by_display_name: str | None = None,
        existing_revision_id: UUID | None = None,
    ) -> PipelineRevision:
        manifest.validate()
        definition = self.session.exec(
            select(PipelineDefinition).where(PipelineDefinition.pipeline_key == manifest.pipeline_key)
        ).first()
        if definition is None:
            definition = PipelineDefinition(
                pipeline_key=manifest.pipeline_key,
                display_name="Production documentaire" if manifest.pipeline_key == "pipeline-p" else manifest.pipeline_key,
                description="Composition logique versionnée.",
            )
            self.session.add(definition)
            self.session.flush()
        revision = self.session.get(PipelineRevision, existing_revision_id) if existing_revision_id else None
        target_revision_number = (
            revision.revision_number
            if revision is not None
            else revision_number or self._next_revision_number(definition.id)
        )
        persisted_manifest = PipelineManifest(
            pipeline_key=manifest.pipeline_key,
            revision=str(target_revision_number),
            bindings=manifest.bindings,
        )
        if revision is None:
            revision = PipelineRevision(
                pipeline_definition_id=definition.id,
                revision_number=target_revision_number,
                status=status,
                manifest_hash=persisted_manifest.manifest_hash,
                rag_template_revision_id=rag_template_revision_id,
                change_reason=change_reason,
                created_by_display_name=created_by_display_name,
            )
            self.session.add(revision)
            self.session.flush()
        else:
            if revision.status == "active":
                raise ValueError("Une PipelineRevision active est immuable ; créez une nouvelle révision.")
            revision.manifest_hash = persisted_manifest.manifest_hash
            revision.status = status
            revision.rag_template_revision_id = rag_template_revision_id or revision.rag_template_revision_id
            revision.change_reason = change_reason or revision.change_reason
            self.session.add(revision)
            self.session.flush()
            old_bindings = self.session.exec(
                select(PipelineBinding).where(PipelineBinding.pipeline_revision_id == revision.id)
            ).all()
            # These association tables intentionally have no ORM cascade.
            # Remove dependants in explicit phases so autoflush cannot delete
            # a binding while a dependency row still references it.
            old_binding_ids = [old.id for old in old_bindings]
            if old_binding_ids:
                for link in self.session.exec(
                    select(PipelineBindingNode).where(
                        PipelineBindingNode.pipeline_binding_id.in_(old_binding_ids)
                    )
                ).all():
                    self.session.delete(link)
                self.session.flush()
                for dependency in self.session.exec(
                    select(PipelineBindingDependency).where(
                        (PipelineBindingDependency.pipeline_binding_id.in_(old_binding_ids))
                        | (PipelineBindingDependency.depends_on_binding_id.in_(old_binding_ids))
                    )
                ).all():
                    self.session.delete(dependency)
                self.session.flush()
                for link in self.session.exec(
                    select(PipelineBindingCapability).where(
                        PipelineBindingCapability.pipeline_binding_id.in_(old_binding_ids)
                    )
                ).all():
                    self.session.delete(link)
                self.session.flush()
            for old in old_bindings:
                self.session.delete(old)
            self.session.flush()
        self._save_bindings(revision, persisted_manifest)
        self.session.flush()
        if status == "active":
            self.validate_revision(revision.id)
        return revision

    def _save_bindings(self, revision: PipelineRevision, manifest: PipelineManifest) -> None:
        version_lookup: dict[tuple[str, str], ComponentVersion] = {}
        for binding in manifest.bindings:
            component = self.session.exec(
                select(Component).where(Component.component_key == binding.component_key)
            ).first()
            version = None
            if component is not None:
                version = self.session.exec(
                    select(ComponentVersion).where(
                        ComponentVersion.component_id == component.id,
                        ComponentVersion.version == binding.component_version,
                    )
                ).first()
            if version is None:
                raise ValueError(
                    f"Version de composant inconnue : {binding.component_key}@{binding.component_version}."
                )
            version_lookup[(binding.component_key, binding.component_version)] = version
        rows: dict[str, PipelineBinding] = {}
        for position, binding in enumerate(manifest.bindings):
            version = version_lookup[(binding.component_key, binding.component_version)]
            row = PipelineBinding(
                pipeline_revision_id=revision.id,
                binding_key=binding.binding_key,
                component_version_id=version.id,
                position=position,
                enabled=binding.enabled,
                configuration=dict(binding.configuration),
            )
            self.session.add(row)
            self.session.flush()
            rows[binding.binding_key] = row
            for capability_key in binding.capabilities:
                capability = _catalog_capability(self.session, capability_key)
                component_capability = self.session.exec(
                    select(ComponentCapability).where(
                        ComponentCapability.component_version_id == version.id,
                        ComponentCapability.capability_id == capability.id,
                    )
                ).first()
                if component_capability is None:
                    raise ValueError(
                        f"{binding.binding_key}: {capability_key} n'est pas fournie par cette version."
                    )
                self.session.add(
                    PipelineBindingCapability(
                        pipeline_binding_id=row.id,
                        component_capability_id=component_capability.id,
                        enabled=True,
                        configuration={},
                    )
                )
        self.session.flush()
        for binding in manifest.bindings:
            for dependency_key in binding.dependencies:
                if dependency_key not in rows:
                    raise ValueError(f"Dépendance de binding inconnue : {dependency_key}.")
                self.session.add(
                    PipelineBindingDependency(
                        pipeline_binding_id=rows[binding.binding_key].id,
                        depends_on_binding_id=rows[dependency_key].id,
                    )
                )

    def clone_active_to_draft(self, pipeline_key: str = "pipeline-p") -> PipelineRevision:
        active = self.active_revision(pipeline_key)
        if active is None:
            raise ValueError(f"Aucune révision active pour {pipeline_key}.")
        manifest = self.load_manifest(active.id)
        draft = self.draft_revision(pipeline_key)
        if draft is not None:
            return draft
        return self.save_revision(
            PipelineManifest(
                pipeline_key=manifest.pipeline_key,
                revision=str(self._next_revision_number(active.pipeline_definition_id)),
                bindings=manifest.bindings,
            ),
            status="draft",
            rag_template_revision_id=active.rag_template_revision_id,
            change_reason="Clonée depuis la production.",
        )

    def reset_draft_from_active(self, pipeline_key: str = "pipeline-p") -> PipelineRevision:
        active = self.active_revision(pipeline_key)
        if active is None:
            raise ValueError(f"Aucune révision active pour {pipeline_key}.")
        draft = self.draft_revision(pipeline_key)
        manifest = self.load_manifest(active.id)
        draft = draft or self.clone_active_to_draft(pipeline_key)
        return self.save_revision(
            PipelineManifest(
                pipeline_key=manifest.pipeline_key,
                revision=str(draft.revision_number),
                bindings=manifest.bindings,
            ),
            status="draft",
            existing_revision_id=draft.id,
            rag_template_revision_id=active.rag_template_revision_id,
            change_reason="Réinitialisée depuis la production.",
        )

    def create_new_draft_revision(self, pipeline_key: str = "pipeline-p") -> PipelineRevision:
        source = self.draft_revision(pipeline_key) or self.active_revision(pipeline_key)
        if source is None:
            raise ValueError(f"Aucune révision source pour {pipeline_key}.")
        manifest = self.load_manifest(source.id)
        return self.save_revision(
            PipelineManifest(
                pipeline_key=manifest.pipeline_key,
                revision=str(self._next_revision_number(source.pipeline_definition_id)),
                bindings=manifest.bindings,
            ),
            status="draft",
            rag_template_revision_id=source.rag_template_revision_id,
            change_reason="Nouvelle révision de travail.",
        )

    def activate(self, revision_id: UUID) -> PipelineRevision:
        revision = self.session.get(PipelineRevision, revision_id)
        if revision is None:
            raise ValueError(f"PipelineRevision introuvable : {revision_id}.")
        self.validate_revision(revision_id)
        active = self.session.exec(
            select(PipelineRevision).where(
                PipelineRevision.pipeline_definition_id == revision.pipeline_definition_id,
                PipelineRevision.status == "active",
            )
        ).all()
        now = _now()
        for old in active:
            if old.id != revision.id:
                old.status = "retired"
                self.session.add(old)
        revision.status = "active"
        revision.activated_at = now
        self.session.add(revision)
        self.session.flush()
        return revision


def build_component_registry_from_db(session: Session) -> ComponentRegistry:
    return PipelinePersistenceService(session).component_registry()


def load_artifact_types(session: Session) -> list[ArtifactType]:
    return PipelinePersistenceService(session).load_artifact_types()


def load_capability_artifact_contracts(
    session: Session,
    capability_id: UUID | None = None,
) -> list[CapabilityArtifactContract]:
    return PipelinePersistenceService(session).load_capability_artifact_contracts(capability_id)


def load_template_nodes(
    session: Session,
    revision_id: UUID | None = None,
) -> list[RagTemplateNode]:
    return PipelinePersistenceService(session).load_template_nodes(revision_id)


def load_template_edges(
    session: Session,
    revision_id: UUID | None = None,
) -> list[RagTemplateEdge]:
    return PipelinePersistenceService(session).load_template_edges(revision_id)


def compare_legacy_vs_graph(
    session: Session,
    revision_id: UUID | None = None,
) -> dict[str, Any]:
    return PipelinePersistenceService(session).compare_legacy_vs_graph(revision_id)


def validate_graph_backfill(
    session: Session,
    revision_id: UUID | None = None,
    *,
    raise_on_error: bool = False,
) -> GraphBackfillValidation:
    return PipelinePersistenceService(session).validate_graph_backfill(
        revision_id,
        raise_on_error=raise_on_error,
    )


def load_pipeline_binding_nodes(
    session: Session,
    revision_id: UUID | None = None,
) -> list[PipelineBindingNode]:
    return PipelineCompositionService(session).load_binding_nodes(revision_id)


def validate_pipeline_composition(
    session: Session,
    revision_id: UUID,
    *,
    raise_on_error: bool = False,
) -> CompositionValidationResult:
    return PipelineCompositionService(session).validate_pipeline_composition(
        revision_id,
        raise_on_error=raise_on_error,
    )


def persist_manifest(
    session: Session,
    manifest: PipelineManifest,
    *,
    status: str = "draft",
    revision_number: int | None = None,
    rag_template_revision_id: UUID | None = None,
    existing_revision_id: UUID | None = None,
) -> PipelineRevision:
    return PipelinePersistenceService(session).save_revision(
        manifest,
        status=status,
        revision_number=revision_number,
        rag_template_revision_id=rag_template_revision_id,
        existing_revision_id=existing_revision_id,
    )


def bootstrap_catalog(session: Session) -> dict[str, Any]:
    """Idempotently persist only the currently qualified Kaliok inventory."""
    from kaliok.pipeline.real_registry import build_static_kaliok_component_registry
    from kaliok.pipeline.production import build_current_production_manifest

    for order, item in enumerate(CAPABILITY_CATALOG):
        capability = session.exec(
            select(Capability).where(Capability.capability_key == item["key"])
        ).first()
        if capability is None:
            capability = Capability(
                capability_key=item["key"],
                display_name=item["display_name"],
                description=f"Fonction Kaliok : {item['display_name']}",
                phase_key=item["phase_key"],
                input_artifact_types=item["input"],
                output_artifact_types=item["output"],
                display_order=order,
            )
            session.add(capability)
        else:
            capability.display_name = item["display_name"]
            capability.phase_key = item["phase_key"]
            capability.input_artifact_types = item["input"]
            capability.output_artifact_types = item["output"]
            session.add(capability)
    session.flush()

    static_registry = build_static_kaliok_component_registry()
    component_count = 0
    for definition in static_registry.definitions:
        component = session.exec(
            select(Component).where(Component.component_key == definition.component_key)
        ).first()
        if component is None:
            component = Component(
                component_key=definition.component_key,
                display_name=COMPONENT_LABELS.get(
                    definition.component_key,
                    definition.component_key.replace("-", " ").title(),
                ),
                description="Composant Kaliok qualifié et raccordé au runtime.",
            )
            session.add(component)
            session.flush()
            component_count += 1
        version = session.exec(
            select(ComponentVersion).where(
                ComponentVersion.component_id == component.id,
                ComponentVersion.version == definition.version,
            )
        ).first()
        if version is None:
            version = ComponentVersion(
                component_id=component.id,
                version=definition.version,
                status="available",
                configuration_schema=dict(definition.configuration_schema),
                extra_data=dict(definition.metadata),
            )
            session.add(version)
            session.flush()
        for capability_key in definition.provides:
            capability = _catalog_capability(session, capability_key)
            link = session.exec(
                select(ComponentCapability).where(
                    ComponentCapability.component_version_id == version.id,
                    ComponentCapability.capability_id == capability.id,
                )
            ).first()
            if link is None:
                session.add(
                    ComponentCapability(
                        component_version_id=version.id,
                        capability_id=capability.id,
                        invocation_mode="independent",
                        configuration_schema={},
                        extra_data={},
                    )
                )
    session.flush()

    template = session.exec(
        select(RagTemplate).where(RagTemplate.template_key == "rag-documentaire-kaliok")
    ).first()
    if template is None:
        template = RagTemplate(
            template_key="rag-documentaire-kaliok",
            display_name="RAG documentaire Kaliok",
            description="Architecture documentaire réellement raccordée au runtime actuel.",
        )
        session.add(template)
        session.flush()
    template_revision = session.exec(
        select(RagTemplateRevision).where(
            RagTemplateRevision.rag_template_id == template.id,
            RagTemplateRevision.revision_number == 1,
        )
    ).first()
    if template_revision is None:
        template_revision = RagTemplateRevision(
            rag_template_id=template.id,
            revision_number=1,
            status="active",
            change_reason="Premier modèle documenté depuis le runtime existant.",
            activated_at=_now(),
        )
        session.add(template_revision)
        session.flush()
        modes = {
            "document_extraction": "required",
            "normalization": "required",
            "entity_discovery": "optional",
            "entity_resolution": "optional",
            "chunking": "optional",
            "indexing": "optional",
        }
        for order, key in enumerate(modes):
            session.add(
                RagTemplateCapability(
                    rag_template_revision_id=template_revision.id,
                    capability_id=_catalog_capability(session, key).id,
                    requirement_mode=modes[key],
                    display_order=order,
                )
            )
        for source, target in (
            ("normalization", "entity_discovery"),
            ("entity_discovery", "entity_resolution"),
        ):
            session.add(
                RagTemplateDependency(
                    rag_template_revision_id=template_revision.id,
                    source_capability_id=_catalog_capability(session, source).id,
                    target_capability_id=_catalog_capability(session, target).id,
                )
            )
    session.flush()

    service = PipelinePersistenceService(session)
    production = build_current_production_manifest()
    active = service.active_revision("pipeline-p")
    if active is None:
        active = service.save_revision(
            production,
            status="active",
            revision_number=1,
            rag_template_revision_id=template_revision.id,
            change_reason="Bootstrap de la composition P réellement connue.",
        )
        active.activated_at = _now()
        session.add(active)
    draft = service.draft_revision("pipeline-p")
    if draft is None:
        draft = service.clone_active_to_draft("pipeline-p")
    session.flush()
    return {
        "capabilities": len(CAPABILITY_CATALOG),
        "components": component_count,
        "active_pipeline_revision_id": active.id,
        "draft_pipeline_revision_id": draft.id,
        "template_revision_id": template_revision.id,
    }


__all__ = [
    "ARTIFACT_STORAGE_KINDS_KNOWN",
    "CARDINALITIES_KNOWN",
    "CAPABILITY_CATALOG",
    "EDGE_TYPES_KNOWN",
    "GraphBackfillValidation",
    "PipelinePersistenceService",
    "bootstrap_catalog",
    "build_component_registry_from_db",
    "compare_legacy_vs_graph",
    "load_artifact_types",
    "load_capability_artifact_contracts",
    "load_template_edges",
    "load_template_nodes",
    "load_pipeline_binding_nodes",
    "persist_manifest",
    "validate_pipeline_composition",
    "validate_graph_backfill",
]
