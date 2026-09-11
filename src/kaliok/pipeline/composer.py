"""Read-only projection of persisted pipeline composition for the RAG Composer."""

from __future__ import annotations

from collections import defaultdict
from typing import Any
from uuid import UUID

from sqlmodel import Session, select

from kaliok.storage.models import (
    ArtifactType,
    Capability,
    CapabilityArtifactContract,
    Component,
    ComponentCapability,
    ComponentVersion,
    PipelineBinding,
    PipelineBindingNode,
    PipelineDefinition,
    PipelineRevision,
    RagTemplate,
    RagTemplateEdge,
    RagTemplateNode,
    RagTemplateRevision,
    ResourceInstance,
    ResourceInstanceCapability,
)


def _configuration_summary(configuration: dict[str, Any] | None) -> dict[str, Any]:
    """Describe configuration shape without returning any stored values."""
    keys = sorted(str(key) for key in (configuration or {}))
    return {"configured": bool(keys), "keys": keys}


class RagComposerReadService:
    """Build the UI read model exclusively from the relational catalogue."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def project(self) -> dict[str, Any]:
        definitions = self.session.exec(
            select(PipelineDefinition).order_by(PipelineDefinition.display_name)
        ).all()
        revisions = self.session.exec(
            select(PipelineRevision).order_by(
                PipelineRevision.pipeline_definition_id,
                PipelineRevision.revision_number.desc(),
            )
        ).all()
        template_revisions = {
            row.id: row for row in self.session.exec(select(RagTemplateRevision)).all()
        }
        templates = {row.id: row for row in self.session.exec(select(RagTemplate)).all()}
        nodes = self.session.exec(
            select(RagTemplateNode).order_by(
                RagTemplateNode.rag_template_revision_id,
                RagTemplateNode.position,
                RagTemplateNode.node_key,
            )
        ).all()
        edges = self.session.exec(
            select(RagTemplateEdge).order_by(
                RagTemplateEdge.rag_template_revision_id,
                RagTemplateEdge.priority,
                RagTemplateEdge.edge_key,
            )
        ).all()
        bindings = {row.id: row for row in self.session.exec(select(PipelineBinding)).all()}
        links = self.session.exec(
            select(PipelineBindingNode).order_by(
                PipelineBindingNode.pipeline_revision_id,
                PipelineBindingNode.rag_template_node_id,
                PipelineBindingNode.priority,
                PipelineBindingNode.pipeline_binding_id,
            )
        ).all()
        versions = {row.id: row for row in self.session.exec(select(ComponentVersion)).all()}
        components = {row.id: row for row in self.session.exec(select(Component)).all()}
        capabilities = {row.id: row for row in self.session.exec(select(Capability)).all()}
        artifact_types = {
            row.id: row for row in self.session.exec(select(ArtifactType)).all()
        }
        artifact_contracts = self.session.exec(
            select(CapabilityArtifactContract).order_by(
                CapabilityArtifactContract.capability_id,
                CapabilityArtifactContract.direction,
                CapabilityArtifactContract.position,
                CapabilityArtifactContract.port_key,
            )
        ).all()
        component_capabilities = self.session.exec(select(ComponentCapability)).all()
        resources = {row.id: row for row in self.session.exec(select(ResourceInstance)).all()}
        resource_capabilities = {
            (row.resource_instance_id, row.component_capability_id): row
            for row in self.session.exec(select(ResourceInstanceCapability)).all()
        }

        revisions_by_pipeline: defaultdict[UUID, list[PipelineRevision]] = defaultdict(list)
        nodes_by_template: defaultdict[UUID, list[RagTemplateNode]] = defaultdict(list)
        edges_by_template: defaultdict[UUID, list[RagTemplateEdge]] = defaultdict(list)
        links_by_revision_node: defaultdict[tuple[UUID, UUID], list[PipelineBindingNode]] = (
            defaultdict(list)
        )
        component_capabilities_by_version_capability: defaultdict[
            tuple[UUID, UUID], list[ComponentCapability]
        ] = defaultdict(list)
        artifact_contracts_by_capability: defaultdict[
            UUID, dict[str, list[dict[str, Any]]]
        ] = defaultdict(lambda: {"input": [], "output": []})
        for revision in revisions:
            revisions_by_pipeline[revision.pipeline_definition_id].append(revision)
        for node in nodes:
            nodes_by_template[node.rag_template_revision_id].append(node)
        for edge in edges:
            edges_by_template[edge.rag_template_revision_id].append(edge)
        for link in links:
            links_by_revision_node[(link.pipeline_revision_id, link.rag_template_node_id)].append(
                link
            )
        for item in component_capabilities:
            component_capabilities_by_version_capability[
                (item.component_version_id, item.capability_id)
            ].append(item)
        for contract in artifact_contracts:
            artifact_type = artifact_types.get(contract.artifact_type_id)
            artifact_contracts_by_capability[contract.capability_id][contract.direction].append(
                {
                    "artifact_type": (
                        {
                            "id": str(artifact_type.id),
                            "key": artifact_type.artifact_type_key,
                            "display_name": artifact_type.display_name,
                            "version": artifact_type.version,
                        }
                        if artifact_type
                        else None
                    ),
                    "port_key": contract.port_key,
                    "required": contract.required,
                    "cardinality": contract.cardinality,
                    "position": contract.position,
                }
            )

        return {
            "pipelines": [
                self._pipeline(
                    definition,
                    revisions_by_pipeline[definition.id],
                    template_revisions,
                    templates,
                    nodes_by_template,
                    edges_by_template,
                    links_by_revision_node,
                    bindings,
                    versions,
                    components,
                    capabilities,
                    artifact_contracts_by_capability,
                    component_capabilities_by_version_capability,
                    resources,
                    resource_capabilities,
                )
                for definition in definitions
            ]
        }

    def _pipeline(
        self,
        definition: PipelineDefinition,
        revisions: list[PipelineRevision],
        template_revisions: dict[UUID, RagTemplateRevision],
        templates: dict[UUID, RagTemplate],
        nodes_by_template: dict[UUID, list[RagTemplateNode]],
        edges_by_template: dict[UUID, list[RagTemplateEdge]],
        links_by_revision_node: dict[tuple[UUID, UUID], list[PipelineBindingNode]],
        bindings: dict[UUID, PipelineBinding],
        versions: dict[UUID, ComponentVersion],
        components: dict[UUID, Component],
        capabilities: dict[UUID, Capability],
        artifact_contracts: dict[UUID, dict[str, list[dict[str, Any]]]],
        component_capabilities: dict[tuple[UUID, UUID], list[ComponentCapability]],
        resources: dict[UUID, ResourceInstance],
        resource_capabilities: dict[tuple[UUID, UUID], ResourceInstanceCapability],
    ) -> dict[str, Any]:
        return {
            "id": str(definition.id),
            "key": definition.pipeline_key,
            "display_name": definition.display_name,
            "description": definition.description,
            "revisions": [
                self._revision(
                    revision,
                    template_revisions,
                    templates,
                    nodes_by_template,
                    edges_by_template,
                    links_by_revision_node,
                    bindings,
                    versions,
                    components,
                    capabilities,
                    artifact_contracts,
                    component_capabilities,
                    resources,
                    resource_capabilities,
                )
                for revision in revisions
            ],
        }

    def _revision(
        self,
        revision: PipelineRevision,
        template_revisions: dict[UUID, RagTemplateRevision],
        templates: dict[UUID, RagTemplate],
        nodes_by_template: dict[UUID, list[RagTemplateNode]],
        edges_by_template: dict[UUID, list[RagTemplateEdge]],
        links_by_revision_node: dict[tuple[UUID, UUID], list[PipelineBindingNode]],
        bindings: dict[UUID, PipelineBinding],
        versions: dict[UUID, ComponentVersion],
        components: dict[UUID, Component],
        capabilities: dict[UUID, Capability],
        artifact_contracts: dict[UUID, dict[str, list[dict[str, Any]]]],
        component_capabilities: dict[tuple[UUID, UUID], list[ComponentCapability]],
        resources: dict[UUID, ResourceInstance],
        resource_capabilities: dict[tuple[UUID, UUID], ResourceInstanceCapability],
    ) -> dict[str, Any]:
        issues: list[str] = []
        template_revision = (
            template_revisions.get(revision.rag_template_revision_id)
            if revision.rag_template_revision_id
            else None
        )
        if template_revision is None:
            issues.append("Cette révision de pipeline n'est associée à aucun modèle RAG.")
            template = None
            revision_nodes: list[RagTemplateNode] = []
            revision_edges: list[RagTemplateEdge] = []
        else:
            template = templates.get(template_revision.rag_template_id)
            revision_nodes = nodes_by_template.get(template_revision.id, [])
            revision_edges = edges_by_template.get(template_revision.id, [])
            if template is None:
                issues.append("Le modèle RAG associé est introuvable.")

        return {
            "id": str(revision.id),
            "revision_number": revision.revision_number,
            "status": revision.status,
            "change_reason": revision.change_reason,
            "created_at": revision.created_at.isoformat(),
            "created_by_display_name": revision.created_by_display_name,
            "template": (
                {
                    "id": str(template_revision.id),
                    "revision_number": template_revision.revision_number,
                    "status": template_revision.status,
                    "key": template.template_key if template else None,
                    "display_name": template.display_name if template else "Modèle RAG introuvable",
                    "description": template.description if template else None,
                }
                if template_revision
                else None
            ),
            "nodes": [
                self._node(
                    revision,
                    node,
                    links_by_revision_node.get((revision.id, node.id), []),
                    bindings,
                    versions,
                    components,
                    capabilities,
                    artifact_contracts,
                    component_capabilities,
                    resources,
                    resource_capabilities,
                )
                for node in revision_nodes
            ],
            "edges": [
                {
                    "id": str(edge.id),
                    "key": edge.edge_key,
                    "source_node_id": str(edge.source_node_id),
                    "target_node_id": str(edge.target_node_id),
                    "type": edge.edge_type,
                    "source_port_key": edge.source_port_key,
                    "target_port_key": edge.target_port_key,
                    "condition_present": edge.condition is not None,
                    "priority": edge.priority,
                    "enabled": edge.enabled,
                    "configuration": _configuration_summary(edge.configuration),
                }
                for edge in revision_edges
            ],
            "issues": issues,
        }

    def _node(
        self,
        revision: PipelineRevision,
        node: RagTemplateNode,
        links: list[PipelineBindingNode],
        bindings: dict[UUID, PipelineBinding],
        versions: dict[UUID, ComponentVersion],
        components: dict[UUID, Component],
        capabilities: dict[UUID, Capability],
        artifact_contracts: dict[UUID, dict[str, list[dict[str, Any]]]],
        component_capabilities: dict[tuple[UUID, UUID], list[ComponentCapability]],
        resources: dict[UUID, ResourceInstance],
        resource_capabilities: dict[tuple[UUID, UUID], ResourceInstanceCapability],
    ) -> dict[str, Any]:
        issues: list[str] = []
        projected = [
            self._assignment(
                link,
                node,
                bindings,
                versions,
                components,
                component_capabilities,
                resources,
                resource_capabilities,
            )
            for link in links
        ]
        selected = [item for item in projected if item["is_selected"]]
        if len(selected) > 1:
            issues.append("Donnée incohérente : plusieurs bindings sont sélectionnés.")
            selected_binding = None
            alternatives = projected
        else:
            selected_binding = selected[0] if selected else None
            alternatives = [item for item in projected if not item["is_selected"]]
            if not selected:
                issues.append(
                    "Aucun binding sélectionné."
                    if projected
                    else "Aucun binding associé."
                )
        if not node.enabled:
            issues.append("Node désactivé.")
        if selected_binding and (
            not selected_binding["enabled"] or not selected_binding["binding"]["enabled"]
        ):
            issues.append("Le binding sélectionné est désactivé.")

        capability = capabilities.get(node.capability_id)
        capability_contracts = artifact_contracts.get(
            node.capability_id, {"input": [], "output": []}
        )
        return {
            "id": str(node.id),
            "key": node.node_key,
            "display_name": node.display_name,
            "description": node.description,
            "zone_key": node.zone_key,
            "requirement_mode": node.requirement_mode,
            "position": node.position,
            "enabled": node.enabled,
            "configuration": _configuration_summary(node.configuration),
            "capability": {
                "id": str(node.capability_id),
                "key": capability.capability_key if capability else None,
                "display_name": capability.display_name if capability else "Capability introuvable",
                "phase_key": capability.phase_key if capability else None,
                "input_contracts": capability_contracts["input"],
                "output_contracts": capability_contracts["output"],
            },
            "selected_binding": selected_binding,
            "alternatives": alternatives,
            "issues": issues,
        }

    def _assignment(
        self,
        link: PipelineBindingNode,
        node: RagTemplateNode,
        bindings: dict[UUID, PipelineBinding],
        versions: dict[UUID, ComponentVersion],
        components: dict[UUID, Component],
        component_capabilities: dict[tuple[UUID, UUID], list[ComponentCapability]],
        resources: dict[UUID, ResourceInstance],
        resource_capabilities: dict[tuple[UUID, UUID], ResourceInstanceCapability],
    ) -> dict[str, Any]:
        binding = bindings.get(link.pipeline_binding_id)
        version = versions.get(binding.component_version_id) if binding else None
        component = components.get(version.component_id) if version else None
        matches = (
            component_capabilities.get((version.id, node.capability_id), []) if version else []
        )
        component_capability = matches[0] if len(matches) == 1 else None
        resource = resources.get(binding.resource_instance_id) if binding else None
        resource_capability = (
            resource_capabilities.get((resource.id, component_capability.id))
            if resource and component_capability
            else None
        )
        assignment_issues: list[str] = []
        if binding is None:
            assignment_issues.append("Binding introuvable.")
        if version is None:
            assignment_issues.append("Version de composant introuvable.")
        if component is None:
            assignment_issues.append("Composant introuvable.")
        if len(matches) != 1:
            assignment_issues.append(
                "Le composant ne fournit pas exactement une capability pour ce node."
            )
        if binding and binding.resource_instance_id and resource is None:
            assignment_issues.append("ResourceInstance introuvable.")

        return {
            "enabled": link.enabled,
            "is_selected": link.is_selected,
            "priority": link.priority,
            "configuration": _configuration_summary(link.configuration),
            "binding": {
                "id": str(binding.id) if binding else str(link.pipeline_binding_id),
                "key": binding.binding_key if binding else "binding-introuvable",
                "enabled": binding.enabled if binding else False,
                "configuration": _configuration_summary(binding.configuration if binding else None),
            },
            "component": (
                {
                    "id": str(component.id),
                    "key": component.component_key,
                    "display_name": component.display_name,
                    "vendor": component.vendor,
                    "active": component.is_active,
                }
                if component
                else None
            ),
            "component_version": (
                {
                    "id": str(version.id),
                    "version": version.version,
                    "status": version.status,
                }
                if version
                else None
            ),
            "component_capability": (
                {
                    "id": str(component_capability.id),
                    "invocation_mode": component_capability.invocation_mode,
                    "execution_bundle_key": component_capability.execution_bundle_key,
                }
                if component_capability
                else None
            ),
            "resource_instance": (
                {
                    "id": str(resource.id),
                    "key": resource.instance_key,
                    "display_name": resource.display_name,
                    "runtime_kind": resource.runtime_kind,
                    "status": resource.status,
                    "health_status": resource.health_status,
                    "capability_availability": (
                        resource_capability.availability_status
                        if resource_capability
                        else None
                    ),
                }
                if resource
                else None
            ),
            "issues": assignment_issues,
        }
