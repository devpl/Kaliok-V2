"""Transactional write operations for the RAG Composer."""

from __future__ import annotations

import re
from typing import Any
from uuid import UUID

from sqlalchemy import func, null
from sqlmodel import Session, select

from kaliok.audit import AuditActor, AuditObject, record_audit_event
from kaliok.storage.models import (
    ArtifactType,
    Capability,
    CapabilityArtifactContract,
    ComponentCapability,
    ComponentVersion,
    PipelineBinding,
    PipelineBindingCapability,
    PipelineBindingDependency,
    PipelineBindingNode,
    PipelineDefinition,
    PipelineRevision,
    RagTemplate,
    RagTemplateCapability,
    RagTemplateDependency,
    RagTemplateEdge,
    RagTemplateNode,
    RagTemplateRevision,
)


def _uuid(value: UUID | str) -> UUID:
    return value if isinstance(value, UUID) else UUID(str(value))


class RagComposerMutationService:
    """Mutate an isolated Composer graph without committing the caller session."""

    def __init__(self, session: Session) -> None:
        self.session = session

    @staticmethod
    def _actor(actor: AuditActor | dict[str, Any] | None):
        return actor or AuditActor(actor_type="system", key="rag-composer")

    def fork_pipeline_revision_with_graph(
        self,
        pipeline_revision_id: UUID | str,
        *,
        actor: AuditActor | dict[str, Any] | None = None,
    ) -> PipelineRevision:
        """Fork a pipeline and its complete graph inside a savepoint."""
        with self.session.begin_nested():
            source = self.session.get(PipelineRevision, _uuid(pipeline_revision_id))
            if source is None:
                raise ValueError("La révision de pipeline demandée est introuvable.")
            if source.rag_template_revision_id is None:
                raise ValueError("Cette révision ne possède pas de modèle RAG à modifier.")

            # Serialize revision-number allocation for this pipeline/template pair.
            definition = self.session.exec(
                select(PipelineDefinition)
                .where(PipelineDefinition.id == source.pipeline_definition_id)
                .with_for_update()
            ).one()
            source_template_revision = self.session.get(
                RagTemplateRevision, source.rag_template_revision_id
            )
            if source_template_revision is None:
                raise ValueError("La révision du modèle RAG source est introuvable.")
            template = self.session.exec(
                select(RagTemplate)
                .where(RagTemplate.id == source_template_revision.rag_template_id)
                .with_for_update()
            ).one()

            next_template_number = self.session.exec(
                select(func.max(RagTemplateRevision.revision_number)).where(
                    RagTemplateRevision.rag_template_id == template.id
                )
            ).one() or 0
            new_template_revision = RagTemplateRevision(
                rag_template_id=template.id,
                revision_number=next_template_number + 1,
                status="draft",
                change_reason="Révision de travail créée depuis le Composer.",
            )
            self.session.add(new_template_revision)
            self.session.flush()

            next_pipeline_number = self.session.exec(
                select(func.max(PipelineRevision.revision_number)).where(
                    PipelineRevision.pipeline_definition_id == definition.id
                )
            ).one() or 0
            new_revision = PipelineRevision(
                pipeline_definition_id=definition.id,
                rag_template_revision_id=new_template_revision.id,
                revision_number=next_pipeline_number + 1,
                status="draft",
                manifest_hash=source.manifest_hash,
                change_reason="Révision de travail créée depuis le Composer.",
                created_by_display_name=source.created_by_display_name,
            )
            self.session.add(new_revision)
            self.session.flush()

            node_id_map: dict[UUID, UUID] = {}
            source_nodes = self.session.exec(
                select(RagTemplateNode)
                .where(
                    RagTemplateNode.rag_template_revision_id
                    == source_template_revision.id
                )
                .order_by(RagTemplateNode.position, RagTemplateNode.node_key)
            ).all()
            for old in source_nodes:
                new = RagTemplateNode(
                    rag_template_revision_id=new_template_revision.id,
                    node_key=old.node_key,
                    capability_id=old.capability_id,
                    display_name=old.display_name,
                    description=old.description,
                    zone_key=old.zone_key,
                    requirement_mode=old.requirement_mode,
                    position=old.position,
                    enabled=old.enabled,
                    configuration=dict(old.configuration or {}),
                )
                self.session.add(new)
                self.session.flush()
                node_id_map[old.id] = new.id

            for old in self.session.exec(
                select(RagTemplateEdge).where(
                    RagTemplateEdge.rag_template_revision_id
                    == source_template_revision.id
                )
            ).all():
                self.session.add(
                    RagTemplateEdge(
                        rag_template_revision_id=new_template_revision.id,
                        source_node_id=node_id_map[old.source_node_id],
                        target_node_id=node_id_map[old.target_node_id],
                        edge_key=old.edge_key,
                        edge_type=old.edge_type,
                        source_port_key=old.source_port_key,
                        target_port_key=old.target_port_key,
                        condition=dict(old.condition) if old.condition is not None else null(),
                        priority=old.priority,
                        enabled=old.enabled,
                        configuration=dict(old.configuration or {}),
                    )
                )

            # Keep the legacy template projection coherent while it remains in use.
            for old in self.session.exec(
                select(RagTemplateCapability).where(
                    RagTemplateCapability.rag_template_revision_id
                    == source_template_revision.id
                )
            ).all():
                self.session.add(
                    RagTemplateCapability(
                        rag_template_revision_id=new_template_revision.id,
                        capability_id=old.capability_id,
                        requirement_mode=old.requirement_mode,
                        display_order=old.display_order,
                        configuration=dict(old.configuration or {}),
                    )
                )
            for old in self.session.exec(
                select(RagTemplateDependency).where(
                    RagTemplateDependency.rag_template_revision_id
                    == source_template_revision.id
                )
            ).all():
                self.session.add(
                    RagTemplateDependency(
                        rag_template_revision_id=new_template_revision.id,
                        source_capability_id=old.source_capability_id,
                        target_capability_id=old.target_capability_id,
                    )
                )

            binding_id_map: dict[UUID, UUID] = {}
            source_bindings = self.session.exec(
                select(PipelineBinding)
                .where(PipelineBinding.pipeline_revision_id == source.id)
                .order_by(PipelineBinding.position)
            ).all()
            for old in source_bindings:
                new = PipelineBinding(
                    pipeline_revision_id=new_revision.id,
                    binding_key=old.binding_key,
                    component_version_id=old.component_version_id,
                    resource_instance_id=old.resource_instance_id,
                    position=old.position,
                    enabled=old.enabled,
                    configuration=dict(old.configuration or {}),
                )
                self.session.add(new)
                self.session.flush()
                binding_id_map[old.id] = new.id

            old_binding_ids = list(binding_id_map)
            if old_binding_ids:
                for old in self.session.exec(
                    select(PipelineBindingCapability).where(
                        PipelineBindingCapability.pipeline_binding_id.in_(old_binding_ids)
                    )
                ).all():
                    self.session.add(
                        PipelineBindingCapability(
                            pipeline_binding_id=binding_id_map[old.pipeline_binding_id],
                            component_capability_id=old.component_capability_id,
                            enabled=old.enabled,
                            configuration=dict(old.configuration or {}),
                        )
                    )
                for old in self.session.exec(
                    select(PipelineBindingDependency).where(
                        PipelineBindingDependency.pipeline_binding_id.in_(old_binding_ids)
                    )
                ).all():
                    self.session.add(
                        PipelineBindingDependency(
                            pipeline_binding_id=binding_id_map[old.pipeline_binding_id],
                            depends_on_binding_id=binding_id_map[old.depends_on_binding_id],
                        )
                    )
                for old in self.session.exec(
                    select(PipelineBindingNode).where(
                        PipelineBindingNode.pipeline_revision_id == source.id
                    )
                ).all():
                    self.session.add(
                        PipelineBindingNode(
                            pipeline_binding_id=binding_id_map[old.pipeline_binding_id],
                            rag_template_node_id=node_id_map[old.rag_template_node_id],
                            pipeline_revision_id=new_revision.id,
                            rag_template_revision_id=new_template_revision.id,
                            enabled=old.enabled,
                            is_selected=old.is_selected,
                            priority=old.priority,
                            configuration=dict(old.configuration or {}),
                        )
                    )

            record_audit_event(
                self.session,
                actor=self._actor(actor),
                action="rag_composer_revision_forked",
                object=AuditObject(
                    "pipeline_revisions",
                    new_revision.id,
                    f"{definition.pipeline_key}:v{new_revision.revision_number}",
                    new_revision.id,
                ),
                after_state={
                    "source_pipeline_revision_id": str(source.id),
                    "pipeline_revision_id": str(new_revision.id),
                    "rag_template_revision_id": str(new_template_revision.id),
                    "node_count": len(node_id_map),
                    "binding_count": len(binding_id_map),
                },
            )
            self.session.flush()
            return new_revision

    def create_node(
        self,
        pipeline_revision_id: UUID | str,
        capability_id: UUID | str,
        *,
        actor: AuditActor | dict[str, Any] | None = None,
    ) -> RagTemplateNode:
        """Create one unconfigured function on a private draft graph."""
        with self.session.begin_nested():
            revision = self.session.get(PipelineRevision, _uuid(pipeline_revision_id))
            if revision is None:
                raise ValueError("La révision de pipeline demandée est introuvable.")
            if revision.status != "draft":
                raise ValueError("Modifier le RAG avant d’ajouter une fonction.")
            if revision.rag_template_revision_id is None:
                raise ValueError("Cette révision ne possède pas de modèle RAG.")
            template_revision = self.session.exec(
                select(RagTemplateRevision)
                .where(RagTemplateRevision.id == revision.rag_template_revision_id)
                .with_for_update()
            ).first()
            if template_revision is None or template_revision.status != "draft":
                raise ValueError("La révision du modèle RAG n’est pas modifiable.")
            owners = self.session.exec(
                select(func.count()).select_from(PipelineRevision).where(
                    PipelineRevision.rag_template_revision_id == template_revision.id
                )
            ).one()
            if owners != 1:
                raise ValueError(
                    "Cette révision de travail n’est pas isolée ; utilisez Modifier le RAG."
                )
            capability = self.session.get(Capability, _uuid(capability_id))
            if capability is None or not capability.is_active:
                raise ValueError("La fonction sélectionnée est introuvable ou inactive.")

            existing = self.session.exec(
                select(RagTemplateNode).where(
                    RagTemplateNode.rag_template_revision_id == template_revision.id
                )
            ).all()
            base_key = re.sub(r"[^a-z0-9_-]+", "-", capability.capability_key.lower()).strip("-") or "function"
            used_keys = {item.node_key for item in existing}
            node_key = base_key
            suffix = 2
            while node_key in used_keys:
                node_key = f"{base_key}-{suffix}"
                suffix += 1
            node = RagTemplateNode(
                rag_template_revision_id=template_revision.id,
                node_key=node_key,
                capability_id=capability.id,
                display_name=capability.display_name,
                description=capability.description,
                zone_key=capability.phase_key,
                requirement_mode="required",
                position=max((item.position for item in existing), default=-1) + 1,
                enabled=True,
                configuration={},
            )
            self.session.add(node)
            self.session.flush()
            record_audit_event(
                self.session,
                actor=self._actor(actor),
                action="rag_composer_node_created",
                object=AuditObject(
                    "rag_template_nodes", node.id, node.node_key, revision.id
                ),
                after_state={
                    "pipeline_revision_id": str(revision.id),
                    "rag_template_revision_id": str(template_revision.id),
                    "capability_id": str(capability.id),
                    "node_key": node.node_key,
                    "configured": False,
                },
            )
            self.session.flush()
            return node

    def create_empty_revision(
        self,
        source_pipeline_revision_id: UUID | str,
        *,
        actor: AuditActor | dict[str, Any] | None = None,
    ) -> PipelineRevision:
        """Create a private draft revision with an intentionally empty graph."""
        with self.session.begin_nested():
            source = self.session.get(PipelineRevision, _uuid(source_pipeline_revision_id))
            if source is None or source.rag_template_revision_id is None:
                raise ValueError("La révision de référence est introuvable ou sans modèle RAG.")
            definition = self.session.exec(select(PipelineDefinition).where(
                PipelineDefinition.id == source.pipeline_definition_id).with_for_update()).one()
            source_template = self.session.get(RagTemplateRevision, source.rag_template_revision_id)
            template = self.session.exec(select(RagTemplate).where(
                RagTemplate.id == source_template.rag_template_id).with_for_update()).one()
            template_number = self.session.exec(select(func.max(RagTemplateRevision.revision_number)).where(
                RagTemplateRevision.rag_template_id == template.id)).one() or 0
            template_revision = RagTemplateRevision(
                rag_template_id=template.id, revision_number=template_number + 1,
                status="draft", change_reason="RAG de travail créé depuis zéro dans le Lab.",
            )
            self.session.add(template_revision)
            self.session.flush()
            pipeline_number = self.session.exec(select(func.max(PipelineRevision.revision_number)).where(
                PipelineRevision.pipeline_definition_id == definition.id)).one() or 0
            revision = PipelineRevision(
                pipeline_definition_id=definition.id,
                rag_template_revision_id=template_revision.id,
                revision_number=pipeline_number + 1,
                status="draft",
                manifest_hash=source.manifest_hash,
                change_reason="RAG de travail créé depuis zéro dans le Lab.",
                created_by_display_name=source.created_by_display_name,
            )
            self.session.add(revision)
            self.session.flush()
            record_audit_event(
                self.session, actor=self._actor(actor), action="rag_composer_empty_revision_created",
                object=AuditObject("pipeline_revisions", revision.id,
                                   f"{definition.pipeline_key}:v{revision.revision_number}", revision.id),
                after_state={"source_pipeline_revision_id": str(source.id),
                             "pipeline_revision_id": str(revision.id), "node_count": 0},
            )
            self.session.flush()
            return revision

    def select_node_tool(
        self, pipeline_revision_id: UUID | str, node_id: UUID | str,
        component_version_id: UUID | str, *, configuration: dict[str, Any] | None = None,
        actor: AuditActor | dict[str, Any] | None = None,
    ) -> PipelineBindingNode:
        """Select a real ComponentVersion for one node in an isolated draft."""
        with self.session.begin_nested():
            revision, node = self._editable_node(pipeline_revision_id, node_id)
            version = self.session.get(ComponentVersion, _uuid(component_version_id))
            if version is None or version.status != "available":
                raise ValueError("La version d'outil sélectionnée n'est pas disponible.")
            component_capability = self.session.exec(select(ComponentCapability).where(
                ComponentCapability.component_version_id == version.id,
                ComponentCapability.capability_id == node.capability_id,
            )).first()
            if component_capability is None:
                raise ValueError("Cet outil ne fournit pas la fonction du node.")
            existing_links = list(self.session.exec(select(PipelineBindingNode).where(
                PipelineBindingNode.pipeline_revision_id == revision.id,
                PipelineBindingNode.rag_template_node_id == node.id,
            )).all())
            for item in existing_links:
                item.is_selected = False
                self.session.add(item)
            selected = next((item for item in existing_links
                             if self.session.get(PipelineBinding, item.pipeline_binding_id).component_version_id == version.id), None)
            if selected is None:
                bindings = list(self.session.exec(select(PipelineBinding).where(
                    PipelineBinding.pipeline_revision_id == revision.id)).all())
                base = re.sub(r"[^a-z0-9_-]+", "-", node.node_key.lower()).strip("-") + "-tool"
                used = {item.binding_key for item in bindings}
                key, suffix = base, 2
                while key in used:
                    key, suffix = f"{base}-{suffix}", suffix + 1
                binding = PipelineBinding(
                    pipeline_revision_id=revision.id, binding_key=key,
                    component_version_id=version.id,
                    position=max((item.position for item in bindings), default=-1) + 1,
                    enabled=True, configuration=dict(configuration or {}),
                )
                self.session.add(binding)
                self.session.flush()
                self.session.add(PipelineBindingCapability(
                    pipeline_binding_id=binding.id,
                    component_capability_id=component_capability.id,
                    enabled=True, configuration={},
                ))
                selected = PipelineBindingNode(
                    pipeline_binding_id=binding.id, rag_template_node_id=node.id,
                    pipeline_revision_id=revision.id,
                    rag_template_revision_id=revision.rag_template_revision_id,
                    enabled=True, is_selected=True, priority=0, configuration={},
                )
                self.session.add(selected)
            else:
                binding = self.session.get(PipelineBinding, selected.pipeline_binding_id)
                binding.configuration = dict(configuration or {})
                binding.enabled = True
                selected.enabled = True
                selected.is_selected = True
                self.session.add(binding)
                self.session.add(selected)
            self.session.flush()
            record_audit_event(
                self.session, actor=self._actor(actor), action="rag_composer_node_tool_selected",
                object=AuditObject("rag_template_nodes", node.id, node.node_key, revision.id),
                after_state={"pipeline_revision_id": str(revision.id),
                             "component_version_id": str(version.id),
                             "binding_id": str(selected.pipeline_binding_id)},
            )
            return selected

    def create_edge(
        self, pipeline_revision_id: UUID | str, source_node_id: UUID | str,
        target_node_id: UUID | str, *, source_port_key: str | None = None,
        target_port_key: str | None = None,
        actor: AuditActor | dict[str, Any] | None = None,
    ) -> RagTemplateEdge:
        """Persist one contract-compatible edge without imposing graph linearity."""
        with self.session.begin_nested():
            revision, source = self._editable_node(pipeline_revision_id, source_node_id)
            _, target = self._editable_node(revision.id, target_node_id)
            if source.id == target.id:
                raise ValueError("Une fonction ne peut pas être reliée à elle-même.")
            output_contracts = self._contracts(source.capability_id, "output")
            input_contracts = self._contracts(target.capability_id, "input")
            matches = [(out, inp) for out in output_contracts for inp in input_contracts
                       if out.artifact_type_id == inp.artifact_type_id
                       and (source_port_key is None or out.port_key == source_port_key)
                       and (target_port_key is None or inp.port_key == target_port_key)]
            if len(matches) != 1:
                raise ValueError("Les ports choisis ne définissent pas une unique connexion d'artefacts compatible.")
            out, inp = matches[0]
            edges = list(self.session.exec(select(RagTemplateEdge).where(
                RagTemplateEdge.rag_template_revision_id == revision.rag_template_revision_id)).all())
            base = f"{source.node_key}-to-{target.node_key}"
            used = {item.edge_key for item in edges}
            key, suffix = base, 2
            while key in used:
                key, suffix = f"{base}-{suffix}", suffix + 1
            edge = RagTemplateEdge(
                rag_template_revision_id=revision.rag_template_revision_id,
                source_node_id=source.id, target_node_id=target.id,
                edge_key=key, edge_type="artifact",
                source_port_key=out.port_key, target_port_key=inp.port_key,
                condition=null(),
                priority=max((item.priority for item in edges), default=-1) + 1,
                enabled=True, configuration={},
            )
            self.session.add(edge)
            self.session.flush()
            record_audit_event(
                self.session, actor=self._actor(actor), action="rag_composer_edge_created",
                object=AuditObject("rag_template_edges", edge.id, edge.edge_key, revision.id),
                after_state={"source_node_id": str(source.id), "target_node_id": str(target.id),
                             "source_port_key": out.port_key, "target_port_key": inp.port_key},
            )
            return edge

    def _editable_node(self, pipeline_revision_id, node_id):
        revision = self.session.get(PipelineRevision, _uuid(pipeline_revision_id))
        if revision is None or revision.status != "draft" or revision.rag_template_revision_id is None:
            raise ValueError("Seul un RAG de travail isolé peut être modifié.")
        owners = self.session.exec(select(func.count()).select_from(PipelineRevision).where(
            PipelineRevision.rag_template_revision_id == revision.rag_template_revision_id)).one()
        if owners != 1:
            raise ValueError("Cette révision de travail n'est pas isolée.")
        node = self.session.get(RagTemplateNode, _uuid(node_id))
        if node is None or node.rag_template_revision_id != revision.rag_template_revision_id:
            raise ValueError("Le node n'appartient pas à ce RAG de travail.")
        return revision, node

    def _contracts(self, capability_id, direction):
        return list(self.session.exec(select(CapabilityArtifactContract).where(
            CapabilityArtifactContract.capability_id == capability_id,
            CapabilityArtifactContract.direction == direction,
        )).all())


__all__ = ["RagComposerMutationService"]
