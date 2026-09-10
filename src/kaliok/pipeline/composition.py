"""Composition services for the descriptive pipeline graph.

This module deliberately stops at composition.  The legacy runtime continues
to consume ``pipeline_binding_capabilities`` and ``PipelineManifest``.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from sqlmodel import Session, select

from kaliok.audit import AuditActor, AuditObject, record_audit_event, validate_configuration
from kaliok.storage.models import (
    Capability,
    ComponentCapability,
    ComponentVersion,
    Connection,
    CredentialReference,
    PipelineBinding,
    PipelineBindingCapability,
    PipelineBindingDependency,
    PipelineBindingNode,
    PipelineRevision,
    RagTemplateNode,
    ResourceInstance,
    ResourceInstanceCapability,
    ResourceInstanceConnection,
    ResourceInstanceCredential,
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _uuid(value: UUID | str) -> UUID:
    return value if isinstance(value, UUID) else UUID(str(value))


@dataclass(frozen=True)
class CompositionValidationResult:
    """Three progressively stronger validation levels for one revision."""

    revision_id: UUID
    structure_valid: bool
    activatable: bool
    runtime_ready: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    runtime_errors: tuple[str, ...] = ()

    @property
    def is_valid(self) -> bool:
        return self.structure_valid


class PipelineCompositionService:
    """Manage descriptive binding-node links and their legacy projection."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def load_binding_nodes(self, revision_id: UUID | None = None) -> list[PipelineBindingNode]:
        statement = select(PipelineBindingNode).order_by(
            PipelineBindingNode.pipeline_revision_id,
            PipelineBindingNode.rag_template_node_id,
            PipelineBindingNode.priority,
            PipelineBindingNode.pipeline_binding_id,
        )
        if revision_id is not None:
            statement = statement.where(PipelineBindingNode.pipeline_revision_id == revision_id)
        return self.session.exec(statement).all()

    def _revision(self, revision_id: UUID) -> PipelineRevision:
        revision = self.session.get(PipelineRevision, revision_id)
        if revision is None:
            raise ValueError(f"PipelineRevision introuvable : {revision_id}.")
        return revision

    def _mutable_revision(self, revision_id: UUID) -> PipelineRevision:
        revision = self._revision(revision_id)
        if revision.status == "active":
            raise ValueError(
                "Une PipelineRevision active est immuable ; créez une nouvelle révision."
            )
        return revision

    def _binding(self, revision_id: UUID, binding_id: UUID) -> PipelineBinding:
        binding = self.session.get(PipelineBinding, binding_id)
        if binding is None or binding.pipeline_revision_id != revision_id:
            raise ValueError("Le binding n'appartient pas à la PipelineRevision.")
        return binding

    def _node(self, revision: PipelineRevision, node_id: UUID) -> RagTemplateNode:
        node = self.session.get(RagTemplateNode, node_id)
        if node is None:
            raise ValueError("RagTemplateNode introuvable.")
        if revision.rag_template_revision_id is None:
            raise ValueError("La PipelineRevision ne possède pas de révision de template.")
        if node.rag_template_revision_id != revision.rag_template_revision_id:
            raise ValueError("Le node n'appartient pas à la révision de template du pipeline.")
        return node

    def _component_capability(self, binding: PipelineBinding, node: RagTemplateNode) -> ComponentCapability:
        candidates = self.session.exec(
            select(ComponentCapability)
            .join(Capability, Capability.id == ComponentCapability.capability_id)
            .where(
                ComponentCapability.component_version_id == binding.component_version_id,
                ComponentCapability.capability_id == node.capability_id,
            )
        ).all()
        if len(candidates) != 1:
            raise ValueError(
                f"{binding.binding_key}: le composant ne fournit pas exactement une "
                f"capability pour le node {node.node_key}."
            )
        return candidates[0]

    @staticmethod
    def _default_actor(actor: AuditActor | dict[str, Any] | None) -> AuditActor | dict[str, Any]:
        return actor or AuditActor(actor_type="system", key="pipeline-composition-service")

    @staticmethod
    def _state(link: PipelineBindingNode) -> dict[str, Any]:
        return {
            "pipeline_revision_id": str(link.pipeline_revision_id),
            "pipeline_binding_id": str(link.pipeline_binding_id),
            "rag_template_revision_id": str(link.rag_template_revision_id),
            "rag_template_node_id": str(link.rag_template_node_id),
            "enabled": link.enabled,
            "is_selected": link.is_selected,
            "priority": link.priority,
            "configuration": link.configuration or {},
        }

    def _audit(
        self,
        *,
        action: str,
        revision: PipelineRevision,
        link: PipelineBindingNode,
        actor: AuditActor | dict[str, Any] | None,
        before: dict[str, Any] | None,
        after: dict[str, Any] | None,
    ) -> None:
        record_audit_event(
            self.session,
            actor=self._default_actor(actor),
            action=action,
            object=AuditObject(
                "pipeline_binding_nodes",
                object_id=link.pipeline_binding_id,
                object_key=f"{link.pipeline_binding_id}:{link.rag_template_node_id}",
                revision_id=revision.id,
            ),
            before_state=before,
            after_state=after,
        )

    def add_binding_node(
        self,
        pipeline_revision_id: UUID,
        pipeline_binding_id: UUID,
        rag_template_node_id: UUID,
        *,
        enabled: bool = True,
        is_selected: bool = False,
        priority: int = 0,
        configuration: dict[str, Any] | None = None,
        actor: AuditActor | dict[str, Any] | None = None,
    ) -> PipelineBindingNode:
        revision = self._mutable_revision(_uuid(pipeline_revision_id))
        binding = self._binding(revision.id, _uuid(pipeline_binding_id))
        node = self._node(revision, _uuid(rag_template_node_id))
        if priority < 0:
            raise ValueError("La priorité d'un lien doit être positive ou nulle.")
        if is_selected and not enabled:
            raise ValueError("Un lien désactivé ne peut pas être sélectionné.")
        self._component_capability(binding, node)
        values = validate_configuration(configuration or {})
        link = PipelineBindingNode(
            pipeline_revision_id=revision.id,
            pipeline_binding_id=binding.id,
            rag_template_revision_id=node.rag_template_revision_id,
            rag_template_node_id=node.id,
            enabled=enabled,
            is_selected=is_selected,
            priority=priority,
            configuration=values,
            created_at=_now(),
            updated_at=_now(),
        )
        self.session.add(link)
        self.session.flush()
        self._audit(
            action="binding_node_added",
            revision=revision,
            link=link,
            actor=actor,
            before=None,
            after=self._state(link),
        )
        return link

    def _link(self, revision_id: UUID, binding_id: UUID, node_id: UUID) -> PipelineBindingNode:
        link = self.session.get(PipelineBindingNode, (binding_id, node_id))
        if link is None or link.pipeline_revision_id != revision_id:
            raise ValueError("Le lien binding-node est introuvable dans la révision.")
        return link

    def remove_binding_node(
        self,
        pipeline_revision_id: UUID,
        pipeline_binding_id: UUID,
        rag_template_node_id: UUID,
        *,
        actor: AuditActor | dict[str, Any] | None = None,
    ) -> None:
        revision = self._mutable_revision(_uuid(pipeline_revision_id))
        link = self._link(revision.id, _uuid(pipeline_binding_id), _uuid(rag_template_node_id))
        before = self._state(link)
        self.session.delete(link)
        self.session.flush()
        self._audit(
            action="binding_node_removed",
            revision=revision,
            link=link,
            actor=actor,
            before=before,
            after=None,
        )

    def disable_binding_node(
        self,
        pipeline_revision_id: UUID,
        pipeline_binding_id: UUID,
        rag_template_node_id: UUID,
        *,
        enabled: bool = False,
        actor: AuditActor | dict[str, Any] | None = None,
    ) -> PipelineBindingNode:
        revision = self._mutable_revision(_uuid(pipeline_revision_id))
        link = self._link(revision.id, _uuid(pipeline_binding_id), _uuid(rag_template_node_id))
        before = self._state(link)
        link.enabled = enabled
        if not enabled:
            link.is_selected = False
        link.updated_at = _now()
        self.session.add(link)
        self.session.flush()
        self._audit(
            action="binding_node_enabled_changed",
            revision=revision,
            link=link,
            actor=actor,
            before=before,
            after=self._state(link),
        )
        return link

    def select_binding_for_node(
        self,
        pipeline_revision_id: UUID,
        rag_template_node_id: UUID,
        pipeline_binding_id: UUID,
        *,
        actor: AuditActor | dict[str, Any] | None = None,
    ) -> PipelineBindingNode:
        revision = self._mutable_revision(_uuid(pipeline_revision_id))
        node = self._node(revision, _uuid(rag_template_node_id))
        selected = self._link(revision.id, _uuid(pipeline_binding_id), node.id)
        if not selected.enabled:
            raise ValueError("Un lien désactivé ne peut pas être sélectionné.")
        before = self._state(selected)
        for candidate in self.session.exec(
            select(PipelineBindingNode).where(
                PipelineBindingNode.pipeline_revision_id == revision.id,
                PipelineBindingNode.rag_template_node_id == node.id,
                PipelineBindingNode.is_selected.is_(True),
            )
        ).all():
            candidate.is_selected = candidate is selected
            candidate.updated_at = _now()
            self.session.add(candidate)
        selected.is_selected = True
        self.session.add(selected)
        self.session.flush()
        self._audit(
            action="binding_node_selection_changed",
            revision=revision,
            link=selected,
            actor=actor,
            before=before,
            after=self._state(selected),
        )
        return selected

    def set_priority(
        self,
        pipeline_revision_id: UUID,
        pipeline_binding_id: UUID,
        rag_template_node_id: UUID,
        priority: int,
        *,
        actor: AuditActor | dict[str, Any] | None = None,
    ) -> PipelineBindingNode:
        revision = self._mutable_revision(_uuid(pipeline_revision_id))
        link = self._link(revision.id, _uuid(pipeline_binding_id), _uuid(rag_template_node_id))
        if priority < 0:
            raise ValueError("La priorité d'un lien doit être positive ou nulle.")
        before = self._state(link)
        link.priority = priority
        link.updated_at = _now()
        self.session.add(link)
        self.session.flush()
        self._audit(
            action="binding_node_priority_changed",
            revision=revision,
            link=link,
            actor=actor,
            before=before,
            after=self._state(link),
        )
        return link

    def update_configuration(
        self,
        pipeline_revision_id: UUID,
        pipeline_binding_id: UUID,
        rag_template_node_id: UUID,
        configuration: dict[str, Any],
        *,
        actor: AuditActor | dict[str, Any] | None = None,
    ) -> PipelineBindingNode:
        revision = self._mutable_revision(_uuid(pipeline_revision_id))
        link = self._link(revision.id, _uuid(pipeline_binding_id), _uuid(rag_template_node_id))
        before = self._state(link)
        link.configuration = validate_configuration(configuration)
        link.updated_at = _now()
        self.session.add(link)
        self.session.flush()
        self._audit(
            action="binding_node_configuration_changed",
            revision=revision,
            link=link,
            actor=actor,
            before=before,
            after=self._state(link),
        )
        return link

    def set_resource_instance(
        self,
        pipeline_revision_id: UUID,
        pipeline_binding_id: UUID,
        resource_instance_id: UUID | None,
        *,
        actor: AuditActor | dict[str, Any] | None = None,
    ) -> PipelineBinding:
        revision = self._mutable_revision(_uuid(pipeline_revision_id))
        binding = self._binding(revision.id, _uuid(pipeline_binding_id))
        before = {"resource_instance_id": str(binding.resource_instance_id) if binding.resource_instance_id else None}
        instance = self.session.get(ResourceInstance, resource_instance_id) if resource_instance_id else None
        if resource_instance_id and instance is None:
            raise ValueError("ResourceInstance introuvable.")
        if instance is not None and instance.component_version_id != binding.component_version_id:
            raise ValueError("La ResourceInstance ne correspond pas au ComponentVersion du binding.")
        binding.resource_instance_id = resource_instance_id
        self.session.add(binding)
        self.session.flush()
        record_audit_event(
            self.session,
            actor=self._default_actor(actor),
            action="binding_resource_instance_changed",
            object=AuditObject(
                "pipeline_bindings", binding.id, binding.binding_key, revision.id
            ),
            before_state=before,
            after_state={"resource_instance_id": str(resource_instance_id) if resource_instance_id else None},
        )
        return binding

    def _projection(self, revision_id: UUID) -> tuple[dict[tuple[UUID, UUID], dict[str, Any]], list[str]]:
        revision = self._revision(revision_id)
        desired: dict[tuple[UUID, UUID], dict[str, Any]] = {}
        issues: list[str] = []
        for link in self.load_binding_nodes(revision.id):
            if not link.enabled or not link.is_selected:
                continue
            binding = self.session.get(PipelineBinding, link.pipeline_binding_id)
            node = self.session.get(RagTemplateNode, link.rag_template_node_id)
            if binding is None or node is None:
                issues.append("un lien binding-node référence une ligne supprimée")
                continue
            try:
                component_capability = self._component_capability(binding, node)
            except ValueError as error:
                issues.append(str(error))
                continue
            key = (binding.id, component_capability.id)
            projected = {"enabled": True, "configuration": dict(link.configuration or {})}
            previous = desired.get(key)
            if previous is not None and previous != projected:
                issues.append(
                    f"plusieurs nodes sélectionnés produisent des configurations divergentes pour {binding.binding_key}"
                )
            else:
                desired[key] = projected
        return desired, issues

    def compare_legacy_projection(self, revision_id: UUID) -> dict[str, Any]:
        desired, issues = self._projection(_uuid(revision_id))
        binding_ids = [row.id for row in self.session.exec(
            select(PipelineBinding).where(PipelineBinding.pipeline_revision_id == revision_id)
        ).all()]
        actual_rows = self.session.exec(
            select(PipelineBindingCapability).where(
                PipelineBindingCapability.pipeline_binding_id.in_(binding_ids)
            )
        ).all() if binding_ids else []
        actual = {
            (row.pipeline_binding_id, row.component_capability_id): {
                "enabled": row.enabled,
                "configuration": dict(row.configuration or {}),
            }
            for row in actual_rows
        }
        missing = sorted(set(desired) - set(actual), key=str)
        extra = sorted(set(actual) - set(desired), key=str)
        mismatched = sorted(
            [key for key in set(desired) & set(actual) if desired[key] != actual[key]],
            key=str,
        )
        return {
            "revision_id": revision_id,
            "missing": missing,
            "extra": extra,
            "mismatched": mismatched,
            "issues": tuple(issues),
            "is_coherent": not (missing or extra or mismatched or issues),
        }

    def project_legacy_capabilities(self, revision_id: UUID) -> list[PipelineBindingCapability]:
        """Persist the explicit graph-to-legacy projection in this transaction."""
        revision = self._mutable_revision(_uuid(revision_id))
        desired, issues = self._projection(revision.id)
        if issues:
            raise ValueError("Projection legacy impossible : " + "; ".join(issues))
        binding_ids = [row.id for row in self.session.exec(
            select(PipelineBinding).where(PipelineBinding.pipeline_revision_id == revision.id)
        ).all()]
        for row in self.session.exec(
            select(PipelineBindingCapability).where(
                PipelineBindingCapability.pipeline_binding_id.in_(binding_ids)
            )
        ).all():
            self.session.delete(row)
        self.session.flush()
        projected_rows: list[PipelineBindingCapability] = []
        for (binding_id, component_capability_id), values in sorted(desired.items(), key=lambda item: str(item[0])):
            row = PipelineBindingCapability(
                pipeline_binding_id=binding_id,
                component_capability_id=component_capability_id,
                enabled=values["enabled"],
                configuration=values["configuration"],
            )
            self.session.add(row)
            projected_rows.append(row)
        self.session.flush()
        return projected_rows

    def validate_pipeline_composition(
        self,
        revision_id: UUID,
        *,
        raise_on_error: bool = False,
    ) -> CompositionValidationResult:
        revision = self._revision(_uuid(revision_id))
        structural: list[str] = []
        activation: list[str] = []
        runtime: list[str] = []
        warnings: list[str] = []
        bindings = self.session.exec(
            select(PipelineBinding).where(PipelineBinding.pipeline_revision_id == revision.id)
        ).all()
        binding_by_id = {row.id: row for row in bindings}
        nodes = (
            self.session.exec(
                select(RagTemplateNode).where(
                    RagTemplateNode.rag_template_revision_id == revision.rag_template_revision_id
                )
            ).all()
            if revision.rag_template_revision_id
            else []
        )
        node_by_id = {row.id: row for row in nodes}
        links = self.load_binding_nodes(revision.id)
        selected_by_node: defaultdict[UUID, list[PipelineBindingNode]] = defaultdict(list)
        for link in links:
            binding = binding_by_id.get(link.pipeline_binding_id)
            node = node_by_id.get(link.rag_template_node_id)
            if binding is None:
                structural.append(f"le binding-node {link.pipeline_binding_id} sort de la révision")
                continue
            if node is None or link.rag_template_revision_id != revision.rag_template_revision_id:
                structural.append(f"le node {link.rag_template_node_id} sort du template du pipeline")
                continue
            if link.pipeline_revision_id != revision.id:
                structural.append("un binding-node porte une PipelineRevision incohérente")
            if link.is_selected:
                selected_by_node[node.id].append(link)
                if not link.enabled:
                    structural.append(f"le lien sélectionné {node.node_key} est désactivé")
                if not node.enabled:
                    structural.append(f"le node sélectionné {node.node_key} est désactivé")
            try:
                self._component_capability(binding, node)
            except ValueError as error:
                structural.append(str(error))
            try:
                validate_configuration(link.configuration or {})
            except (TypeError, ValueError) as error:
                structural.append(f"configuration sensible ou invalide pour {node.node_key}: {error}")
            if binding.resource_instance_id is not None:
                instance = self.session.get(ResourceInstance, binding.resource_instance_id)
                if instance is None:
                    structural.append(f"ResourceInstance absente pour {binding.binding_key}")
                elif instance.component_version_id != binding.component_version_id:
                    structural.append(f"ResourceInstance incompatible pour {binding.binding_key}")

        for node in nodes:
            selected = selected_by_node.get(node.id, [])
            if len(selected) > 1:
                structural.append(f"plusieurs bindings sélectionnés pour le node {node.node_key}")
            if node.requirement_mode == "required" and node.enabled and not selected:
                activation.append(f"le node requis n'est pas couvert : {node.node_key}")

        for binding in bindings:
            for dependency in self.session.exec(
                select(PipelineBindingDependency).where(
                    PipelineBindingDependency.pipeline_binding_id == binding.id
                )
            ).all():
                if dependency.depends_on_binding_id not in binding_by_id:
                    warnings.append(
                        f"la dépendance du binding {binding.binding_key} sort de la révision"
                    )

        projection = self.compare_legacy_projection(revision.id)
        if projection["issues"]:
            structural.extend(projection["issues"])
        if not projection["is_coherent"]:
            structural.append("la projection legacy diffère de la composition descriptive")

        # Availability is intentionally kept out of structure-valid.  An
        # unreachable service is an operational problem, not a bad design.
        for node_id, selected_links in selected_by_node.items():
            for link in selected_links:
                binding = binding_by_id[link.pipeline_binding_id]
                if binding.resource_instance_id is None:
                    runtime.append(f"aucune ResourceInstance sélectionnée pour {binding.binding_key}")
                    continue
                instance = self.session.get(ResourceInstance, binding.resource_instance_id)
                if instance is None:
                    continue
                if instance.status in {"unavailable", "disabled"}:
                    activation.append(f"ResourceInstance indisponible pour {binding.binding_key}")
                if instance.health_status in {"unreachable", "unhealthy", "failed"}:
                    runtime.append(f"ResourceInstance non joignable pour {binding.binding_key}")
                if instance.health_status not in {"healthy", "available", "ok", "ready"}:
                    warnings.append(f"health_status non confirmé pour {binding.binding_key}: {instance.health_status}")
                component_capability = None
                node = node_by_id.get(node_id)
                if node is not None:
                    try:
                        component_capability = self._component_capability(binding, node)
                    except ValueError:
                        pass
                if component_capability is not None:
                    qualification = self.session.get(
                        ResourceInstanceCapability,
                        (instance.id, component_capability.id),
                    )
                    if qualification is None:
                        warnings.append(f"qualification absente pour {binding.binding_key}/{node.node_key}")
                    elif qualification.availability_status in {"unavailable", "disabled"}:
                        activation.append(f"capability indisponible pour {binding.binding_key}/{node.node_key}")
                    elif qualification.availability_status != "available":
                        runtime.append(f"capability non disponible pour {binding.binding_key}/{node.node_key}")

                connections = self.session.exec(
                    select(ResourceInstanceConnection, Connection)
                    .join(Connection, Connection.id == ResourceInstanceConnection.connection_id)
                    .where(
                        ResourceInstanceConnection.resource_instance_id == instance.id,
                        ResourceInstanceConnection.enabled.is_(True),
                        ResourceInstanceConnection.required.is_(True),
                    )
                ).all()
                for _, connection in connections:
                    if connection.status in {"unavailable", "disabled"} or connection.health_status in {"unreachable", "unhealthy", "failed"}:
                        runtime.append(f"connexion indisponible pour {binding.binding_key}: {connection.connection_key}")
                credentials = self.session.exec(
                    select(ResourceInstanceCredential, CredentialReference)
                    .join(CredentialReference, CredentialReference.id == ResourceInstanceCredential.credential_reference_id)
                    .where(
                        ResourceInstanceCredential.resource_instance_id == instance.id,
                        ResourceInstanceCredential.enabled.is_(True),
                        ResourceInstanceCredential.required.is_(True),
                    )
                ).all()
                for _, credential in credentials:
                    if credential.status in {"unavailable", "disabled"}:
                        runtime.append(f"credential non résoluble pour {binding.binding_key}: {credential.credential_key}")

        result = CompositionValidationResult(
            revision_id=revision.id,
            structure_valid=not structural,
            activatable=not structural and not activation,
            runtime_ready=not structural and not activation and not runtime,
            errors=tuple(dict.fromkeys(structural + activation)),
            warnings=tuple(dict.fromkeys(warnings)),
            runtime_errors=tuple(dict.fromkeys(runtime)),
        )
        if raise_on_error and not result.structure_valid:
            raise ValueError("Composition pipeline invalide : " + "; ".join(result.errors))
        return result


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


def load_pipeline_binding_nodes(
    session: Session,
    revision_id: UUID | None = None,
) -> list[PipelineBindingNode]:
    return PipelineCompositionService(session).load_binding_nodes(revision_id)


__all__ = [
    "CompositionValidationResult",
    "PipelineCompositionService",
    "load_pipeline_binding_nodes",
    "validate_pipeline_composition",
]
