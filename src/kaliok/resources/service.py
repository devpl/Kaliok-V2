from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any
from uuid import UUID

from sqlmodel import Session, select

from kaliok.audit import AuditActor, AuditObject, record_audit_event, validate_configuration, validate_endpoint
from kaliok.storage.models import (
    Connection,
    ConnectionCredential,
    CredentialReference,
    ResourceInstance,
    ResourceInstanceCapability,
    ResourceInstanceConnection,
    ResourceInstanceCredential,
    utc_now,
)

SYSTEM_ACTOR = AuditActor(actor_type="system", key="kaliok")


def _required(value: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} ne peut pas être vide.")
    return value


def _connection_state(item: Connection) -> dict[str, Any]:
    return {
        "id": str(item.id), "connection_key": item.connection_key,
        "display_name": item.display_name, "connection_kind": item.connection_kind,
        "endpoint": item.endpoint, "status": item.status, "health_status": item.health_status,
        "configuration": item.configuration, "health_detail": item.health_detail,
    }


def _resource_state(item: ResourceInstance) -> dict[str, Any]:
    return {
        "id": str(item.id), "component_version_id": str(item.component_version_id),
        "instance_key": item.instance_key, "display_name": item.display_name,
        "runtime_kind": item.runtime_kind, "status": item.status,
        "health_status": item.health_status, "configuration": item.configuration,
    }


class ConnectionService:
    def __init__(self, session: Session):
        self.session = session

    def create(self, *, actor: AuditActor | Mapping[str, Any] = SYSTEM_ACTOR, **values: Any) -> Connection:
        configuration = validate_configuration(values.pop("configuration", {}))
        if "endpoint" in values:
            values["endpoint"] = validate_endpoint(values["endpoint"])
        connection = Connection(
            connection_key=_required(values.pop("connection_key"), "connection_key"),
            display_name=_required(values.pop("display_name"), "display_name"),
            connection_kind=_required(values.pop("connection_kind"), "connection_kind"),
            configuration=configuration,
            **values,
        )
        self.session.add(connection)
        self.session.flush()
        record_audit_event(
            self.session, actor=actor, action="created",
            object=AuditObject("connections", connection.id, connection.connection_key),
            after_state=_connection_state(connection),
        )
        return connection

    def get(self, connection_id: UUID) -> Connection | None:
        return self.session.get(Connection, connection_id)

    def list(self) -> list[Connection]:
        return list(self.session.exec(select(Connection).order_by(Connection.connection_key)).all())

    def update(self, connection_id: UUID, *, actor: AuditActor | Mapping[str, Any] = SYSTEM_ACTOR, **changes: Any) -> Connection:
        connection = self.session.get(Connection, connection_id)
        if connection is None:
            raise ValueError("Connexion introuvable.")
        before = _connection_state(connection)
        if "configuration" in changes:
            changes["configuration"] = validate_configuration(changes["configuration"])
        if "endpoint" in changes:
            changes["endpoint"] = validate_endpoint(changes["endpoint"])
        for field, value in changes.items():
            if field in {"connection_key", "display_name", "connection_kind"}:
                value = _required(value, field)
            if not hasattr(connection, field):
                raise ValueError(f"Champ de connexion inconnu: {field}.")
            setattr(connection, field, value)
        connection.updated_at = utc_now()
        self.session.add(connection)
        self.session.flush()
        after = _connection_state(connection)
        record_audit_event(
            self.session, actor=actor,
            action="configuration_changed" if "configuration" in changes else "updated",
            object=AuditObject("connections", connection.id, connection.connection_key),
            before_state=before, after_state=after,
        )
        return connection

    def attach_credential(self, connection_id: UUID, credential_reference_id: UUID, *, role_key: str = "default", actor: AuditActor | Mapping[str, Any] = SYSTEM_ACTOR, required: bool = True, enabled: bool = True) -> ConnectionCredential:
        link = ConnectionCredential(connection_id=connection_id, credential_reference_id=credential_reference_id, role_key=_required(role_key, "role_key"), required=required, enabled=enabled)
        self.session.add(link)
        self.session.flush()
        record_audit_event(self.session, actor=actor, action="updated", object=AuditObject("connection_credentials", connection_id, role_key), after_state={"credential_reference_id": str(credential_reference_id), "role_key": role_key, "required": required, "enabled": enabled})
        return link

    def record_health(self, connection_id: UUID, *, health_status: str, health_detail: dict[str, Any] | None = None, actor: AuditActor | Mapping[str, Any] = SYSTEM_ACTOR, checked_at: datetime | None = None) -> Connection:
        connection = self.get(connection_id)
        if connection is None:
            raise ValueError("Connexion introuvable.")
        before = _connection_state(connection)
        connection.health_status = _required(health_status, "health_status")
        connection.health_checked_at = checked_at or utc_now()
        connection.health_detail = validate_configuration(health_detail or {})
        connection.updated_at = utc_now()
        self.session.add(connection)
        self.session.flush()
        record_audit_event(self.session, actor=actor, action="health_checked", object=AuditObject("connections", connection.id, connection.connection_key), before_state=before, after_state=_connection_state(connection))
        return connection


class CredentialReferenceService:
    def __init__(self, session: Session):
        self.session = session

    def create(self, *, actor: AuditActor | Mapping[str, Any] = SYSTEM_ACTOR, **values: Any) -> CredentialReference:
        item = CredentialReference(
            credential_key=_required(values.pop("credential_key"), "credential_key"),
            display_name=_required(values.pop("display_name"), "display_name"),
            credential_type=_required(values.pop("credential_type"), "credential_type"),
            backend=_required(values.pop("backend"), "backend"),
            secret_locator=_required(values.pop("secret_locator"), "secret_locator"),
            **values,
        )
        self.session.add(item)
        self.session.flush()
        record_audit_event(
            self.session, actor=actor, action="created",
            object=AuditObject("credential_references", item.id, item.credential_key),
            after_state={"id": str(item.id), "credential_key": item.credential_key,
                         "credential_type": item.credential_type, "backend": item.backend},
        )
        return item


class ResourceInstanceService:
    def __init__(self, session: Session):
        self.session = session

    def create(self, *, actor: AuditActor | Mapping[str, Any] = SYSTEM_ACTOR, **values: Any) -> ResourceInstance:
        configuration = validate_configuration(values.pop("configuration", {}))
        item = ResourceInstance(
            component_version_id=values.pop("component_version_id"),
            instance_key=_required(values.pop("instance_key"), "instance_key"),
            display_name=_required(values.pop("display_name"), "display_name"),
            runtime_kind=_required(values.pop("runtime_kind"), "runtime_kind"),
            configuration=configuration,
            **values,
        )
        self.session.add(item)
        self.session.flush()
        record_audit_event(
            self.session, actor=actor, action="created",
            object=AuditObject("resource_instances", item.id, item.instance_key),
            after_state=_resource_state(item),
        )
        return item

    def get(self, instance_id: UUID) -> ResourceInstance | None:
        return self.session.get(ResourceInstance, instance_id)

    def list(self, *, component_version_id: UUID | None = None) -> list[ResourceInstance]:
        statement = select(ResourceInstance).order_by(ResourceInstance.instance_key)
        if component_version_id is not None:
            statement = statement.where(ResourceInstance.component_version_id == component_version_id)
        return list(self.session.exec(statement).all())

    def update(self, instance_id: UUID, *, actor: AuditActor | Mapping[str, Any] = SYSTEM_ACTOR, **changes: Any) -> ResourceInstance:
        item = self.session.get(ResourceInstance, instance_id)
        if item is None:
            raise ValueError("ResourceInstance introuvable.")
        before = _resource_state(item)
        if "configuration" in changes:
            changes["configuration"] = validate_configuration(changes["configuration"])
        for field, value in changes.items():
            if field in {"instance_key", "display_name", "runtime_kind"}:
                value = _required(value, field)
            if not hasattr(item, field):
                raise ValueError(f"Champ de ResourceInstance inconnu: {field}.")
            setattr(item, field, value)
        item.updated_at = utc_now()
        self.session.add(item)
        self.session.flush()
        record_audit_event(
            self.session, actor=actor,
            action="configuration_changed" if "configuration" in changes else "updated",
            object=AuditObject("resource_instances", item.id, item.instance_key),
            before_state=before, after_state=_resource_state(item),
        )
        return item

    def record_health(self, instance_id: UUID, *, health_status: str, health_detail: dict[str, Any] | None = None, actor: AuditActor | Mapping[str, Any] = SYSTEM_ACTOR, checked_at: datetime | None = None) -> ResourceInstance:
        item = self.get(instance_id)
        if item is None:
            raise ValueError("ResourceInstance introuvable.")
        before = _resource_state(item)
        item.health_status = _required(health_status, "health_status")
        item.health_checked_at = checked_at or utc_now()
        item.health_detail = validate_configuration(health_detail or {})
        item.updated_at = utc_now()
        self.session.add(item)
        self.session.flush()
        record_audit_event(self.session, actor=actor, action="health_checked", object=AuditObject("resource_instances", item.id, item.instance_key), before_state=before, after_state=_resource_state(item))
        return item

    def attach_connection(self, instance_id: UUID, connection_id: UUID, *, role_key: str = "default", actor: AuditActor | Mapping[str, Any] = SYSTEM_ACTOR, required: bool = True, enabled: bool = True, configuration: dict[str, Any] | None = None) -> ResourceInstanceConnection:
        validate_configuration(configuration or {})
        link = ResourceInstanceConnection(resource_instance_id=instance_id, connection_id=connection_id, role_key=_required(role_key, "role_key"), required=required, enabled=enabled, configuration=configuration or {})
        self.session.add(link)
        self.session.flush()
        record_audit_event(self.session, actor=actor, action="updated", object=AuditObject("resource_instance_connections", instance_id, role_key), after_state={"connection_id": str(connection_id), "role_key": role_key, "required": required, "enabled": enabled})
        return link

    def attach_credential(self, instance_id: UUID, credential_reference_id: UUID, *, role_key: str = "default", actor: AuditActor | Mapping[str, Any] = SYSTEM_ACTOR, required: bool = True, enabled: bool = True) -> ResourceInstanceCredential:
        link = ResourceInstanceCredential(resource_instance_id=instance_id, credential_reference_id=credential_reference_id, role_key=_required(role_key, "role_key"), required=required, enabled=enabled)
        self.session.add(link)
        self.session.flush()
        record_audit_event(self.session, actor=actor, action="updated", object=AuditObject("resource_instance_credentials", instance_id, role_key), after_state={"credential_reference_id": str(credential_reference_id), "role_key": role_key, "required": required, "enabled": enabled})
        return link

    def qualify(self, instance_id: UUID, component_capability_id: UUID, *, availability_status: str = "available", actor: AuditActor | Mapping[str, Any] = SYSTEM_ACTOR, configuration: dict[str, Any] | None = None) -> ResourceInstanceCapability:
        validate_configuration(configuration or {})
        link = ResourceInstanceCapability(resource_instance_id=instance_id, component_capability_id=component_capability_id, availability_status=_required(availability_status, "availability_status"), configuration=configuration or {})
        self.session.add(link)
        self.session.flush()
        record_audit_event(self.session, actor=actor, action="qualified", object=AuditObject("resource_instance_capabilities", instance_id), after_state={"component_capability_id": str(component_capability_id), "availability_status": availability_status, "configuration": configuration or {}})
        return link


__all__ = ["ConnectionService", "CredentialReferenceService", "ResourceInstanceService"]
