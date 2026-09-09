from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID

from sqlmodel import Session

from kaliok.audit.sanitization import diff_states, sanitize_audit_state
from kaliok.storage.models import AuditEvent, utc_now


@dataclass(frozen=True)
class AuditActor:
    actor_type: str
    user_id: UUID | None = None
    key: str | None = None
    display_name: str | None = None
    identifier: str | None = None


@dataclass(frozen=True)
class AuditObject:
    object_type: str
    object_id: UUID | None = None
    object_key: str | None = None
    revision_id: UUID | None = None


def _actor(value: AuditActor | dict[str, Any] | Any) -> AuditActor:
    if isinstance(value, AuditActor):
        return value
    if isinstance(value, dict):
        return AuditActor(
            actor_type=value["actor_type"],
            user_id=value.get("actor_user_id", value.get("user_id")),
            key=value.get("actor_key", value.get("key")),
            display_name=value.get("actor_display_name_snapshot", value.get("display_name")),
            identifier=value.get("actor_identifier_snapshot", value.get("identifier")),
        )
    return AuditActor(
        actor_type=getattr(value, "actor_type"),
        user_id=getattr(value, "actor_user_id", getattr(value, "user_id", None)),
        key=getattr(value, "actor_key", getattr(value, "key", None)),
        display_name=getattr(value, "actor_display_name_snapshot", getattr(value, "display_name", None)),
        identifier=getattr(value, "actor_identifier_snapshot", getattr(value, "identifier", None)),
    )


def _object(value: AuditObject | dict[str, Any] | Any) -> AuditObject:
    if isinstance(value, AuditObject):
        return value
    if isinstance(value, dict):
        return AuditObject(
            object_type=value.get("object_type", value.get("type", "object")),
            object_id=value.get("object_id", value.get("id")),
            object_key=value.get("object_key", value.get("key")),
            revision_id=value.get("object_revision_id", value.get("revision_id")),
        )
    return AuditObject(
        object_type=getattr(value, "__tablename__", value.__class__.__name__),
        object_id=getattr(value, "id", None),
        object_key=getattr(value, "instance_key", getattr(value, "connection_key", getattr(value, "key", None))),
    )


def record_audit_event(
    session: Session,
    *,
    actor: AuditActor | dict[str, Any] | Any,
    action: str,
    object: AuditObject | dict[str, Any] | Any,
    before_state: dict[str, Any] | None = None,
    after_state: dict[str, Any] | None = None,
    reason: str | None = None,
    metadata: dict[str, Any] | None = None,
    changed_fields: dict[str, Any] | None = None,
    occurred_at: datetime | None = None,
    request_id: UUID | None = None,
    execution_group_id: UUID | None = None,
) -> AuditEvent:
    """Stage one audit row in the caller's transaction; never commits."""
    actor_value = _actor(actor)
    object_value = _object(object)
    before = sanitize_audit_state(before_state) if before_state is not None else None
    after = sanitize_audit_state(after_state) if after_state is not None else None
    changes = (
        sanitize_audit_state(changed_fields)
        if changed_fields is not None
        else diff_states(before, after)
    )
    event = AuditEvent(
        occurred_at=occurred_at or utc_now(),
        actor_type=actor_value.actor_type,
        actor_user_id=actor_value.user_id,
        actor_key=actor_value.key,
        actor_display_name_snapshot=actor_value.display_name,
        actor_identifier_snapshot=actor_value.identifier,
        action=action,
        object_type=object_value.object_type,
        object_id=object_value.object_id,
        object_key=object_value.object_key,
        object_revision_id=object_value.revision_id,
        before_state=before,
        after_state=after,
        changed_fields=changes,
        reason=reason,
        request_id=request_id,
        execution_group_id=execution_group_id,
        extra_data=sanitize_audit_state(metadata or {}),
    )
    session.add(event)
    return event


class AuditService:
    def __init__(self, session: Session):
        self.session = session

    def record(self, **kwargs: Any) -> AuditEvent:
        return record_audit_event(self.session, **kwargs)
