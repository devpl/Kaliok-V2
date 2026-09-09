from __future__ import annotations

from collections.abc import Iterator
from uuid import uuid4

import pytest
from sqlalchemy import inspect
from sqlmodel import Session, select

from kaliok.audit import AuditActor, AuditObject, SensitiveConfigurationError, record_audit_event
from kaliok.resources import ConnectionService, CredentialReferenceService, ResourceInstanceService
from kaliok.storage.database import create_database_engine
from kaliok.storage.models import (
    AuditEvent,
    ComponentCapability,
    ComponentVersion,
    ConnectionCredential,
    ResourceInstanceCapability,
)


@pytest.fixture
def resource_session() -> Iterator[Session]:
    engine = create_database_engine()
    with engine.connect() as connection:
        transaction = connection.begin()
        try:
            if "audit_events" not in inspect(connection).get_table_names():
                pytest.skip("La migration 2A n'est pas appliquée dans cette base.")
            with Session(bind=connection) as session:
                yield session
        finally:
            if transaction.is_active:
                transaction.rollback()
    engine.dispose()


def _catalog_ids(session: Session) -> tuple[ComponentVersion, ComponentCapability]:
    version = session.exec(select(ComponentVersion)).first()
    capability = session.exec(select(ComponentCapability)).first()
    assert version is not None and capability is not None
    return version, capability


def test_migration_2a_schema_and_legacy_bindings(resource_session: Session):
    tables = set(inspect(resource_session.connection()).get_table_names())
    assert {
        "connections", "credential_references", "connection_credentials",
        "resource_instances", "resource_instance_connections",
        "resource_instance_credentials", "resource_instance_capabilities",
        "audit_events",
    } <= tables
    assert "resource_instance_id" in {
        column["name"] for column in inspect(resource_session.connection()).get_columns("pipeline_bindings")
    }
    assert resource_session.exec(select(AuditEvent)).all() == []


def test_connections_instances_credentials_and_roles(resource_session: Session):
    version, capability = _catalog_ids(resource_session)
    actor = AuditActor("user", user_id=uuid4(), display_name="Alice", identifier="alice@example.test")
    connections = ConnectionService(resource_session)
    credentials = CredentialReferenceService(resource_session)
    instances = ResourceInstanceService(resource_session)

    shared = connections.create(connection_key=f"shared-{uuid4().hex}", display_name="Shared", connection_kind="local_service", endpoint="http://localhost:11434", actor=actor)
    secondary = connections.create(connection_key=f"secondary-{uuid4().hex}", display_name="Secondary", connection_kind="database", actor=actor)
    credential_a = credentials.create(credential_key=f"cred-a-{uuid4().hex}", display_name="API ref", credential_type="api_key", backend="env", secret_locator="MISTRAL_API_KEY", actor=actor)
    credential_b = credentials.create(credential_key=f"cred-b-{uuid4().hex}", display_name="DB ref", credential_type="token", backend="vault", secret_locator="vault://kaliok/db", actor=actor)
    first = instances.create(component_version_id=version.id, instance_key=f"instance-a-{uuid4().hex}", display_name="A", runtime_kind="local_service", actor=actor)
    second = instances.create(component_version_id=version.id, instance_key=f"instance-b-{uuid4().hex}", display_name="B", runtime_kind="remote_service", actor=actor)

    instances.attach_connection(first.id, shared.id, role_key="model", actor=actor)
    instances.attach_connection(first.id, secondary.id, role_key="storage", actor=actor)
    instances.attach_connection(second.id, shared.id, role_key="default", actor=actor)
    connections.attach_credential(shared.id, credential_a.id, role_key="api", actor=actor)
    connections.attach_credential(shared.id, credential_b.id, role_key="fallback", actor=actor)
    instances.attach_credential(first.id, credential_a.id, role_key="model", actor=actor)
    instances.attach_credential(first.id, credential_b.id, role_key="storage", actor=actor)

    assert len(resource_session.exec(select(ConnectionCredential)).all()) == 2
    assert resource_session.exec(select(ResourceInstanceCapability)).all() == []
    unknown = ResourceInstanceCapability(resource_instance_id=first.id, component_capability_id=capability.id)
    resource_session.add(unknown)
    resource_session.flush()
    assert unknown.availability_status == "unknown"


def test_audit_snapshots_diff_redaction_and_transaction_rollback(resource_session: Session):
    version, _ = _catalog_ids(resource_session)
    actor = AuditActor("user", user_id=uuid4(), display_name="Before Rename", identifier="user-1")
    instances = ResourceInstanceService(resource_session)
    item = instances.create(component_version_id=version.id, instance_key=f"audit-{uuid4().hex}", display_name="Before", runtime_kind="api", actor=actor)
    instances.update(item.id, display_name="After", status="available", actor=actor)
    event = resource_session.exec(select(AuditEvent).where(AuditEvent.object_id == item.id).order_by(AuditEvent.created_at.desc())).first()
    assert event is not None
    assert event.actor_display_name_snapshot == "Before Rename"
    assert event.before_state["display_name"] == "Before"
    assert event.after_state["display_name"] == "After"
    assert event.changed_fields["display_name"] == {"before": "Before", "after": "After"}

    redacted = record_audit_event(
        resource_session,
        actor=AuditActor("system", key="test"),
        action="updated",
        object=AuditObject("resource_instances", item.id, item.instance_key),
        after_state={"safe": "ok", "password": "not-persisted", "nested": {"API_KEY": "also-not-persisted"}},
        metadata={"token": "not-persisted"},
    )
    resource_session.flush()
    assert redacted.after_state == {"safe": "ok", "password": "[REDACTED]", "nested": {"API_KEY": "[REDACTED]"}}
    assert redacted.extra_data == {"token": "[REDACTED]"}

    resource_session.rollback()
    assert resource_session.exec(select(AuditEvent)).all() == []


def test_new_configuration_rejects_secrets_and_embedded_endpoint(resource_session: Session):
    version, _ = _catalog_ids(resource_session)
    with pytest.raises(SensitiveConfigurationError):
        ResourceInstanceService(resource_session).create(
            component_version_id=version.id,
            instance_key=f"bad-{uuid4().hex}",
            display_name="Bad",
            runtime_kind="api",
            configuration={"password": "secret"},
        )
    with pytest.raises(SensitiveConfigurationError):
        ConnectionService(resource_session).create(
            connection_key=f"bad-{uuid4().hex}",
            display_name="Bad",
            connection_kind="api",
            endpoint="https://user:password@example.test",
        )
