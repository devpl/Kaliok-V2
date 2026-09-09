"""Add resource instances, reusable connections, credential references and audit."""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "a2b3c4d5e6f7"
down_revision: str | None = "f7c8d9e0a1b2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

JSONB = postgresql.JSONB()
NULLABLE_JSONB = postgresql.JSONB(none_as_null=True)


def _ts(*, updated: bool = True):
    columns = [sa.Column("created_at", sa.DateTime(timezone=True), nullable=False)]
    if updated:
        columns.append(sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False))
    return columns


def upgrade() -> None:
    op.create_table(
        "connections",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("connection_key", sa.String(), nullable=False),
        sa.Column("display_name", sa.String(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("connection_kind", sa.String(), nullable=False),
        sa.Column("endpoint", sa.String(), nullable=True),
        sa.Column("configuration", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("status", sa.String(), nullable=False, server_default=sa.text("'unknown'")),
        sa.Column("health_status", sa.String(), nullable=False, server_default=sa.text("'unknown'")),
        sa.Column("health_checked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("health_detail", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("metadata", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        *_ts(),
        sa.CheckConstraint("btrim(connection_key) != ''", name="ck_connections_key_nonempty"),
        sa.CheckConstraint("btrim(display_name) != ''", name="ck_connections_display_name_nonempty"),
        sa.CheckConstraint("btrim(connection_kind) != ''", name="ck_connections_kind_nonempty"),
        sa.CheckConstraint("jsonb_typeof(configuration) = 'object'", name="ck_connections_configuration_object"),
        sa.CheckConstraint("jsonb_typeof(health_detail) = 'object'", name="ck_connections_health_detail_object"),
        sa.CheckConstraint("jsonb_typeof(metadata) = 'object'", name="ck_connections_metadata_object"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("connection_key", name="uq_connections_key"),
    )
    op.create_index("ix_connections_connection_key", "connections", ["connection_key"])

    op.create_table(
        "credential_references",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("credential_key", sa.String(), nullable=False),
        sa.Column("display_name", sa.String(), nullable=False),
        sa.Column("credential_type", sa.String(), nullable=False),
        sa.Column("backend", sa.String(), nullable=False),
        sa.Column("secret_locator", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False, server_default=sa.text("'unknown'")),
        sa.Column("metadata", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        *_ts(),
        sa.CheckConstraint("btrim(credential_key) != ''", name="ck_credential_references_key_nonempty"),
        sa.CheckConstraint("btrim(display_name) != ''", name="ck_credential_references_display_name_nonempty"),
        sa.CheckConstraint("btrim(credential_type) != ''", name="ck_credential_references_type_nonempty"),
        sa.CheckConstraint("btrim(backend) != ''", name="ck_credential_references_backend_nonempty"),
        sa.CheckConstraint("btrim(secret_locator) != ''", name="ck_credential_references_locator_nonempty"),
        sa.CheckConstraint("jsonb_typeof(metadata) = 'object'", name="ck_credential_references_metadata_object"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("credential_key", name="uq_credential_references_key"),
    )
    op.create_index("ix_credential_references_credential_key", "credential_references", ["credential_key"])

    op.create_table(
        "connection_credentials",
        sa.Column("connection_id", sa.Uuid(), nullable=False),
        sa.Column("credential_reference_id", sa.Uuid(), nullable=False),
        sa.Column("role_key", sa.String(), nullable=False, server_default=sa.text("'default'")),
        sa.Column("required", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("metadata", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        *_ts(),
        sa.CheckConstraint("btrim(role_key) != ''", name="ck_connection_credentials_role_nonempty"),
        sa.CheckConstraint("jsonb_typeof(metadata) = 'object'", name="ck_connection_credentials_metadata_object"),
        sa.ForeignKeyConstraint(["connection_id"], ["connections.id"]),
        sa.ForeignKeyConstraint(["credential_reference_id"], ["credential_references.id"]),
        sa.PrimaryKeyConstraint("connection_id", "credential_reference_id", "role_key"),
    )
    op.create_index("ix_connection_credentials_credential_id", "connection_credentials", ["credential_reference_id"])

    op.create_table(
        "resource_instances",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("component_version_id", sa.Uuid(), nullable=False),
        sa.Column("instance_key", sa.String(), nullable=False),
        sa.Column("display_name", sa.String(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("runtime_kind", sa.String(), nullable=False),
        sa.Column("configuration", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("status", sa.String(), nullable=False, server_default=sa.text("'unknown'")),
        sa.Column("health_status", sa.String(), nullable=False, server_default=sa.text("'unknown'")),
        sa.Column("health_checked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("health_detail", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("metadata", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        *_ts(),
        sa.CheckConstraint("btrim(instance_key) != ''", name="ck_resource_instances_key_nonempty"),
        sa.CheckConstraint("btrim(display_name) != ''", name="ck_resource_instances_display_name_nonempty"),
        sa.CheckConstraint("btrim(runtime_kind) != ''", name="ck_resource_instances_runtime_kind_nonempty"),
        sa.CheckConstraint("jsonb_typeof(configuration) = 'object'", name="ck_resource_instances_configuration_object"),
        sa.CheckConstraint("jsonb_typeof(health_detail) = 'object'", name="ck_resource_instances_health_detail_object"),
        sa.CheckConstraint("jsonb_typeof(metadata) = 'object'", name="ck_resource_instances_metadata_object"),
        sa.ForeignKeyConstraint(["component_version_id"], ["component_versions.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("instance_key", name="uq_resource_instances_key"),
    )
    op.create_index("ix_resource_instances_component_version_id", "resource_instances", ["component_version_id"])
    op.create_index("ix_resource_instances_instance_key", "resource_instances", ["instance_key"])

    op.create_table(
        "resource_instance_connections",
        sa.Column("resource_instance_id", sa.Uuid(), nullable=False),
        sa.Column("connection_id", sa.Uuid(), nullable=False),
        sa.Column("role_key", sa.String(), nullable=False, server_default=sa.text("'default'")),
        sa.Column("required", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("configuration", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("metadata", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        *_ts(),
        sa.CheckConstraint("btrim(role_key) != ''", name="ck_resource_instance_connections_role_nonempty"),
        sa.CheckConstraint("jsonb_typeof(configuration) = 'object'", name="ck_resource_instance_connections_configuration_object"),
        sa.CheckConstraint("jsonb_typeof(metadata) = 'object'", name="ck_resource_instance_connections_metadata_object"),
        sa.ForeignKeyConstraint(["resource_instance_id"], ["resource_instances.id"]),
        sa.ForeignKeyConstraint(["connection_id"], ["connections.id"]),
        sa.PrimaryKeyConstraint("resource_instance_id", "connection_id", "role_key"),
    )
    op.create_index("ix_resource_instance_connections_connection_id", "resource_instance_connections", ["connection_id"])

    op.create_table(
        "resource_instance_credentials",
        sa.Column("resource_instance_id", sa.Uuid(), nullable=False),
        sa.Column("credential_reference_id", sa.Uuid(), nullable=False),
        sa.Column("role_key", sa.String(), nullable=False, server_default=sa.text("'default'")),
        sa.Column("required", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("metadata", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        *_ts(),
        sa.CheckConstraint("btrim(role_key) != ''", name="ck_resource_instance_credentials_role_nonempty"),
        sa.CheckConstraint("jsonb_typeof(metadata) = 'object'", name="ck_resource_instance_credentials_metadata_object"),
        sa.ForeignKeyConstraint(["resource_instance_id"], ["resource_instances.id"]),
        sa.ForeignKeyConstraint(["credential_reference_id"], ["credential_references.id"]),
        sa.PrimaryKeyConstraint("resource_instance_id", "credential_reference_id", "role_key"),
    )
    op.create_index("ix_resource_instance_credentials_credential_id", "resource_instance_credentials", ["credential_reference_id"])

    op.create_table(
        "resource_instance_capabilities",
        sa.Column("resource_instance_id", sa.Uuid(), nullable=False),
        sa.Column("component_capability_id", sa.Uuid(), nullable=False),
        sa.Column("availability_status", sa.String(), nullable=False, server_default=sa.text("'unknown'")),
        sa.Column("configuration", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("metadata", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        *_ts(),
        sa.CheckConstraint("jsonb_typeof(configuration) = 'object'", name="ck_resource_instance_capabilities_configuration_object"),
        sa.CheckConstraint("jsonb_typeof(metadata) = 'object'", name="ck_resource_instance_capabilities_metadata_object"),
        sa.ForeignKeyConstraint(["resource_instance_id"], ["resource_instances.id"]),
        sa.ForeignKeyConstraint(["component_capability_id"], ["component_capabilities.id"]),
        sa.PrimaryKeyConstraint("resource_instance_id", "component_capability_id"),
    )
    op.create_index("ix_resource_instance_capabilities_component_capability_id", "resource_instance_capabilities", ["component_capability_id"])

    op.create_table(
        "audit_events",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("actor_type", sa.String(), nullable=False),
        sa.Column("actor_user_id", sa.Uuid(), nullable=True),
        sa.Column("actor_key", sa.String(), nullable=True),
        sa.Column("actor_display_name_snapshot", sa.String(), nullable=True),
        sa.Column("actor_identifier_snapshot", sa.String(), nullable=True),
        sa.Column("action", sa.String(), nullable=False),
        sa.Column("object_type", sa.String(), nullable=False),
        sa.Column("object_id", sa.Uuid(), nullable=True),
        sa.Column("object_key", sa.String(), nullable=True),
        sa.Column("object_revision_id", sa.Uuid(), nullable=True),
        sa.Column("before_state", NULLABLE_JSONB, nullable=True),
        sa.Column("after_state", NULLABLE_JSONB, nullable=True),
        sa.Column("changed_fields", NULLABLE_JSONB, nullable=True),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("request_id", sa.Uuid(), nullable=True),
        sa.Column("execution_group_id", sa.Uuid(), nullable=True),
        sa.Column("metadata", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("btrim(actor_type) != ''", name="ck_audit_events_actor_type_nonempty"),
        sa.CheckConstraint("btrim(action) != ''", name="ck_audit_events_action_nonempty"),
        sa.CheckConstraint("btrim(object_type) != ''", name="ck_audit_events_object_type_nonempty"),
        sa.CheckConstraint("before_state IS NULL OR jsonb_typeof(before_state) = 'object'", name="ck_audit_events_before_state_object"),
        sa.CheckConstraint("after_state IS NULL OR jsonb_typeof(after_state) = 'object'", name="ck_audit_events_after_state_object"),
        sa.CheckConstraint("changed_fields IS NULL OR jsonb_typeof(changed_fields) = 'object'", name="ck_audit_events_changed_fields_object"),
        sa.CheckConstraint("jsonb_typeof(metadata) = 'object'", name="ck_audit_events_metadata_object"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_audit_events_occurred_at", "audit_events", ["occurred_at"])
    op.create_index("ix_audit_events_object", "audit_events", ["object_type", "object_id"])
    op.create_index("ix_audit_events_request_id", "audit_events", ["request_id"])

    op.add_column("pipeline_bindings", sa.Column("resource_instance_id", sa.Uuid(), nullable=True))
    op.create_foreign_key("fk_pipeline_bindings_resource_instance_id", "pipeline_bindings", "resource_instances", ["resource_instance_id"], ["id"])
    op.create_index("ix_pipeline_bindings_resource_instance_id", "pipeline_bindings", ["resource_instance_id"])


def downgrade() -> None:
    op.drop_index("ix_pipeline_bindings_resource_instance_id", table_name="pipeline_bindings")
    op.drop_constraint("fk_pipeline_bindings_resource_instance_id", "pipeline_bindings", type_="foreignkey")
    op.drop_column("pipeline_bindings", "resource_instance_id")
    op.drop_index("ix_audit_events_request_id", table_name="audit_events")
    op.drop_index("ix_audit_events_object", table_name="audit_events")
    op.drop_index("ix_audit_events_occurred_at", table_name="audit_events")
    op.drop_table("audit_events")
    op.drop_index("ix_resource_instance_capabilities_component_capability_id", table_name="resource_instance_capabilities")
    op.drop_table("resource_instance_capabilities")
    op.drop_index("ix_resource_instance_credentials_credential_id", table_name="resource_instance_credentials")
    op.drop_table("resource_instance_credentials")
    op.drop_index("ix_resource_instance_connections_connection_id", table_name="resource_instance_connections")
    op.drop_table("resource_instance_connections")
    op.drop_index("ix_resource_instances_instance_key", table_name="resource_instances")
    op.drop_index("ix_resource_instances_component_version_id", table_name="resource_instances")
    op.drop_table("resource_instances")
    op.drop_index("ix_connection_credentials_credential_id", table_name="connection_credentials")
    op.drop_table("connection_credentials")
    op.drop_index("ix_credential_references_credential_key", table_name="credential_references")
    op.drop_table("credential_references")
    op.drop_index("ix_connections_connection_key", table_name="connections")
    op.drop_table("connections")
