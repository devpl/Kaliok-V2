"""add persistent catalogue and versioned pipelines

Revision ID: c7d8e9f0a1b2
Revises: a6b7c8d9e0f1

This migration only creates the descriptive schema.  Catalogue/bootstrap data
is deliberately supplied by an explicit application service afterwards.
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "c7d8e9f0a1b2"
down_revision: str | None = "a6b7c8d9e0f1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

JSONB = postgresql.JSONB()


def _timestamps():
    return (
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )


def upgrade() -> None:
    op.create_table(
        "capabilities",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("capability_key", sa.String(), nullable=False),
        sa.Column("display_name", sa.String(), nullable=False),
        sa.Column("description", sa.String(), nullable=True),
        sa.Column("phase_key", sa.String(), nullable=True),
        sa.Column("input_artifact_types", JSONB, nullable=False),
        sa.Column("output_artifact_types", JSONB, nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("display_order", sa.Integer(), nullable=False),
        *_timestamps(),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("capability_key", name="uq_capabilities_key"),
    )
    op.create_index("ix_capabilities_capability_key", "capabilities", ["capability_key"])

    op.create_table(
        "components",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("component_key", sa.String(), nullable=False),
        sa.Column("display_name", sa.String(), nullable=False),
        sa.Column("vendor", sa.String(), nullable=True),
        sa.Column("description", sa.String(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        *_timestamps(),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("component_key", name="uq_components_key"),
    )
    op.create_index("ix_components_component_key", "components", ["component_key"])

    op.create_table(
        "component_versions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("component_id", sa.Uuid(), nullable=False),
        sa.Column("version", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("configuration_schema", JSONB, nullable=False),
        sa.Column("metadata", JSONB, nullable=False),
        *_timestamps(),
        sa.CheckConstraint(
            "status IN ('available', 'deprecated', 'unavailable')",
            name="ck_component_versions_status",
        ),
        sa.ForeignKeyConstraint(["component_id"], ["components.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "component_id", "version",
            name="uq_component_versions_component_version",
        ),
    )
    op.create_index("ix_component_versions_component_id", "component_versions", ["component_id"])

    op.create_table(
        "component_capabilities",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("component_version_id", sa.Uuid(), nullable=False),
        sa.Column("capability_id", sa.Uuid(), nullable=False),
        sa.Column("invocation_mode", sa.String(), nullable=False),
        sa.Column("execution_bundle_key", sa.String(), nullable=True),
        sa.Column("configuration_schema", JSONB, nullable=False),
        sa.Column("metadata", JSONB, nullable=False),
        *_timestamps(),
        sa.CheckConstraint(
            "invocation_mode IN ('independent', 'all_or_none', 'produced_with_bundle')",
            name="ck_component_capabilities_invocation_mode",
        ),
        sa.ForeignKeyConstraint(["capability_id"], ["capabilities.id"]),
        sa.ForeignKeyConstraint(["component_version_id"], ["component_versions.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "component_version_id", "capability_id",
            name="uq_component_capabilities_version_capability",
        ),
    )
    op.create_index("ix_component_capabilities_version_id", "component_capabilities", ["component_version_id"])
    op.create_index("ix_component_capabilities_capability_id", "component_capabilities", ["capability_id"])

    op.create_table(
        "rag_templates",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("template_key", sa.String(), nullable=False),
        sa.Column("display_name", sa.String(), nullable=False),
        sa.Column("description", sa.String(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        *_timestamps(),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("template_key", name="uq_rag_templates_key"),
    )
    op.create_index("ix_rag_templates_template_key", "rag_templates", ["template_key"])

    op.create_table(
        "rag_template_revisions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("rag_template_id", sa.Uuid(), nullable=False),
        sa.Column("revision_number", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("change_reason", sa.String(), nullable=True),
        *_timestamps(),
        sa.Column("activated_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint("revision_number > 0", name="ck_rag_template_revisions_number_positive"),
        sa.CheckConstraint("status IN ('draft', 'active', 'retired')", name="ck_rag_template_revisions_status"),
        sa.ForeignKeyConstraint(["rag_template_id"], ["rag_templates.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("rag_template_id", "revision_number", name="uq_rag_template_revisions_number"),
    )
    op.create_index("ix_rag_template_revisions_template_id", "rag_template_revisions", ["rag_template_id"])
    op.create_index(
        "uq_rag_template_revisions_single_active",
        "rag_template_revisions",
        ["rag_template_id"],
        unique=True,
        postgresql_where=sa.text("status = 'active'"),
    )

    op.create_table(
        "rag_template_capabilities",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("rag_template_revision_id", sa.Uuid(), nullable=False),
        sa.Column("capability_id", sa.Uuid(), nullable=False),
        sa.Column("requirement_mode", sa.String(), nullable=False),
        sa.Column("display_order", sa.Integer(), nullable=False),
        sa.Column("configuration", JSONB, nullable=False),
        *_timestamps(),
        sa.CheckConstraint("requirement_mode IN ('required', 'optional')", name="ck_rag_template_capabilities_requirement_mode"),
        sa.ForeignKeyConstraint(["capability_id"], ["capabilities.id"]),
        sa.ForeignKeyConstraint(["rag_template_revision_id"], ["rag_template_revisions.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("rag_template_revision_id", "capability_id", name="uq_rag_template_capabilities_revision_capability"),
    )
    op.create_index("ix_rag_template_capabilities_revision_id", "rag_template_capabilities", ["rag_template_revision_id"])
    op.create_index("ix_rag_template_capabilities_capability_id", "rag_template_capabilities", ["capability_id"])

    op.create_table(
        "rag_template_dependencies",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("rag_template_revision_id", sa.Uuid(), nullable=False),
        sa.Column("source_capability_id", sa.Uuid(), nullable=False),
        sa.Column("target_capability_id", sa.Uuid(), nullable=False),
        *_timestamps(),
        sa.CheckConstraint("source_capability_id <> target_capability_id", name="ck_rag_template_dependencies_distinct"),
        sa.ForeignKeyConstraint(["rag_template_revision_id"], ["rag_template_revisions.id"]),
        sa.ForeignKeyConstraint(["source_capability_id"], ["capabilities.id"]),
        sa.ForeignKeyConstraint(["target_capability_id"], ["capabilities.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("rag_template_revision_id", "source_capability_id", "target_capability_id", name="uq_rag_template_dependencies_edge"),
    )
    op.create_index("ix_rag_template_dependencies_revision_id", "rag_template_dependencies", ["rag_template_revision_id"])

    op.create_table(
        "pipeline_definitions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("pipeline_key", sa.String(), nullable=False),
        sa.Column("display_name", sa.String(), nullable=False),
        sa.Column("description", sa.String(), nullable=True),
        *_timestamps(),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("pipeline_key", name="uq_pipeline_definitions_key"),
    )
    op.create_index("ix_pipeline_definitions_pipeline_key", "pipeline_definitions", ["pipeline_key"])

    op.create_table(
        "pipeline_revisions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("pipeline_definition_id", sa.Uuid(), nullable=False),
        sa.Column("rag_template_revision_id", sa.Uuid(), nullable=True),
        sa.Column("revision_number", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("manifest_hash", sa.String(), nullable=False),
        sa.Column("change_reason", sa.String(), nullable=True),
        *_timestamps(),
        sa.Column("activated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_by_display_name", sa.String(), nullable=True),
        sa.CheckConstraint("revision_number > 0", name="ck_pipeline_revisions_number_positive"),
        sa.CheckConstraint("status IN ('draft', 'active', 'retired')", name="ck_pipeline_revisions_status"),
        sa.ForeignKeyConstraint(["pipeline_definition_id"], ["pipeline_definitions.id"]),
        sa.ForeignKeyConstraint(["rag_template_revision_id"], ["rag_template_revisions.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("pipeline_definition_id", "revision_number", name="uq_pipeline_revisions_number"),
    )
    op.create_index("ix_pipeline_revisions_definition_id", "pipeline_revisions", ["pipeline_definition_id"])
    op.create_index("ix_pipeline_revisions_template_revision_id", "pipeline_revisions", ["rag_template_revision_id"])
    op.create_index(
        "uq_pipeline_revisions_single_active",
        "pipeline_revisions",
        ["pipeline_definition_id"],
        unique=True,
        postgresql_where=sa.text("status = 'active'"),
    )

    op.create_table(
        "pipeline_bindings",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("pipeline_revision_id", sa.Uuid(), nullable=False),
        sa.Column("binding_key", sa.String(), nullable=False),
        sa.Column("component_version_id", sa.Uuid(), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("configuration", JSONB, nullable=False),
        *_timestamps(),
        sa.ForeignKeyConstraint(["component_version_id"], ["component_versions.id"]),
        sa.ForeignKeyConstraint(["pipeline_revision_id"], ["pipeline_revisions.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("pipeline_revision_id", "binding_key", name="uq_pipeline_bindings_revision_key"),
        sa.UniqueConstraint("pipeline_revision_id", "position", name="uq_pipeline_bindings_revision_position"),
    )
    op.create_index("ix_pipeline_bindings_revision_id", "pipeline_bindings", ["pipeline_revision_id"])
    op.create_index("ix_pipeline_bindings_component_version_id", "pipeline_bindings", ["component_version_id"])

    op.create_table(
        "pipeline_binding_capabilities",
        sa.Column("pipeline_binding_id", sa.Uuid(), nullable=False),
        sa.Column("component_capability_id", sa.Uuid(), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("configuration", JSONB, nullable=False),
        *_timestamps(),
        sa.ForeignKeyConstraint(["component_capability_id"], ["component_capabilities.id"]),
        sa.ForeignKeyConstraint(["pipeline_binding_id"], ["pipeline_bindings.id"]),
        sa.PrimaryKeyConstraint("pipeline_binding_id", "component_capability_id"),
        sa.UniqueConstraint("pipeline_binding_id", "component_capability_id", name="uq_pipeline_binding_capabilities_binding_capability"),
    )

    op.create_table(
        "pipeline_binding_dependencies",
        sa.Column("pipeline_binding_id", sa.Uuid(), nullable=False),
        sa.Column("depends_on_binding_id", sa.Uuid(), nullable=False),
        *_timestamps(),
        sa.CheckConstraint("pipeline_binding_id <> depends_on_binding_id", name="ck_pipeline_binding_dependencies_distinct"),
        sa.ForeignKeyConstraint(["depends_on_binding_id"], ["pipeline_bindings.id"]),
        sa.ForeignKeyConstraint(["pipeline_binding_id"], ["pipeline_bindings.id"]),
        sa.PrimaryKeyConstraint("pipeline_binding_id", "depends_on_binding_id"),
        sa.UniqueConstraint("pipeline_binding_id", "depends_on_binding_id", name="uq_pipeline_binding_dependencies_edge"),
    )

    op.add_column(
        "processing_runs",
        sa.Column("pipeline_revision_id", sa.Uuid(), nullable=True),
    )
    op.create_foreign_key(
        "fk_processing_runs_pipeline_revision_id",
        "processing_runs",
        "pipeline_revisions",
        ["pipeline_revision_id"],
        ["id"],
    )
    op.create_index(
        "ix_processing_runs_pipeline_revision_id",
        "processing_runs",
        ["pipeline_revision_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_processing_runs_pipeline_revision_id", table_name="processing_runs")
    op.drop_constraint("fk_processing_runs_pipeline_revision_id", "processing_runs", type_="foreignkey")
    op.drop_column("processing_runs", "pipeline_revision_id")

    op.drop_table("pipeline_binding_dependencies")
    op.drop_table("pipeline_binding_capabilities")
    op.drop_index("ix_pipeline_bindings_component_version_id", table_name="pipeline_bindings")
    op.drop_index("ix_pipeline_bindings_revision_id", table_name="pipeline_bindings")
    op.drop_table("pipeline_bindings")
    op.drop_index("uq_pipeline_revisions_single_active", table_name="pipeline_revisions")
    op.drop_index("ix_pipeline_revisions_template_revision_id", table_name="pipeline_revisions")
    op.drop_index("ix_pipeline_revisions_definition_id", table_name="pipeline_revisions")
    op.drop_table("pipeline_revisions")
    op.drop_index("ix_pipeline_definitions_pipeline_key", table_name="pipeline_definitions")
    op.drop_table("pipeline_definitions")
    op.drop_index("ix_rag_template_dependencies_revision_id", table_name="rag_template_dependencies")
    op.drop_table("rag_template_dependencies")
    op.drop_index("ix_rag_template_capabilities_capability_id", table_name="rag_template_capabilities")
    op.drop_index("ix_rag_template_capabilities_revision_id", table_name="rag_template_capabilities")
    op.drop_table("rag_template_capabilities")
    op.drop_index("uq_rag_template_revisions_single_active", table_name="rag_template_revisions")
    op.drop_index("ix_rag_template_revisions_template_id", table_name="rag_template_revisions")
    op.drop_table("rag_template_revisions")
    op.drop_index("ix_rag_templates_template_key", table_name="rag_templates")
    op.drop_table("rag_templates")
    op.drop_index("ix_component_capabilities_capability_id", table_name="component_capabilities")
    op.drop_index("ix_component_capabilities_version_id", table_name="component_capabilities")
    op.drop_table("component_capabilities")
    op.drop_index("ix_component_versions_component_id", table_name="component_versions")
    op.drop_table("component_versions")
    op.drop_index("ix_components_component_key", table_name="components")
    op.drop_table("components")
    op.drop_index("ix_capabilities_capability_key", table_name="capabilities")
    op.drop_table("capabilities")
