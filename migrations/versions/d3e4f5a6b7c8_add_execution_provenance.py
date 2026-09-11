"""Add durable execution, step and artifact provenance.

Revision ID: d3e4f5a6b7c8
Revises: b2c3d4e5f6a7

This migration is strictly additive and deliberately performs no historical
backfill. Existing ProcessingRun and artifact rows retain partial provenance.
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "d3e4f5a6b7c8"
down_revision: str | None = "b2c3d4e5f6a7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

JSONB = postgresql.JSONB()


def upgrade() -> None:
    op.create_unique_constraint(
        "uq_pipeline_binding_nodes_execution_reference",
        "pipeline_binding_nodes",
        [
            "pipeline_binding_id",
            "pipeline_revision_id",
            "rag_template_node_id",
            "rag_template_revision_id",
        ],
    )

    op.create_table(
        "executions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("scope", sa.String(), nullable=False),
        sa.Column("execution_mode", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False, server_default=sa.text("'pending'")),
        sa.Column("actor_type", sa.String(), nullable=False),
        sa.Column("actor_user_id", sa.Uuid(), nullable=True),
        sa.Column("configuration_revision_id", sa.Uuid(), nullable=True),
        sa.Column("pipeline_revision_id", sa.Uuid(), nullable=False),
        sa.Column("rag_template_revision_id", sa.Uuid(), nullable=False),
        sa.Column("requested_rag_template_node_id", sa.Uuid(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("metadata", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.CheckConstraint("scope IN ('production', 'lab', 'evaluation')", name="ck_executions_scope"),
        sa.CheckConstraint("execution_mode IN ('step', 'prerequisites', 'zone', 'pipeline')", name="ck_executions_mode"),
        sa.CheckConstraint("status IN ('pending', 'running', 'completed', 'failed', 'cancelled')", name="ck_executions_status"),
        sa.CheckConstraint("btrim(actor_type) != ''", name="ck_executions_actor_type_nonempty"),
        sa.CheckConstraint("jsonb_typeof(metadata) = 'object'", name="ck_executions_metadata_object"),
        sa.ForeignKeyConstraint(["configuration_revision_id"], ["configuration_profile_revisions.id"], name="fk_executions_configuration_revision"),
        sa.ForeignKeyConstraint(
            ["pipeline_revision_id", "rag_template_revision_id"],
            ["pipeline_revisions.id", "pipeline_revisions.rag_template_revision_id"],
            name="fk_executions_pipeline_template_revision",
        ),
        sa.ForeignKeyConstraint(
            ["requested_rag_template_node_id", "rag_template_revision_id"],
            ["rag_template_nodes.id", "rag_template_nodes.rag_template_revision_id"],
            name="fk_executions_requested_node_template_revision",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("id", "pipeline_revision_id", "rag_template_revision_id", name="uq_executions_id_pipeline_template_revision"),
    )
    op.create_index("ix_executions_scope", "executions", ["scope"])
    op.create_index("ix_executions_status", "executions", ["status"])
    op.create_index("ix_executions_pipeline_revision_id", "executions", ["pipeline_revision_id"])
    op.create_index("ix_executions_rag_template_revision_id", "executions", ["rag_template_revision_id"])
    op.create_index("ix_executions_created_at", "executions", ["created_at"])

    op.create_table(
        "execution_steps",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("execution_id", sa.Uuid(), nullable=False),
        sa.Column("sequence_no", sa.Integer(), nullable=False),
        sa.Column("pipeline_revision_id", sa.Uuid(), nullable=False),
        sa.Column("rag_template_revision_id", sa.Uuid(), nullable=False),
        sa.Column("rag_template_node_id", sa.Uuid(), nullable=False),
        sa.Column("pipeline_binding_id", sa.Uuid(), nullable=True),
        sa.Column("resource_instance_id", sa.Uuid(), nullable=True),
        sa.Column("status", sa.String(), nullable=False, server_default=sa.text("'pending'")),
        sa.Column("configuration", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.CheckConstraint("sequence_no >= 0", name="ck_execution_steps_sequence_nonnegative"),
        sa.CheckConstraint("status IN ('pending', 'running', 'completed', 'failed', 'skipped', 'cancelled')", name="ck_execution_steps_status"),
        sa.CheckConstraint("jsonb_typeof(configuration) = 'object'", name="ck_execution_steps_configuration_object"),
        sa.UniqueConstraint("execution_id", "sequence_no", name="uq_execution_steps_execution_sequence"),
        sa.ForeignKeyConstraint(
            ["execution_id", "pipeline_revision_id", "rag_template_revision_id"],
            ["executions.id", "executions.pipeline_revision_id", "executions.rag_template_revision_id"],
            name="fk_execution_steps_execution_revisions",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["rag_template_node_id", "rag_template_revision_id"],
            ["rag_template_nodes.id", "rag_template_nodes.rag_template_revision_id"],
            name="fk_execution_steps_node_template_revision",
        ),
        sa.ForeignKeyConstraint(
            ["pipeline_binding_id", "pipeline_revision_id", "rag_template_node_id", "rag_template_revision_id"],
            ["pipeline_binding_nodes.pipeline_binding_id", "pipeline_binding_nodes.pipeline_revision_id", "pipeline_binding_nodes.rag_template_node_id", "pipeline_binding_nodes.rag_template_revision_id"],
            name="fk_execution_steps_binding_node",
        ),
        sa.ForeignKeyConstraint(["resource_instance_id"], ["resource_instances.id"], name="fk_execution_steps_resource_instance"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_execution_steps_execution_id", "execution_steps", ["execution_id"])
    op.create_index("ix_execution_steps_rag_template_node_id", "execution_steps", ["rag_template_node_id"])
    op.create_index("ix_execution_steps_pipeline_binding_id", "execution_steps", ["pipeline_binding_id"])
    op.create_index("ix_execution_steps_status", "execution_steps", ["status"])

    op.add_column("processing_runs", sa.Column("execution_step_id", sa.Uuid(), nullable=True))
    op.create_foreign_key(
        "fk_processing_runs_execution_step_id",
        "processing_runs", "execution_steps",
        ["execution_step_id"], ["id"],
        ondelete="SET NULL",
    )
    op.create_index("ix_processing_runs_execution_step_id", "processing_runs", ["execution_step_id"])

    op.create_table(
        "execution_artifacts",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("execution_step_id", sa.Uuid(), nullable=False),
        sa.Column("role", sa.String(), nullable=False),
        sa.Column("artifact_type_id", sa.Uuid(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("metadata", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.CheckConstraint("role IN ('input', 'output')", name="ck_execution_artifacts_role"),
        sa.CheckConstraint("jsonb_typeof(metadata) = 'object'", name="ck_execution_artifacts_metadata_object"),
        sa.ForeignKeyConstraint(["execution_step_id"], ["execution_steps.id"], name="fk_execution_artifacts_step", ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["artifact_type_id"], ["artifact_types.id"], name="fk_execution_artifacts_type"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_execution_artifacts_execution_step_id", "execution_artifacts", ["execution_step_id"])
    op.create_index("ix_execution_artifacts_role", "execution_artifacts", ["role"])
    op.create_index("ix_execution_artifacts_artifact_type_id", "execution_artifacts", ["artifact_type_id"])

    links = (
        ("execution_artifact_content_blocks", "content_block_id", "content_blocks", "ix_execution_artifact_content_block_target"),
        ("execution_artifact_normalized_content_units", "normalized_content_unit_id", "normalized_content_units", "ix_execution_artifact_normalized_unit_target"),
        ("execution_artifact_discovered_candidates", "discovered_candidate_id", "discovered_candidates", "ix_execution_artifact_candidate_target"),
        ("execution_artifact_entities", "entity_id", "entities", "ix_execution_artifact_entity_target"),
    )
    for table_name, target_column, target_table, index_name in links:
        op.create_table(
            table_name,
            sa.Column("execution_artifact_id", sa.Uuid(), nullable=False),
            sa.Column(target_column, sa.Uuid(), nullable=False),
            sa.ForeignKeyConstraint(["execution_artifact_id"], ["execution_artifacts.id"], ondelete="CASCADE"),
            sa.ForeignKeyConstraint([target_column], [f"{target_table}.id"]),
            sa.PrimaryKeyConstraint("execution_artifact_id"),
        )
        op.create_index(index_name, table_name, [target_column])


def downgrade() -> None:
    links = (
        ("execution_artifact_entities", "ix_execution_artifact_entity_target"),
        ("execution_artifact_discovered_candidates", "ix_execution_artifact_candidate_target"),
        ("execution_artifact_normalized_content_units", "ix_execution_artifact_normalized_unit_target"),
        ("execution_artifact_content_blocks", "ix_execution_artifact_content_block_target"),
    )
    for table_name, index_name in links:
        op.drop_index(index_name, table_name=table_name)
        op.drop_table(table_name)
    op.drop_index("ix_execution_artifacts_artifact_type_id", table_name="execution_artifacts")
    op.drop_index("ix_execution_artifacts_role", table_name="execution_artifacts")
    op.drop_index("ix_execution_artifacts_execution_step_id", table_name="execution_artifacts")
    op.drop_table("execution_artifacts")
    op.drop_index("ix_processing_runs_execution_step_id", table_name="processing_runs")
    op.drop_constraint("fk_processing_runs_execution_step_id", "processing_runs", type_="foreignkey")
    op.drop_column("processing_runs", "execution_step_id")
    op.drop_index("ix_execution_steps_status", table_name="execution_steps")
    op.drop_index("ix_execution_steps_pipeline_binding_id", table_name="execution_steps")
    op.drop_index("ix_execution_steps_rag_template_node_id", table_name="execution_steps")
    op.drop_index("ix_execution_steps_execution_id", table_name="execution_steps")
    op.drop_table("execution_steps")
    op.drop_index("ix_executions_created_at", table_name="executions")
    op.drop_index("ix_executions_rag_template_revision_id", table_name="executions")
    op.drop_index("ix_executions_pipeline_revision_id", table_name="executions")
    op.drop_index("ix_executions_status", table_name="executions")
    op.drop_index("ix_executions_scope", table_name="executions")
    op.drop_table("executions")
    op.drop_constraint("uq_pipeline_binding_nodes_execution_reference", "pipeline_binding_nodes", type_="unique")
