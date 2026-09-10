"""Add descriptive pipeline binding-to-template-node composition.

Migration 2B is deliberately additive.  The legacy capability projection and
runtime remain untouched; the new table records which binding is a candidate
or the selected provider for each descriptive template node.
"""

from collections.abc import Sequence

from alembic import context, op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "b2c3d4e5f6a7"
down_revision: str | None = "a2b3c4d5e6f7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

JSONB = postgresql.JSONB()


def upgrade() -> None:
    # Composite parent keys make every cross-revision relationship explicit.
    op.create_unique_constraint(
        "uq_pipeline_bindings_id_revision",
        "pipeline_bindings",
        ["id", "pipeline_revision_id"],
    )
    op.create_unique_constraint(
        "uq_pipeline_revisions_id_template_revision",
        "pipeline_revisions",
        ["id", "rag_template_revision_id"],
    )
    op.create_unique_constraint(
        "uq_resource_instances_id_component_version",
        "resource_instances",
        ["id", "component_version_id"],
    )
    op.create_foreign_key(
        "fk_pipeline_bindings_resource_instance_component_version",
        "pipeline_bindings",
        "resource_instances",
        ["resource_instance_id", "component_version_id"],
        ["id", "component_version_id"],
    )

    op.create_table(
        "pipeline_binding_nodes",
        sa.Column("pipeline_revision_id", sa.Uuid(), nullable=False),
        sa.Column("pipeline_binding_id", sa.Uuid(), nullable=False),
        sa.Column("rag_template_revision_id", sa.Uuid(), nullable=False),
        sa.Column("rag_template_node_id", sa.Uuid(), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("is_selected", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("priority", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("configuration", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "priority >= 0",
            name="ck_pipeline_binding_nodes_priority_nonnegative",
        ),
        sa.CheckConstraint(
            "jsonb_typeof(configuration) = 'object'",
            name="ck_pipeline_binding_nodes_configuration_object",
        ),
        sa.CheckConstraint(
            "NOT is_selected OR enabled",
            name="ck_pipeline_binding_nodes_selected_enabled",
        ),
        sa.ForeignKeyConstraint(
            ["pipeline_binding_id", "pipeline_revision_id"],
            ["pipeline_bindings.id", "pipeline_bindings.pipeline_revision_id"],
            name="fk_pipeline_binding_nodes_binding_revision",
        ),
        sa.ForeignKeyConstraint(
            ["rag_template_node_id", "rag_template_revision_id"],
            ["rag_template_nodes.id", "rag_template_nodes.rag_template_revision_id"],
            name="fk_pipeline_binding_nodes_node_template_revision",
        ),
        sa.ForeignKeyConstraint(
            ["pipeline_revision_id", "rag_template_revision_id"],
            ["pipeline_revisions.id", "pipeline_revisions.rag_template_revision_id"],
            name="fk_pipeline_binding_nodes_pipeline_template_revision",
        ),
        sa.PrimaryKeyConstraint(
            "pipeline_binding_id",
            "rag_template_node_id",
            name="pk_pipeline_binding_nodes",
        ),
    )
    op.create_index(
        "ix_pipeline_binding_nodes_binding_revision",
        "pipeline_binding_nodes",
        ["pipeline_binding_id", "pipeline_revision_id"],
    )
    op.create_index(
        "ix_pipeline_binding_nodes_node_template_revision",
        "pipeline_binding_nodes",
        ["rag_template_node_id", "rag_template_revision_id"],
    )
    op.create_index(
        "ix_pipeline_binding_nodes_revision_node",
        "pipeline_binding_nodes",
        ["pipeline_revision_id", "rag_template_node_id"],
    )
    op.create_index(
        "uq_pipeline_binding_nodes_selected_node",
        "pipeline_binding_nodes",
        ["pipeline_revision_id", "rag_template_node_id"],
        unique=True,
        postgresql_where=sa.text("is_selected IS TRUE"),
    )

    # Only template-backed, enabled legacy capability mappings are projected.
    # A missing or ambiguous node is an architecture error, so the migration
    # aborts atomically instead of inventing a node or silently dropping data.
    connection = op.get_bind()
    if not context.is_offline_mode():
        connection.execute(
            sa.text(
                """
                DO $$
                DECLARE
                    item RECORD;
                    matching_nodes INTEGER;
                BEGIN
                    FOR item IN
                        SELECT
                            pb.pipeline_revision_id,
                            pb.id AS pipeline_binding_id,
                            pr.rag_template_revision_id,
                            c.capability_id
                        FROM pipeline_binding_capabilities pbc
                        JOIN pipeline_bindings pb
                          ON pb.id = pbc.pipeline_binding_id
                        JOIN pipeline_revisions pr
                          ON pr.id = pb.pipeline_revision_id
                        JOIN component_capabilities c
                          ON c.id = pbc.component_capability_id
                        WHERE pbc.enabled IS TRUE
                          AND pr.rag_template_revision_id IS NOT NULL
                    LOOP
                        SELECT count(*)
                          INTO matching_nodes
                          FROM rag_template_nodes n
                         WHERE n.rag_template_revision_id = item.rag_template_revision_id
                           AND n.capability_id = item.capability_id;

                        IF matching_nodes <> 1 THEN
                            RAISE EXCEPTION
                                'Migration 2B: mapping capability % for binding % has % matching template nodes',
                                item.capability_id, item.pipeline_binding_id, matching_nodes;
                        END IF;

                        INSERT INTO pipeline_binding_nodes (
                            pipeline_revision_id,
                            pipeline_binding_id,
                            rag_template_revision_id,
                            rag_template_node_id,
                            enabled,
                            is_selected,
                            priority,
                            configuration,
                            created_at,
                            updated_at
                        )
                        SELECT
                            item.pipeline_revision_id,
                            item.pipeline_binding_id,
                            item.rag_template_revision_id,
                            n.id,
                            TRUE,
                            TRUE,
                            0,
                            '{}'::jsonb,
                            CURRENT_TIMESTAMP,
                            CURRENT_TIMESTAMP
                        FROM rag_template_nodes n
                        WHERE n.rag_template_revision_id = item.rag_template_revision_id
                          AND n.capability_id = item.capability_id;
                    END LOOP;
                END
                $$;
                """
            )
        )
def downgrade() -> None:
    op.drop_index(
        "uq_pipeline_binding_nodes_selected_node",
        table_name="pipeline_binding_nodes",
    )
    op.drop_index(
        "ix_pipeline_binding_nodes_revision_node",
        table_name="pipeline_binding_nodes",
    )
    op.drop_index(
        "ix_pipeline_binding_nodes_node_template_revision",
        table_name="pipeline_binding_nodes",
    )
    op.drop_index(
        "ix_pipeline_binding_nodes_binding_revision",
        table_name="pipeline_binding_nodes",
    )
    op.drop_table("pipeline_binding_nodes")
    op.drop_constraint(
        "fk_pipeline_bindings_resource_instance_component_version",
        "pipeline_bindings",
        type_="foreignkey",
    )
    op.drop_constraint(
        "uq_resource_instances_id_component_version",
        "resource_instances",
        type_="unique",
    )
    op.drop_constraint(
        "uq_pipeline_revisions_id_template_revision",
        "pipeline_revisions",
        type_="unique",
    )
    op.drop_constraint(
        "uq_pipeline_bindings_id_revision",
        "pipeline_bindings",
        type_="unique",
    )
