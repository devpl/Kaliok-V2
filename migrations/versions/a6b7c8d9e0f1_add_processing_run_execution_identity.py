"""add processing run execution identity

Revision ID: a6b7c8d9e0f1
Revises: f1a2b3c4d5e6
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "a6b7c8d9e0f1"
down_revision: str | None = "f1a2b3c4d5e6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "processing_runs",
        sa.Column("execution_environment", sa.String(), nullable=True),
    )
    op.add_column(
        "processing_runs",
        sa.Column("configuration_revision_id", sa.Uuid(), nullable=True),
    )
    op.add_column(
        "processing_runs",
        sa.Column("execution_group_id", sa.Uuid(), nullable=True),
    )
    op.add_column(
        "processing_runs",
        sa.Column("configuration_hash", sa.String(), nullable=True),
    )
    op.create_foreign_key(
        "fk_processing_runs_configuration_revision_id",
        "processing_runs",
        "configuration_profile_revisions",
        ["configuration_revision_id"],
        ["id"],
    )
    op.create_index(
        "ix_processing_runs_configuration_revision_id",
        "processing_runs",
        ["configuration_revision_id"],
    )
    op.create_index(
        "ix_processing_runs_execution_group_id",
        "processing_runs",
        ["execution_group_id"],
    )
    op.create_check_constraint(
        "ck_processing_runs_execution_environment",
        "processing_runs",
        "execution_environment IS NULL OR execution_environment IN "
        "('production', 'experiment')",
    )


def downgrade() -> None:
    op.drop_constraint(
        "ck_processing_runs_execution_environment",
        "processing_runs",
        type_="check",
    )
    op.drop_index(
        "ix_processing_runs_execution_group_id",
        table_name="processing_runs",
    )
    op.drop_index(
        "ix_processing_runs_configuration_revision_id",
        table_name="processing_runs",
    )
    op.drop_constraint(
        "fk_processing_runs_configuration_revision_id",
        "processing_runs",
        type_="foreignkey",
    )
    op.drop_column("processing_runs", "configuration_hash")
    op.drop_column("processing_runs", "execution_group_id")
    op.drop_column("processing_runs", "configuration_revision_id")
    op.drop_column("processing_runs", "execution_environment")
