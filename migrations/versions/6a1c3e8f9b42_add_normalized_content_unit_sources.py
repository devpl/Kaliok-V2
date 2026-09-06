"""add normalized content unit sources

Revision ID: 6a1c3e8f9b42
Revises: d4f7a9c2e681
Create Date: 2026-09-04
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "6a1c3e8f9b42"
down_revision: str | None = "d4f7a9c2e681"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "normalized_content_units",
        sa.Column("processing_run_id", sa.Uuid(), nullable=True),
    )
    op.create_foreign_key(
        "fk_normalized_content_units_processing_run_id",
        "normalized_content_units",
        "processing_runs",
        ["processing_run_id"],
        ["id"],
    )
    op.create_index(
        "ix_normalized_content_units_processing_run_id",
        "normalized_content_units",
        ["processing_run_id"],
    )

    op.create_table(
        "normalized_content_unit_sources",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("normalized_content_unit_id", sa.Uuid(), nullable=False),
        sa.Column("content_block_id", sa.Uuid(), nullable=False),
        sa.Column("source_order", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.CheckConstraint(
            "source_order >= 0",
            name="ck_normalized_content_unit_sources_order_nonnegative",
        ),
        sa.ForeignKeyConstraint(
            ["normalized_content_unit_id"],
            ["normalized_content_units.id"],
        ),
        sa.ForeignKeyConstraint(
            ["content_block_id"],
            ["content_blocks.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "normalized_content_unit_id",
            "content_block_id",
            name="uq_normalized_content_unit_sources_unit_block",
        ),
        sa.UniqueConstraint(
            "normalized_content_unit_id",
            "source_order",
            name="uq_normalized_content_unit_sources_unit_order",
        ),
    )
    op.create_index(
        "ix_normalized_content_unit_sources_content_block_id",
        "normalized_content_unit_sources",
        ["content_block_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_normalized_content_unit_sources_content_block_id",
        table_name="normalized_content_unit_sources",
    )
    op.drop_table("normalized_content_unit_sources")

    op.drop_index(
        "ix_normalized_content_units_processing_run_id",
        table_name="normalized_content_units",
    )
    op.drop_constraint(
        "fk_normalized_content_units_processing_run_id",
        "normalized_content_units",
        type_="foreignkey",
    )
    op.drop_column("normalized_content_units", "processing_run_id")
