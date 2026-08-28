"""add normalized content units

Revision ID: a84c7e2f19d1
Revises: 7f3a2c91d6e4
Create Date: 2026-08-28
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "a84c7e2f19d1"
down_revision: Union[str, Sequence[str], None] = "7f3a2c91d6e4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "normalized_content_units",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("document_version_id", sa.Uuid(), nullable=False),
        sa.Column("parent_unit_id", sa.Uuid(), nullable=True),
        sa.Column("unit_index", sa.Integer(), nullable=False),
        sa.Column("content_type", sa.String(), nullable=False),
        sa.Column("content", sa.String(), nullable=False),
        sa.Column("source_reference", sa.String(), nullable=True),
        sa.Column("source_unit_id", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.CheckConstraint(
            "unit_index >= 0",
            name="ck_normalized_content_units_index_nonnegative",
        ),
        sa.ForeignKeyConstraint(
            ["document_version_id"],
            ["document_versions.id"],
        ),
        sa.ForeignKeyConstraint(
            ["parent_unit_id"],
            ["normalized_content_units.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "document_version_id",
            "unit_index",
            name="uq_normalized_content_units_version_index",
        ),
        sa.UniqueConstraint(
            "document_version_id",
            "source_unit_id",
            name="uq_normalized_content_units_version_source_unit",
        ),
    )


def downgrade() -> None:
    op.drop_table("normalized_content_units")
