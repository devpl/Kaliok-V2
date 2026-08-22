"""add current perception run to pages

Revision ID: 7f3a2c91d6e4
Revises: 3b73d74e239f
Create Date: 2026-08-22
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "7f3a2c91d6e4"
down_revision: Union[str, Sequence[str], None] = "3b73d74e239f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "pages",
        sa.Column(
            "perception_processing_run_id",
            sa.Uuid(),
            nullable=True,
        ),
    )

    op.create_foreign_key(
        "fk_pages_perception_processing_run_id",
        "pages",
        "processing_runs",
        ["perception_processing_run_id"],
        ["id"],
    )


def downgrade() -> None:
    op.drop_constraint(
        "fk_pages_perception_processing_run_id",
        "pages",
        type_="foreignkey",
    )

    op.drop_column(
        "pages",
        "perception_processing_run_id",
    )
