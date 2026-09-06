"""enforce single active configuration revision

Revision ID: c4e8a12d9f33
Revises: f2b1d9c7a6e0
Create Date: 2026-09-01
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "c4e8a12d9f33"
down_revision: str | None = "f2b1d9c7a6e0"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_index(
        "uq_configuration_profile_revisions_single_active",
        "configuration_profile_revisions",
        ["profile_id"],
        unique=True,
        postgresql_where=sa.text("status = 'active'"),
    )


def downgrade() -> None:
    op.drop_index(
        "uq_configuration_profile_revisions_single_active",
        table_name="configuration_profile_revisions",
    )