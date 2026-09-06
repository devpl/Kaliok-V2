"""allow multi-document processing runs

Revision ID: e9f3a7b1c2d4
Revises: c4e8a12d9f33
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "e9f3a7b1c2d4"
down_revision: str | None = "9c4e2b7a1d60"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.alter_column(
        "processing_runs",
        "document_version_id",
        existing_type=sa.Uuid(),
        nullable=True,
    )


def downgrade() -> None:
    bind = op.get_bind()
    has_null = bind.execute(
        sa.text("SELECT 1 FROM processing_runs WHERE document_version_id IS NULL LIMIT 1")
    ).first()
    if has_null is not None:
        raise RuntimeError(
            "Impossible de rendre document_version_id NOT NULL : des runs sans document existent."
        )
    op.alter_column(
        "processing_runs",
        "document_version_id",
        existing_type=sa.Uuid(),
        nullable=False,
    )
