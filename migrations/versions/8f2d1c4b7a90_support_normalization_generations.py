"""support normalization generations

Revision ID: 8f2d1c4b7a90
Revises: 6a1c3e8f9b42
Create Date: 2026-09-04
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "8f2d1c4b7a90"
down_revision: str | None = "6a1c3e8f9b42"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_constraint(
        "uq_normalized_content_units_version_index",
        "normalized_content_units",
        type_="unique",
    )
    op.drop_constraint(
        "uq_normalized_content_units_version_source_unit",
        "normalized_content_units",
        type_="unique",
    )

    op.create_index(
        "uq_normalized_content_units_historical_version_index",
        "normalized_content_units",
        ["document_version_id", "unit_index"],
        unique=True,
        postgresql_where=sa.text("processing_run_id IS NULL"),
    )
    op.create_index(
        "uq_normalized_content_units_historical_version_source_unit",
        "normalized_content_units",
        ["document_version_id", "source_unit_id"],
        unique=True,
        postgresql_where=sa.text(
            "processing_run_id IS NULL AND source_unit_id IS NOT NULL"
        ),
    )
    op.create_index(
        "uq_normalized_content_units_run_version_index",
        "normalized_content_units",
        ["processing_run_id", "document_version_id", "unit_index"],
        unique=True,
        postgresql_where=sa.text("processing_run_id IS NOT NULL"),
    )
    op.create_index(
        "uq_normalized_content_units_run_version_source_unit",
        "normalized_content_units",
        ["processing_run_id", "document_version_id", "source_unit_id"],
        unique=True,
        postgresql_where=sa.text(
            "processing_run_id IS NOT NULL AND source_unit_id IS NOT NULL"
        ),
    )


def downgrade() -> None:
    op.drop_index(
        "uq_normalized_content_units_run_version_source_unit",
        table_name="normalized_content_units",
    )
    op.drop_index(
        "uq_normalized_content_units_run_version_index",
        table_name="normalized_content_units",
    )
    op.drop_index(
        "uq_normalized_content_units_historical_version_source_unit",
        table_name="normalized_content_units",
    )
    op.drop_index(
        "uq_normalized_content_units_historical_version_index",
        table_name="normalized_content_units",
    )

    op.create_unique_constraint(
        "uq_normalized_content_units_version_index",
        "normalized_content_units",
        ["document_version_id", "unit_index"],
    )
    op.create_unique_constraint(
        "uq_normalized_content_units_version_source_unit",
        "normalized_content_units",
        ["document_version_id", "source_unit_id"],
    )
