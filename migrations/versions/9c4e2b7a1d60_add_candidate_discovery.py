"""add candidate discovery

Revision ID: 9c4e2b7a1d60
Revises: 8f2d1c4b7a90
Create Date: 2026-09-04
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "9c4e2b7a1d60"
down_revision: str | None = "8f2d1c4b7a90"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "discovered_candidates",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("document_version_id", sa.Uuid(), nullable=False),
        sa.Column("processing_run_id", sa.Uuid(), nullable=False),
        sa.Column("candidate_type", sa.String(), nullable=False),
        sa.Column("raw_value", sa.String(), nullable=False),
        sa.Column("normalized_value", sa.String(), nullable=True),
        sa.Column(
            "payload",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
        ),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("detector_key", sa.String(), nullable=False),
        sa.Column("detector_version", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["document_version_id"], ["document_versions.id"]),
        sa.ForeignKeyConstraint(["processing_run_id"], ["processing_runs.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_discovered_candidates_document_version_id",
        "discovered_candidates",
        ["document_version_id"],
    )
    op.create_index(
        "ix_discovered_candidates_processing_run_id",
        "discovered_candidates",
        ["processing_run_id"],
    )
    op.create_table(
        "candidate_source_fragments",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("candidate_id", sa.Uuid(), nullable=False),
        sa.Column("normalized_content_unit_id", sa.Uuid(), nullable=False),
        sa.Column("fragment_order", sa.Integer(), nullable=False),
        sa.Column("start_offset", sa.Integer(), nullable=True),
        sa.Column("end_offset", sa.Integer(), nullable=True),
        sa.Column("exact_text", sa.String(), nullable=False),
        sa.Column("role", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.CheckConstraint(
            "fragment_order >= 0",
            name="ck_candidate_source_fragments_order_nonnegative",
        ),
        sa.CheckConstraint(
            "start_offset IS NULL OR start_offset >= 0",
            name="ck_candidate_source_fragments_start_nonnegative",
        ),
        sa.CheckConstraint(
            "end_offset IS NULL OR start_offset IS NULL OR end_offset >= start_offset",
            name="ck_candidate_source_fragments_offsets_ordered",
        ),
        sa.ForeignKeyConstraint(["candidate_id"], ["discovered_candidates.id"]),
        sa.ForeignKeyConstraint(
            ["normalized_content_unit_id"],
            ["normalized_content_units.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "candidate_id",
            "fragment_order",
            name="uq_candidate_source_fragments_candidate_order",
        ),
    )
    op.create_index(
        "ix_candidate_source_fragments_candidate_id",
        "candidate_source_fragments",
        ["candidate_id"],
    )
    op.create_index(
        "ix_candidate_source_fragments_normalized_content_unit_id",
        "candidate_source_fragments",
        ["normalized_content_unit_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_candidate_source_fragments_normalized_content_unit_id",
        table_name="candidate_source_fragments",
    )
    op.drop_index(
        "ix_candidate_source_fragments_candidate_id",
        table_name="candidate_source_fragments",
    )
    op.drop_table("candidate_source_fragments")
    op.drop_index(
        "ix_discovered_candidates_processing_run_id",
        table_name="discovered_candidates",
    )
    op.drop_index(
        "ix_discovered_candidates_document_version_id",
        table_name="discovered_candidates",
    )
    op.drop_table("discovered_candidates")
