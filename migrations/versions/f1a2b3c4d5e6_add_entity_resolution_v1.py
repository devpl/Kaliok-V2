"""add persistent entity resolution v1 tables

Revision ID: f1a2b3c4d5e6
Revises: e9f3a7b1c2d4
"""
from collections.abc import Sequence
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "f1a2b3c4d5e6"
down_revision = "e9f3a7b1c2d4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

def upgrade() -> None:
    op.create_table("entity_resolution_scope_items",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True), sa.Column("processing_run_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("processing_runs.id"), nullable=False), sa.Column("discovered_candidate_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("discovered_candidates.id"), nullable=False), sa.Column("scope_order", sa.Integer(), nullable=False), sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("processing_run_id", "discovered_candidate_id", name="uq_er_scope_run_candidate"), sa.UniqueConstraint("processing_run_id", "scope_order", name="uq_er_scope_run_order"), sa.CheckConstraint("scope_order >= 0", name="ck_er_scope_order_nonnegative"))
    op.create_index("ix_entity_resolution_scope_items_processing_run_id", "entity_resolution_scope_items", ["processing_run_id"])
    op.create_index("ix_entity_resolution_scope_items_discovered_candidate_id", "entity_resolution_scope_items", ["discovered_candidate_id"])
    op.create_table("entities",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True), sa.Column("processing_run_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("processing_runs.id"), nullable=False), sa.Column("entity_index", sa.Integer(), nullable=False), sa.Column("entity_type", sa.String(), nullable=False), sa.Column("canonical_label", sa.String(), nullable=False), sa.Column("status", sa.String(), nullable=False), sa.Column("confidence", sa.Float()), sa.Column("metadata", postgresql.JSONB(), nullable=False), sa.Column("created_at", sa.DateTime(timezone=True), nullable=False), sa.UniqueConstraint("processing_run_id", "entity_index", name="uq_entities_run_index"), sa.CheckConstraint("entity_index >= 0", name="ck_entities_index_nonnegative"))
    op.create_index("ix_entities_processing_run_id", "entities", ["processing_run_id"])
    op.create_table("entity_memberships",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True), sa.Column("entity_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("entities.id"), nullable=False), sa.Column("discovered_candidate_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("discovered_candidates.id"), nullable=False), sa.Column("processing_run_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("processing_runs.id"), nullable=False), sa.Column("membership_status", sa.String(), nullable=False), sa.Column("confidence", sa.Float()), sa.Column("decision_origin", sa.String(), nullable=False), sa.Column("metadata", postgresql.JSONB(), nullable=False), sa.Column("created_at", sa.DateTime(timezone=True), nullable=False), sa.UniqueConstraint("processing_run_id", "discovered_candidate_id", name="uq_entity_membership_run_candidate"))
    for name, col in (("entity_id", "entity_id"), ("discovered_candidate_id", "discovered_candidate_id"), ("processing_run_id", "processing_run_id")):
        op.create_index(f"ix_entity_memberships_{name}", "entity_memberships", [col])
    op.create_table("entity_resolution_evidence",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True), sa.Column("membership_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("entity_memberships.id"), nullable=False), sa.Column("evidence_order", sa.Integer(), nullable=False), sa.Column("signal_key", sa.String(), nullable=False), sa.Column("method", sa.String(), nullable=False), sa.Column("score", sa.Float()), sa.Column("explanation", sa.Text()), sa.Column("metadata", postgresql.JSONB(), nullable=False), sa.Column("created_at", sa.DateTime(timezone=True), nullable=False), sa.UniqueConstraint("membership_id", "evidence_order", name="uq_er_evidence_membership_order"), sa.CheckConstraint("evidence_order >= 0", name="ck_er_evidence_order_nonnegative"))
    op.create_index("ix_entity_resolution_evidence_membership_id", "entity_resolution_evidence", ["membership_id"])

def downgrade() -> None:
    op.drop_index("ix_entity_resolution_evidence_membership_id", table_name="entity_resolution_evidence")
    op.drop_table("entity_resolution_evidence")
    for name in ("entity_id", "discovered_candidate_id", "processing_run_id"):
        op.drop_index(f"ix_entity_memberships_{name}", table_name="entity_memberships")
    op.drop_table("entity_memberships")
    op.drop_index("ix_entities_processing_run_id", table_name="entities")
    op.drop_table("entities")
    op.drop_index("ix_entity_resolution_scope_items_discovered_candidate_id", table_name="entity_resolution_scope_items")
    op.drop_index("ix_entity_resolution_scope_items_processing_run_id", table_name="entity_resolution_scope_items")
    op.drop_table("entity_resolution_scope_items")
