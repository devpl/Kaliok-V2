"""add content block fragments

Revision ID: d4f7a9c2e681
Revises: b71e3c9f2a10
Create Date: 2026-09-04
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "d4f7a9c2e681"
down_revision: str | None = "b71e3c9f2a10"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "content_block_fragments",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("content_block_id", sa.Uuid(), nullable=False),
        sa.Column("page_id", sa.Uuid(), nullable=False),
        sa.Column("fragment_index", sa.Integer(), nullable=False),
        sa.Column("reading_order", sa.Integer(), nullable=True),
        sa.Column("content", sa.String(), nullable=False),
        sa.Column("bbox_x", sa.Float(), nullable=True),
        sa.Column("bbox_y", sa.Float(), nullable=True),
        sa.Column("bbox_width", sa.Float(), nullable=True),
        sa.Column("bbox_height", sa.Float(), nullable=True),
        sa.Column("coordinate_system", sa.String(), nullable=True),
        sa.Column("metadata", postgresql.JSONB(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["content_block_id"],
            ["content_blocks.id"],
        ),
        sa.ForeignKeyConstraint(["page_id"], ["pages.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "content_block_id",
            "fragment_index",
            name="uq_content_block_fragments_block_index",
        ),
    )
    op.create_index(
        "ix_content_block_fragments_content_block_id",
        "content_block_fragments",
        ["content_block_id"],
    )
    op.create_index(
        "ix_content_block_fragments_page_id",
        "content_block_fragments",
        ["page_id"],
    )

    op.execute(
        """
        INSERT INTO content_block_fragments (
            id,
            content_block_id,
            page_id,
            fragment_index,
            reading_order,
            content,
            bbox_x,
            bbox_y,
            bbox_width,
            bbox_height,
            coordinate_system,
            metadata,
            created_at
        )
        SELECT
            gen_random_uuid(),
            id,
            page_id,
            0,
            reading_order,
            content,
            bbox_x,
            bbox_y,
            bbox_width,
            bbox_height,
            coordinate_system,
            '{}'::jsonb,
            created_at
        FROM content_blocks
        """
    )


def downgrade() -> None:
    op.drop_index(
        "ix_content_block_fragments_page_id",
        table_name="content_block_fragments",
    )
    op.drop_index(
        "ix_content_block_fragments_content_block_id",
        table_name="content_block_fragments",
    )
    op.drop_table("content_block_fragments")
