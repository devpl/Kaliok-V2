"""add document perception fields

Revision ID: 3b73d74e239f
Revises: d9a9a5ad7ac9
Create Date: 2026-08-21 16:20:11.973479

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "3b73d74e239f"
down_revision: Union[str, Sequence[str], None] = "d9a9a5ad7ac9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add document perception fields without modifying existing objects."""

    op.add_column(
        "content_blocks",
        sa.Column(
            "processing_run_id",
            sa.Uuid(),
            nullable=True,
        ),
    )

    op.add_column(
        "content_blocks",
        sa.Column(
            "bbox_x",
            sa.Float(),
            nullable=True,
        ),
    )

    op.add_column(
        "content_blocks",
        sa.Column(
            "bbox_y",
            sa.Float(),
            nullable=True,
        ),
    )

    op.add_column(
        "content_blocks",
        sa.Column(
            "bbox_width",
            sa.Float(),
            nullable=True,
        ),
    )

    op.add_column(
        "content_blocks",
        sa.Column(
            "bbox_height",
            sa.Float(),
            nullable=True,
        ),
    )

    op.add_column(
        "content_blocks",
        sa.Column(
            "coordinate_system",
            sa.String(),
            nullable=True,
        ),
    )

    op.create_foreign_key(
        "fk_content_blocks_processing_run_id",
        "content_blocks",
        "processing_runs",
        ["processing_run_id"],
        ["id"],
    )

    op.add_column(
        "pages",
        sa.Column(
            "perception_mode",
            sa.String(),
            nullable=False,
            server_default="unknown",
        ),
    )

    op.add_column(
        "pages",
        sa.Column(
            "ocr_performed",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
    )

    op.add_column(
        "pages",
        sa.Column(
            "ocr_reason",
            sa.String(),
            nullable=True,
        ),
    )

    op.add_column(
        "pages",
        sa.Column(
            "ocr_processing_run_id",
            sa.Uuid(),
            nullable=True,
        ),
    )

    op.create_foreign_key(
        "fk_pages_ocr_processing_run_id",
        "pages",
        "processing_runs",
        ["ocr_processing_run_id"],
        ["id"],
    )


def downgrade() -> None:
    """Remove only the fields introduced by this revision."""

    op.drop_constraint(
        "fk_pages_ocr_processing_run_id",
        "pages",
        type_="foreignkey",
    )

    op.drop_column(
        "pages",
        "ocr_processing_run_id",
    )

    op.drop_column(
        "pages",
        "ocr_reason",
    )

    op.drop_column(
        "pages",
        "ocr_performed",
    )

    op.drop_column(
        "pages",
        "perception_mode",
    )

    op.drop_constraint(
        "fk_content_blocks_processing_run_id",
        "content_blocks",
        type_="foreignkey",
    )

    op.drop_column(
        "content_blocks",
        "coordinate_system",
    )

    op.drop_column(
        "content_blocks",
        "bbox_height",
    )

    op.drop_column(
        "content_blocks",
        "bbox_width",
    )

    op.drop_column(
        "content_blocks",
        "bbox_y",
    )

    op.drop_column(
        "content_blocks",
        "bbox_x",
    )

    op.drop_column(
        "content_blocks",
        "processing_run_id",
    )