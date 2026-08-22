"""add hnsw index for chunk embeddings

Revision ID: e47bea513fde
Revises: e601828759a0
Create Date: 2026-08-18 18:42:21.164224

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e47bea513fde'
down_revision: Union[str, Sequence[str], None] = 'e601828759a0'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.execute(
        """
        CREATE INDEX ix_chunk_embeddings_embedding_hnsw
        ON chunk_embeddings
        USING hnsw (embedding vector_cosine_ops)
        """
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.execute(
        """
        DROP INDEX IF EXISTS ix_chunk_embeddings_embedding_hnsw
        """
    )
