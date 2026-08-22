from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import text

from kaliok.storage.database import create_database_engine


@dataclass
class LexicalChunk:
    chunk_id: UUID
    content: str
    rank: float


def search_lexical_chunks(
    question: str,
    *,
    document_version_id: UUID,
    limit: int = 5,
) -> list[LexicalChunk]:
    query = text(
        """
        WITH query_data AS (
            SELECT
                to_tsquery(
                    'french',
                    array_to_string(
                        tsvector_to_array(
                            to_tsvector(
                                'french',
                                :question
                            )
                        ),
                        ' | '
                    )
                ) AS query
        )
        SELECT
            dc.id AS chunk_id,
            dc.content,
            ts_rank_cd(
                to_tsvector(
                    'french',
                    coalesce(dc.content, '')
                ),
                q.query
            ) AS rank
        FROM document_chunks dc
        CROSS JOIN query_data q
        WHERE dc.document_version_id = :document_version_id
          AND q.query IS NOT NULL
          AND to_tsvector(
                'french',
                coalesce(dc.content, '')
              ) @@ q.query
        ORDER BY rank DESC
        LIMIT :limit
        """
    )

    engine = create_database_engine()

    with engine.connect() as connection:
        rows = connection.execute(
            query,
            {
                "question": question,
                "document_version_id": document_version_id,
                "limit": limit,
            },
        ).all()

    return [
        LexicalChunk(
            chunk_id=row.chunk_id,
            content=row.content,
            rank=float(row.rank),
        )
        for row in rows
    ]
