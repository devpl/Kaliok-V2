from dataclasses import dataclass
from uuid import UUID

from pgvector.sqlalchemy import Vector
from sqlalchemy import bindparam, text

from kaliok.storage.database import create_database_engine


EMBEDDING_DIMENSIONS = 1024


@dataclass
class SimilarChunk:
    chunk_id: UUID
    content: str
    distance: float


def search_similar_chunks(
    query_embedding: list[float],
    embedding_model_id: UUID,
    limit: int = 5,
    document_version_id: UUID | None = None,
) -> list[SimilarChunk]:
    if len(query_embedding) != EMBEDDING_DIMENSIONS:
        raise ValueError(
            f"Dimension d'embedding invalide : "
            f"{len(query_embedding)} au lieu de {EMBEDDING_DIMENSIONS}"
        )

    search_query = text(
        """
        SELECT
            dc.id AS chunk_id,
            dc.content,
            ce.embedding <=> :query_embedding AS distance
        FROM chunk_embeddings ce
        JOIN document_chunks dc
            ON dc.id = ce.chunk_id
        WHERE ce.embedding_model_id = :embedding_model_id
          AND (
                :document_version_id IS NULL
                OR dc.document_version_id = :document_version_id
              )
        ORDER BY ce.embedding <=> :query_embedding
        LIMIT :limit
        """
    ).bindparams(
        bindparam(
            "query_embedding",
            type_=Vector(EMBEDDING_DIMENSIONS),
        )
    )

    engine = create_database_engine()

    with engine.connect() as connection:
        rows = connection.execute(
            search_query,
            {
                "query_embedding": query_embedding,
                "embedding_model_id": embedding_model_id,
                "document_version_id": document_version_id,
                "limit": limit,
            },
        ).all()

    return [
        SimilarChunk(
            chunk_id=row.chunk_id,
            content=row.content,
            distance=float(row.distance),
        )
        for row in rows
    ]
