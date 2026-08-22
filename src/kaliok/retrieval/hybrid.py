from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from kaliok.embeddings.ollama import embed_text
from kaliok.embeddings.service import search_similar_chunks
from kaliok.retrieval.lexical import search_lexical_chunks


@dataclass
class HybridChunk:
    chunk_id: UUID
    content: str

    rrf_score: float

    vector_rank: int | None = None
    lexical_rank: int | None = None

    vector_distance: float | None = None
    lexical_score: float | None = None


def search_hybrid_chunks(
    question: str,
    *,
    embedding_model_id: UUID,
    document_version_id: UUID,
    limit: int = 5,
    candidate_limit: int = 10,
    rrf_k: int = 60,
) -> list[HybridChunk]:
    query_embedding = embed_text(
        question
    )

    vector_results = search_similar_chunks(
        query_embedding=query_embedding,
        embedding_model_id=embedding_model_id,
        document_version_id=document_version_id,
        limit=candidate_limit,
    )

    lexical_results = search_lexical_chunks(
        question,
        document_version_id=document_version_id,
        limit=candidate_limit,
    )

    merged: dict[UUID, HybridChunk] = {}

    # ---------------------------------------------------------
    # Vectoriel
    # ---------------------------------------------------------

    for rank, result in enumerate(
        vector_results,
        start=1,
    ):
        rrf_score = 1.0 / (
            rrf_k + rank
        )

        merged[result.chunk_id] = HybridChunk(
            chunk_id=result.chunk_id,
            content=result.content,
            rrf_score=rrf_score,
            vector_rank=rank,
            vector_distance=result.distance,
        )

    # ---------------------------------------------------------
    # Lexical
    # ---------------------------------------------------------

    for rank, result in enumerate(
        lexical_results,
        start=1,
    ):
        rrf_score = 1.0 / (
            rrf_k + rank
        )

        existing = merged.get(
            result.chunk_id
        )

        if existing is None:
            merged[result.chunk_id] = HybridChunk(
                chunk_id=result.chunk_id,
                content=result.content,
                rrf_score=rrf_score,
                lexical_rank=rank,
                lexical_score=result.rank,
            )

            continue

        existing.rrf_score += rrf_score
        existing.lexical_rank = rank
        existing.lexical_score = result.rank

    results = list(
        merged.values()
    )

    results.sort(
        key=lambda item: item.rrf_score,
        reverse=True,
    )

    return results[:limit]
