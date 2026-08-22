from __future__ import annotations

from uuid import UUID

from kaliok.embeddings.ollama import embed_text
from kaliok.embeddings.service import search_similar_chunks
from kaliok.qa.service import (
    add_evidence,
    complete_attempt,
    create_attempt,
    create_question,
)


def answer_question_with_vector_retrieval(
    question_text: str,
    *,
    embedding_model_id: UUID,
    document_version_id: UUID,
    limit: int = 3,
    origin: str = "user",
):
    question = create_question(
        question_text,
        origin=origin,
        document_version_id=document_version_id,
    )

    attempt = create_attempt(
        question.id,
        strategy="vector",
        pipeline_version="vector-retrieval-1",
    )

    try:
        query_embedding = embed_text(
            question_text
        )

        results = search_similar_chunks(
            query_embedding=query_embedding,
            embedding_model_id=embedding_model_id,
            limit=limit,
            document_version_id=document_version_id,
        )

        if not results:
            complete_attempt(
                attempt.id,
                answer_text=None,
                confidence=None,
                resolved=False,
                resolution_reason="no_relevant_evidence",
                failure_reason="no_results",
            )

            return {
                "question_id": question.id,
                "attempt_id": attempt.id,
                "status": "unresolved",
                "answer": None,
                "results": [],
            }

        for rank, result in enumerate(
            results,
            start=1,
        ):
            add_evidence(
                attempt.id,
                document_version_id=document_version_id,
                chunk_id=result.chunk_id,
                rank=rank,
                score=result.distance,
                evidence_text=result.content,
            )

        best_result = results[0]

        # Première version retrieval-only :
        # on retourne directement le meilleur passage.
        answer_text = best_result.content

        # Une distance cosinus faible est meilleure.
        # Ce score est volontairement simple pour ce premier test.
        confidence = max(
            0.0,
            min(
                1.0,
                1.0 - best_result.distance,
            ),
        )

        complete_attempt(
            attempt.id,
            answer_text=answer_text,
            confidence=confidence,
            resolved=True,
            resolution_reason="best_vector_evidence",
        )

        return {
            "question_id": question.id,
            "attempt_id": attempt.id,
            "status": "resolved",
            "answer": answer_text,
            "confidence": confidence,
            "results": results,
        }

    except Exception:
        complete_attempt(
            attempt.id,
            answer_text=None,
            confidence=None,
            resolved=False,
            resolution_reason="retrieval_error",
            failure_reason="exception",
        )
        raise
