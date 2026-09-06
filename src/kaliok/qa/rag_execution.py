from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any
from uuid import UUID

from sqlalchemy import func
from sqlmodel import Session, select

from kaliok.rag.types import RagAnswer, RankedCandidate
from kaliok.rag_runtime import (
    NormalizedContentReference,
    RagRuntimeConfiguration,
    create_normalized_rag_runtime,
)
from kaliok.storage.models import (
    Question,
    QuestionAttempt,
    QuestionEvidence,
    utc_now,
)


@dataclass(frozen=True)
class RagExecutionSource:
    rank: int
    score: float | None
    document_version_id: UUID
    text: str
    extra_data: dict[str, Any]


@dataclass(frozen=True)
class RagExecutionResult:
    question_id: UUID
    question_attempt_id: UUID
    document_id: UUID
    document_version_id: UUID
    answer: str
    sources: tuple[RagExecutionSource, ...]
    configuration_revision_id: UUID
    configuration: dict[str, object]
    metrics: dict[str, Any]


class RagExecutionService:
    """Run the existing RAG pipeline and persist its durable QA trace.

    Once an attempt has been committed, a RAG failure is recorded on that
    attempt and the original exception is re-raised unchanged.
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    def execute(
        self,
        question_id: UUID,
        *,
        profile_key: str = "production-default",
        configuration_revision_id: UUID | None = None,
        evaluation_campaign_id: UUID | None = None,
    ) -> RagExecutionResult:
        question = self._session.get(Question, question_id)
        if question is None:
            raise ValueError(f"Question introuvable : {question_id}.")
        if question.document_id is None and question.document_version_id is None:
            raise ValueError(
                "La question ne référence aucun document ni aucune version."
            )

        reference = NormalizedContentReference(
            document_id=question.document_id,
            document_version_id=question.document_version_id,
        )
        runtime_options: dict[str, Any]
        if configuration_revision_id is None:
            runtime_options = {"profile_key": profile_key}
        else:
            runtime_options = {
                "configuration_revision_id": configuration_revision_id
            }
        runtime = create_normalized_rag_runtime(
            self._session,
            reference=reference,
            **runtime_options,
        )
        snapshot = runtime.configuration.snapshot()
        attempt = QuestionAttempt(
            question_id=question.id,
            attempt_number=self._next_attempt_number(question.id),
            strategy="normalized-rag",
            status="processing",
            configuration_revision_id=runtime.configuration.revision_id,
            evaluation_campaign_id=evaluation_campaign_id,
            configuration=snapshot,
            metrics={},
        )
        self._session.add(attempt)
        self._session.commit()
        self._session.refresh(attempt)

        started_at = perf_counter()
        try:
            answer = runtime.answer(question.question_text)
            duration_ms = (perf_counter() - started_at) * 1000.0
            sources = self._persist_evidence(
                attempt.id,
                runtime.document_version_id,
                answer,
            )
            metrics = self._metrics(
                answer,
                runtime.configuration,
                duration_ms,
                len(sources),
            )
            attempt.answer_text = answer.text
            attempt.status = "completed"
            attempt.failure_reason = None
            attempt.metrics = metrics
            attempt.completed_at = utc_now()
            self._session.add(attempt)
            self._session.commit()
            self._session.refresh(attempt)
        except Exception as error:
            self._session.rollback()
            failed_attempt = self._session.get(QuestionAttempt, attempt.id)
            if failed_attempt is not None:
                failed_attempt.status = "failed"
                failed_attempt.failure_reason = (
                    f"{type(error).__name__}: {error}"
                )
                failed_attempt.completed_at = utc_now()
                self._session.add(failed_attempt)
                self._session.commit()
            raise

        return RagExecutionResult(
            question_id=question.id,
            question_attempt_id=attempt.id,
            document_id=runtime.document_id,
            document_version_id=runtime.document_version_id,
            answer=answer.text,
            sources=sources,
            configuration_revision_id=runtime.configuration.revision_id,
            configuration=snapshot,
            metrics=metrics,
        )

    def _next_attempt_number(self, question_id: UUID) -> int:
        maximum = self._session.exec(
            select(func.max(QuestionAttempt.attempt_number)).where(
                QuestionAttempt.question_id == question_id
            )
        ).one()
        return (maximum or 0) + 1

    def _persist_evidence(
        self,
        attempt_id: UUID,
        document_version_id: UUID,
        answer: RagAnswer,
    ) -> tuple[RagExecutionSource, ...]:
        sources: list[RagExecutionSource] = []
        for ranked in answer.context.candidates:
            extra_data = self._source_metadata(ranked)
            evidence = QuestionEvidence(
                question_attempt_id=attempt_id,
                document_version_id=document_version_id,
                rank=ranked.rank,
                score=ranked.score,
                evidence_text=ranked.unit.text,
                extra_data=extra_data,
            )
            self._session.add(evidence)
            sources.append(
                RagExecutionSource(
                    rank=ranked.rank,
                    score=ranked.score,
                    document_version_id=document_version_id,
                    text=ranked.unit.text,
                    extra_data=extra_data,
                )
            )
        return tuple(sources)

    @staticmethod
    def _source_metadata(ranked: RankedCandidate) -> dict[str, Any]:
        provenance = ranked.unit.provenance
        values = {
            "normalized_content_unit_id": provenance.metadata.get(
                "normalized_content_unit_id"
            ),
            "source_unit_id": provenance.metadata.get("source_unit_id"),
            "representation": provenance.representation,
            "embedding_model": provenance.embedding_model,
        }
        return {
            key: str(value) if isinstance(value, UUID) else value
            for key, value in values.items()
            if value is not None
        }

    @staticmethod
    def _metrics(
        answer: RagAnswer,
        configuration: RagRuntimeConfiguration,
        duration_ms: float,
        source_count: int,
    ) -> dict[str, Any]:
        generation_model = answer.metadata.get("model")
        return {
            "retrieved_passage_count": source_count,
            "generation_model": (
                generation_model
                if generation_model is not None
                else configuration.generation_model
            ),
            "top_k": configuration.retrieval_top_k,
            "duration_ms": duration_ms,
        }
