from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlmodel import Session, select

from kaliok.storage.database import create_database_engine
from kaliok.storage.models import (
    Question,
    QuestionAttempt,
    QuestionEvidence,
    utc_now,
)


def create_question(
    question_text: str,
    *,
    origin: str = "user",
    document_id: UUID | None = None,
    document_version_id: UUID | None = None,
    priority: int = 0,
) -> Question:
    engine = create_database_engine()

    question = Question(
        origin=origin,
        question_text=question_text,
        document_id=document_id,
        document_version_id=document_version_id,
        priority=priority,
        status="pending",
    )

    with Session(engine) as session:
        session.add(question)
        session.commit()
        session.refresh(question)

    return question


def create_attempt(
    question_id: UUID,
    *,
    strategy: str,
    pipeline_version: str | None = None,
) -> QuestionAttempt:
    engine = create_database_engine()

    with Session(engine) as session:
        existing_attempts = session.exec(
            select(QuestionAttempt).where(
                QuestionAttempt.question_id == question_id
            )
        ).all()

        attempt = QuestionAttempt(
            question_id=question_id,
            attempt_number=len(existing_attempts) + 1,
            strategy=strategy,
            pipeline_version=pipeline_version,
            status="processing",
        )

        question = session.get(Question, question_id)

        if question is None:
            raise ValueError(
                f"Question introuvable : {question_id}"
            )

        question.status = "processing"
        question.updated_at = utc_now()

        session.add(attempt)
        session.add(question)
        session.commit()
        session.refresh(attempt)

    return attempt


def add_evidence(
    question_attempt_id: UUID,
    *,
    document_version_id: UUID | None = None,
    page_id: UUID | None = None,
    chunk_id: UUID | None = None,
    content_block_id: UUID | None = None,
    rank: int | None = None,
    score: float | None = None,
    evidence_text: str | None = None,
) -> QuestionEvidence:
    engine = create_database_engine()

    evidence = QuestionEvidence(
        question_attempt_id=question_attempt_id,
        document_version_id=document_version_id,
        page_id=page_id,
        chunk_id=chunk_id,
        content_block_id=content_block_id,
        rank=rank,
        score=score,
        evidence_text=evidence_text,
    )

    with Session(engine) as session:
        attempt = session.get(
            QuestionAttempt,
            question_attempt_id,
        )

        if attempt is None:
            raise ValueError(
                "Tentative introuvable : "
                f"{question_attempt_id}"
            )

        session.add(evidence)
        session.commit()
        session.refresh(evidence)

    return evidence


def complete_attempt(
    question_attempt_id: UUID,
    *,
    answer_text: str | None,
    confidence: float | None = None,
    resolved: bool,
    resolution_reason: str | None = None,
    failure_reason: str | None = None,
) -> QuestionAttempt:
    engine = create_database_engine()

    with Session(engine) as session:
        attempt = session.get(
            QuestionAttempt,
            question_attempt_id,
        )

        if attempt is None:
            raise ValueError(
                "Tentative introuvable : "
                f"{question_attempt_id}"
            )

        question = session.get(
            Question,
            attempt.question_id,
        )

        if question is None:
            raise ValueError(
                f"Question introuvable : {attempt.question_id}"
            )

        now: datetime = utc_now()

        attempt.answer_text = answer_text
        attempt.confidence = confidence
        attempt.completed_at = now

        if resolved:
            attempt.status = "resolved"
            attempt.failure_reason = None

            question.status = "resolved"
            question.resolution_reason = resolution_reason
            question.resolved_at = now
        else:
            attempt.status = "unresolved"
            attempt.failure_reason = failure_reason

            question.status = "unresolved"
            question.resolution_reason = resolution_reason
            question.resolved_at = None

        question.updated_at = now

        session.add(attempt)
        session.add(question)
        session.commit()
        session.refresh(attempt)

    return attempt


def get_question(
    question_id: UUID,
) -> Question | None:
    engine = create_database_engine()

    with Session(engine) as session:
        return session.get(
            Question,
            question_id,
        )


def get_attempts(
    question_id: UUID,
) -> list[QuestionAttempt]:
    engine = create_database_engine()

    with Session(engine) as session:
        return list(
            session.exec(
                select(QuestionAttempt)
                .where(
                    QuestionAttempt.question_id == question_id
                )
                .order_by(
                    QuestionAttempt.attempt_number
                )
            ).all()
        )
    