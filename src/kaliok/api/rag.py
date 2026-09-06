from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlmodel import Session

from kaliok.api.dependencies import get_session
from kaliok.rag_runtime import (
    NormalizedContentReference,
    create_normalized_rag_runtime,
)

router = APIRouter(prefix="/rag", tags=["rag"])


class RagQuestionRequest(BaseModel):
    document_id: UUID
    question: str = Field(min_length=1)


class RagSourceResponse(BaseModel):
    rank: int
    score: float
    document_id: UUID | None
    document_version_id: UUID | None
    normalized_content_unit_id: UUID | None
    source_unit_id: str | None
    text: str


class RagAnswerResponse(BaseModel):
    document_id: UUID
    document_version_id: UUID
    answer: str
    generation_model: str | None
    indexed_now: bool
    sources: list[RagSourceResponse]


@router.post("/answer", response_model=RagAnswerResponse)
def answer_question(
    body: RagQuestionRequest,
    session: Session = Depends(get_session),
) -> RagAnswerResponse:
    try:
        runtime = create_normalized_rag_runtime(
            session,
            reference=NormalizedContentReference(
                document_id=body.document_id,
            ),
        )

        indexed_now = False

        if not runtime.is_indexed():
            runtime.index()
            session.commit()
            indexed_now = True

        answer = runtime.answer(body.question)

    except ValueError as error:
        session.rollback()
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error
    except Exception:
        session.rollback()
        raise

    sources = [
        RagSourceResponse(
            rank=ranked.rank,
            score=ranked.score,
            document_id=(
                UUID(str(ranked.unit.provenance.document_id))
                if ranked.unit.provenance.document_id is not None
                else None
            ),
            document_version_id=(
                UUID(str(ranked.unit.provenance.document_version_id))
                if ranked.unit.provenance.document_version_id is not None
                else None
            ),
            normalized_content_unit_id=(
                UUID(
                    str(
                        ranked.unit.provenance.metadata.get(
                            "normalized_content_unit_id"
                        )
                    )
                )
                if ranked.unit.provenance.metadata.get(
                    "normalized_content_unit_id"
                )
                is not None
                else None
            ),
            source_unit_id=ranked.unit.provenance.metadata.get(
                "source_unit_id"
            ),
            text=ranked.unit.text,
        )
        for ranked in answer.context.candidates
    ]

    return RagAnswerResponse(
        document_id=runtime.document_id,
        document_version_id=runtime.document_version_id,
        answer=answer.text,
        generation_model=answer.metadata.get("model"),
        indexed_now=indexed_now,
        sources=sources,
    )