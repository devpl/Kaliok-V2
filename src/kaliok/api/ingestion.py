from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlmodel import Session

from kaliok.api.dependencies import get_session
from kaliok.ingestion.detection import (
    NoSourceIngestorError,
    SourceDetectionError,
)
from kaliok.ingestion_runtime.factory import create_ingestion_orchestrator
from kaliok.ingestion.ingestors.txt import (
    TxtDecodingError,
    TxtSourceLocationError,
)
from kaliok.ingestion.stores.postgres import (
    NormalizedContentConflictError,
)
from kaliok.ingestion.types import (
    IngestionRequest,
    SourceReference,
)


router = APIRouter(
    prefix="/ingestion",
    tags=["ingestion"],
)


class SourceReferenceRequest(BaseModel):
    name: str
    uri: str
    media_type: str

    size: int | None = None
    external_id: str | None = None


class IngestionRequestBody(BaseModel):
    source: SourceReferenceRequest

    source_id: UUID | None = None
    document_id: UUID | None = None


class IngestionResponse(BaseModel):
    document_id: UUID
    document_version_id: UUID

    status: str
    source_type: str
    media_type: str | None


@router.post(
    "",
    response_model=IngestionResponse,
)
def ingest(
    body: IngestionRequestBody,
    session: Session = Depends(get_session),
):
    request = IngestionRequest(
        source=SourceReference(
            name=body.source.name,
            uri=body.source.uri,
            media_type=body.source.media_type,
            size=body.source.size,
            external_id=body.source.external_id,
        ),
        source_id=body.source_id,
        document_id=body.document_id,
    )

    orchestrator = create_ingestion_orchestrator(session)

    try:
        result = orchestrator.ingest(request)
        session.commit()
    except NormalizedContentConflictError as error:
        session.rollback()
        raise HTTPException(
            status_code=409,
            detail=str(error),
        ) from error
    except (
        SourceDetectionError,
        NoSourceIngestorError,
        TxtSourceLocationError,
        TxtDecodingError,
        ValueError,
    ) as error:
        session.rollback()
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error
    except Exception:
        session.rollback()
        raise

    return IngestionResponse(
        document_id=UUID(str(result.document_id)),
        document_version_id=UUID(str(result.document_version_id)),
        status=result.status,
        source_type=result.detected_source.source_type,
        media_type=result.detected_source.media_type,
    )