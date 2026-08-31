from datetime import datetime
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlmodel import Session, select

from kaliok.api.dependencies import get_session
from kaliok.storage.models import Document, DocumentVersion


router = APIRouter(
    prefix="/documents",
    tags=["documents"],
)


class DocumentVersionResponse(BaseModel):
    id: UUID
    version_number: int

    filename: str
    mime_type: str | None
    file_size: int | None
    storage_uri: str

    page_count: int | None

    document_type: str | None
    document_subtype: str | None

    version_status: str
    processing_status: str
    readability_status: str
    readability_score: float | None

    is_current: bool

    created_at: datetime
    processed_at: datetime | None


class DocumentSummaryResponse(BaseModel):
    id: UUID

    title: str | None
    document_family: str | None
    status: str
    language: str | None

    created_at: datetime
    updated_at: datetime


class DocumentDetailResponse(BaseModel):
    id: UUID

    source_id: UUID | None
    external_id: str | None
    title: str | None

    document_family: str | None
    status: str
    language: str | None

    created_at: datetime
    updated_at: datetime

    current_version: DocumentVersionResponse | None
    versions: list[DocumentVersionResponse]


@router.get(
    "",
    response_model=list[DocumentSummaryResponse],
)
def list_documents(
    session: Session = Depends(get_session),
):
    documents = session.exec(
        select(Document).order_by(Document.created_at.desc())
    ).all()

    return [
        DocumentSummaryResponse.model_validate(
            document,
            from_attributes=True,
        )
        for document in documents
    ]


@router.get(
    "/{document_id}",
    response_model=DocumentDetailResponse,
)
def get_document(
    document_id: UUID,
    session: Session = Depends(get_session),
):
    document = session.get(Document, document_id)

    if document is None:
        raise HTTPException(
            status_code=404,
            detail="Document introuvable",
        )

    versions = session.exec(
        select(DocumentVersion)
        .where(DocumentVersion.document_id == document_id)
        .order_by(DocumentVersion.version_number.desc())
    ).all()

    current_version = next(
        (version for version in versions if version.is_current),
        None,
    )

    return DocumentDetailResponse(
        id=document.id,
        source_id=document.source_id,
        external_id=document.external_id,
        title=document.title,
        document_family=document.document_family,
        status=document.status,
        language=document.language,
        created_at=document.created_at,
        updated_at=document.updated_at,
        current_version=(
            DocumentVersionResponse.model_validate(
                current_version,
                from_attributes=True,
            )
            if current_version
            else None
        ),
        versions=[
            DocumentVersionResponse.model_validate(
                version,
                from_attributes=True,
            )
            for version in versions
        ],
    )