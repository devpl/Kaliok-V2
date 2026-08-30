from uuid import UUID

import httpx
from django.http import Http404
from django.shortcuts import render
from sqlmodel import Session, select

from kaliok.storage.database import create_database_engine
from kaliok.storage.models import Document, DocumentVersion


engine = create_database_engine()

KALIOK_API_BASE_URL = "http://127.0.0.1:8010"


def get_api_status() -> dict[str, str]:
    try:
        response = httpx.get(
            f"{KALIOK_API_BASE_URL}/health",
            timeout=2.0,
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError:
        return {
            "status": "error",
            "service": "kaliok-api",
        }


def home(request):
    with Session(engine) as session:
        documents = session.exec(
            select(Document).order_by(Document.created_at.desc())
        ).all()

    return render(
        request,
        "core_ui/home.html",
        {
            "documents": documents,
            "api_status": get_api_status(),
        },
    )


def document_detail(request, document_id: UUID):
    with Session(engine) as session:
        document = session.get(Document, document_id)

        if document is None:
            raise Http404("Document introuvable")

        versions = session.exec(
            select(DocumentVersion)
            .where(DocumentVersion.document_id == document_id)
            .order_by(DocumentVersion.version_number.desc())
        ).all()

    current_version = next(
        (version for version in versions if version.is_current),
        None,
    )

    return render(
        request,
        "core_ui/document_detail.html",
        {
            "document": document,
            "current_version": current_version,
            "versions": versions,
        },
    )