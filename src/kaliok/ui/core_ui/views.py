from uuid import UUID

from django.http import Http404
from django.shortcuts import render
from sqlmodel import Session, select

from kaliok.storage.database import create_database_engine
from kaliok.storage.models import Document, DocumentVersion


engine = create_database_engine()


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