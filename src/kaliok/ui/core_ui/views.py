from uuid import UUID

import httpx
from django.conf import settings
from django.http import Http404, HttpResponse
from django.shortcuts import render
from sqlmodel import Session, select

from kaliok.storage.database import create_database_engine
from kaliok.storage.models import Document


engine = create_database_engine()


def get_api_status() -> dict[str, str]:
    try:
        response = httpx.get(
            f"{settings.KALIOK_API_BASE_URL}/health",
            timeout=2.0,
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError:
        return {
            "status": "error",
            "service": "kaliok-api",
        }


def get_api_document(document_id: UUID) -> dict | None:
    try:
        response = httpx.get(
            f"{settings.KALIOK_API_BASE_URL}/documents/{document_id}",
            timeout=5.0,
        )
    except httpx.RequestError:
        return None

    if response.status_code == 404:
        raise Http404("Document introuvable")

    try:
        response.raise_for_status()
    except httpx.HTTPError:
        return None

    try:
        payload = response.json()
    except ValueError:
        return None

    if not isinstance(payload, dict):
        return None

    return payload


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
    document = get_api_document(document_id)

    if document is None:
        return HttpResponse(
            "API technique indisponible",
            status=503,
            content_type="text/plain; charset=utf-8",
        )

    return render(
        request,
        "core_ui/document_detail.html",
        {
            "document": document,
            "current_version": document.get("current_version"),
            "versions": document.get("versions", []),
        },
    )