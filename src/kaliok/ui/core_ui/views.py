from pathlib import Path
from uuid import UUID, uuid4

import httpx
from django.conf import settings
from django.http import Http404, HttpResponse
from django.shortcuts import redirect, render

from .forms import DocumentUploadForm


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


def get_api_documents() -> list[dict] | None:
    try:
        response = httpx.get(
            f"{settings.KALIOK_API_BASE_URL}/documents",
            timeout=5.0,
        )
        response.raise_for_status()
    except httpx.HTTPError:
        return None

    try:
        payload = response.json()
    except ValueError:
        return None

    if not isinstance(payload, list):
        return None

    return payload


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


def ingest_api_document(
    *,
    path: Path,
    original_name: str,
    size: int,
) -> dict | None:
    body = {
        "source": {
            "name": original_name,
            "uri": path.resolve().as_uri(),
            "media_type": "text/plain",
            "size": size,
        }
    }

    try:
        response = httpx.post(
            f"{settings.KALIOK_API_BASE_URL}/ingestion",
            json=body,
            timeout=30.0,
        )
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
    documents = get_api_documents()

    if documents is None:
        documents = []

    return render(
        request,
        "core_ui/home.html",
        {
            "documents": documents,
            "api_status": get_api_status(),
            "upload_form": DocumentUploadForm(),
        },
    )


def upload_document(request):
    if request.method != "POST":
        return redirect("home")

    form = DocumentUploadForm(request.POST, request.FILES)

    if not form.is_valid():
        documents = get_api_documents()

        if documents is None:
            documents = []

        return render(
            request,
            "core_ui/home.html",
            {
                "documents": documents,
                "api_status": get_api_status(),
                "upload_form": form,
            },
            status=400,
        )

    uploaded_file = form.cleaned_data["file"]

    upload_dir = Path(settings.KALIOK_UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)

    original_name = Path(uploaded_file.name).name
    stored_name = f"{uuid4()}_{original_name}"
    stored_path = upload_dir / stored_name

    try:
        with stored_path.open("wb") as destination:
            for chunk in uploaded_file.chunks():
                destination.write(chunk)

        result = ingest_api_document(
            path=stored_path,
            original_name=original_name,
            size=uploaded_file.size,
        )

        if result is None:
            stored_path.unlink(missing_ok=True)
            return HttpResponse(
                "L'ingestion par l'API technique a échoué.",
                status=503,
                content_type="text/plain; charset=utf-8",
            )

        document_id = result.get("document_id")
        if document_id is None:
            stored_path.unlink(missing_ok=True)
            return HttpResponse(
                "Réponse d'ingestion invalide.",
                status=502,
                content_type="text/plain; charset=utf-8",
            )

        return redirect(
            "document_detail",
            document_id=document_id,
        )
    except OSError:
        stored_path.unlink(missing_ok=True)
        return HttpResponse(
            "Impossible d'enregistrer le fichier envoyé.",
            status=500,
            content_type="text/plain; charset=utf-8",
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