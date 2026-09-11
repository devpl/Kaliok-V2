import httpx
from django.conf import settings
from django.shortcuts import render


def get_composer_data() -> tuple[dict, str | None]:
    """Read the Composer projection through the Kaliok API."""
    try:
        response = httpx.get(
            f"{settings.KALIOK_API_BASE_URL}/rag/composer",
            timeout=5.0,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or not isinstance(payload.get("pipelines"), list):
            raise ValueError("invalid Composer payload")
        return payload, None
    except (httpx.HTTPError, ValueError, TypeError):
        return {"pipelines": []}, "Le service de composition RAG est indisponible."


def rag_prototype(request):
    """Render the read-only RAG Composer backed by persisted data."""
    composer_data, composer_error = get_composer_data()
    return render(
        request,
        "core_ui/rag_prototype.html",
        {"composer_data": composer_data, "composer_error": composer_error,
         "composer_api_base_url": settings.KALIOK_API_BASE_URL.rstrip("/")},
    )
