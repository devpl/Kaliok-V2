import httpx
from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_GET, require_POST


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
    """Render the RAG Composer backed by persisted data."""
    composer_data, composer_error = get_composer_data()
    return render(
        request,
        "core_ui/rag_prototype.html",
        {"composer_data": composer_data, "composer_error": composer_error,
         "composer_api_base_url": settings.KALIOK_API_BASE_URL.rstrip("/")},
    )


@require_GET
def composer_data(request):
    """Relay a same-origin projection refresh to the Composer API."""
    payload, error = get_composer_data()
    if error:
        return JsonResponse({"status": "failed", "error": error}, status=503)
    return JsonResponse(payload)


def _relay_mutation(request, api_path: str) -> JsonResponse:
    try:
        response = httpx.post(
            f"{settings.KALIOK_API_BASE_URL}/rag/composer/{api_path}",
            content=request.body,
            headers={"Content-Type": "application/json"},
            timeout=10.0,
        )
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("invalid Composer mutation payload")
        return JsonResponse(payload, status=response.status_code)
    except (httpx.HTTPError, ValueError, TypeError):
        return JsonResponse(
            {"status": "failed", "error": "Le service de composition RAG est indisponible."},
            status=503,
        )


@require_POST
def composer_fork_revision(request):
    return _relay_mutation(request, "revisions/fork")


@require_POST
def composer_create_node(request):
    return _relay_mutation(request, "nodes")


@require_POST
def composer_create_empty_revision(request):
    return _relay_mutation(request, "revisions/empty")


@require_POST
def composer_select_tool(request):
    return _relay_mutation(request, "nodes/select-tool")


@require_POST
def composer_create_edge(request):
    return _relay_mutation(request, "edges")


@require_POST
def composer_step_test(request):
    """Relay a same-origin Composer test request to the existing FastAPI endpoint."""
    try:
        response = httpx.post(
            f"{settings.KALIOK_API_BASE_URL}/rag/composer/execute-step",
            content=request.body,
            headers={"Content-Type": "application/json"},
            timeout=30.0,
        )
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("invalid step-test payload")
        return JsonResponse(payload, status=response.status_code)
    except (httpx.HTTPError, ValueError, TypeError):
        return JsonResponse(
            {"status": "failed", "error": "Le service d’exécution est indisponible."},
            status=503,
        )
