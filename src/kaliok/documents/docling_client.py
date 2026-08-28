from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import requests

from dotenv import load_dotenv

from kaliok.paths import PROJECT_ROOT


load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_DOCLING_TIMEOUT = 300


class DoclingClientError(RuntimeError):
    """Raised when the optional Docling HTTP conversion fails."""


def convert_pdf_with_docling(
    pdf_path: Path | str,
    *,
    base_url: str | None = None,
    timeout: float = DEFAULT_DOCLING_TIMEOUT,
) -> dict[str, Any]:
    """Send a PDF to Docling and return document.json_content only."""
    path = Path(pdf_path)
    url = base_url or os.getenv("KALIOK_DOCLING_URL")
    if not url:
        raise DoclingClientError(
            "URL Docling absente : définir KALIOK_DOCLING_URL "
            "ou fournir base_url."
        )

    try:
        with path.open("rb") as pdf_file:
            response = requests.post(
                f"{url.rstrip('/')}/v1/convert/file",
                files={
                    "files": (
                        path.name,
                        pdf_file,
                        "application/pdf",
                    )
                },
                data={
                    "to_formats": "json",
                    "image_export_mode": "placeholder",
                    "do_ocr": "true",
                    "ocr_engine": "auto",
                    "table_mode": "accurate",
                },
                timeout=timeout,
            )
        response.raise_for_status()
    except requests.Timeout as error:
        raise DoclingClientError(
            f"Délai dépassé lors de l'appel Docling : {error}"
        ) from error
    except requests.RequestException as error:
        raise DoclingClientError(
            f"Échec HTTP lors de l'appel Docling : {error}"
        ) from error
    except OSError as error:
        raise DoclingClientError(
            f"Impossible de lire le PDF {path} : {error}"
        ) from error

    try:
        payload = response.json()
    except (requests.JSONDecodeError, ValueError) as error:
        raise DoclingClientError(
            "Réponse Docling invalide : le corps n'est pas du JSON."
        ) from error

    document = payload.get("document") if isinstance(payload, dict) else None
    json_content = (
        document.get("json_content")
        if isinstance(document, dict)
        else None
    )
    if not isinstance(json_content, dict):
        raise DoclingClientError(
            "Réponse Docling invalide : document.json_content "
            "est absent ou n'est pas un objet JSON."
        )

    return json_content


def convert_pdf_with_docling_async(
    pdf_path: Path | str,
    *,
    base_url: str | None = None,
    submit_timeout: float = 60,
    poll_interval: float = 2.0,
    overall_timeout: float = 1800,
) -> dict[str, Any]:
    """Use Docling's asynchronous HTTP workflow and return json_content."""
    path = Path(pdf_path)
    url = base_url or os.getenv("KALIOK_DOCLING_URL")
    if not url:
        raise DoclingClientError(
            "URL Docling absente : définir KALIOK_DOCLING_URL "
            "ou fournir base_url."
        )
    if submit_timeout <= 0 or overall_timeout <= 0 or poll_interval < 0:
        raise DoclingClientError(
            "Délais Docling invalides : submit_timeout et overall_timeout "
            "doivent être positifs, poll_interval doit être positif ou nul."
        )

    started_at = time.perf_counter()

    def remaining_timeout() -> float:
        remaining = overall_timeout - (time.perf_counter() - started_at)
        if remaining <= 0:
            raise DoclingClientError(
                "Délai global dépassé lors de la conversion Docling "
                f"asynchrone ({overall_timeout} s)."
            )
        return min(submit_timeout, remaining)

    try:
        with path.open("rb") as pdf_file:
            response = requests.post(
                f"{url.rstrip('/')}/v1/convert/file/async",
                files={
                    "files": (
                        path.name,
                        pdf_file,
                        "application/pdf",
                    )
                },
                data={
                    "to_formats": "json",
                    "image_export_mode": "placeholder",
                    "do_ocr": "true",
                    "ocr_engine": "auto",
                    "table_mode": "accurate",
                },
                timeout=remaining_timeout(),
            )
        response.raise_for_status()
        payload = _read_json_response(response, stage="soumission")
    except DoclingClientError:
        raise
    except requests.Timeout as error:
        raise DoclingClientError(
            f"Délai dépassé lors de la soumission Docling : {error}"
        ) from error
    except requests.RequestException as error:
        raise DoclingClientError(
            f"Échec HTTP lors de la soumission Docling : {error}"
        ) from error
    except OSError as error:
        raise DoclingClientError(
            f"Impossible de lire le PDF {path} : {error}"
        ) from error

    task_id = payload.get("task_id") if isinstance(payload, dict) else None
    if not isinstance(task_id, str) or not task_id.strip():
        raise DoclingClientError(
            "Réponse Docling invalide : task_id absent après soumission."
        )

    while True:
        status_payload = _get_async_response(
            f"{url.rstrip('/')}/v1/status/poll/{task_id}",
            timeout=remaining_timeout(),
            stage=f"polling de la tâche {task_id}",
        )
        status = (
            status_payload.get("task_status", status_payload.get("status"))
            if isinstance(status_payload, dict)
            else None
        )
        if status == "success":
            break
        if status == "failure":
            detail = status_payload.get("error") or status_payload.get(
                "message"
            )
            suffix = f" : {detail}" if detail else ""
            raise DoclingClientError(
                f"La tâche Docling {task_id} a échoué{suffix}."
            )
        if status not in {"pending", "started"}:
            raise DoclingClientError(
                "Réponse Docling invalide : statut de tâche inconnu "
                f"pour {task_id} : {status!r}."
            )
        remaining = overall_timeout - (time.perf_counter() - started_at)
        if remaining <= 0:
            remaining_timeout()
        time.sleep(min(poll_interval, remaining))

    result_payload = _get_async_response(
        f"{url.rstrip('/')}/v1/result/{task_id}",
        timeout=remaining_timeout(),
        stage=f"récupération du résultat de la tâche {task_id}",
    )
    document = (
        result_payload.get("document")
        if isinstance(result_payload, dict)
        else None
    )
    json_content = (
        document.get("json_content")
        if isinstance(document, dict)
        else None
    )
    if not isinstance(json_content, dict):
        raise DoclingClientError(
            "Réponse Docling invalide : document.json_content "
            "est absent ou n'est pas un objet JSON."
        )
    return json_content


def _get_async_response(
    url: str,
    *,
    timeout: float,
    stage: str,
) -> dict[str, Any]:
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return _read_json_response(response, stage=stage)
    except DoclingClientError:
        raise
    except requests.Timeout as error:
        raise DoclingClientError(
            f"Délai dépassé lors du {stage} : {error}"
        ) from error
    except requests.RequestException as error:
        raise DoclingClientError(
            f"Échec HTTP lors du {stage} : {error}"
        ) from error


def _read_json_response(response, *, stage: str) -> dict[str, Any]:
    try:
        payload = response.json()
    except (requests.JSONDecodeError, ValueError) as error:
        raise DoclingClientError(
            f"Réponse Docling invalide lors du {stage} : corps non JSON."
        ) from error
    if not isinstance(payload, dict):
        raise DoclingClientError(
            f"Réponse Docling invalide lors du {stage} : objet JSON attendu."
        )
    return payload
