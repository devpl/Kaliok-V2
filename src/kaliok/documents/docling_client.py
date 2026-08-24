from __future__ import annotations

import os
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
