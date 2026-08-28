from __future__ import annotations

import os
import time

import requests
from dotenv import load_dotenv

from kaliok.paths import PROJECT_ROOT


load_dotenv(PROJECT_ROOT / ".env")


OLLAMA_URL = os.environ["KALIOK_OLLAMA_URL"]
EMBEDDING_MODEL = os.environ["KALIOK_EMBEDDING_MODEL"]

EMBEDDING_DIMENSIONS = 1024
DEFAULT_EMBEDDING_BATCH_SIZE = 32


def embed_texts(
    texts: list[str],
    *,
    model: str | None = None,
    batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
) -> list[list[float]]:
    if batch_size <= 0:
        raise ValueError("batch_size doit être strictement positif.")
    if not texts:
        return []

    selected_model = model or EMBEDDING_MODEL
    expected_dimensions = (
        EMBEDDING_DIMENSIONS
        if selected_model == EMBEDDING_MODEL
        else None
    )
    observed_dimensions: int | None = None
    all_embeddings: list[list[float]] = []
    batch_count = (len(texts) + batch_size - 1) // batch_size

    for batch_index, start_index in enumerate(
        range(0, len(texts), batch_size),
        start=1,
    ):
        batch = texts[start_index : start_index + batch_size]
        started = time.perf_counter()
        response = requests.post(
            f"{OLLAMA_URL}/api/embed",
            json={
                "model": selected_model,
                "input": batch,
            },
            timeout=300,
        )
        response.raise_for_status()
        duration = time.perf_counter() - started
        embeddings = response.json().get("embeddings")

        if not isinstance(embeddings, list) or len(embeddings) != len(batch):
            returned_count = (
                len(embeddings) if isinstance(embeddings, list) else "invalide"
            )
            raise ValueError(
                "Nombre d'embeddings inattendu dans le lot "
                f"{batch_index}/{batch_count} : {returned_count} pour "
                f"{len(batch)} textes."
            )

        for embedding in embeddings:
            if not isinstance(embedding, list) or not embedding:
                raise ValueError(
                    "Embedding Ollama invalide pour "
                    f"le modèle {selected_model}."
                )
            if not all(
                isinstance(value, (int, float)) for value in embedding
            ):
                raise ValueError(
                    "Embedding Ollama non numérique pour "
                    f"le modèle {selected_model}."
                )
            observed_dimensions = observed_dimensions or len(embedding)
            if len(embedding) != observed_dimensions:
                raise ValueError(
                    "Dimensions d'embedding hétérogènes pour "
                    f"le modèle {selected_model}."
                )
            if (
                expected_dimensions is not None
                and len(embedding) != expected_dimensions
            ):
                raise ValueError(
                    "Dimension d'embedding inattendue : "
                    f"{len(embedding)}"
                )

        all_embeddings.extend(embeddings)
        print(
            "    Ollama embed : "
            f"lot {batch_index:>3}/{batch_count:<3} | "
            f"{len(batch):>4} texte(s) | "
            f"{sum(len(text) for text in batch):>8} caractères | "
            f"{duration:>7.2f} s"
        )

    if len(all_embeddings) != len(texts):
        raise ValueError(
            "Nombre total d'embeddings inattendu : "
            f"{len(all_embeddings)} pour {len(texts)} textes."
        )

    return all_embeddings


def embed_text(
    text: str,
    *,
    model: str | None = None,
) -> list[float]:
    return embed_texts(
        [text],
        model=model,
    )[0]
