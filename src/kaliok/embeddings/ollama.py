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


def embed_texts(
    texts: list[str],
) -> list[list[float]]:
    if not texts:
        return []

    total_characters = sum(
        len(text)
        for text in texts
    )

    start = time.perf_counter()

    response = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={
            "model": EMBEDDING_MODEL,
            "input": texts,
        },
        timeout=300,
    )

    response.raise_for_status()

    duration = (
        time.perf_counter()
        - start
    )

    data = response.json()

    embeddings = data["embeddings"]

    if len(embeddings) != len(texts):
        raise ValueError(
            "Nombre d'embeddings inattendu : "
            f"{len(embeddings)} pour "
            f"{len(texts)} textes."
        )

    for embedding in embeddings:
        if (
            len(embedding)
            != EMBEDDING_DIMENSIONS
        ):
            raise ValueError(
                "Dimension d'embedding inattendue : "
                f"{len(embedding)}"
            )

    print(
        "    Ollama embed : "
        f"{len(texts):>4} texte(s) | "
        f"{total_characters:>8} caractères | "
        f"{duration:>7.2f} s"
    )

    return embeddings


def embed_text(
    text: str,
) -> list[float]:
    return embed_texts(
        [text]
    )[0]
