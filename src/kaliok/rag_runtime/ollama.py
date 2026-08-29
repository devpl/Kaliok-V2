from __future__ import annotations

import os
from collections.abc import Sequence

import requests

from kaliok.embeddings.ollama import (
    EMBEDDING_MODEL,
    OLLAMA_URL,
    embed_text,
    embed_texts,
)
from kaliok.rag.types import (
    ContextBundle,
    EmbeddingRecord,
    RagAnswer,
    RetrievalUnit,
)


class OllamaRagEmbedder:
    def __init__(self, model: str = EMBEDDING_MODEL) -> None:
        self.model = model

    def embed_units(
        self, units: Sequence[RetrievalUnit]
    ) -> tuple[EmbeddingRecord, ...]:
        vectors = embed_texts(
            [unit.text for unit in units],
            model=self.model,
        )
        return tuple(
            EmbeddingRecord(unit=unit, vector=vector, model=self.model)
            for unit, vector in zip(units, vectors, strict=True)
        )

    def embed_query(self, question: str) -> list[float]:
        return embed_text(question, model=self.model)


class OllamaGenerator:
    def __init__(
        self,
        model: str | None = None,
        *,
        base_url: str = OLLAMA_URL,
        timeout: float = 300,
    ) -> None:
        self.model = model or os.getenv("KALIOK_GENERATION_MODEL")
        if not self.model:
            raise ValueError(
                "Un modèle est requis via --generation-model ou "
                "KALIOK_GENERATION_MODEL."
            )
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    def generate(self, question: str, context: ContextBundle) -> RagAnswer:
        prompt = (
            "Réponds uniquement à partir du contexte fourni. "
            "Si le contexte ne suffit pas, indique-le explicitement. "
            "N'invente aucune information. Cite les identifiants de source "
            "présents dans le contexte.\n\n"
            f"Question :\n{question}\n\nContexte :\n{context.text}\n\nRéponse :"
        )
        response = requests.post(
            f"{self._base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0},
            },
            timeout=self._timeout,
        )
        response.raise_for_status()
        text = response.json().get("response")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Réponse Ollama de génération invalide.")
        return RagAnswer(
            text=text.strip(),
            context=context,
            metadata={"model": self.model},
        )
