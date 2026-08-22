from typing import Any

from llama_index.core.embeddings import BaseEmbedding

from kaliok.embeddings.ollama import (
    embed_text,
    embed_texts,
)


class KaliokLlamaEmbedding(BaseEmbedding):
    model_name: str = "kaliok-ollama"

    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            embed_batch_size=32,
            **kwargs,
        )

    def _get_query_embedding(
        self,
        query: str,
    ) -> list[float]:
        return embed_text(
            query
        )

    def _get_text_embedding(
        self,
        text: str,
    ) -> list[float]:
        return embed_text(
            text
        )

    def _get_text_embeddings(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        return embed_texts(
            texts
        )

    async def _aget_query_embedding(
        self,
        query: str,
    ) -> list[float]:
        return self._get_query_embedding(
            query
        )

    async def _aget_text_embedding(
        self,
        text: str,
    ) -> list[float]:
        return self._get_text_embedding(
            text
        )
