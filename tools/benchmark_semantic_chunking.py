from __future__ import annotations

import hashlib
import time

from llama_index.core import Document
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.node_parser import SemanticSplitterNodeParser
from pydantic import PrivateAttr

from kaliok.documents.models import DocumentContent, TextBlock
from kaliok.documents.semantic_chunking import (
    _chunk_document_semantically_with_model,
)


class CountingEmbedding(BaseEmbedding):
    model_name: str = "counting"

    _calls: int = PrivateAttr(default=0)

    def __init__(self) -> None:
        super().__init__(embed_batch_size=32)

    @property
    def calls(self) -> int:
        return self._calls

    @staticmethod
    def _vector(text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [byte / 255 for byte in digest[:16]]

    def _get_query_embedding(self, query: str):
        return self._vector(query)

    def _get_text_embedding(self, text: str):
        return self._vector(text)

    def _get_text_embeddings(self, texts: list[str]):
        self._calls += 1
        return [self._vector(text) for text in texts]

    async def _aget_query_embedding(self, query: str):
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str):
        return self._get_text_embedding(text)


def legacy(document, model):
    splitter = SemanticSplitterNodeParser.from_defaults(
        embed_model=model,
        breakpoint_percentile_threshold=95,
        buffer_size=1,
        include_metadata=False,
        include_prev_next_rel=False,
    )

    result = []

    for block in document.blocks:
        result.extend(
            splitter.get_nodes_from_documents(
                [Document(text=block.text)]
            )
        )

    return result


def main():
    document = DocumentContent(
        source="benchmark.pdf",
        page_count=50,
        blocks=[
            TextBlock(
                text=" ".join(
                    f"Phrase {i} de la page {page}."
                    for i in range(1, 13)
                ),
                page=page,
            )
            for page in range(1, 51)
        ],
    )

    old_model = CountingEmbedding()
    start = time.perf_counter()
    legacy(document, old_model)
    old_time = time.perf_counter() - start

    new_model = CountingEmbedding()
    start = time.perf_counter()
    _chunk_document_semantically_with_model(
        document,
        new_model,
    )
    new_time = time.perf_counter() - start

    print(
        f"legacy_calls={old_model.calls} "
        f"legacy_time={old_time:.4f}s"
    )
    print(
        f"batched_calls={new_model.calls} "
        f"batched_time={new_time:.4f}s"
    )


if __name__ == "__main__":
    main()
