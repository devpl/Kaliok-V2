from __future__ import annotations

import hashlib
from typing import Any

from llama_index.core import Document
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.node_parser import SemanticSplitterNodeParser
from pydantic import PrivateAttr

from kaliok.documents.chunking import DocumentChunk
from kaliok.documents.models import DocumentContent, TextBlock
from kaliok.documents.semantic_chunking import (
    _chunk_document_semantically_with_model,
)


class DeterministicEmbedding(BaseEmbedding):
    model_name: str = "deterministic-test"

    _batch_calls: int = PrivateAttr(default=0)

    def __init__(self) -> None:
        super().__init__(embed_batch_size=32)

    @property
    def batch_calls(self) -> int:
        return self._batch_calls

    @staticmethod
    def _vector(text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [
            int.from_bytes(digest[i:i + 4], "big") / 2**32
            for i in range(0, 32, 4)
        ]

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._vector(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._vector(text)

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        self._batch_calls += 1
        return [self._vector(text) for text in texts]

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return self._get_text_embedding(text)


def _legacy_chunking(
    document: DocumentContent,
    embed_model: BaseEmbedding,
) -> list[DocumentChunk]:
    splitter = SemanticSplitterNodeParser.from_defaults(
        embed_model=embed_model,
        breakpoint_percentile_threshold=95,
        buffer_size=1,
        include_metadata=False,
        include_prev_next_rel=False,
    )

    chunks: list[DocumentChunk] = []
    chunk_index = 0

    for source_block_index, block in enumerate(document.blocks):
        nodes = splitter.get_nodes_from_documents(
            [Document(text=block.text)]
        )

        for node in nodes:
            text = node.get_content().strip()

            if not text:
                continue

            chunks.append(
                DocumentChunk(
                    text=text,
                    page=block.page,
                    index=chunk_index,
                    source_block_index=source_block_index,
                )
            )
            chunk_index += 1

    return chunks


def _signature(chunks: list[DocumentChunk]) -> list[tuple[Any, ...]]:
    return [
        (
            chunk.index,
            chunk.page,
            chunk.source_block_index,
            chunk.text,
        )
        for chunk in chunks
    ]


def test_batched_chunking_is_strictly_equivalent():
    document = DocumentContent(
        source="equivalence.pdf",
        page_count=3,
        blocks=[
            TextBlock(
                text=(
                    "La première phrase parle du budget. "
                    "La deuxième poursuit sur les finances. "
                    "Puis le sujet change vers les travaux. "
                    "Enfin on revient aux dépenses."
                ),
                page=1,
            ),
            TextBlock(
                text=(
                    "Un contrat est signé. "
                    "Le délai est de trente jours. "
                    "Le fournisseur livre ensuite. "
                    "Une réception clôture le dossier."
                ),
                page=2,
            ),
            TextBlock(
                text=(
                    "Texte court mais distinct. "
                    "Deuxième phrase distincte."
                ),
                page=3,
            ),
        ],
    )

    legacy_model = DeterministicEmbedding()
    batched_model = DeterministicEmbedding()

    legacy = _legacy_chunking(document, legacy_model)
    batched = _chunk_document_semantically_with_model(
        document,
        batched_model,
    )

    assert _signature(batched) == _signature(legacy)
    assert batched_model.batch_calls <= legacy_model.batch_calls
