from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from llama_index.core.node_parser import SemanticSplitterNodeParser

from kaliok.documents.chunking import DocumentChunk
from kaliok.documents.models import DocumentContent
from kaliok.embeddings.llama_adapter import KaliokLlamaEmbedding


@dataclass
class _PreparedBlock:
    source_block_index: int
    page: int
    sentences: list[dict[str, Any]]


def chunk_document_semantically(
    document: DocumentContent,
    breakpoint_percentile_threshold: int = 95,
    buffer_size: int = 1,
) -> list[DocumentChunk]:
    return _chunk_document_semantically_with_model(
        document=document,
        embed_model=KaliokLlamaEmbedding(),
        breakpoint_percentile_threshold=breakpoint_percentile_threshold,
        buffer_size=buffer_size,
    )


def _chunk_document_semantically_with_model(
    document: DocumentContent,
    embed_model: Any,
    breakpoint_percentile_threshold: int = 95,
    buffer_size: int = 1,
) -> list[DocumentChunk]:
    splitter = SemanticSplitterNodeParser.from_defaults(
        embed_model=embed_model,
        breakpoint_percentile_threshold=breakpoint_percentile_threshold,
        buffer_size=buffer_size,
        include_metadata=False,
        include_prev_next_rel=False,
    )

    prepared_blocks: list[_PreparedBlock] = []
    all_combined_sentences: list[str] = []

    for source_block_index, block in enumerate(document.blocks):
        text_splits = splitter.sentence_splitter(block.text)
        if not text_splits:
            continue

        sentences = splitter._build_sentence_groups(text_splits)
        if not sentences:
            continue

        prepared_blocks.append(
            _PreparedBlock(
                source_block_index=source_block_index,
                page=block.page,
                sentences=sentences,
            )
        )
        all_combined_sentences.extend(
            sentence["combined_sentence"]
            for sentence in sentences
        )

    if not prepared_blocks:
        return []

    embeddings = embed_model.get_text_embedding_batch(
        all_combined_sentences,
        show_progress=False,
    )

    if len(embeddings) != len(all_combined_sentences):
        raise RuntimeError(
            "Nombre d'embeddings incohérent pour le chunking sémantique."
        )

    chunks: list[DocumentChunk] = []
    chunk_index = 0
    offset = 0

    for prepared in prepared_blocks:
        count = len(prepared.sentences)
        block_embeddings = embeddings[offset:offset + count]
        offset += count

        for sentence, embedding in zip(
            prepared.sentences,
            block_embeddings,
            strict=True,
        ):
            sentence["combined_sentence_embedding"] = embedding

        distances = splitter._calculate_distances_between_sentence_groups(
            prepared.sentences
        )
        block_chunks = splitter._build_node_chunks(
            prepared.sentences,
            distances,
        )

        for text in block_chunks:
            text = text.strip()
            if not text:
                continue

            chunks.append(
                DocumentChunk(
                    text=text,
                    page=prepared.page,
                    index=chunk_index,
                    source_block_index=prepared.source_block_index,
                )
            )
            chunk_index += 1

    return chunks
