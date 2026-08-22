from llama_index.core.node_parser import SentenceSplitter

from kaliok.documents.chunking import DocumentChunk
from kaliok.documents.models import DocumentContent


def chunk_document_with_llamaindex(
    document: DocumentContent,
    chunk_size: int = 256,
    chunk_overlap: int = 32,
) -> list[DocumentChunk]:
    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks: list[DocumentChunk] = []
    chunk_index = 0

    for block in document.blocks:
        texts = splitter.split_text(block.text)

        for text in texts:
            cleaned_text = text.strip()

            if not cleaned_text:
                continue

            chunks.append(
                DocumentChunk(
                    text=cleaned_text,
                    page=block.page,
                    index=chunk_index,
                )
            )

            chunk_index += 1

    return chunks
