from dataclasses import dataclass

from kaliok.documents.models import DocumentContent


@dataclass
class DocumentChunk:
    text: str
    page: int
    index: int


def chunk_document(
    document: DocumentContent,
    max_chars: int = 1000,
) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    chunk_index = 0

    for block in document.blocks:
        text = block.text.strip()

        if not text:
            continue

        for start in range(0, len(text), max_chars):
            part = text[start:start + max_chars].strip()

            if not part:
                continue

            chunks.append(
                DocumentChunk(
                    text=part,
                    page=block.page,
                    index=chunk_index,
                )
            )

            chunk_index += 1

    return chunks
