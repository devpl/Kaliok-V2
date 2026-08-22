import re
from dataclasses import dataclass

from kaliok.documents.models import DocumentContent


@dataclass
class DocumentChunk:
    text: str
    page: int
    index: int
    source_block_index: int


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part.strip() for part in parts if part.strip()]


def chunk_document(
    document: DocumentContent,
    max_chars: int = 1000,
    overlap_chars: int = 0,
) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    chunk_index = 0

    for source_block_index, block in enumerate(
        document.blocks
    ):
        sentences = split_sentences(block.text)

        current_chunk: list[str] = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            if (
                current_chunk
                and current_length
                + 1
                + sentence_length
                > max_chars
            ):
                chunks.append(
                    DocumentChunk(
                        text=" ".join(current_chunk),
                        page=block.page,
                        index=chunk_index,
                        source_block_index=source_block_index,
                    )
                )

                chunk_index += 1
                current_chunk = []
                current_length = 0

            if sentence_length > max_chars:
                for part in split_long_text(
                    sentence,
                    max_chars,
                ):
                    chunks.append(
                        DocumentChunk(
                            text=part,
                            page=block.page,
                            index=chunk_index,
                            source_block_index=source_block_index,
                        )
                    )
                    chunk_index += 1

                continue

            current_chunk.append(sentence)

            if current_length == 0:
                current_length = sentence_length
            else:
                current_length += 1 + sentence_length

        if current_chunk:
            chunks.append(
                DocumentChunk(
                    text=" ".join(current_chunk),
                    page=block.page,
                    index=chunk_index,
                    source_block_index=source_block_index,
                )
            )

            chunk_index += 1

    return apply_overlap(
        chunks,
        overlap_chars,
    )


def split_long_text(
    text: str,
    max_chars: int,
) -> list[str]:
    parts: list[str] = []
    remaining = text.strip()

    while len(remaining) > max_chars:
        cut = remaining.rfind(
            " ",
            0,
            max_chars + 1,
        )

        if cut <= 0:
            cut = max_chars

        parts.append(
            remaining[:cut].strip()
        )
        remaining = remaining[cut:].strip()

    if remaining:
        parts.append(remaining)

    return parts


def apply_overlap(
    chunks: list[DocumentChunk],
    overlap_chars: int,
) -> list[DocumentChunk]:
    if overlap_chars <= 0:
        return chunks

    overlapped: list[DocumentChunk] = []

    for index, chunk in enumerate(chunks):
        if index == 0:
            overlapped.append(chunk)
            continue

        previous = chunks[index - 1]

        if (
            previous.source_block_index
            != chunk.source_block_index
        ):
            overlapped.append(chunk)
            continue

        overlap = previous.text[
            -overlap_chars:
        ].strip()

        text = chunk.text

        if overlap:
            text = f"{overlap} {text}"

        overlapped.append(
            DocumentChunk(
                text=text,
                page=chunk.page,
                index=chunk.index,
                source_block_index=(
                    chunk.source_block_index
                ),
            )
        )

    return overlapped
