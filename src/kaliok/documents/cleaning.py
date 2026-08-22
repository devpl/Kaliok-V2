from collections import Counter
from dataclasses import replace

from kaliok.documents.models import DocumentContent, TextBlock


def normalize_line(line: str) -> str:
    return " ".join(line.strip().split()).lower()


def find_repeated_lines(
    document: DocumentContent,
    min_page_ratio: float = 0.5,
    min_length: int = 8,
) -> set[str]:
    lines_by_page: dict[int, set[str]] = {}

    for block in document.blocks:
        page_lines = lines_by_page.setdefault(
            block.page,
            set(),
        )

        page_lines.update(
            normalize_line(line)
            for line in block.text.splitlines()
            if len(normalize_line(line)) >= min_length
        )

    counts = Counter()

    for lines in lines_by_page.values():
        counts.update(lines)

    minimum_pages = max(
        2,
        int(document.page_count * min_page_ratio),
    )

    return {
        line
        for line, count in counts.items()
        if count >= minimum_pages
    }


def remove_vertical_letter_sequences(
    lines: list[str],
    min_sequence_length: int = 4,
) -> list[str]:
    cleaned: list[str] = []

    index = 0

    while index < len(lines):
        current = lines[index].strip()

        if len(current) == 1 and current.isalpha():
            sequence_start = index
            sequence_end = index

            while sequence_end < len(lines):
                candidate = lines[sequence_end].strip()

                if len(candidate) == 1 and candidate.isalpha():
                    sequence_end += 1
                    continue

                break

            sequence_length = sequence_end - sequence_start

            if sequence_length >= min_sequence_length:
                if sequence_end < len(lines):
                    candidate = lines[sequence_end].strip()

                    parts = candidate.split(maxsplit=1)

                    if (
                        len(parts) == 2
                        and len(parts[0]) == 1
                        and parts[0].isalpha()
                    ):
                        cleaned.append(parts[1])
                        sequence_end += 1

                index = sequence_end
                continue

        cleaned.append(lines[index])
        index += 1

    return cleaned


def clean_block_text(
    text: str,
    repeated_lines: set[str],
) -> str:
    kept_lines: list[str] = []

    for line in text.splitlines():
        normalized = normalize_line(line)

        if normalized in repeated_lines:
            continue

        kept_lines.append(line)

    kept_lines = remove_vertical_letter_sequences(
        kept_lines,
    )

    return "\n".join(kept_lines).strip()


def clean_document(
    document: DocumentContent,
    min_page_ratio: float = 0.5,
) -> DocumentContent:
    repeated_lines = find_repeated_lines(
        document,
        min_page_ratio=min_page_ratio,
    )

    cleaned_blocks: list[TextBlock] = []

    for block in document.blocks:
        cleaned_text = clean_block_text(
            block.text,
            repeated_lines,
        )

        cleaned_blocks.append(
            replace(
                block,
                text=cleaned_text,
            )
        )

    return DocumentContent(
        source=document.source,
        page_count=document.page_count,
        blocks=cleaned_blocks,
        pages=list(document.pages),
    )
