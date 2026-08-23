from __future__ import annotations

import re
import unicodedata

from kaliok.documents.models import TextBlock


MIN_CONTAINED_OCR_TOKENS = 3
MIN_CONTAINED_OCR_CHARACTERS = 12


def normalize_text_for_dedup(
    text: str,
) -> str:
    normalized = unicodedata.normalize(
        "NFKC",
        text,
    ).casefold()

    normalized = re.sub(
        r"\s+",
        " ",
        normalized,
    )

    normalized = re.sub(
        r"[^\w\s]",
        "",
        normalized,
    )

    return normalized.strip()


def _is_informative_contained_text(
    normalized_ocr: str,
) -> bool:
    tokens = normalized_ocr.split()

    return (
        len(tokens) >= MIN_CONTAINED_OCR_TOKENS
        and len(normalized_ocr)
        >= MIN_CONTAINED_OCR_CHARACTERS
    )


def deduplicate_ocr_against_native(
    blocks: list[TextBlock],
) -> list[TextBlock]:
    native_texts = {
        normalize_text_for_dedup(block.text)
        for block in blocks
        if (
            block.extraction_method == "native"
            and block.text.strip()
        )
    }

    if not native_texts:
        return blocks

    deduplicated: list[TextBlock] = []

    for block in blocks:
        if block.extraction_method != "ocr":
            deduplicated.append(block)
            continue

        normalized = normalize_text_for_dedup(
            block.text
        )

        if not normalized:
            deduplicated.append(block)
            continue

        if normalized in native_texts:
            continue

        if (
            _is_informative_contained_text(
                normalized
            )
            and any(
                normalized in native_text
                for native_text in native_texts
            )
        ):
            continue

        deduplicated.append(block)

    return deduplicated
