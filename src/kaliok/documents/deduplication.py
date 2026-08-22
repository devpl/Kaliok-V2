from __future__ import annotations

import re
import unicodedata

from kaliok.documents.models import TextBlock


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

        if (
            normalized
            and normalized in native_texts
        ):
            continue

        deduplicated.append(block)

    return deduplicated
