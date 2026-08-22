from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PageReadabilityAssessment:
    status: str
    score: float
    reason: str | None
    ocr_required: bool


def assess_native_text(
    text: str,
) -> PageReadabilityAssessment:
    stripped = text.strip()

    if not stripped:
        return PageReadabilityAssessment(
            status="unreadable",
            score=0.0,
            reason="empty_native_text",
            ocr_required=True,
        )

    score = 1.0
    reasons: list[str] = []

    text_length = len(stripped)

    if text_length < 20:
        score -= 0.60
        reasons.append("very_short_native_text")
    elif text_length < 80:
        score -= 0.20
        reasons.append("short_native_text")

    non_space_chars = [
        char
        for char in stripped
        if not char.isspace()
    ]

    alnum_ratio = 1.0

    if non_space_chars:
        alnum_ratio = (
            sum(
                char.isalnum()
                for char in non_space_chars
            )
            / len(non_space_chars)
        )

        if alnum_ratio < 0.35:
            score -= 0.30
            reasons.append("low_alphanumeric_ratio")

        suspicious_ratio = (
            sum(
                char == "\ufffd"
                or (
                    not char.isprintable()
                    and char not in "\r\n\t"
                )
                for char in stripped
            )
            / len(stripped)
        )

        if suspicious_ratio > 0.01:
            score -= 0.30
            reasons.append("suspicious_characters")

    lines = [
        line.strip()
        for line in stripped.splitlines()
        if line.strip()
    ]

    layout_fragmentation = False

    if lines:
        single_char_ratio = (
            sum(
                len(line) == 1
                for line in lines
            )
            / len(lines)
        )

        short_line_ratio = (
            sum(
                len(line) <= 3
                for line in lines
            )
            / len(lines)
        )

        average_line_length = (
            sum(len(line) for line in lines)
            / len(lines)
        )

        layout_fragmentation = (
            len(lines) >= 8
            and (
                single_char_ratio > 0.35
                or short_line_ratio > 0.45
                or average_line_length < 8
            )
        )

        if layout_fragmentation:
            reasons.append("fragmented_layout")

            # La fragmentation seule ne suffit pas à imposer l'OCR :
            # des tableaux, formulaires ou textes verticaux peuvent produire
            # ce motif tout en conservant un texte natif exploitable.
            if text_length < 120 or alnum_ratio < 0.55:
                score -= 0.30

    score = max(
        0.0,
        min(
            1.0,
            score,
        ),
    )

    if score >= 0.75:
        status = "readable"
        ocr_required = False
    elif score >= 0.45:
        status = "degraded"
        ocr_required = True
    else:
        status = "unreadable"
        ocr_required = True

    reason = (
        ";".join(reasons)
        if reasons
        else None
    )

    return PageReadabilityAssessment(
        status=status,
        score=score,
        reason=reason,
        ocr_required=ocr_required,
    )
