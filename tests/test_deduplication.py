from kaliok.documents.deduplication import (
    deduplicate_ocr_against_native,
)
from kaliok.documents.models import TextBlock


def test_exact_ocr_duplicate_is_removed():
    blocks = [
        TextBlock(
            text="NATIF PARTIEL 26",
            page=1,
            extraction_method="native",
        ),
        TextBlock(
            text="Natif partiel 26",
            page=1,
            confidence=0.99,
            extraction_method="ocr",
        ),
        TextBlock(
            text="Texte uniquement OCR",
            page=1,
            confidence=0.98,
            extraction_method="ocr",
        ),
    ]

    result = deduplicate_ocr_against_native(
        blocks
    )

    assert len(result) == 2
    assert result[0].extraction_method == "native"
    assert result[1].text == "Texte uniquement OCR"


def test_punctuation_and_spacing_do_not_prevent_dedup():
    blocks = [
        TextBlock(
            text="NATIF   PARTIEL 26",
            page=1,
            extraction_method="native",
        ),
        TextBlock(
            text="NATIF PARTIEL 26.",
            page=1,
            confidence=0.99,
            extraction_method="ocr",
        ),
    ]

    result = deduplicate_ocr_against_native(
        blocks
    )

    assert len(result) == 1
    assert result[0].extraction_method == "native"


def test_different_ocr_text_is_kept():
    blocks = [
        TextBlock(
            text="Texte natif",
            page=1,
            extraction_method="native",
        ),
        TextBlock(
            text="Zone image différente",
            page=1,
            confidence=0.95,
            extraction_method="ocr",
        ),
    ]

    result = deduplicate_ocr_against_native(
        blocks
    )

    assert len(result) == 2
