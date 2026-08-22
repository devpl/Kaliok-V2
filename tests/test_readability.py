from kaliok.documents.readability import (
    assess_native_text,
)


def test_empty_text_requires_ocr():
    result = assess_native_text("")

    assert result.status == "unreadable"
    assert result.score == 0.0
    assert result.ocr_required is True
    assert result.reason == "empty_native_text"


def test_normal_text_is_readable():
    result = assess_native_text(
        "Ceci est un paragraphe normal avec "
        "suffisamment de texte exploitable pour "
        "une extraction native."
    )

    assert result.status == "readable"
    assert result.score >= 0.75
    assert result.ocr_required is False


def test_fragmented_text_requires_ocr():
    result = assess_native_text(
        "A\nB\nC\nD\nE\nF\nG\nH\nI\nJ"
    )

    assert result.status in {
        "degraded",
        "unreadable",
    }
    assert result.ocr_required is True
