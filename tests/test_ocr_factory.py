import pytest

from kaliok.ocr.factory import create_ocr_engine


def test_create_rapidocr_engine(monkeypatch):
    class FakeRapidOcrEngine:
        pass

    monkeypatch.setenv("KALIOK_OCR_ENGINE", "rapidocr")
    monkeypatch.setattr(
        "kaliok.ocr.factory.RapidOcrEngine",
        FakeRapidOcrEngine,
    )

    engine = create_ocr_engine()

    assert isinstance(engine, FakeRapidOcrEngine)


def test_create_doctr_engine(monkeypatch):
    class FakeDocTrEngine:
        pass

    monkeypatch.setenv("KALIOK_OCR_ENGINE", "doctr")
    monkeypatch.setattr(
        "kaliok.ocr.factory.DocTrEngine",
        FakeDocTrEngine,
    )

    engine = create_ocr_engine()

    assert isinstance(engine, FakeDocTrEngine)


def test_unknown_ocr_engine(monkeypatch):
    monkeypatch.setenv("KALIOK_OCR_ENGINE", "unknown")

    with pytest.raises(ValueError, match="Moteur OCR inconnu"):
        create_ocr_engine()
