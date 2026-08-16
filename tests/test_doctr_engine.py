from kaliok.ocr.base import OcrEngine
from kaliok.ocr.doctr_engine import DocTrEngine


def test_doctr_engine_implements_ocr_engine():
    assert issubclass(DocTrEngine, OcrEngine)
