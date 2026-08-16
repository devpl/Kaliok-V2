from kaliok.ocr.rapidocr_engine import RapidOcrEngine
from kaliok.ocr.base import OcrEngine


def test_rapidocr_engine_implements_ocr_engine():
    engine = RapidOcrEngine()

    assert isinstance(engine, OcrEngine)
