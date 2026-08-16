import os

from kaliok.ocr.base import OcrEngine
from kaliok.ocr.doctr_engine import DocTrEngine
from kaliok.ocr.rapidocr_engine import RapidOcrEngine


def create_ocr_engine() -> OcrEngine:
    engine_name = os.getenv(
        "KALIOK_OCR_ENGINE",
        "rapidocr",
    ).strip().lower()

    if engine_name == "rapidocr":
        return RapidOcrEngine()

    if engine_name == "doctr":
        return DocTrEngine()

    raise ValueError(
        f"Moteur OCR inconnu : {engine_name}"
    )
