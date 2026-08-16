from rapidocr import RapidOCR

from kaliok.ocr.base import OcrEngine, OcrResult


class RapidOcrEngine(OcrEngine):

    def __init__(self):
        self.engine = RapidOCR()

    def recognize(self, image) -> list[OcrResult]:
        result = self.engine(image)

        return [
            OcrResult(
                text=text,
                confidence=float(score),
            )
            for text, score in zip(result.txts, result.scores)
        ]
    