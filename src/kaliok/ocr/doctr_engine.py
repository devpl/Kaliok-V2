import numpy as np
from doctr.models import ocr_predictor

from kaliok.ocr.base import OcrEngine, OcrResult


class DocTrEngine(OcrEngine):

    def __init__(self):
        self.engine = ocr_predictor(
            det_arch="db_mobilenet_v3_large",
            reco_arch="crnn_vgg16_bn",
            pretrained=True,
            assume_straight_pages=True,
        )

    def recognize(self, image) -> list[OcrResult]:
        image_array = np.array(image.convert("RGB"))

        result = self.engine([image_array])

        ocr_results: list[OcrResult] = []

        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    for word in line.words:
                        ocr_results.append(
                            OcrResult(
                                text=word.value,
                                confidence=float(word.confidence),
                            )
                        )

        return ocr_results
