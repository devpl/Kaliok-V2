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
        image_rgb = image.convert("RGB")
        image_array = np.array(image_rgb)
        image_width, image_height = image_rgb.size

        result = self.engine([image_array])

        ocr_results: list[OcrResult] = []

        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    words = [
                        word
                        for word in line.words
                        if word.value.strip()
                    ]

                    if not words:
                        continue

                    text = " ".join(
                        word.value.strip()
                        for word in words
                    )

                    confidences = [
                        float(word.confidence)
                        for word in words
                        if word.confidence is not None
                    ]

                    confidence = (
                        sum(confidences) / len(confidences)
                        if confidences
                        else None
                    )

                    (
                        bbox_x,
                        bbox_y,
                        bbox_width,
                        bbox_height,
                    ) = _line_bbox_pixels(
                        line=line,
                        image_width=image_width,
                        image_height=image_height,
                    )

                    ocr_results.append(
                        OcrResult(
                            text=text,
                            confidence=confidence,
                            bbox_x=bbox_x,
                            bbox_y=bbox_y,
                            bbox_width=bbox_width,
                            bbox_height=bbox_height,
                            coordinate_system=(
                                "pixels"
                                if bbox_x is not None
                                else None
                            ),
                        )
                    )

        return ocr_results


def _line_bbox_pixels(
    *,
    line,
    image_width: int,
    image_height: int,
) -> tuple[
    float | None,
    float | None,
    float | None,
    float | None,
]:
    geometry = getattr(
        line,
        "geometry",
        None,
    )

    if (
        not geometry
        or len(geometry) != 2
    ):
        return None, None, None, None

    (x_min, y_min), (x_max, y_max) = geometry

    bbox_x = float(x_min) * image_width
    bbox_y = float(y_min) * image_height
    bbox_width = (
        float(x_max) - float(x_min)
    ) * image_width
    bbox_height = (
        float(y_max) - float(y_min)
    ) * image_height

    return (
        bbox_x,
        bbox_y,
        bbox_width,
        bbox_height,
    )
