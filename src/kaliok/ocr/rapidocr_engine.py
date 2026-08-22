from __future__ import annotations

from typing import Any

from rapidocr import RapidOCR

from kaliok.ocr.base import OcrEngine, OcrResult


class RapidOcrEngine(OcrEngine):

    def __init__(self) -> None:
        self.engine = RapidOCR()

    def recognize(self, image: Any) -> list[OcrResult]:
        result = self.engine(image)

        if result is None:
            return []

        if result.txts is None or result.scores is None:
            return []

        boxes = result.boxes

        results: list[OcrResult] = []

        for index, (text, score) in enumerate(
            zip(
                result.txts,
                result.scores,
            )
        ):
            bbox_x: float | None = None
            bbox_y: float | None = None
            bbox_width: float | None = None
            bbox_height: float | None = None
            coordinate_system: str | None = None

            if boxes is not None and index < len(boxes):
                bbox = _bounding_box_from_points(
                    boxes[index]
                )

                if bbox is not None:
                    (
                        bbox_x,
                        bbox_y,
                        bbox_width,
                        bbox_height,
                    ) = bbox

                    coordinate_system = "pixels"

            results.append(
                OcrResult(
                    text=text,
                    confidence=float(score),
                    bbox_x=bbox_x,
                    bbox_y=bbox_y,
                    bbox_width=bbox_width,
                    bbox_height=bbox_height,
                    coordinate_system=coordinate_system,
                )
            )

        return results


def _bounding_box_from_points(
    points: Any,
) -> tuple[float, float, float, float] | None:
    try:
        if len(points) == 0:
            return None
    except TypeError:
        return None

    x_values: list[float] = []
    y_values: list[float] = []

    for point in points:
        try:
            if len(point) < 2:
                return None

            x = float(point[0])
            y = float(point[1])
        except (TypeError, ValueError, IndexError):
            return None

        x_values.append(x)
        y_values.append(y)

    if not x_values or not y_values:
        return None

    min_x = min(x_values)
    min_y = min(y_values)
    max_x = max(x_values)
    max_y = max(y_values)

    return (
        min_x,
        min_y,
        max_x - min_x,
        max_y - min_y,
    )

