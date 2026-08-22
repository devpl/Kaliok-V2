from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class OcrResult:
    text: str
    confidence: float | None = None

    bbox_x: float | None = None
    bbox_y: float | None = None
    bbox_width: float | None = None
    bbox_height: float | None = None

    coordinate_system: str | None = None


class OcrEngine(ABC):

    @abstractmethod
    def recognize(self, image: Any) -> list[OcrResult]:
        raise NotImplementedError
