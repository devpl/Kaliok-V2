from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class OcrResult:
    text: str
    confidence: float | None = None


class OcrEngine(ABC):

    @abstractmethod
    def recognize(self, image: Any) -> list[OcrResult]:
        raise NotImplementedError
