from dataclasses import dataclass, field
from typing import Any


@dataclass
class TextBlock:
    text: str
    page: int
    confidence: float | None = None

    extraction_method: str = "unknown"
    extraction_engine: str | None = None
    extraction_engine_version: str | None = None

    bbox_x: float | None = None
    bbox_y: float | None = None
    bbox_width: float | None = None
    bbox_height: float | None = None

    coordinate_system: str | None = None

    block_type: str = "text"
    reading_order: int | None = None

    self_ref: str | None = None
    parent_ref: str | None = None
    content_layer: str | None = None
    heading_level: int | None = None

    bbox: dict[str, Any] | None = None
    provenances: list[dict[str, Any]] = field(default_factory=list)
    indexable: bool = True
    extra_data: dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentPage:
    page: int

    width: float | None = None
    height: float | None = None

    native_text_length: int = 0

    readability_status: str = "unknown"
    readability_score: float | None = None
    readability_reason: str | None = None

    ocr_required: bool = False
    ocr_performed: bool = False
    ocr_reason: str | None = None

    perception_mode: str = "unknown"

    ocr_engine: str | None = None
    ocr_confidence_mean: float | None = None


@dataclass
class DocumentContent:
    source: str
    page_count: int
    blocks: list[TextBlock] = field(default_factory=list)
    pages: list[DocumentPage] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(block.text for block in self.blocks)
