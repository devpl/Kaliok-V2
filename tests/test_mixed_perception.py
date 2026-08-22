from __future__ import annotations

from dataclasses import dataclass

from kaliok.documents.reader import read_document
from kaliok.ocr.base import OcrEngine, OcrResult


class FakeTextPage:
    def get_text_range(self) -> str:
        return "Texte natif partiel"

    def close(self) -> None:
        pass


class FakeImage:
    def close(self) -> None:
        pass


class FakeBitmap:
    def to_pil(self) -> FakeImage:
        return FakeImage()

    def close(self) -> None:
        pass


class FakePage:
    def get_size(self) -> tuple[float, float]:
        return 595.0, 842.0

    def get_textpage(self) -> FakeTextPage:
        return FakeTextPage()

    def render(self, scale: float = 1.0) -> FakeBitmap:
        return FakeBitmap()

    def close(self) -> None:
        pass


class FakePdfDocument:
    def __init__(self, path) -> None:
        self._pages = [FakePage()]

    def __len__(self) -> int:
        return len(self._pages)

    def __getitem__(self, index: int) -> FakePage:
        return self._pages[index]

    def close(self) -> None:
        pass


class FakeOcrEngine(OcrEngine):
    def recognize(self, image) -> list[OcrResult]:
        return [
            OcrResult(
                text="Texte récupéré par OCR",
                confidence=0.92,
                bbox_x=10.0,
                bbox_y=20.0,
                bbox_width=120.0,
                bbox_height=30.0,
                coordinate_system="pixels",
            )
        ]


@dataclass(frozen=True)
class FakeAssessment:
    status: str = "degraded"
    score: float = 0.50
    reason: str = "synthetic_degraded_native_text"
    ocr_required: bool = True


def test_reader_keeps_native_and_ocr_for_mixed_page(
    monkeypatch,
    tmp_path,
):
    pdf_path = tmp_path / "mixed.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 synthetic")

    monkeypatch.setattr(
        "kaliok.documents.reader.pdfium.PdfDocument",
        FakePdfDocument,
    )

    monkeypatch.setattr(
        "kaliok.documents.reader.assess_native_text",
        lambda text: FakeAssessment(),
    )

    document = read_document(
        pdf_path,
        ocr_engine=FakeOcrEngine(),
    )

    assert document.page_count == 1
    assert len(document.pages) == 1
    assert len(document.blocks) == 2

    page = document.pages[0]

    assert page.perception_mode == "mixed"
    assert page.ocr_required is True
    assert page.ocr_performed is True
    assert page.ocr_reason == "synthetic_degraded_native_text"
    assert page.readability_status == "readable"
    assert page.readability_score == 0.92
    assert page.readability_reason == "ocr_recovered_text"

    native_block = document.blocks[0]
    ocr_block = document.blocks[1]

    assert native_block.extraction_method == "native"
    assert native_block.text == "Texte natif partiel"

    assert ocr_block.extraction_method == "ocr"
    assert ocr_block.text == "Texte récupéré par OCR"
    assert ocr_block.confidence == 0.92
    assert ocr_block.bbox_x == 10.0
    assert ocr_block.bbox_y == 20.0
    assert ocr_block.bbox_width == 120.0
    assert ocr_block.bbox_height == 30.0
    assert ocr_block.coordinate_system == "pixels"
