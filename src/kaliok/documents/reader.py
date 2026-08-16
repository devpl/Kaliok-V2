from pathlib import Path

import pypdfium2 as pdfium

from kaliok.documents.models import DocumentContent, TextBlock
from kaliok.ocr.base import OcrEngine
from kaliok.ocr.factory import create_ocr_engine


MIN_NATIVE_TEXT_LENGTH = 20


def read_document(
    pdf_path: str | Path,
    scale: float = 1.0,
    ocr_engine: OcrEngine | None = None,
) -> DocumentContent:
    pdf_path = Path(pdf_path)
    pdf = pdfium.PdfDocument(pdf_path)

    active_ocr_engine = ocr_engine
    blocks: list[TextBlock] = []

    try:
        for page_index in range(len(pdf)):
            page_number = page_index + 1
            page = pdf[page_index]

            text_page = page.get_textpage()
            native_text = text_page.get_text_range().strip()

            if len(native_text) >= MIN_NATIVE_TEXT_LENGTH:
                blocks.append(
                    TextBlock(
                        text=native_text,
                        page=page_number,
                        confidence=None,
                    )
                )
                continue

            if active_ocr_engine is None:
                active_ocr_engine = create_ocr_engine()

            bitmap = page.render(scale=scale)
            image = bitmap.to_pil()

            results = active_ocr_engine.recognize(image)

            for result in results:
                blocks.append(
                    TextBlock(
                        text=result.text,
                        page=page_number,
                        confidence=result.confidence,
                    )
                )

        return DocumentContent(
            source=pdf_path.name,
            page_count=len(pdf),
            blocks=blocks,
        )

    finally:
        pdf.close()
