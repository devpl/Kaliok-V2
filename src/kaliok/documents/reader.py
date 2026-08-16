from pathlib import Path

import pypdfium2 as pdfium
from rapidocr import RapidOCR

from kaliok.documents.models import DocumentContent, TextBlock


MIN_NATIVE_TEXT_LENGTH = 20


def read_document(pdf_path: str | Path, scale: float = 1.0) -> DocumentContent:
    pdf_path = Path(pdf_path)

    pdf = pdfium.PdfDocument(pdf_path)
    ocr_engine = None

    blocks: list[TextBlock] = []

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

        if ocr_engine is None:
            ocr_engine = RapidOCR()

        bitmap = page.render(scale=scale)
        image = bitmap.to_pil()

        result = ocr_engine(image)

        for text, score in zip(result.txts, result.scores):
            blocks.append(
                TextBlock(
                    text=text,
                    page=page_number,
                    confidence=float(score),
                )
            )

    return DocumentContent(
        source=pdf_path.name,
        page_count=len(pdf),
        blocks=blocks,
    )
