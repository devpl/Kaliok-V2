from pathlib import Path

import pypdfium2 as pdfium

from kaliok.documents.models import DocumentContent, TextBlock


def read_pdf_with_pdfium(pdf_path: str | Path) -> DocumentContent:
    pdf_path = Path(pdf_path)

    pdf = pdfium.PdfDocument(pdf_path)

    blocks: list[TextBlock] = []

    for page_index in range(len(pdf)):
        page = pdf[page_index]
        text_page = page.get_textpage()
        text = text_page.get_text_range().strip()

        if text:
            blocks.append(
                TextBlock(
                    text=text,
                    page=page_index + 1,
                    confidence=None,
                )
            )

    return DocumentContent(
        source=pdf_path.name,
        page_count=len(pdf),
        blocks=blocks,
    )
