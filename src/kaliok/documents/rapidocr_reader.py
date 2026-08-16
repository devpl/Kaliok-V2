from pathlib import Path

import pypdfium2 as pdfium
from rapidocr import RapidOCR

from kaliok.documents.models import DocumentContent, TextBlock


def read_pdf_with_rapidocr(
    pdf_path: str | Path,
    scale: float = 1.0,
) -> DocumentContent:
    pdf_path = Path(pdf_path)

    pdf = pdfium.PdfDocument(pdf_path)
    engine = RapidOCR()

    blocks: list[TextBlock] = []

    for page_index in range(len(pdf)):
        page = pdf[page_index]

        bitmap = page.render(scale=scale)
        image = bitmap.to_pil()

        result = engine(image)

        for text, score in zip(result.txts, result.scores):
            blocks.append(
                TextBlock(
                    text=text,
                    page=page_index + 1,
                    confidence=float(score),
                )
            )

    return DocumentContent(
        source=pdf_path.name,
        page_count=len(pdf),
        blocks=blocks,
    )
