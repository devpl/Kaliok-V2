from pathlib import Path
from statistics import mean

import pypdfium2 as pdfium

from kaliok.documents.deduplication import (
    deduplicate_ocr_against_native,
)
from kaliok.documents.models import (
    DocumentContent,
    DocumentPage,
    TextBlock,
)
from kaliok.documents.readability import assess_native_text
from kaliok.ocr.base import OcrEngine
from kaliok.ocr.factory import create_ocr_engine


OCR_READABLE_CONFIDENCE = 0.60
OCR_DEGRADED_CONFIDENCE = 0.35


def read_document(
    pdf_path: str | Path,
    scale: float = 1.0,
    ocr_engine: OcrEngine | None = None,
) -> DocumentContent:
    pdf_path = Path(pdf_path)
    pdf = pdfium.PdfDocument(pdf_path)

    active_ocr_engine = ocr_engine
    blocks: list[TextBlock] = []
    pages: list[DocumentPage] = []

    try:
        for page_index in range(len(pdf)):
            page_number = page_index + 1
            page = pdf[page_index]

            try:
                width, height = page.get_size()

                text_page = page.get_textpage()

                try:
                    native_text = (
                        text_page
                        .get_text_range()
                        .strip()
                    )
                finally:
                    text_page.close()

                native_assessment = assess_native_text(
                    native_text
                )

                page_blocks: list[TextBlock] = []

                if native_text:
                    page_blocks.append(
                        TextBlock(
                            text=native_text,
                            page=page_number,
                            confidence=None,
                            extraction_method="native",
                            extraction_engine="pdfium",
                        )
                    )

                ocr_performed = False
                ocr_engine_name: str | None = None
                ocr_confidences: list[float] = []
                ocr_text_found = False

                if native_assessment.ocr_required:
                    if active_ocr_engine is None:
                        active_ocr_engine = (
                            create_ocr_engine()
                        )

                    ocr_performed = True
                    ocr_engine_name = _ocr_engine_name(
                        active_ocr_engine
                    )

                    bitmap = page.render(
                        scale=scale
                    )

                    try:
                        image = bitmap.to_pil()

                        try:
                            results = (
                                active_ocr_engine
                                .recognize(image)
                            )
                        finally:
                            image.close()
                    finally:
                        bitmap.close()

                    for result in results:
                        if result.text.strip():
                            ocr_text_found = True

                        if result.confidence is not None:
                            ocr_confidences.append(
                                result.confidence
                            )

                        page_blocks.append(
                            TextBlock(
                                text=result.text,
                                page=page_number,
                                confidence=(
                                    result.confidence
                                ),
                                extraction_method="ocr",
                                extraction_engine=(
                                    ocr_engine_name
                                ),
                                bbox_x=result.bbox_x,
                                bbox_y=result.bbox_y,
                                bbox_width=(
                                    result.bbox_width
                                ),
                                bbox_height=(
                                    result.bbox_height
                                ),
                                coordinate_system=(
                                    result.coordinate_system
                                ),
                            )
                        )

                page_blocks = (
                    deduplicate_ocr_against_native(
                        page_blocks
                    )
                )

                ocr_text_found = any(
                    block.extraction_method == "ocr"
                    and bool(block.text.strip())
                    for block in page_blocks
                )

                if not page_blocks:
                    page_blocks.append(
                        TextBlock(
                            text="",
                            page=page_number,
                            confidence=None,
                            extraction_method="none",
                            extraction_engine=None,
                        )
                    )

                perception_mode = (
                    _perception_mode(
                        page_blocks
                    )
                )

                ocr_confidence_mean = (
                    mean(
                        block.confidence
                        for block in page_blocks
                        if (
                            block.extraction_method == "ocr"
                            and block.confidence
                            is not None
                        )
                    )
                    if any(
                        block.extraction_method == "ocr"
                        and block.confidence is not None
                        for block in page_blocks
                    )
                    else None
                )

                (
                    readability_status,
                    readability_score,
                    readability_reason,
                ) = _final_readability(
                    native_status=(
                        native_assessment.status
                    ),
                    native_score=(
                        native_assessment.score
                    ),
                    native_reason=(
                        native_assessment.reason
                    ),
                    ocr_performed=ocr_performed,
                    ocr_text_found=ocr_text_found,
                    ocr_confidence_mean=(
                        ocr_confidence_mean
                    ),
                )

                pages.append(
                    DocumentPage(
                        page=page_number,
                        width=float(width),
                        height=float(height),
                        native_text_length=(
                            len(native_text)
                        ),
                        readability_status=(
                            readability_status
                        ),
                        readability_score=(
                            readability_score
                        ),
                        readability_reason=(
                            readability_reason
                        ),
                        ocr_required=(
                            native_assessment.ocr_required
                        ),
                        ocr_performed=ocr_performed,
                        ocr_reason=(
                            native_assessment.reason
                            if native_assessment.ocr_required
                            else None
                        ),
                        perception_mode=(
                            perception_mode
                        ),
                        ocr_engine=(
                            ocr_engine_name
                        ),
                        ocr_confidence_mean=(
                            ocr_confidence_mean
                        ),
                    )
                )

                blocks.extend(page_blocks)

            finally:
                page.close()

        return DocumentContent(
            source=pdf_path.name,
            page_count=len(pdf),
            blocks=blocks,
            pages=pages,
        )

    finally:
        pdf.close()


def _final_readability(
    *,
    native_status: str,
    native_score: float,
    native_reason: str | None,
    ocr_performed: bool,
    ocr_text_found: bool,
    ocr_confidence_mean: float | None,
) -> tuple[str, float, str | None]:
    if not ocr_performed:
        return (
            native_status,
            native_score,
            native_reason,
        )

    if not ocr_text_found:
        if native_status == "readable":
            return (
                native_status,
                native_score,
                native_reason,
            )

        return (
            "unreadable",
            0.0,
            "ocr_no_text",
        )

    if ocr_confidence_mean is None:
        return (
            "degraded",
            0.5,
            "ocr_text_without_confidence",
        )

    score = max(
        0.0,
        min(
            1.0,
            float(ocr_confidence_mean),
        ),
    )

    if score >= OCR_READABLE_CONFIDENCE:
        return (
            "readable",
            score,
            "ocr_recovered_text",
        )

    if score >= OCR_DEGRADED_CONFIDENCE:
        return (
            "degraded",
            score,
            "low_ocr_confidence",
        )

    return (
        "unreadable",
        score,
        "very_low_ocr_confidence",
    )


def _perception_mode(
    blocks: list[TextBlock],
) -> str:
    has_native = any(
        block.extraction_method == "native"
        and bool(block.text.strip())
        for block in blocks
    )

    has_ocr = any(
        block.extraction_method == "ocr"
        and bool(block.text.strip())
        for block in blocks
    )

    if has_native and has_ocr:
        return "mixed"

    if has_ocr:
        return "ocr"

    if has_native:
        return "native"

    return "none"


def _ocr_engine_name(
    ocr_engine: OcrEngine,
) -> str:
    engine_name = type(ocr_engine).__name__

    if engine_name == "RapidOcrEngine":
        return "rapidocr"

    return engine_name
