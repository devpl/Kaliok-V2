from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from sqlmodel import Session, select

from kaliok.documents.docling_adapter import (
    document_content_from_docling,
)
from kaliok.documents.docling_client import convert_pdf_with_docling
from kaliok.documents.models import DocumentContent
from kaliok.storage.models import (
    ContentBlock,
    DocumentVersion,
    Page,
    ProcessingRun,
    utc_now,
)


DOCLING_PROCESS_TYPE = "document_extraction"
DOCLING_ENGINE = "docling"


def store_pdf_with_docling(
    session: Session,
    document_version: DocumentVersion,
    pdf_path: Path | str,
    *,
    base_url: str | None = None,
    timeout: float = 300,
    engine_version: str | None = None,
) -> ProcessingRun:
    """Explicitly convert a PDF with Docling, then persist its output."""
    document = convert_pdf_with_docling(
        pdf_path,
        base_url=base_url,
        timeout=timeout,
    )
    return store_docling_document(
        session,
        document_version,
        document,
        engine_version=engine_version,
    )


def store_docling_document(
    session: Session,
    document_version: DocumentVersion,
    document: dict[str, Any],
    *,
    engine_version: str | None = None,
) -> ProcessingRun:
    """Persist Docling output as an optional, non-current perception."""
    content = document_content_from_docling(document)
    fingerprint = hashlib.sha256(
        json.dumps(
            document,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    return store_docling_perception(
        session,
        document_version,
        content,
        engine_version=engine_version,
        idempotency_key=fingerprint,
    )


def store_docling_perception(
    session: Session,
    document_version: DocumentVersion,
    content: DocumentContent,
    *,
    engine_version: str | None = None,
    idempotency_key: str | None = None,
) -> ProcessingRun:
    """Store enriched blocks without changing Page's current perception."""
    _validate_page_numbers(document_version, content)

    with session.begin_nested():
        session.exec(
            select(DocumentVersion)
            .where(DocumentVersion.id == document_version.id)
            .with_for_update()
        ).one()

        if idempotency_key is not None:
            existing_run = _find_equivalent_run(
                session,
                document_version,
                engine_version,
                idempotency_key,
            )
            if existing_run is not None:
                return existing_run

        return _store_docling_perception(
            session,
            document_version,
            content,
            engine_version=engine_version,
            idempotency_key=idempotency_key,
        )


def _find_equivalent_run(
    session: Session,
    document_version: DocumentVersion,
    engine_version: str | None,
    idempotency_key: str,
) -> ProcessingRun | None:
    runs = session.exec(
        select(ProcessingRun).where(
            ProcessingRun.document_version_id == document_version.id,
            ProcessingRun.process_type == DOCLING_PROCESS_TYPE,
            ProcessingRun.status == "completed",
            ProcessingRun.engine == DOCLING_ENGINE,
        )
    ).all()
    return next(
        (
            run
            for run in runs
            if run.engine_version == engine_version
            and run.configuration.get("content_fingerprint")
            == idempotency_key
        ),
        None,
    )


def _validate_page_numbers(
    document_version: DocumentVersion,
    content: DocumentContent,
) -> None:
    if document_version.page_count is None:
        raise ValueError(
            "DocumentVersion.page_count doit être défini avant "
            "la persistance Docling."
        )

    invalid_page_numbers = sorted(
        {
            block.page
            for block in content.blocks
            if not 1 <= block.page <= document_version.page_count
        }
    )
    if invalid_page_numbers:
        raise ValueError(
            "Numéro(s) de page Docling hors limites : "
            f"{invalid_page_numbers}."
        )


def _store_docling_perception(
    session: Session,
    document_version: DocumentVersion,
    content: DocumentContent,
    *,
    engine_version: str | None,
    idempotency_key: str | None,
) -> ProcessingRun:
    run = ProcessingRun(
        document_version_id=document_version.id,
        process_type=DOCLING_PROCESS_TYPE,
        status="running",
        engine=DOCLING_ENGINE,
        engine_version=engine_version,
        configuration={
            "source": "docling_document_json",
            "content_fingerprint": idempotency_key,
        },
        metrics={"block_count": len(content.blocks)},
    )
    session.add(run)
    session.flush()

    pages = {
        page.page_number: page
        for page in session.exec(
            select(Page).where(
                Page.document_version_id == document_version.id
            )
        ).all()
    }
    page_metadata = {page.page: page for page in content.pages}

    for page_number in sorted({block.page for block in content.blocks}):
        if page_number in pages:
            continue
        metadata = page_metadata.get(page_number)
        page = Page(
            document_version_id=document_version.id,
            page_number=page_number,
            width=metadata.width if metadata else None,
            height=metadata.height if metadata else None,
            perception_mode="docling",
        )
        session.add(page)
        session.flush()
        pages[page_number] = page

    stored_by_ref: dict[str, ContentBlock] = {}
    stored_pairs: list[tuple[ContentBlock, str | None]] = []

    for block_index, block in enumerate(content.blocks):
        extra_data = dict(block.extra_data)
        extra_data.update(
            {
                "docling_self_ref": block.self_ref,
                "docling_parent_ref": block.parent_ref,
                "content_layer": block.content_layer,
                "heading_level": block.heading_level,
                "provenances": block.provenances,
                "indexable": block.indexable,
            }
        )
        stored = ContentBlock(
            page_id=pages[block.page].id,
            processing_run_id=run.id,
            block_index=block_index,
            reading_order=(
                block.reading_order
                if block.reading_order is not None
                else block_index
            ),
            block_type=block.block_type,
            content=block.text,
            extraction_method=block.extraction_method,
            extraction_engine=block.extraction_engine,
            extraction_engine_version=(
                block.extraction_engine_version or engine_version
            ),
            confidence=block.confidence,
            bbox=block.bbox,
            bbox_x=block.bbox_x,
            bbox_y=block.bbox_y,
            bbox_width=block.bbox_width,
            bbox_height=block.bbox_height,
            coordinate_system=block.coordinate_system,
            extra_data=extra_data,
        )
        session.add(stored)
        session.flush()
        if block.self_ref is not None:
            stored_by_ref[block.self_ref] = stored
        stored_pairs.append((stored, block.parent_ref))

    for stored, parent_ref in stored_pairs:
        parent = stored_by_ref.get(parent_ref) if parent_ref else None
        if parent is not None:
            stored.parent_block_id = parent.id

    run.status = "completed"
    run.completed_at = utc_now()
    session.flush()
    return run
