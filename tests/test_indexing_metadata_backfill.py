from __future__ import annotations

from pathlib import Path

from sqlalchemy import func
from sqlmodel import Session, select

from kaliok.documents.chunking import DocumentChunk as PerceivedChunk
from kaliok.documents.models import (
    DocumentContent,
    DocumentPage,
    TextBlock,
)
from kaliok.indexing import service
from kaliok.storage.database import create_database_engine
from kaliok.storage.models import (
    ChunkContentBlock,
    ChunkEmbedding,
    ContentBlock,
    DocumentChunk,
    DocumentVersion,
    Page,
    ProcessingRun,
    utc_now,
)


def test_page_metadata_backfill_keeps_a_non_empty_current_perception(
    monkeypatch,
    tmp_path,
):
    engine = create_database_engine()

    with engine.connect() as connection:
        transaction = connection.begin()

        def transactional_session(_engine):
            return Session(
                bind=connection,
                join_transaction_mode="create_savepoint",
            )

        monkeypatch.setattr(
            service,
            "Session",
            transactional_session,
        )
        monkeypatch.setattr(
            service,
            "create_database_engine",
            lambda: engine,
        )

        pdf_path = tmp_path / "metadata-backfill.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 metadata backfill")
        file_hash = service.calculate_sha256(pdf_path)

        document_content = DocumentContent(
            source=pdf_path.name,
            page_count=2,
            blocks=[
                TextBlock(
                    text="Contenu historique page 1",
                    page=1,
                    extraction_method="native",
                    extraction_engine="pdfium",
                ),
                TextBlock(
                    text="Contenu historique page 2",
                    page=2,
                    extraction_method="native",
                    extraction_engine="pdfium",
                ),
            ],
            pages=[
                DocumentPage(
                    page=1,
                    width=595.0,
                    height=842.0,
                    native_text_length=27,
                    readability_status="readable",
                    readability_score=1.0,
                    perception_mode="native",
                ),
                DocumentPage(
                    page=2,
                    width=595.0,
                    height=842.0,
                    native_text_length=27,
                    readability_status="readable",
                    readability_score=1.0,
                    perception_mode="native",
                ),
            ],
        )

        monkeypatch.setattr(
            service,
            "read_document",
            lambda path: document_content,
        )
        monkeypatch.setattr(
            service,
            "clean_document",
            lambda document: document,
        )
        monkeypatch.setattr(
            service,
            "chunk_document_semantically",
            lambda document, **kwargs: [
                PerceivedChunk(
                    text="Contenu historique indexé",
                    page=1,
                    index=0,
                    source_block_index=0,
                )
            ],
        )

        try:
            with transactional_session(engine) as session:
                source = service.get_or_create_source(session)
                document = service.get_or_create_document(
                    session,
                    source,
                    pdf_path,
                )
                embedding_model = (
                    service.get_or_create_embedding_model(
                        session
                    )
                )
                version = DocumentVersion(
                    document_id=document.id,
                    version_number=1,
                    filename=pdf_path.name,
                    mime_type="application/pdf",
                    file_hash=file_hash,
                    file_size=pdf_path.stat().st_size,
                    storage_uri=str(pdf_path),
                    page_count=2,
                    document_type="pdf",
                    version_status="active",
                    processing_status="completed",
                    readability_status="unknown",
                    is_current=True,
                    processed_at=utc_now(),
                )
                session.add(version)
                session.flush()

                historical_run = ProcessingRun(
                    document_version_id=version.id,
                    process_type=service.PROCESS_TYPE,
                    status="completed",
                    engine="kaliok-reader",
                    engine_version="2",
                    completed_at=utc_now(),
                )
                session.add(historical_run)
                session.flush()

                first_historical_block = None

                for page_number in (1, 2):
                    page = Page(
                        document_version_id=version.id,
                        page_number=page_number,
                        page_status="active",
                        width=None,
                        height=None,
                        readability_status="unknown",
                        readability_score=None,
                        perception_mode="unknown",
                        perception_processing_run_id=historical_run.id,
                    )
                    session.add(page)
                    session.flush()

                    historical_block = ContentBlock(
                        page_id=page.id,
                        processing_run_id=historical_run.id,
                        block_index=page_number - 1,
                        reading_order=0,
                        block_type="text",
                        content=(
                            f"Contenu historique page {page_number}"
                        ),
                        extraction_method="native",
                        extraction_engine="pdfium",
                    )
                    session.add(historical_block)
                    session.flush()

                    if first_historical_block is None:
                        first_historical_block = historical_block

                assert first_historical_block is not None

                chunk = DocumentChunk(
                    document_version_id=version.id,
                    chunk_index=0,
                    content="Contenu historique indexé",
                    char_count=26,
                    page_start=1,
                    page_end=1,
                    chunking_strategy=service.CHUNKING_STRATEGY,
                    chunking_version=service.CHUNKING_VERSION,
                )
                session.add(chunk)
                session.flush()

                session.add(
                    ChunkEmbedding(
                        chunk_id=chunk.id,
                        embedding_model_id=embedding_model.id,
                        embedding=(
                            [0.0]
                            * service.EMBEDDING_DIMENSIONS
                        ),
                    )
                )
                session.add(
                    ChunkContentBlock(
                        chunk_id=chunk.id,
                        content_block_id=(
                            first_historical_block.id
                        ),
                        block_order=0,
                    )
                )

                session.commit()
                version_id = version.id
                historical_run_id = historical_run.id
                historical_block_id = first_historical_block.id
                chunk_id = chunk.id

            first_result = service.index_document(
                pdf_path,
                verbose=False,
            )

            assert first_result.already_indexed is False

            with transactional_session(engine) as session:
                version = session.get(DocumentVersion, version_id)
                assert version is not None

                pages = session.exec(
                    select(Page).where(
                        Page.document_version_id == version_id
                    )
                ).all()

                assert len(pages) == 2

                for page in pages:
                    assert page.perception_processing_run_id is not None

                    run = session.get(
                        ProcessingRun,
                        page.perception_processing_run_id,
                    )
                    assert run is not None
                    assert run.status == "completed"

                    block_count = session.exec(
                        select(func.count(ContentBlock.id)).where(
                            ContentBlock.page_id == page.id,
                            ContentBlock.processing_run_id == run.id,
                        )
                    ).one()
                    assert block_count > 0

                assert (
                    service.get_perception_storage_state(
                        session,
                        version,
                    )
                    == "complete"
                )

                links = session.exec(
                    select(ChunkContentBlock).where(
                        ChunkContentBlock.chunk_id == chunk_id
                    )
                ).all()

                assert links

                linked_blocks = [
                    session.get(
                        ContentBlock,
                        link.content_block_id,
                    )
                    for link in links
                ]

                assert all(
                    block is not None
                    for block in linked_blocks
                )
                assert all(
                    block.processing_run_id
                    == session.get(
                        Page,
                        block.page_id,
                    ).perception_processing_run_id
                    for block in linked_blocks
                    if block is not None
                )
                assert all(
                    link.content_block_id
                    != historical_block_id
                    for link in links
                )

                embedding_model = (
                    service.get_or_create_embedding_model(
                        session
                    )
                )
                assert (
                    service.get_index_storage_state(
                        session,
                        version,
                        embedding_model,
                    )
                    == "complete"
                )

                run_count_after_first_call = session.exec(
                    select(func.count(ProcessingRun.id)).where(
                        ProcessingRun.document_version_id == version_id,
                    )
                ).one()
                block_count_after_first_call = session.exec(
                    select(func.count(ContentBlock.id)).where(
                        ContentBlock.page_id.in_(
                            {
                                page.id
                                for page in pages
                            }
                        )
                    )
                ).one()
                link_count_after_first_call = session.exec(
                    select(func.count(ChunkContentBlock.chunk_id)).where(
                        ChunkContentBlock.chunk_id == chunk_id
                    )
                ).one()

                assert all(
                    page.perception_processing_run_id
                    != historical_run_id
                    for page in pages
                )

            second_result = service.index_document(
                pdf_path,
                verbose=False,
            )

            assert second_result.already_indexed is True

            with transactional_session(engine) as session:
                run_count_after_second_call = session.exec(
                    select(func.count(ProcessingRun.id)).where(
                        ProcessingRun.document_version_id == version_id,
                    )
                ).one()
                page_ids = {
                    page.id
                    for page in session.exec(
                        select(Page).where(
                            Page.document_version_id == version_id
                        )
                    ).all()
                }
                block_count_after_second_call = session.exec(
                    select(func.count(ContentBlock.id)).where(
                        ContentBlock.page_id.in_(page_ids)
                    )
                ).one()
                link_count_after_second_call = session.exec(
                    select(func.count(ChunkContentBlock.chunk_id)).where(
                        ChunkContentBlock.chunk_id == chunk_id
                    )
                ).one()

                assert (
                    run_count_after_second_call
                    == run_count_after_first_call
                )
                assert (
                    block_count_after_second_call
                    == block_count_after_first_call
                )
                assert (
                    link_count_after_second_call
                    == link_count_after_first_call
                )
        finally:
            transaction.rollback()
            engine.dispose()
