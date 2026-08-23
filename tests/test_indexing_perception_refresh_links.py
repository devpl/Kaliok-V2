from __future__ import annotations

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


def test_perception_refresh_replaces_chunk_links_but_keeps_history(
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

        pdf_path = tmp_path / "perception-refresh.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 perception refresh")

        refreshed_content = DocumentContent(
            source=pdf_path.name,
            page_count=1,
            blocks=[
                TextBlock(
                    text="Contenu de la perception courante",
                    page=1,
                    extraction_method="native",
                    extraction_engine="pdfium",
                )
            ],
            pages=[
                DocumentPage(
                    page=1,
                    width=595.0,
                    height=842.0,
                    native_text_length=35,
                    readability_status="readable",
                    readability_score=1.0,
                    perception_mode="native",
                )
            ],
        )

        monkeypatch.setattr(
            service,
            "read_document",
            lambda path: refreshed_content,
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
                    text="Chunk stable entre les perceptions",
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
                    file_hash=service.calculate_sha256(pdf_path),
                    file_size=pdf_path.stat().st_size,
                    storage_uri=str(pdf_path),
                    page_count=1,
                    document_type="pdf",
                    version_status="active",
                    processing_status="completed",
                    readability_status="readable",
                    is_current=True,
                    processed_at=utc_now(),
                )
                session.add(version)
                session.flush()

                historical_run = ProcessingRun(
                    document_version_id=version.id,
                    process_type=service.PROCESS_TYPE,
                    status="completed",
                    engine=service.PERCEPTION_ENGINE,
                    engine_version="2",
                    completed_at=utc_now(),
                )
                session.add(historical_run)
                session.flush()

                page = Page(
                    document_version_id=version.id,
                    page_number=1,
                    page_status="active",
                    width=595.0,
                    height=842.0,
                    has_native_text=True,
                    native_text_length=25,
                    readability_status="readable",
                    readability_score=1.0,
                    perception_mode="native",
                    perception_processing_run_id=historical_run.id,
                )
                session.add(page)
                session.flush()

                historical_block = ContentBlock(
                    page_id=page.id,
                    processing_run_id=historical_run.id,
                    block_index=0,
                    reading_order=0,
                    block_type="text",
                    content="Contenu de la perception historique",
                    extraction_method="native",
                    extraction_engine="pdfium",
                )
                session.add(historical_block)
                session.flush()

                chunk = DocumentChunk(
                    document_version_id=version.id,
                    chunk_index=0,
                    content="Chunk stable entre les perceptions",
                    char_count=35,
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
                        content_block_id=historical_block.id,
                        block_order=0,
                    )
                )
                session.commit()

                version_id = version.id
                page_id = page.id
                chunk_id = chunk.id
                historical_run_id = historical_run.id
                historical_block_id = historical_block.id

            first_result = service.index_document(
                pdf_path,
                verbose=False,
            )
            assert first_result.already_indexed is False

            first_current_run_id = _assert_only_current_links(
                transactional_session,
                engine,
                version_id=version_id,
                page_id=page_id,
                chunk_id=chunk_id,
                expected_link_count=1,
            )

            with transactional_session(engine) as session:
                assert session.get(
                    ProcessingRun,
                    historical_run_id,
                ) is not None
                assert session.get(
                    ContentBlock,
                    historical_block_id,
                ) is not None

                first_current_run = session.get(
                    ProcessingRun,
                    first_current_run_id,
                )
                assert first_current_run is not None
                first_current_run.engine_version = "2"
                session.add(first_current_run)
                session.commit()

            second_result = service.index_document(
                pdf_path,
                verbose=False,
            )
            assert second_result.already_indexed is False

            second_current_run_id = _assert_only_current_links(
                transactional_session,
                engine,
                version_id=version_id,
                page_id=page_id,
                chunk_id=chunk_id,
                expected_link_count=1,
            )

            assert second_current_run_id != first_current_run_id

            with transactional_session(engine) as session:
                assert session.get(
                    ProcessingRun,
                    first_current_run_id,
                ) is not None

                historical_blocks = session.exec(
                    select(ContentBlock).where(
                        ContentBlock.page_id == page_id
                    )
                ).all()
                assert len(historical_blocks) == 3
        finally:
            transaction.rollback()
            engine.dispose()


def _assert_only_current_links(
    session_factory,
    engine,
    *,
    version_id,
    page_id,
    chunk_id,
    expected_link_count,
):
    with session_factory(engine) as session:
        page = session.get(Page, page_id)
        assert page is not None
        assert page.document_version_id == version_id

        current_run_id = page.perception_processing_run_id
        assert current_run_id is not None

        links = session.exec(
            select(ChunkContentBlock).where(
                ChunkContentBlock.chunk_id == chunk_id
            )
        ).all()

        assert len(links) == expected_link_count

        linked_blocks = [
            session.get(ContentBlock, link.content_block_id)
            for link in links
        ]

        assert all(block is not None for block in linked_blocks)
        assert all(
            block.processing_run_id == current_run_id
            for block in linked_blocks
            if block is not None
        )

        return current_run_id
