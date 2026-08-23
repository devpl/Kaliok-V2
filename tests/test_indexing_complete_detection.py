from __future__ import annotations

import pytest
from sqlmodel import Session

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


@pytest.mark.parametrize(
    ("state", "chunk_indices", "embedding_indices", "link_indices"),
    [
        ("missing_chunk", (0, 2), (0, 2), (0, 2)),
        ("missing_embedding", (0, 1), (0,), (0, 1)),
        ("missing_link", (0, 1), (0, 1), (0,)),
    ],
)
def test_complete_perception_is_not_already_indexed_when_index_is_incomplete(
    monkeypatch,
    tmp_path,
    state,
    chunk_indices,
    embedding_indices,
    link_indices,
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

        try:
            pdf_path = tmp_path / f"{state}.pdf"
            pdf_path.write_bytes(b"%PDF-1.4 index state")

            _seed_complete_perception_and_index(
                transactional_session,
                engine,
                pdf_path,
                chunk_indices=chunk_indices,
                embedding_indices=embedding_indices,
                link_indices=link_indices,
            )

            with pytest.raises(
                RuntimeError,
                match="Index incomplet",
            ):
                service.index_document(
                    pdf_path,
                    verbose=False,
                )
        finally:
            transaction.rollback()
            engine.dispose()


def test_complete_perception_and_index_is_already_indexed(
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

        try:
            pdf_path = tmp_path / "complete-index.pdf"
            pdf_path.write_bytes(b"%PDF-1.4 complete index")

            _, expected_version_id = (
                _seed_complete_perception_and_index(
                    transactional_session,
                    engine,
                    pdf_path,
                    chunk_indices=(0, 1),
                    embedding_indices=(0, 1),
                    link_indices=(0, 1),
                )
            )

            result = service.index_document(
                pdf_path,
                verbose=False,
            )

            assert result.already_indexed is True
            assert result.document_version_id == expected_version_id
            assert result.chunk_count == 2
        finally:
            transaction.rollback()
            engine.dispose()


def _seed_complete_perception_and_index(
    session_factory,
    engine,
    pdf_path,
    *,
    chunk_indices,
    embedding_indices,
    link_indices,
):
    with session_factory(engine) as session:
        source = service.get_or_create_source(session)
        document = service.get_or_create_document(
            session,
            source,
            pdf_path,
        )
        embedding_model = service.get_or_create_embedding_model(
            session
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

        run = ProcessingRun(
            document_version_id=version.id,
            process_type=service.PROCESS_TYPE,
            status="completed",
            engine=service.PERCEPTION_ENGINE,
            engine_version=service.PERCEPTION_VERSION,
            completed_at=utc_now(),
        )
        session.add(run)
        session.flush()

        page = Page(
            document_version_id=version.id,
            page_number=1,
            page_status="active",
            width=595.0,
            height=842.0,
            has_native_text=True,
            native_text_length=100,
            readability_status="readable",
            readability_score=1.0,
            perception_mode="native",
            perception_processing_run_id=run.id,
        )
        session.add(page)
        session.flush()

        block = ContentBlock(
            page_id=page.id,
            processing_run_id=run.id,
            block_index=0,
            reading_order=0,
            block_type="text",
            content="Bloc courant de la perception",
            extraction_method="native",
            extraction_engine="pdfium",
        )
        session.add(block)
        session.flush()

        stored_chunks = {}

        for chunk_index in chunk_indices:
            chunk = DocumentChunk(
                document_version_id=version.id,
                chunk_index=chunk_index,
                content=f"Contenu du chunk {chunk_index}",
                char_count=18,
                page_start=1,
                page_end=1,
                chunking_strategy=service.CHUNKING_STRATEGY,
                chunking_version=service.CHUNKING_VERSION,
            )
            session.add(chunk)
            session.flush()
            stored_chunks[chunk_index] = chunk

        for chunk_index in embedding_indices:
            session.add(
                ChunkEmbedding(
                    chunk_id=stored_chunks[chunk_index].id,
                    embedding_model_id=embedding_model.id,
                    embedding=[0.0] * service.EMBEDDING_DIMENSIONS,
                )
            )

        for chunk_index in link_indices:
            session.add(
                ChunkContentBlock(
                    chunk_id=stored_chunks[chunk_index].id,
                    content_block_id=block.id,
                    block_order=0,
                )
            )

        session.commit()

        return document.id, version.id
