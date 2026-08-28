from __future__ import annotations

from copy import deepcopy

import pytest
from sqlalchemy import func
from sqlmodel import Session, select

from kaliok.documents.docling_adapter import (
    document_content_from_docling,
)
from kaliok.indexing import docling
from kaliok.indexing.docling import store_docling_document
from kaliok.storage.database import create_database_engine
from kaliok.storage.models import (
    ContentBlock,
    ChunkContentBlock,
    ChunkEmbedding,
    Document,
    DocumentChunk,
    DocumentVersion,
    EmbeddingModel,
    Page,
    ProcessingRun,
    Source,
    utc_now,
)


@pytest.fixture
def docling_session():
    engine = create_database_engine()
    with engine.connect() as connection:
        transaction = connection.begin()
        try:
            with Session(
                bind=connection,
                join_transaction_mode="create_savepoint",
            ) as session:
                yield session
        finally:
            transaction.rollback()


def _create_version(
    session: Session,
    *,
    page_count: int = 1,
    create_page: bool = True,
) -> tuple[DocumentVersion, Page | None]:
    source = Source(name="docling-test", source_type="test")
    session.add(source)
    session.flush()
    document = Document(source_id=source.id, external_id="structured.pdf")
    session.add(document)
    session.flush()
    version = DocumentVersion(
        document_id=document.id,
        version_number=1,
        filename="structured.pdf",
        file_hash="docling-test",
        storage_uri="memory://structured.pdf",
        page_count=page_count,
    )
    session.add(version)
    session.flush()
    page = None
    if create_page:
        page = Page(document_version_id=version.id, page_number=1)
        session.add(page)
        session.flush()
    return version, page


def _docling_document() -> dict:
    first_bbox = {
        "l": 10.0,
        "t": 40.0,
        "r": 210.0,
        "b": 20.0,
        "coord_origin": "BOTTOMLEFT",
    }
    return {
        "name": "structured.pdf",
        "pages": {
            "1": {"size": {"width": 595.0, "height": 842.0}}
        },
        "body": {
            "children": [
                {"$ref": "#/groups/0"},
                {"$ref": "#/tables/0"},
            ]
        },
        "furniture": {
            "children": [{"$ref": "#/texts/2"}]
        },
        "groups": [
            {
                "self_ref": "#/groups/0",
                "label": "group",
                "children": [
                    {"$ref": "#/texts/0"},
                    {"$ref": "#/texts/1"},
                ],
            }
        ],
        "texts": [
            {
                "self_ref": "#/texts/0",
                "parent": {"$ref": "#/groups/0"},
                "label": "section_header",
                "level": 2,
                "text": "Conditions générales",
                "prov": [
                    {"page_no": 1, "bbox": first_bbox},
                    {
                        "page_no": 1,
                        "bbox": {
                            "l": 10.0,
                            "t": 18.0,
                            "r": 80.0,
                            "b": 10.0,
                            "coord_origin": "BOTTOMLEFT",
                        },
                    },
                ],
            },
            {
                "self_ref": "#/texts/1",
                "parent": {"$ref": "#/groups/0"},
                "label": "list_item",
                "text": "Premier élément",
                "prov": [{"page_no": 1}],
            },
            {
                "self_ref": "#/texts/2",
                "label": "page_header",
                "text": "Kaliok — document confidentiel",
                "content_layer": "furniture",
                "prov": [{"page_no": 1}],
            },
        ],
        "tables": [
            {
                "self_ref": "#/tables/0",
                "label": "table",
                "prov": [{"page_no": 1}],
                "data": {
                    "num_rows": 2,
                    "num_cols": 2,
                    "table_cells": [
                        {
                            "start_row_offset_idx": 0,
                            "start_col_offset_idx": 0,
                            "text": "Nom",
                        },
                        {
                            "start_row_offset_idx": 0,
                            "start_col_offset_idx": 1,
                            "text": "Valeur",
                        },
                        {
                            "start_row_offset_idx": 1,
                            "start_col_offset_idx": 0,
                            "text": "Durée",
                        },
                        {
                            "start_row_offset_idx": 1,
                            "start_col_offset_idx": 1,
                            "text": "12 mois",
                        },
                    ],
                },
            }
        ],
        "pictures": [],
    }


def test_docling_json_is_mapped_to_enriched_blocks():
    content = document_content_from_docling(_docling_document())
    blocks = {block.self_ref: block for block in content.blocks}

    group = blocks["#/groups/0"]
    heading = blocks["#/texts/0"]
    list_item = blocks["#/texts/1"]
    header = blocks["#/texts/2"]
    table = blocks["#/tables/0"]

    assert group.block_type == "group"
    assert heading.block_type == "section_header"
    assert heading.heading_level == 2
    assert heading.parent_ref == group.self_ref
    assert heading.bbox == _docling_document()["texts"][0]["prov"][0]["bbox"]
    assert heading.coordinate_system == "BOTTOMLEFT"
    assert heading.bbox_x == 10.0
    assert heading.bbox_y == 20.0
    assert heading.bbox_width == 200.0
    assert heading.bbox_height == 20.0
    assert len(heading.provenances) == 2
    assert list_item.block_type == "list_item"
    assert header.block_type == "page_header"
    assert header.content_layer == "furniture"
    assert header.indexable is False
    assert table.block_type == "table"
    assert table.text == "Nom | Valeur\nDurée | 12 mois"
    assert table.extra_data["table"]["num_rows"] == 2


def test_docling_blocks_are_persisted_without_becoming_current():
    engine = create_database_engine()

    with engine.connect() as connection:
        transaction = connection.begin()
        try:
            with Session(
                bind=connection,
                join_transaction_mode="create_savepoint",
            ) as session:
                source = Source(
                    name="docling-adapter-test",
                    source_type="test",
                )
                session.add(source)
                session.flush()
                document = Document(
                    source_id=source.id,
                    external_id="structured.pdf",
                )
                session.add(document)
                session.flush()
                version = DocumentVersion(
                    document_id=document.id,
                    version_number=1,
                    filename="structured.pdf",
                    file_hash="docling-adapter-test",
                    storage_uri="memory://structured.pdf",
                    page_count=1,
                )
                session.add(version)
                session.flush()
                historical_run = ProcessingRun(
                    document_version_id=version.id,
                    process_type="document_extraction",
                    status="completed",
                    engine="kaliok-reader",
                    completed_at=utc_now(),
                )
                session.add(historical_run)
                session.flush()
                page = Page(
                    document_version_id=version.id,
                    page_number=1,
                    perception_processing_run_id=historical_run.id,
                )
                session.add(page)
                session.flush()

                run = store_docling_document(
                    session,
                    version,
                    _docling_document(),
                    engine_version="test",
                )
                session.commit()

                stored_page = session.get(Page, page.id)
                blocks = session.exec(
                    select(ContentBlock)
                    .where(ContentBlock.processing_run_id == run.id)
                    .order_by(ContentBlock.block_index)
                ).all()
                blocks_by_ref = {
                    block.extra_data["docling_self_ref"]: block
                    for block in blocks
                }

                assert run.status == "completed"
                assert run.engine == "docling"
                assert stored_page is not None
                assert (
                    stored_page.perception_processing_run_id
                    == historical_run.id
                )
                assert session.get(ProcessingRun, historical_run.id) is not None

                group = blocks_by_ref["#/groups/0"]
                heading = blocks_by_ref["#/texts/0"]
                header = blocks_by_ref["#/texts/2"]
                table = blocks_by_ref["#/tables/0"]

                assert heading.block_type == "section_header"
                assert heading.parent_block_id == group.id
                assert heading.coordinate_system == "BOTTOMLEFT"
                assert len(heading.extra_data["provenances"]) == 2
                assert header.extra_data["indexable"] is False
                assert header.extra_data["content_layer"] == "furniture"
                assert table.block_type == "table"
                assert table.extra_data["table"]["num_cols"] == 2
                assert not any(
                    block.block_type == "table_cell" for block in blocks
                )
        finally:
            transaction.rollback()


def test_second_identical_docling_call_is_idempotent(docling_session):
    version, page = _create_version(docling_session)

    first_run = store_docling_document(
        docling_session,
        version,
        _docling_document(),
        engine_version="test",
    )
    first_block_count = docling_session.exec(
        select(func.count(ContentBlock.id)).where(
            ContentBlock.processing_run_id == first_run.id
        )
    ).one()
    second_run = store_docling_document(
        docling_session,
        version,
        deepcopy(_docling_document()),
        engine_version="test",
    )

    assert second_run.id == first_run.id
    assert docling_session.exec(
        select(func.count(ProcessingRun.id)).where(
            ProcessingRun.document_version_id == version.id,
            ProcessingRun.engine == "docling",
        )
    ).one() == 1
    assert docling_session.exec(
        select(func.count(ContentBlock.id)).where(
            ContentBlock.processing_run_id == first_run.id
        )
    ).one() == first_block_count
    assert docling_session.exec(
        select(func.count(Page.id)).where(
            Page.document_version_id == version.id
        )
    ).one() == 1
    assert page is not None
    assert page.perception_processing_run_id is None


def test_two_identical_pdf_conversions_reuse_the_same_run(
    docling_session,
    monkeypatch,
):
    version, _ = _create_version(docling_session)
    network_calls = 0

    def fake_convert(*args, **kwargs):
        nonlocal network_calls
        network_calls += 1
        document = deepcopy(_docling_document())
        document["version"] = "1.10.0"
        return document

    monkeypatch.setattr(docling, "convert_pdf_with_docling", fake_convert)

    first_run = docling.store_pdf_with_docling(
        docling_session,
        version,
        "structured.pdf",
    )
    second_run = docling.store_pdf_with_docling(
        docling_session,
        version,
        "structured.pdf",
    )

    assert network_calls == 2
    assert second_run.id == first_run.id
    assert first_run.engine_version == "1.10.0"
    assert docling_session.exec(
        select(func.count(ProcessingRun.id)).where(
            ProcessingRun.document_version_id == version.id,
            ProcessingRun.engine == "docling",
        )
    ).one() == 1


def test_different_docling_json_creates_a_new_generation(docling_session):
    version, _ = _create_version(docling_session)
    first_document = _docling_document()
    second_document = deepcopy(first_document)
    second_document["texts"][0]["text"] = "Conditions particulières"

    first_run = store_docling_document(
        docling_session,
        version,
        first_document,
        engine_version="test",
    )
    second_run = store_docling_document(
        docling_session,
        version,
        second_document,
        engine_version="test",
    )

    assert second_run.id != first_run.id
    assert docling_session.exec(
        select(func.count(ProcessingRun.id)).where(
            ProcessingRun.document_version_id == version.id,
            ProcessingRun.engine == "docling",
        )
    ).one() == 2


@pytest.mark.parametrize("page_number", [0, 2])
def test_docling_page_number_outside_version_is_rejected(
    docling_session,
    page_number,
):
    version, _ = _create_version(docling_session)
    document = _docling_document()
    document["texts"][0]["prov"][0]["page_no"] = page_number

    with pytest.raises(ValueError, match="hors limites"):
        store_docling_document(
            docling_session,
            version,
            document,
            engine_version="test",
        )

    assert docling_session.exec(
        select(func.count(ProcessingRun.id)).where(
            ProcessingRun.document_version_id == version.id,
            ProcessingRun.engine == "docling",
        )
    ).one() == 0


def test_docling_partial_failure_is_rolled_back(
    docling_session,
    monkeypatch,
):
    version, _ = _create_version(docling_session, create_page=False)
    original_content_block = docling.ContentBlock
    calls = 0

    def failing_content_block(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("synthetic block failure")
        return original_content_block(*args, **kwargs)

    monkeypatch.setattr(docling, "ContentBlock", failing_content_block)
    with pytest.raises(RuntimeError, match="synthetic block failure"):
        store_docling_document(
            docling_session,
            version,
            _docling_document(),
            engine_version="test",
        )
    monkeypatch.setattr(docling, "ContentBlock", original_content_block)

    docling_session.commit()
    assert docling_session.exec(
        select(func.count(ProcessingRun.id)).where(
            ProcessingRun.document_version_id == version.id,
            ProcessingRun.engine == "docling",
        )
    ).one() == 0
    assert docling_session.exec(
        select(func.count(Page.id)).where(
            Page.document_version_id == version.id
        )
    ).one() == 0


def test_docling_does_not_touch_existing_search_index(docling_session):
    version, page = _create_version(docling_session)
    assert page is not None
    historical_run = ProcessingRun(
        document_version_id=version.id,
        process_type="document_extraction",
        status="completed",
        engine="kaliok-reader",
        completed_at=utc_now(),
    )
    docling_session.add(historical_run)
    docling_session.flush()
    page.perception_processing_run_id = historical_run.id
    historical_block = ContentBlock(
        page_id=page.id,
        processing_run_id=historical_run.id,
        block_index=0,
        content="Contenu historique",
        extraction_method="native",
    )
    chunk = DocumentChunk(
        document_version_id=version.id,
        chunk_index=0,
        content="Contenu historique",
        char_count=19,
        chunking_strategy="semantic",
    )
    embedding_model = EmbeddingModel(
        provider="test",
        model_name="docling-test",
        dimensions=1024,
    )
    docling_session.add(historical_block)
    docling_session.add(chunk)
    docling_session.add(embedding_model)
    docling_session.flush()
    link = ChunkContentBlock(
        chunk_id=chunk.id,
        content_block_id=historical_block.id,
        block_order=0,
    )
    embedding = ChunkEmbedding(
        chunk_id=chunk.id,
        embedding_model_id=embedding_model.id,
        embedding=[0.0] * 1024,
    )
    docling_session.add(link)
    docling_session.add(embedding)
    docling_session.flush()

    store_docling_document(
        docling_session,
        version,
        _docling_document(),
        engine_version="test",
    )

    assert page.perception_processing_run_id == historical_run.id
    assert docling_session.get(DocumentChunk, chunk.id) is not None
    assert docling_session.get(
        ChunkEmbedding,
        (chunk.id, embedding_model.id),
    ) is not None
    assert docling_session.get(
        ChunkContentBlock,
        (chunk.id, historical_block.id),
    ) is not None
