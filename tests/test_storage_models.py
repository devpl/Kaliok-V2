from kaliok.storage.models import Document, DocumentVersion, Page, Source

from kaliok.storage.models import (
    ChunkContentBlock,
    ContentBlock,
    Document,
    DocumentChunk,
    DocumentVersion,
    EmbeddingModel,
    Page,
    ProcessingRun,
    Source,
    ChunkEmbedding,
)

def test_source_creation():
    source = Source(
        name="Upload manuel",
        source_type="upload",
    )

    assert source.name == "Upload manuel"
    assert source.source_type == "upload"


def test_document_creation():
    document = Document(
        title="Contrat de prestation",
        document_family="contractual",
    )

    assert document.title == "Contrat de prestation"
    assert document.document_family == "contractual"


def test_document_version_creation():
    version = DocumentVersion(
        document_id="00000000-0000-0000-0000-000000000001",
        version_number=1,
        filename="contrat.docx",
        file_hash="abc123",
        storage_uri="file://contrat.docx",
        document_type="contrat",
        version_status="draft",
    )

    assert version.version_number == 1
    assert version.filename == "contrat.docx"
    assert version.document_type == "contrat"
    assert version.version_status == "draft"

def test_page_creation():
    page = Page(
        document_version_id="00000000-0000-0000-0000-000000000001",
        page_number=3,
        page_status="active",
        readability_status="partially_readable",
        readability_reason="scan_too_dark",
        ocr_required=True,
        ocr_engine="rapidocr",
        ocr_confidence_mean=0.61,
    )

    assert page.page_number == 3
    assert page.page_status == "active"
    assert page.readability_status == "partially_readable"
    assert page.ocr_required is True
    assert page.ocr_engine == "rapidocr"

def test_content_block_creation():
    block = ContentBlock(
        page_id="00000000-0000-0000-0000-000000000001",
        block_index=0,
        reading_order=0,
        block_type="paragraph",
        content="Le présent contrat prend effet le 5 août 2026.",
        extraction_method="ocr",
        extraction_engine="rapidocr",
        confidence=0.94,
        bbox={
            "x0": 0.10,
            "y0": 0.20,
            "x1": 0.85,
            "y1": 0.28,
        },
    )

    assert block.block_type == "paragraph"
    assert block.extraction_method == "ocr"
    assert block.extraction_engine == "rapidocr"
    assert block.confidence == 0.94
    assert block.bbox["x0"] == 0.10

def test_document_chunk_creation():
    chunk = DocumentChunk(
        document_version_id="00000000-0000-0000-0000-000000000001",
        chunk_index=0,
        content="Le présent contrat prend effet le 5 août 2026.",
        char_count=49,
        token_count=11,
        page_start=1,
        page_end=1,
        breadcrumb=[
            "Contrat",
            "Durée",
        ],
        chunking_strategy="semantic",
        chunking_version="1",
    )

    assert chunk.chunk_index == 0
    assert chunk.page_start == 1
    assert chunk.page_end == 1
    assert chunk.breadcrumb == ["Contrat", "Durée"]
    assert chunk.chunking_strategy == "semantic"


def test_chunk_content_block_creation():
    link = ChunkContentBlock(
        chunk_id="00000000-0000-0000-0000-000000000001",
        content_block_id="00000000-0000-0000-0000-000000000002",
        block_order=0,
    )

    assert link.block_order == 0

def test_processing_run_creation():
    run = ProcessingRun(
        document_version_id="00000000-0000-0000-0000-000000000001",
        process_type="ocr",
        status="success",
        engine="rapidocr",
        configuration={
            "scale": 1.0,
        },
        metrics={
            "pages": 4,
            "duration_seconds": 8.5,
        },
    )

    assert run.process_type == "ocr"
    assert run.status == "success"
    assert run.engine == "rapidocr"
    assert run.configuration["scale"] == 1.0
    assert run.metrics["pages"] == 4

def test_embedding_model_creation():
    model = EmbeddingModel(
        provider="ollama",
        model_name="test-model",
        dimensions=768,
        distance_metric="cosine",
    )

    assert model.provider == "ollama"
    assert model.model_name == "test-model"
    assert model.dimensions == 768
    assert model.distance_metric == "cosine"

def test_chunk_embedding_with_vector():
    embedding = ChunkEmbedding(
        chunk_id="00000000-0000-0000-0000-000000000001",
        embedding_model_id="00000000-0000-0000-0000-000000000002",
        embedding=[0.0] * 1024,
    )

    assert len(embedding.embedding) == 1024
    assert embedding.embedding[0] == 0.0

