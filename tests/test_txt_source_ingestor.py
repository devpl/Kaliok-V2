from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from kaliok.ingestion import (
    DetectedSource,
    IngestionOrchestrator,
    IngestionRequest,
    SourceIngestorSelector,
    SourceReference,
)
from kaliok.ingestion.ingestors import (
    PLAIN_TEXT_MEDIA_TYPE,
    PLAIN_TEXT_SOURCE_TYPE,
    TxtDecodingError,
    TxtSourceIngestor,
)
from kaliok.ingestion.stores import PostgresDocumentStore
from test_postgres_document_store import FakeSession


def make_source(path: Path) -> tuple[IngestionRequest, DetectedSource]:
    reference = SourceReference(
        name=path.name,
        uri=str(path),
        media_type=PLAIN_TEXT_MEDIA_TYPE,
        size=path.stat().st_size,
        external_id="text-1",
    )
    request = IngestionRequest(source=reference)
    detected = DetectedSource(
        source=reference,
        source_type=PLAIN_TEXT_SOURCE_TYPE,
        media_type=PLAIN_TEXT_MEDIA_TYPE,
        confidence=1.0,
    )
    return request, detected


def test_supports_only_detected_plain_text(tmp_path):
    path = tmp_path / "source.txt"
    path.write_text("Texte", encoding="utf-8")
    request, detected = make_source(path)
    ingestor = TxtSourceIngestor()

    assert ingestor.supports(detected) is True
    assert ingestor.supports(
        DetectedSource(
            request.source,
            source_type="other",
            media_type=PLAIN_TEXT_MEDIA_TYPE,
        )
    ) is False
    assert ingestor.supports(
        DetectedSource(
            request.source,
            source_type=PLAIN_TEXT_SOURCE_TYPE,
            media_type="application/example",
        )
    ) is False


def test_reads_utf8_and_preserves_ordered_paragraphs(tmp_path):
    path = tmp_path / "accents.txt"
    path.write_text(
        "Premier paragraphe.\nSuite du premier.\n\nDeuxième paragraphe.\n\n\n"
        "Troisième paragraphe.",
        encoding="utf-8",
    )
    request, detected = make_source(path)

    normalized = TxtSourceIngestor().ingest(request, detected)

    assert [unit.order for unit in normalized.units] == [0, 1, 2]
    assert [unit.content_type for unit in normalized.units] == [
        "paragraph",
        "paragraph",
        "paragraph",
    ]
    assert [unit.content for unit in normalized.units] == [
        "Premier paragraphe.\nSuite du premier.",
        "Deuxième paragraphe.",
        "Troisième paragraphe.",
    ]
    assert [unit.source_unit_id for unit in normalized.units] == [
        "paragraph-0",
        "paragraph-1",
        "paragraph-2",
    ]


def test_reads_utf8_bom_without_preserving_bom(tmp_path):
    path = tmp_path / "bom.txt"
    path.write_bytes("Contenu avec BOM".encode("utf-8-sig"))
    request, detected = make_source(path)

    normalized = TxtSourceIngestor().ingest(request, detected)

    assert normalized.units[0].content == "Contenu avec BOM"
    assert not normalized.units[0].content.startswith("\ufeff")


def test_reads_local_file_uri(tmp_path):
    path = tmp_path / "file-uri.txt"
    path.write_text("Source via URI locale", encoding="utf-8")
    reference = SourceReference(
        name=path.name,
        uri=path.resolve().as_uri(),
        media_type=PLAIN_TEXT_MEDIA_TYPE,
    )
    request = IngestionRequest(source=reference)
    detected = DetectedSource(
        reference,
        source_type=PLAIN_TEXT_SOURCE_TYPE,
        media_type=PLAIN_TEXT_MEDIA_TYPE,
    )

    normalized = TxtSourceIngestor().ingest(request, detected)

    assert normalized.units[0].content == "Source via URI locale"
    assert normalized.storage_uri == path.resolve().as_uri()


def test_invalid_utf8_raises_explicit_error(tmp_path):
    path = tmp_path / "invalid.txt"
    path.write_bytes(b"texte \xff invalide")
    request, detected = make_source(path)

    with pytest.raises(TxtDecodingError, match="UTF-8"):
        TxtSourceIngestor().ingest(request, detected)


def test_normalized_metadata_and_hash_are_source_based(tmp_path):
    path = tmp_path / "metadata.txt"
    raw_content = "Métadonnées déterministes".encode("utf-8")
    path.write_bytes(raw_content)
    request, detected = make_source(path)
    ingestor = TxtSourceIngestor()

    first = ingestor.ingest(request, detected)
    second = ingestor.ingest(request, detected)

    expected_hash = hashlib.sha256(raw_content).hexdigest()
    assert first.content_hash == expected_hash == second.content_hash
    assert first.filename == "metadata.txt"
    assert first.storage_uri == path.resolve().as_uri()
    assert first.file_size == len(raw_content)
    assert first.mime_type == "text/plain"
    assert first.title == "metadata"
    assert first.document_family is None
    assert first.document_type == "text"
    assert first.document_subtype is None
    assert first.page_count is None


class PlainTextDetector:
    def detect(self, source):
        return DetectedSource(
            source,
            source_type=PLAIN_TEXT_SOURCE_TYPE,
            media_type=PLAIN_TEXT_MEDIA_TYPE,
        )


class RecordingStore:
    def __init__(self):
        self.document = None

    def store(self, request, document):
        self.document = document
        return type(
            "Result",
            (),
            {
                "document_id": "document-1",
                "document_version_id": "version-1",
                "processing_run_id": None,
                "detected_source": document.source,
                "status": "created",
            },
        )()


def test_txt_ingestor_integrates_with_ingestion_orchestrator(tmp_path):
    path = tmp_path / "orchestrator.txt"
    path.write_text("Premier.\n\nSecond.", encoding="utf-8")
    request, _ = make_source(path)
    store = RecordingStore()
    orchestrator = IngestionOrchestrator(
        detector=PlainTextDetector(),
        ingestor_selector=SourceIngestorSelector([TxtSourceIngestor()]),
        document_store=store,
    )

    result = orchestrator.ingest(request)

    assert result.status == "created"
    assert [unit.content for unit in store.document.units] == [
        "Premier.",
        "Second.",
    ]


def test_full_txt_chain_persists_without_pages_or_content_blocks(tmp_path):
    path = tmp_path / "stored.txt"
    path.write_text("Un.\n\nDeux.", encoding="utf-8")
    request, _ = make_source(path)
    session = FakeSession()
    orchestrator = IngestionOrchestrator(
        detector=PlainTextDetector(),
        ingestor_selector=SourceIngestorSelector([TxtSourceIngestor()]),
        document_store=PostgresDocumentStore(session),
    )

    result = orchestrator.ingest(request)

    assert result.status == "created"
    assert len(session.documents) == 1
    assert len(session.versions) == 1
    assert [unit.content for unit in session.units] == ["Un.", "Deux."]
    assert not hasattr(session, "pages")
    assert not hasattr(session, "content_blocks")


def test_txt_ingestor_has_no_rag_or_disallowed_format_dependencies():
    path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "kaliok"
        / "ingestion"
        / "ingestors"
        / "txt.py"
    )
    source = path.read_text(encoding="utf-8").lower()

    assert "kaliok.rag" not in source
    for forbidden in ("pdf", "docx", "html", "mail", "ocr", "docling"):
        assert forbidden not in source
