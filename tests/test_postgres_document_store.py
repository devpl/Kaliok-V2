from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import pytest

from kaliok.ingestion import (
    DetectedSource,
    IngestionOrchestrator,
    IngestionRequest,
    NormalizedContentUnit,
    NormalizedDocument,
    SourceIngestorSelector,
    SourceReference,
)
from kaliok.ingestion.stores import PostgresDocumentStore
from kaliok.storage.models import (
    Document,
    DocumentVersion,
    NormalizedContentUnit as StoredContentUnit,
    Source,
)


class FakeResult:
    def __init__(self, values):
        self.values = list(values)

    def first(self):
        return self.values[0] if self.values else None

    def one_or_none(self):
        if len(self.values) > 1:
            raise AssertionError("Plus d'un résultat inattendu.")
        return self.first()

    def all(self):
        return list(self.values)


class FakeSavepoint(AbstractContextManager):
    def __init__(self, session):
        self.session = session
        self.documents = list(session.documents)
        self.versions = list(session.versions)
        self.units = list(session.units)
        self.added = list(session.added)
        self.current_states = {
            version.id: version.is_current for version in session.versions
        }

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            self.session.documents = self.documents
            self.session.versions = self.versions
            self.session.units = self.units
            self.session.added = self.added
            for version in self.session.versions:
                version.is_current = self.current_states[version.id]
        return False


class FakeSession:
    def __init__(
        self,
        *,
        sources=None,
        documents=None,
        versions=None,
        units=None,
    ):
        self.sources = {source.id: source for source in (sources or [])}
        self.documents = list(documents or [])
        self.versions = list(versions or [])
        self.units = list(units or [])
        self.added = []
        self.flush_count = 0
        self.fail_on_flush = None
        self.savepoint_count = 0

    def begin_nested(self):
        self.savepoint_count += 1
        return FakeSavepoint(self)

    def get(self, model, identifier):
        assert model is Source
        return self.sources.get(identifier)

    def exec(self, statement):
        entity = statement.column_descriptions[0]["entity"]
        if entity is Document:
            return FakeResult(self.documents[:1])
        if entity is DocumentVersion:
            if len(statement._where_criteria) > 1:
                parameters = statement.compile().params.values()
                content_hash = next(
                    value for value in parameters if isinstance(value, str)
                )
                return FakeResult(
                    [
                        version
                        for version in self.versions
                        if version.file_hash == content_hash
                    ]
                )
            return FakeResult(self.versions)
        if entity is StoredContentUnit:
            return FakeResult(
                sorted(self.units, key=lambda unit: unit.unit_index)
            )
        raise AssertionError(f"Requête inattendue pour {entity}.")

    def add(self, value):
        self.added.append(value)
        if isinstance(value, Document) and value not in self.documents:
            self.documents.append(value)
        if isinstance(value, DocumentVersion) and value not in self.versions:
            self.versions.append(value)
        if isinstance(value, StoredContentUnit) and value not in self.units:
            self.units.append(value)

    def flush(self):
        self.flush_count += 1
        if self.flush_count == self.fail_on_flush:
            raise RuntimeError("échec de persistance")


def make_normalized(*, content_hash="hash-1"):
    reference = SourceReference(
        name="source normalisée",
        uri="storage://sources/item-1",
        media_type="application/example",
        size=321,
        external_id="external-1",
    )
    detected = DetectedSource(
        source=reference,
        source_type="generic",
        media_type=reference.media_type,
    )
    return NormalizedDocument(
        source=detected,
        units=(
            NormalizedContentUnit(
                order=0,
                content_type="section",
                content="Introduction",
                source_reference="source://root",
                source_unit_id="unit-0",
            ),
            NormalizedContentUnit(
                order=1,
                content_type="text",
                content="Contenu normalisé",
                source_reference="source://body/1",
                source_unit_id="unit-1",
                parent_source_unit_id="unit-0",
            ),
        ),
        filename="source.bin",
        storage_uri=reference.uri,
        content_hash=content_hash,
        file_size=reference.size,
        mime_type=reference.media_type,
        title="Titre normalisé",
        language="fr",
        page_count=3,
        document_family="generic",
        document_type="structured",
    )


def test_store_creates_document_and_document_version():
    session = FakeSession()
    normalized = make_normalized()
    request = IngestionRequest(source=normalized.source.source)

    result = PostgresDocumentStore(session).store(request, normalized)

    assert result.status == "created"
    assert result.processing_run_id is None
    assert len(session.documents) == 1
    assert len(session.versions) == 1
    assert len(session.units) == 2
    document = session.documents[0]
    version = session.versions[0]
    assert result.document_id == document.id
    assert result.document_version_id == version.id
    assert document.external_id == "external-1"
    assert document.title == "Titre normalisé"
    assert version.document_id == document.id
    assert version.version_number == 1
    assert version.file_hash == "hash-1"
    assert version.storage_uri == "storage://sources/item-1"
    assert version.processing_status == "pending"
    assert version.is_current is True
    assert [unit.unit_index for unit in session.units] == [0, 1]
    assert all(
        unit.document_version_id == version.id for unit in session.units
    )
    assert session.units[1].parent_unit_id == session.units[0].id
    assert session.units[1].source_reference == "source://body/1"
    assert session.savepoint_count == 1


def test_explicit_document_and_hash_reuse_existing_version_without_writes():
    session = FakeSession()
    normalized = make_normalized()
    first = PostgresDocumentStore(session).store(
        IngestionRequest(source=normalized.source.source),
        normalized,
    )
    added_count = len(session.added)
    request = IngestionRequest(
        source=normalized.source.source,
        document_id=first.document_id,
    )

    result = PostgresDocumentStore(session).store(request, normalized)

    assert result.status == "already_exists"
    assert result.document_id == first.document_id
    assert result.document_version_id == first.document_version_id
    assert len(session.added) == added_count
    assert len(session.documents) == 1
    assert len(session.versions) == 1
    assert len(session.units) == 2


def test_explicit_document_with_new_hash_creates_next_version():
    document = Document(id=uuid4(), title="Existant")
    first = DocumentVersion(
        id=uuid4(),
        document_id=document.id,
        version_number=1,
        filename="source.bin",
        file_hash="hash-1",
        storage_uri="storage://sources/item-1",
        is_current=True,
    )
    session = FakeSession(documents=[document], versions=[first])
    normalized = make_normalized(content_hash="hash-2")
    request = IngestionRequest(
        source=normalized.source.source,
        document_id=document.id,
    )

    result = PostgresDocumentStore(session).store(request, normalized)

    assert result.status == "created"
    assert len(session.documents) == 1
    assert len(session.versions) == 2
    assert first.is_current is False
    assert session.versions[-1].version_number == 2
    assert session.versions[-1].is_current is True


def test_store_respects_existing_source_identifier():
    source = Source(id=uuid4(), name="Source", source_type="generic")
    session = FakeSession(sources=[source])
    normalized = make_normalized()
    request = IngestionRequest(
        source=normalized.source.source,
        source_id=source.id,
    )

    result = PostgresDocumentStore(session).store(request, normalized)

    assert result.status == "created"
    assert session.documents[0].source_id == source.id


def test_savepoint_rolls_back_document_and_version_on_failure():
    session = FakeSession()
    session.fail_on_flush = 3
    normalized = make_normalized()
    request = IngestionRequest(source=normalized.source.source)

    with pytest.raises(RuntimeError, match="échec de persistance"):
        PostgresDocumentStore(session).store(request, normalized)

    assert session.documents == []
    assert session.versions == []
    assert session.units == []
    assert session.added == []
    assert session.savepoint_count == 1


class StaticDetector:
    def detect(self, source):
        return DetectedSource(source, source_type="generic")


class StaticIngestor:
    def __init__(self, normalized):
        self.normalized = normalized

    def supports(self, source):
        return True

    def ingest(self, request, source):
        return replace(self.normalized, source=source)


def test_store_is_compatible_with_ingestion_orchestrator():
    session = FakeSession()
    normalized = make_normalized()
    orchestrator = IngestionOrchestrator(
        detector=StaticDetector(),
        ingestor_selector=SourceIngestorSelector(
            [StaticIngestor(normalized)]
        ),
        document_store=PostgresDocumentStore(session),
    )

    result = orchestrator.ingest(
        IngestionRequest(source=normalized.source.source)
    )

    assert result.status == "created"
    assert len(session.documents) == 1
    assert len(session.versions) == 1


def test_postgres_store_has_no_rag_or_format_specific_dependencies():
    path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "kaliok"
        / "ingestion"
        / "stores"
        / "postgres.py"
    )
    source = path.read_text(encoding="utf-8").lower()

    assert "kaliok.rag" not in source
    for forbidden in ("pdf", "mail", "docx", "html", "ocr", "docling"):
        assert forbidden not in source
