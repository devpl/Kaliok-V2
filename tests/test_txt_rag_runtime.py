from __future__ import annotations

from uuid import uuid4

from kaliok.embeddings.ollama import EMBEDDING_MODEL
from kaliok.embeddings.service import SimilarChunk
from kaliok.rag import EmbeddingRecord
from kaliok.rag_runtime import (
    NormalizedContentProvider,
    NormalizedContentReference,
    NormalizedContentRepresentationBuilder,
    OllamaGenerator,
    PostgresVectorIndexStore,
    PostgresVectorRetriever,
    RankedContextBuilder,
)
from kaliok.rag_runtime.postgres import (
    NORMALIZED_CHUNKING_STRATEGY,
    NORMALIZED_CHUNKING_VERSION,
)
from kaliok.storage.models import (
    ChunkEmbedding,
    Document,
    DocumentChunk,
    DocumentVersion,
    EmbeddingModel,
    NormalizedContentUnit,
)


class Result:
    def __init__(self, values):
        self.values = values

    def all(self):
        return self.values

    def first(self):
        return self.values[0] if self.values else None

    def one_or_none(self):
        if len(self.values) > 1:
            raise RuntimeError("plusieurs résultats")
        return self.first()


class ProviderSession:
    def __init__(self, document, version, units):
        self.document = document
        self.version = version
        self.units = units

    def get(self, model, identity):
        if model is DocumentVersion and identity == self.version.id:
            return self.version
        if model is Document and identity == self.document.id:
            return self.document
        return None

    def exec(self, statement):
        return Result(self.units)


def make_stored_content():
    document = Document(title="notes")
    version = DocumentVersion(
        document_id=document.id,
        version_number=2,
        filename="notes.txt",
        file_hash="hash",
        storage_uri="file:///notes.txt",
        is_current=True,
    )
    units = [
        NormalizedContentUnit(
            document_version_id=version.id,
            unit_index=index,
            content_type="paragraph",
            content=text,
            source_unit_id=f"paragraph-{index}",
        )
        for index, text in enumerate(("Premier paragraphe.", "Second."))
    ]
    return document, version, units


def test_provider_and_representation_preserve_order_and_provenance():
    document, version, units = make_stored_content()
    provider = NormalizedContentProvider(
        ProviderSession(document, version, units)
    )

    extracted = provider.provide(
        NormalizedContentReference(document_version_id=version.id)
    )
    represented = NormalizedContentRepresentationBuilder().build(extracted)

    assert [unit.text for unit in represented] == [
        "Premier paragraphe.",
        "Second.",
    ]
    assert [unit.unit_id for unit in represented] == [unit.id for unit in units]
    assert represented[0].provenance.document_id == document.id
    assert represented[0].provenance.document_version_id == version.id
    assert represented[0].provenance.page is None
    assert represented[0].provenance.source_ids == (units[0].id,)
    source_uuid = represented[0].provenance.metadata[
        "normalized_content_unit_id"
    ]
    assert source_uuid == units[0].id
    assert isinstance(source_uuid, type(units[0].id))
    assert not isinstance(source_uuid, tuple)
    assert represented[0].provenance.metadata["source_unit_id"] == "paragraph-0"


def test_provider_selects_current_version_from_document_id():
    document, version, units = make_stored_content()

    class CurrentVersionSession(ProviderSession):
        def __init__(self):
            super().__init__(document, version, units)
            self.exec_count = 0

        def exec(self, statement):
            self.exec_count += 1
            return Result([version] if self.exec_count == 1 else units)

    extracted = NormalizedContentProvider(CurrentVersionSession()).provide(
        NormalizedContentReference(document_id=document.id)
    )

    assert extracted.provenance.document_id == document.id
    assert extracted.provenance.document_version_id == version.id


class MemorySession:
    def __init__(self, source_unit, version, model):
        self.source_unit = source_unit
        self.version = version
        self.model = model
        self.objects = {}
        self.added = []

    def get(self, model, identity):
        if model is NormalizedContentUnit and identity == self.source_unit.id:
            return self.source_unit
        if model is DocumentVersion and identity == self.version.id:
            return self.version
        if model is DocumentChunk:
            return self.objects.get((DocumentChunk, identity))
        if model is ChunkEmbedding:
            return self.objects.get((ChunkEmbedding, identity))
        return None

    def add(self, value):
        self.added.append(value)
        if isinstance(value, DocumentChunk):
            self.objects[(DocumentChunk, value.id)] = value
        if isinstance(value, ChunkEmbedding):
            self.objects[
                (ChunkEmbedding, (value.chunk_id, value.embedding_model_id))
            ] = value

    def flush(self):
        return None

    def exec(self, statement):
        return Result([self.model])


def test_index_store_is_idempotent_and_retriever_restores_provenance(
    monkeypatch,
):
    document_id = uuid4()
    version = DocumentVersion(
        document_id=document_id,
        version_number=1,
        filename="notes.txt",
        file_hash="hash",
        storage_uri="file:///notes.txt",
    )
    source_unit = NormalizedContentUnit(
        document_version_id=version.id,
        unit_index=0,
        content_type="paragraph",
        content="Une information précise.",
        source_unit_id="paragraph-0",
    )
    model = EmbeddingModel(
        provider="ollama",
        model_name=EMBEDDING_MODEL,
        dimensions=1024,
        is_active=True,
    )
    session = MemorySession(source_unit, version, model)
    monkeypatch.setattr(
        "kaliok.rag_runtime.postgres._get_or_create_embedding_model",
        lambda unused_session: model,
    )
    extracted = NormalizedContentProvider(
        ProviderSession(Document(id=document_id), version, [source_unit])
    ).provide(NormalizedContentReference(document_version_id=version.id))
    unit = NormalizedContentRepresentationBuilder().build(extracted)[0]
    record = EmbeddingRecord(unit, [0.1] * 1024, EMBEDDING_MODEL)
    store = PostgresVectorIndexStore(session, model_name=EMBEDDING_MODEL)

    store.write([record])
    store.write([record])

    assert len([item for item in session.added if isinstance(item, DocumentChunk)]) == 1
    assert len([item for item in session.added if isinstance(item, ChunkEmbedding)]) == 1

    def search(**kwargs):
        assert kwargs["document_version_id"] == version.id
        return [SimilarChunk(source_unit.id, source_unit.content, 0.125)]

    candidates = PostgresVectorRetriever(
        session,
        version.id,
        model_name=EMBEDDING_MODEL,
        search=search,
    ).retrieve([0.1] * 1024, top_k=5)

    assert candidates[0].score == 0.875
    assert candidates[0].unit.provenance.document_id == document_id
    assert candidates[0].unit.provenance.page is None
    source_uuid = candidates[0].unit.provenance.metadata[
        "normalized_content_unit_id"
    ]
    assert source_uuid == source_unit.id
    assert not isinstance(source_uuid, tuple)
    assert candidates[0].unit.provenance.metadata["source_unit_id"] == "paragraph-0"


def test_retriever_ignores_chunks_outside_normalized_strategy_or_version():
    document_id = uuid4()
    version = DocumentVersion(
        document_id=document_id,
        version_number=1,
        filename="notes.txt",
        file_hash="hash",
        storage_uri="file:///notes.txt",
    )
    source_unit = NormalizedContentUnit(
        document_version_id=version.id,
        unit_index=0,
        content_type="paragraph",
        content="Contenu valide.",
        source_unit_id="paragraph-0",
    )
    model = EmbeddingModel(
        provider="ollama",
        model_name=EMBEDDING_MODEL,
        dimensions=1024,
        is_active=True,
    )
    session = MemorySession(source_unit, version, model)
    valid_chunk = DocumentChunk(
        id=source_unit.id,
        document_version_id=version.id,
        chunk_index=0,
        content=source_unit.content,
        char_count=len(source_unit.content),
        chunking_strategy=NORMALIZED_CHUNKING_STRATEGY,
        chunking_version=NORMALIZED_CHUNKING_VERSION,
    )
    wrong_strategy = DocumentChunk(
        document_version_id=version.id,
        chunk_index=1,
        content="Autre stratégie.",
        char_count=16,
        chunking_strategy="other-strategy",
        chunking_version=NORMALIZED_CHUNKING_VERSION,
    )
    wrong_version = DocumentChunk(
        document_version_id=uuid4(),
        chunk_index=2,
        content="Autre version.",
        char_count=14,
        chunking_strategy=NORMALIZED_CHUNKING_STRATEGY,
        chunking_version=NORMALIZED_CHUNKING_VERSION,
    )
    for chunk in (valid_chunk, wrong_strategy, wrong_version):
        session.add(chunk)

    def search(**kwargs):
        return [
            SimilarChunk(wrong_strategy.id, wrong_strategy.content, 0.01),
            SimilarChunk(wrong_version.id, wrong_version.content, 0.02),
            SimilarChunk(valid_chunk.id, valid_chunk.content, 0.10),
        ]

    candidates = PostgresVectorRetriever(
        session,
        version.id,
        model_name=EMBEDDING_MODEL,
        search=search,
    ).retrieve([0.1] * 1024, top_k=5)

    assert len(candidates) == 1
    assert candidates[0].unit.unit_id == valid_chunk.id


class FakeResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {"response": "Réponse avec source paragraph-0."}


def test_context_and_generator_use_candidates_without_network(monkeypatch):
    document, version, units = make_stored_content()
    extracted = NormalizedContentProvider(
        ProviderSession(document, version, units)
    ).provide(NormalizedContentReference(document_version_id=version.id))
    unit = NormalizedContentRepresentationBuilder().build(extracted)[0]
    from kaliok.rag import Candidate, RankedCandidate

    ranked = RankedCandidate(Candidate(unit, score=0.9), rank=1, score=0.9)
    context = RankedContextBuilder().build("Question ?", [ranked])
    captured = {}

    def post(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return FakeResponse()

    monkeypatch.setattr("kaliok.rag_runtime.ollama.requests.post", post)

    answer = OllamaGenerator(
        model="mistral", base_url="http://ollama.local"
    ).generate("Question ?", context)

    assert "source_unit_id=paragraph-0" in context.text
    assert answer.context is context
    assert answer.metadata["model"] == "mistral"
    assert captured["url"] == "http://ollama.local/api/generate"
    assert captured["json"]["options"]["temperature"] == 0
