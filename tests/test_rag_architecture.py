from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from kaliok.rag import (
    Candidate,
    ContextBundle,
    EmbeddingRecord,
    ExtractedDocument,
    Provenance,
    RagAnswer,
    RagOrchestrator,
    RagPipelineConfig,
    RankedCandidate,
    RetrievalUnit,
)
from kaliok.rag.context import ContextBuilder
from kaliok.rag.embedding import Embedder
from kaliok.rag.extraction import Extractor
from kaliok.rag.fusion import FusionStrategy
from kaliok.rag.generation import Generator
from kaliok.rag.indexing import IndexStore
from kaliok.rag.representation import RepresentationBuilder
from kaliok.rag.reranking import Reranker
from kaliok.rag.retrieval import Retriever


class FakeExtractor:
    def __init__(self, calls, provenance):
        self.calls = calls
        self.provenance = provenance

    def extract(self, document):
        self.calls.append(("extract", document))
        return ExtractedDocument("contenu", self.provenance)


class FakeRepresentationBuilder:
    def __init__(self, calls):
        self.calls = calls

    def build(self, document):
        self.calls.append(("represent", document.content))
        return [RetrievalUnit("unit-1", document.content, document.provenance)]


class FakeEmbedder:
    def __init__(self, calls, vector=(1.0, 0.0)):
        self.calls = calls
        self.vector = vector

    def embed_units(self, units):
        self.calls.append(("embed_units", tuple(unit.unit_id for unit in units)))
        return [EmbeddingRecord(unit, self.vector, "fake-model") for unit in units]

    def embed_query(self, question):
        self.calls.append(("embed_query", question))
        return self.vector


class FakeIndexStore:
    def __init__(self, calls):
        self.calls = calls
        self.records = ()

    def write(self, records):
        self.records = tuple(records)
        self.calls.append(("index", len(records)))


class FakeRetriever:
    def __init__(self, calls, unit):
        self.calls = calls
        self.unit = unit

    def retrieve(self, query_embedding, *, top_k):
        self.calls.append(("retrieve", tuple(query_embedding), top_k))
        return [Candidate(self.unit, score=0.8)]


class FakeFusion:
    def __init__(self, calls):
        self.calls = calls

    def fuse(self, candidates):
        self.calls.append(("fusion", tuple(item.rank for item in candidates)))
        return candidates


class FakeReranker:
    def __init__(self, calls):
        self.calls = calls

    def rerank(self, question, candidates):
        self.calls.append(("rerank", question, len(candidates)))
        return candidates


class FakeContextBuilder:
    def __init__(self, calls):
        self.calls = calls
        self.received = ()

    def build(self, question, candidates):
        self.received = tuple(candidates)
        self.calls.append(("context", question, len(candidates)))
        return ContextBundle(question, "contexte", self.received)


class FakeGenerator:
    def __init__(self, calls):
        self.calls = calls

    def generate(self, question, context):
        self.calls.append(("generate", question, context.text))
        return RagAnswer("réponse", context)


def _orchestrator(calls, *, fusion=None, reranker=None, retriever=None):
    provenance = Provenance(document_id="document-1", page=4)
    unit = RetrievalUnit("unit-1", "passage", provenance)
    return RagOrchestrator(
        extractor=FakeExtractor(calls, provenance),
        representation_builder=FakeRepresentationBuilder(calls),
        embedder=FakeEmbedder(calls),
        index_store=FakeIndexStore(calls),
        retriever=retriever or FakeRetriever(calls, unit),
        fusion=fusion,
        reranker=reranker,
        context_builder=FakeContextBuilder(calls),
        generator=FakeGenerator(calls),
        retrieval_top_k=7,
    )


def test_rag_package_and_protocols_are_importable():
    assert all(
        contract is not None
        for contract in (
            Extractor,
            RepresentationBuilder,
            Embedder,
            IndexStore,
            Retriever,
            FusionStrategy,
            Reranker,
            ContextBuilder,
            Generator,
        )
    )
    assert RagPipelineConfig(
        extractor="extractor",
        representation="representation",
        embedder="embedder",
        index_store="index-store",
        retriever="retriever",
        context_builder="context",
        generator="generator",
    ).retrieval_top_k == 10


def test_index_calls_components_in_architectural_order():
    calls = []
    orchestrator = _orchestrator(calls)

    records = orchestrator.index("document brut")

    assert [call[0] for call in calls] == [
        "extract",
        "represent",
        "embed_units",
        "index",
    ]
    assert len(records) == 1
    assert records[0].unit.provenance.document_id == "document-1"


def test_answer_calls_optional_fusion_and_reranking_in_order():
    calls = []
    orchestrator = _orchestrator(
        calls,
        fusion=FakeFusion(calls),
        reranker=FakeReranker(calls),
    )

    answer = orchestrator.answer("question")

    assert [call[0] for call in calls] == [
        "embed_query",
        "retrieve",
        "fusion",
        "rerank",
        "context",
        "generate",
    ]
    assert answer.text == "réponse"
    assert calls[1] == ("retrieve", (1.0, 0.0), 7)


def test_answer_skips_optional_components_when_absent():
    calls = []

    _orchestrator(calls).answer("question")

    assert [call[0] for call in calls] == [
        "embed_query",
        "retrieve",
        "context",
        "generate",
    ]


def test_component_can_be_replaced_without_changing_orchestrator():
    calls = []
    replacement_unit = RetrievalUnit(
        "replacement",
        "autre passage",
        Provenance(document_id="document-2", page=9),
    )
    replacement = FakeRetriever(calls, replacement_unit)
    orchestrator = _orchestrator(calls, retriever=replacement)

    answer = orchestrator.answer("question")

    assert answer.context.candidates[0].unit.unit_id == "replacement"
    assert answer.context.candidates[0].unit.provenance.page == 9


def test_transport_objects_preserve_end_to_end_provenance():
    document_id = uuid4()
    version_id = uuid4()
    run_id = uuid4()
    source_ids = (uuid4(), uuid4())
    provenance = Provenance(
        document_id=document_id,
        document_version_id=version_id,
        processing_run_id=run_id,
        page=6,
        source_ids=source_ids,
        representation="structured",
        embedding_model="embedding-model",
        metadata={"bbox": [1, 2, 3, 4]},
    )
    unit = RetrievalUnit("unit", "texte", provenance)
    record = EmbeddingRecord(unit, (0.1, 0.2), "embedding-model")
    candidate = Candidate(record.unit, score=0.9)
    ranked = RankedCandidate(candidate, rank=1, score=0.9)
    context = ContextBundle("question", "texte", (ranked,))
    answer = RagAnswer("réponse", context)

    restored = answer.context.candidates[0].unit.provenance
    assert restored.document_id == document_id
    assert restored.document_version_id == version_id
    assert restored.processing_run_id == run_id
    assert restored.page == 6
    assert restored.source_ids == source_ids
    assert restored.representation == "structured"
    assert restored.embedding_model == "embedding-model"
    assert restored.metadata["bbox"] == [1, 2, 3, 4]


def test_rag_package_has_no_dependency_on_experimental_implementations():
    package = Path(__file__).resolve().parents[1] / "src" / "kaliok" / "rag"
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in package.rglob("*.py")
    )

    assert "kaliok.experiments" not in source
    assert "sqlmodel" not in source.lower()
    assert "requests" not in source.lower()
