from __future__ import annotations

from uuid import uuid4

import pytest

import kaliok.rag_runtime.factory as factory_module
from kaliok.configuration import ConfigurationRevisionInfo
from kaliok.rag_runtime.normalized import NormalizedContentReference


class FakeSession:
    pass


class FakeConfigurationReader:
    def __init__(
        self,
        session,
        profile_key="production-default",
        revision_id=None,
    ):
        self.session = session
        self.profile_key = profile_key
        self.requested_revision_id = revision_id
        self.revision_info = ConfigurationRevisionInfo(
            profile_id=uuid4(),
            profile_key=profile_key,
            revision_id=revision_id or uuid4(),
            revision_number=3,
        )

    def get_choice(self, category_key, setting_key):
        values = {
            ("rag.generation", "provider"): "ollama",
            ("rag.generation", "model"): "generation-test",
            ("rag.context", "builder"): "ranked",
        }
        return values[(category_key, setting_key)]

    def get_float(self, category_key, setting_key):
        assert (category_key, setting_key) == (
            "rag.generation",
            "temperature",
        )
        return 0.25

    def get_integer(self, category_key, setting_key):
        assert (category_key, setting_key) == (
            "rag.retrieval",
            "top_k",
        )
        return 7


class FakeProvenance:
    def __init__(self, document_id, document_version_id):
        self.document_id = document_id
        self.document_version_id = document_version_id


class FakeExtractedDocument:
    def __init__(self, document_id, document_version_id):
        self.provenance = FakeProvenance(
            document_id,
            document_version_id,
        )


class FakeProvider:
    instances = []

    def __init__(self, session):
        self.session = session
        self.document_id = uuid4()
        self.document_version_id = uuid4()
        self.provided_reference = None
        type(self).instances.append(self)

    def provide(self, reference):
        self.provided_reference = reference
        return FakeExtractedDocument(
            self.document_id,
            self.document_version_id,
        )


class FakeRepresentationBuilder:
    pass


class FakeEmbedder:
    pass


class FakeIndexStore:
    def __init__(self, session):
        self.session = session


class FakeRetriever:
    def __init__(self, session, document_version_id):
        self.session = session
        self.document_version_id = document_version_id


class FakeContextBuilder:
    pass


class FakeGenerator:
    def __init__(self, model=None, *, temperature=0.0):
        self.model = model
        self.temperature = temperature


class FakeOrchestrator:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.index_reference = None
        self.answer_question = None

    def index(self, reference):
        self.index_reference = reference
        return ("record-1", "record-2")

    def answer(self, question):
        self.answer_question = question
        return "answer"


def _patch_runtime(monkeypatch):
    FakeProvider.instances.clear()

    monkeypatch.setattr(
        factory_module,
        "ConfigurationReader",
        FakeConfigurationReader,
    )
    monkeypatch.setattr(
        factory_module,
        "NormalizedContentProvider",
        FakeProvider,
    )
    monkeypatch.setattr(
        factory_module,
        "NormalizedContentRepresentationBuilder",
        FakeRepresentationBuilder,
    )
    monkeypatch.setattr(
        factory_module,
        "OllamaRagEmbedder",
        FakeEmbedder,
    )
    monkeypatch.setattr(
        factory_module,
        "PostgresVectorIndexStore",
        FakeIndexStore,
    )
    monkeypatch.setattr(
        factory_module,
        "PostgresVectorRetriever",
        FakeRetriever,
    )
    monkeypatch.setattr(
        factory_module,
        "RankedContextBuilder",
        FakeContextBuilder,
    )
    monkeypatch.setattr(
        factory_module,
        "OllamaGenerator",
        FakeGenerator,
    )
    monkeypatch.setattr(
        factory_module,
        "RagOrchestrator",
        FakeOrchestrator,
    )


def test_create_normalized_rag_runtime_wires_existing_components(monkeypatch):
    _patch_runtime(monkeypatch)

    session = FakeSession()
    reference = NormalizedContentReference(
        document_id=uuid4(),
    )
    observer = object()

    runtime = factory_module.create_normalized_rag_runtime(
        session,
        reference=reference,
        observer=observer,
    )

    provider = FakeProvider.instances[0]

    assert provider.session is session
    assert provider.provided_reference is reference

    assert runtime.reference is reference
    assert runtime.document_id == provider.document_id
    assert runtime.document_version_id == provider.document_version_id
    assert runtime.session is session
    assert runtime.configuration.profile_key == "production-default"
    assert runtime.configuration.revision_number == 3
    assert runtime.configuration.generation_provider == "ollama"
    assert runtime.configuration.generation_model == "generation-test"
    assert runtime.configuration.generation_temperature == 0.25
    assert runtime.configuration.retrieval_top_k == 7
    assert runtime.configuration.context_builder == "ranked"
    assert runtime.configuration.embedding_model == factory_module.EMBEDDING_MODEL
    assert runtime.configuration.snapshot() == {
        "generation": {
            "provider": "ollama",
            "model": "generation-test",
            "temperature": 0.25,
        },
        "retrieval": {"top_k": 7},
        "context": {"builder": "ranked"},
        "embedding": {"model": factory_module.EMBEDDING_MODEL},
    }

    components = runtime.orchestrator.kwargs

    assert components["content_provider"] is provider
    assert isinstance(
        components["representation_builder"],
        FakeRepresentationBuilder,
    )
    assert isinstance(components["embedder"], FakeEmbedder)

    assert isinstance(components["index_store"], FakeIndexStore)
    assert components["index_store"].session is session

    assert isinstance(components["retriever"], FakeRetriever)
    assert components["retriever"].session is session
    assert (
        components["retriever"].document_version_id
        == provider.document_version_id
    )

    assert isinstance(
        components["context_builder"],
        FakeContextBuilder,
    )

    assert isinstance(components["generator"], FakeGenerator)
    assert components["generator"].model == "generation-test"
    assert components["generator"].temperature == 0.25

    assert components["retrieval_top_k"] == 7
    assert components["observer"] is observer


def test_create_normalized_rag_runtime_rejects_invalid_top_k(monkeypatch):
    class InvalidTopKConfigurationReader(FakeConfigurationReader):
        def get_integer(self, category_key, setting_key):
            return 0

    _patch_runtime(monkeypatch)

    monkeypatch.setattr(
        factory_module,
        "ConfigurationReader",
        InvalidTopKConfigurationReader,
    )

    with pytest.raises(
        ValueError,
        match="top_k doit être strictement positif",
    ):
        factory_module.create_normalized_rag_runtime(
            FakeSession(),
            reference=NormalizedContentReference(
                document_id=uuid4(),
            ),
        )


def test_create_normalized_rag_runtime_rejects_unsupported_provider(
    monkeypatch,
):
    class UnsupportedProviderConfigurationReader(
        FakeConfigurationReader
    ):
        def get_choice(self, category_key, setting_key):
            if (category_key, setting_key) == (
                "rag.generation",
                "provider",
            ):
                return "unsupported"

            return super().get_choice(category_key, setting_key)

    _patch_runtime(monkeypatch)

    monkeypatch.setattr(
        factory_module,
        "ConfigurationReader",
        UnsupportedProviderConfigurationReader,
    )

    with pytest.raises(
        ValueError,
        match="Fournisseur de génération RAG non pris en charge",
    ):
        factory_module.create_normalized_rag_runtime(
            FakeSession(),
            reference=NormalizedContentReference(
                document_id=uuid4(),
            ),
        )


def test_create_normalized_rag_runtime_rejects_unsupported_context_builder(
    monkeypatch,
):
    class UnsupportedBuilderConfigurationReader(
        FakeConfigurationReader
    ):
        def get_choice(self, category_key, setting_key):
            if (category_key, setting_key) == (
                "rag.context",
                "builder",
            ):
                return "unsupported"

            return super().get_choice(category_key, setting_key)

    _patch_runtime(monkeypatch)

    monkeypatch.setattr(
        factory_module,
        "ConfigurationReader",
        UnsupportedBuilderConfigurationReader,
    )

    with pytest.raises(
        ValueError,
        match="Constructeur de contexte RAG non pris en charge",
    ):
        factory_module.create_normalized_rag_runtime(
            FakeSession(),
            reference=NormalizedContentReference(
                document_id=uuid4(),
            ),
        )


def test_runtime_index_uses_selected_reference(monkeypatch):
    _patch_runtime(monkeypatch)

    reference = NormalizedContentReference(
        document_id=uuid4(),
    )

    runtime = factory_module.create_normalized_rag_runtime(
        FakeSession(),
        reference=reference,
    )

    count = runtime.index()

    assert count == 2
    assert runtime.orchestrator.index_reference is reference


def test_runtime_answer_delegates_to_orchestrator(monkeypatch):
    _patch_runtime(monkeypatch)

    runtime = factory_module.create_normalized_rag_runtime(
        FakeSession(),
        reference=NormalizedContentReference(
            document_id=uuid4(),
        ),
    )

    result = runtime.answer("Quelle est la réponse ?")

    assert result == "answer"
    assert (
        runtime.orchestrator.answer_question
        == "Quelle est la réponse ?"
    )


def test_runtime_is_indexed_uses_selected_version(monkeypatch):
    _patch_runtime(monkeypatch)

    checked = {}

    def fake_is_indexed(session, document_version_id, *, model_name):
        checked["session"] = session
        checked["document_version_id"] = document_version_id
        checked["model_name"] = model_name
        return True

    monkeypatch.setattr(
        factory_module,
        "normalized_version_is_indexed",
        fake_is_indexed,
    )

    session = FakeSession()

    runtime = factory_module.create_normalized_rag_runtime(
        session,
        reference=NormalizedContentReference(
            document_id=uuid4(),
        ),
    )

    assert runtime.is_indexed() is True
    assert checked["session"] is session
    assert (
        checked["document_version_id"]
        == runtime.document_version_id
    )
    assert checked["model_name"] == factory_module.EMBEDDING_MODEL


def test_runtime_requests_an_explicit_configuration_revision(monkeypatch):
    _patch_runtime(monkeypatch)
    revision_id = uuid4()

    runtime = factory_module.create_normalized_rag_runtime(
        FakeSession(),
        reference=NormalizedContentReference(document_id=uuid4()),
        configuration_revision_id=revision_id,
    )

    assert runtime.configuration.revision_id == revision_id
