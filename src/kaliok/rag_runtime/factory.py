from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from sqlmodel import Session

from kaliok.configuration import ConfigurationReader
from kaliok.embeddings.ollama import EMBEDDING_MODEL
from kaliok.observability.base import Observer
from kaliok.rag import RagOrchestrator
from kaliok.rag.types import RagAnswer
from kaliok.rag_runtime.normalized import (
    NormalizedContentProvider,
    NormalizedContentReference,
    NormalizedContentRepresentationBuilder,
)
from kaliok.rag_runtime.ollama import OllamaGenerator, OllamaRagEmbedder
from kaliok.rag_runtime.postgres import (
    PostgresVectorIndexStore,
    PostgresVectorRetriever,
    normalized_version_is_indexed,
)
from kaliok.rag_runtime.simple_context import RankedContextBuilder


@dataclass(frozen=True)
class RagRuntimeConfiguration:
    profile_id: UUID
    profile_key: str
    revision_id: UUID
    revision_number: int
    generation_provider: str
    generation_model: str
    generation_temperature: float
    retrieval_top_k: int
    context_builder: str
    embedding_model: str

    def snapshot(self) -> dict[str, object]:
        return {
            "generation": {
                "provider": self.generation_provider,
                "model": self.generation_model,
                "temperature": self.generation_temperature,
            },
            "retrieval": {"top_k": self.retrieval_top_k},
            "context": {"builder": self.context_builder},
            "embedding": {"model": self.embedding_model},
        }


@dataclass(frozen=True)
class NormalizedRagRuntime:
    orchestrator: RagOrchestrator
    reference: NormalizedContentReference
    document_id: UUID
    document_version_id: UUID
    session: Session
    configuration: RagRuntimeConfiguration

    def is_indexed(self) -> bool:
        return normalized_version_is_indexed(
            self.session,
            self.document_version_id,
            model_name=EMBEDDING_MODEL,
        )

    def index(self) -> int:
        records = self.orchestrator.index(self.reference)
        return len(records)

    def answer(self, question: str) -> RagAnswer:
        return self.orchestrator.answer(question)


def create_normalized_rag_runtime(
    session: Session,
    *,
    reference: NormalizedContentReference,
    observer: Observer | None = None,
    profile_key: str = "production-default",
    configuration_revision_id: UUID | None = None,
) -> NormalizedRagRuntime:
    reader_options: dict[str, object]
    if configuration_revision_id is None:
        reader_options = {"profile_key": profile_key}
    else:
        reader_options = {"revision_id": configuration_revision_id}
    configuration = ConfigurationReader(session, **reader_options)

    generation_provider = configuration.get_choice(
        "rag.generation",
        "provider",
    )
    generation_model = configuration.get_choice(
        "rag.generation",
        "model",
    )
    generation_temperature = configuration.get_float(
        "rag.generation",
        "temperature",
    )
    retrieval_top_k = configuration.get_integer(
        "rag.retrieval",
        "top_k",
    )
    context_builder = configuration.get_choice(
        "rag.context",
        "builder",
    )
    revision = configuration.revision_info
    runtime_configuration = RagRuntimeConfiguration(
        profile_id=revision.profile_id,
        profile_key=revision.profile_key,
        revision_id=revision.revision_id,
        revision_number=revision.revision_number,
        generation_provider=generation_provider,
        generation_model=generation_model,
        generation_temperature=generation_temperature,
        retrieval_top_k=retrieval_top_k,
        context_builder=context_builder,
        embedding_model=EMBEDDING_MODEL,
    )

    if generation_provider != "ollama":
        raise ValueError(
            "Fournisseur de génération RAG non pris en charge : "
            f"{generation_provider}."
        )

    if retrieval_top_k <= 0:
        raise ValueError(
            "rag.retrieval.top_k doit être strictement positif."
        )

    if context_builder != "ranked":
        raise ValueError(
            "Constructeur de contexte RAG non pris en charge : "
            f"{context_builder}."
        )

    provider = NormalizedContentProvider(session)
    selected = provider.provide(reference)

    document_id = UUID(str(selected.provenance.document_id))
    document_version_id = UUID(
        str(selected.provenance.document_version_id)
    )

    orchestrator = RagOrchestrator(
        content_provider=provider,
        representation_builder=NormalizedContentRepresentationBuilder(),
        embedder=OllamaRagEmbedder(),
        index_store=PostgresVectorIndexStore(session),
        retriever=PostgresVectorRetriever(
            session,
            document_version_id,
        ),
        context_builder=RankedContextBuilder(),
        generator=OllamaGenerator(
            model=generation_model,
            temperature=generation_temperature,
        ),
        retrieval_top_k=retrieval_top_k,
        observer=observer,
    )

    return NormalizedRagRuntime(
        orchestrator=orchestrator,
        reference=reference,
        document_id=document_id,
        document_version_id=document_version_id,
        session=session,
        configuration=runtime_configuration,
    )
