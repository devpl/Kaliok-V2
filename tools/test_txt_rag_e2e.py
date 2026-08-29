"""Manual TXT normalized-content to RAG end-to-end validation tool."""

from __future__ import annotations

import argparse
import sys
from typing import Sequence
from uuid import UUID

from sqlmodel import Session

from kaliok.embeddings.ollama import EMBEDDING_MODEL
from kaliok.observability import ObservabilityEvent
from kaliok.rag import RagOrchestrator
from kaliok.rag_runtime import (
    NormalizedContentProvider,
    NormalizedContentReference,
    NormalizedContentRepresentationBuilder,
    OllamaGenerator,
    OllamaRagEmbedder,
    PostgresVectorIndexStore,
    PostgresVectorRetriever,
    RankedContextBuilder,
    normalized_version_is_indexed,
)
from kaliok.storage.database import create_database_engine


class ConsoleObserver:
    def __init__(self) -> None:
        self.events: list[ObservabilityEvent] = []

    def emit(self, event: ObservabilityEvent) -> None:
        self.events.append(event)
        duration = (
            "-" if event.duration_ms is None else f"{event.duration_ms:.2f} ms"
        )
        print(
            f"[event] {event.event_name} | execution={event.execution_id} "
            f"| correlation={event.correlation_id} | duration={duration} "
            f"| success={event.success}"
        )


def run(
    *,
    question: str,
    document_id: UUID | None,
    document_version_id: UUID | None,
    generation_model: str | None,
    top_k: int,
) -> None:
    reference = NormalizedContentReference(
        document_id=document_id,
        document_version_id=document_version_id,
    )
    engine = create_database_engine()
    observer = ConsoleObserver()
    try:
        with Session(engine) as session:
            provider = NormalizedContentProvider(session)
            selected = provider.provide(reference)
            selected_document_id = UUID(
                str(selected.provenance.document_id)
            )
            selected_version_id = UUID(
                str(selected.provenance.document_version_id)
            )
            orchestrator = RagOrchestrator(
                content_provider=provider,
                representation_builder=(
                    NormalizedContentRepresentationBuilder()
                ),
                embedder=OllamaRagEmbedder(),
                index_store=PostgresVectorIndexStore(session),
                retriever=PostgresVectorRetriever(
                    session, selected_version_id
                ),
                context_builder=RankedContextBuilder(),
                generator=OllamaGenerator(model=generation_model),
                retrieval_top_k=top_k,
                observer=observer,
            )

            if normalized_version_is_indexed(
                session,
                selected_version_id,
                model_name=EMBEDDING_MODEL,
            ):
                print("Index pgvector            : déjà complet, réutilisé")
                session.rollback()
            else:
                print("Index pgvector            : incomplet, indexation...")
                try:
                    records = orchestrator.index(reference)
                    session.commit()
                except Exception:
                    session.rollback()
                    raise
                print(f"Embeddings persistés      : {len(records)}")

            answer = orchestrator.answer(question)

            print("\nRéponse")
            print(answer.text)
            print("\nIdentité")
            print(f"document_id               : {selected_document_id}")
            print(f"document_version_id       : {selected_version_id}")
            print(f"modèle embedding          : {EMBEDDING_MODEL}")
            print(f"modèle génération         : {answer.metadata.get('model')}")
            print("\nCandidats récupérés")
            for ranked in answer.context.candidates:
                provenance = ranked.unit.provenance
                print(f"\n--- rang {ranked.rank} ---")
                print(f"score                     : {ranked.score}")
                print(
                    "distance                  : "
                    f"{ranked.candidate.metadata.get('distance')}"
                )
                print(f"document_id               : {provenance.document_id}")
                print(
                    "document_version_id       : "
                    f"{provenance.document_version_id}"
                )
                print(
                    "source_unit_id            : "
                    f"{provenance.metadata.get('source_unit_id')}"
                )
                print(
                    "source UUID               : "
                    f"{provenance.metadata.get('normalized_content_unit_id')}"
                )
                print("texte :")
                print(ranked.unit.text)
            print("\nContexte utilisé")
            print(answer.context.text)
    finally:
        engine.dispose()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Valide manuellement le RAG TXT réel de bout en bout."
    )
    identity = parser.add_mutually_exclusive_group(required=True)
    identity.add_argument("--document-id", type=UUID)
    identity.add_argument("--document-version-id", type=UUID)
    parser.add_argument("--question", required=True)
    parser.add_argument(
        "--generation-model",
        help="Modèle Ollama ; sinon KALIOK_GENERATION_MODEL.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.top_k <= 0:
        print("ERREUR : --top-k doit être strictement positif.", file=sys.stderr)
        return 2
    try:
        run(
            question=args.question,
            document_id=args.document_id,
            document_version_id=args.document_version_id,
            generation_model=args.generation_model,
            top_k=args.top_k,
        )
    except Exception as error:
        print(f"ERREUR : {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
