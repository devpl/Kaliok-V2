from __future__ import annotations

import argparse
from dataclasses import dataclass
from statistics import mean
from time import perf_counter
from uuid import UUID

from sqlmodel import Session

from kaliok.observability.events import ObservabilityEvent
from kaliok.rag_runtime import (
    NormalizedContentReference,
    create_normalized_rag_runtime,
)
from kaliok.storage.database import create_database_engine


DEFAULT_MODELS = (
    "qwen3:8b",
    "mistral",
    "gemma3:4b",
    "phi4-mini",
)


@dataclass(frozen=True)
class RunResult:
    model: str
    run_number: int
    query_embedding_ms: float
    retrieval_ms: float
    context_ms: float
    generation_ms: float
    rag_total_ms: float
    http_total_ms: float
    answer: str


class EventCollector:
    def __init__(self) -> None:
        self.events: list[ObservabilityEvent] = []

    def emit(self, event: ObservabilityEvent) -> None:
        self.events.append(event)

    def duration(self, event_name: str) -> float:
        matching = [
            event
            for event in self.events
            if event.event_name == event_name
        ]

        if not matching:
            raise RuntimeError(
                f"Événement absent : {event_name}"
            )

        event = matching[-1]

        if event.duration_ms is None:
            raise RuntimeError(
                f"Durée absente : {event_name}"
            )

        return event.duration_ms


def benchmark_one(
    *,
    session: Session,
    document_id: UUID,
    question: str,
    generation_model: str,
    top_k: int,
    run_number: int,
) -> RunResult:
    observer = EventCollector()

    runtime = create_normalized_rag_runtime(
        session,
        reference=NormalizedContentReference(
            document_id=document_id,
        ),
        generation_model=generation_model,
        top_k=top_k,
        observer=observer,
    )

    if not runtime.is_indexed():
        raise RuntimeError(
            "Le document n'est pas déjà indexé. "
            "Le benchmark doit être exécuté à chaud."
        )

    wall_start = perf_counter()
    answer = runtime.answer(question)
    wall_ms = (perf_counter() - wall_start) * 1000.0

    return RunResult(
        model=generation_model,
        run_number=run_number,
        query_embedding_ms=observer.duration(
            "rag.query_embedding.completed"
        ),
        retrieval_ms=observer.duration(
            "rag.retrieval.completed"
        ),
        context_ms=observer.duration(
            "rag.context.completed"
        ),
        generation_ms=observer.duration(
            "rag.generation.completed"
        ),
        rag_total_ms=observer.duration(
            "rag.answer.completed"
        ),
        http_total_ms=wall_ms,
        answer=answer.text,
    )


def print_run(result: RunResult) -> None:
    print()
    print(
        f"{result.model} — passage {result.run_number}"
    )
    print(
        f"  embedding question : "
        f"{result.query_embedding_ms:9.2f} ms"
    )
    print(
        f"  recherche pgvector : "
        f"{result.retrieval_ms:9.2f} ms"
    )
    print(
        f"  construction contexte : "
        f"{result.context_ms:9.2f} ms"
    )
    print(
        f"  génération : "
        f"{result.generation_ms:9.2f} ms"
    )
    print(
        f"  total RAG : "
        f"{result.rag_total_ms:9.2f} ms"
    )
    print(
        f"  total mesuré : "
        f"{result.http_total_ms:9.2f} ms"
    )
    print(f"  réponse : {result.answer}")


def print_summary(
    results: list[RunResult],
    models: tuple[str, ...],
) -> None:
    print()
    print("=" * 105)
    print("SYNTHÈSE")
    print("=" * 105)

    header = (
        f"{'Modèle':<18}"
        f"{'Embed ms':>12}"
        f"{'Recherche':>12}"
        f"{'Contexte':>12}"
        f"{'Génération':>14}"
        f"{'RAG total':>14}"
        f"{'Mesuré':>12}"
    )
    print(header)
    print("-" * len(header))

    for model in models:
        model_results = [
            result
            for result in results
            if result.model == model
        ]

        if not model_results:
            continue

        print(
            f"{model:<18}"
            f"{mean(r.query_embedding_ms for r in model_results):>12.2f}"
            f"{mean(r.retrieval_ms for r in model_results):>12.2f}"
            f"{mean(r.context_ms for r in model_results):>12.2f}"
            f"{mean(r.generation_ms for r in model_results):>14.2f}"
            f"{mean(r.rag_total_ms for r in model_results):>14.2f}"
            f"{mean(r.http_total_ms for r in model_results):>12.2f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare les modèles de génération RAG dans "
            "les mêmes conditions."
        )
    )

    parser.add_argument(
        "--document-id",
        type=UUID,
        required=True,
    )
    parser.add_argument(
        "--question",
        required=True,
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=2,
        help=(
            "Nombre de passages par modèle. "
            "Le premier peut inclure le chargement du modèle."
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
    )

    args = parser.parse_args()

    if args.top_k <= 0:
        parser.error("--top-k doit être strictement positif.")

    if args.runs <= 0:
        parser.error("--runs doit être strictement positif.")

    return args


def main() -> None:
    args = parse_args()
    models = tuple(args.models)

    engine = create_database_engine()
    results: list[RunResult] = []

    print("Benchmark RAG à chaud")
    print(f"Document : {args.document_id}")
    print(f"Question : {args.question}")
    print(f"top_k    : {args.top_k}")
    print(f"Passages : {args.runs}")
    print(f"Modèles  : {', '.join(models)}")

    with Session(engine) as session:
        for model in models:
            for run_number in range(1, args.runs + 1):
                result = benchmark_one(
                    session=session,
                    document_id=args.document_id,
                    question=args.question,
                    generation_model=model,
                    top_k=args.top_k,
                    run_number=run_number,
                )

                results.append(result)
                print_run(result)

    print_summary(results, models)


if __name__ == "__main__":
    main()
