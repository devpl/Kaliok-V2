from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from sqlmodel import Session, select

from kaliok.embeddings.ollama import (
    embed_text,
)
from kaliok.embeddings.service import (
    search_similar_chunks,
)
from kaliok.indexing.service import (
    IndexDocumentResult,
    index_document,
)
from kaliok.paths import TEST_DOCUMENTS
from kaliok.retrieval.hybrid import (
    search_hybrid_chunks,
)
from kaliok.retrieval.lexical import (
    search_lexical_chunks,
)
from kaliok.storage.database import (
    create_database_engine,
)
from kaliok.storage.models import (
    ChunkEmbedding,
    DocumentChunk,
)


VALIDATION_DIR = (
    TEST_DOCUMENTS / "validation"
)

DATASET_PATH = (
    VALIDATION_DIR
    / "qa_validation.json"
)

RETRIEVAL_LIMIT = 10

STRATEGIES = (
    "vector",
    "lexical",
    "hybrid",
)


@dataclass
class RetrievedChunk:
    chunk_id: UUID
    page_start: int | None
    page_end: int | None
    vector_distance: float | None = None


@dataclass
class AbstentionObservation:
    question_id: str
    filename: str
    answerable: bool
    expected_pages: list[int]
    top1_page: int | None
    top1_distance: float | None
    top2_distance: float | None
    top1_top2_gap: float | None
    correct_rank: int | None
    correct_distance: float | None


@dataclass
class StrategyMetrics:
    evaluated: int = 0
    hit_at_1: int = 0
    hit_at_3: int = 0
    reciprocal_rank_sum: float = 0.0

    def add_rank(
        self,
        rank: int | None,
    ) -> None:
        self.evaluated += 1

        if rank is None:
            return

        if rank == 1:
            self.hit_at_1 += 1

        if rank <= 3:
            self.hit_at_3 += 1

        self.reciprocal_rank_sum += (
            1.0 / rank
        )

    @property
    def hit_at_1_rate(
        self,
    ) -> float:
        if not self.evaluated:
            return 0.0

        return (
            self.hit_at_1
            / self.evaluated
        )

    @property
    def hit_at_3_rate(
        self,
    ) -> float:
        if not self.evaluated:
            return 0.0

        return (
            self.hit_at_3
            / self.evaluated
        )

    @property
    def mrr(
        self,
    ) -> float:
        if not self.evaluated:
            return 0.0

        return (
            self.reciprocal_rank_sum
            / self.evaluated
        )


def load_dataset() -> dict:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            "Dataset introuvable : "
            f"{DATASET_PATH}"
        )

    with DATASET_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(
            file
        )


def load_chunk_pages(
    chunk_ids: list[UUID],
) -> dict[UUID, RetrievedChunk]:
    if not chunk_ids:
        return {}

    engine = create_database_engine()

    with Session(engine) as session:
        chunks = session.exec(
            select(DocumentChunk).where(
                DocumentChunk.id.in_(
                    chunk_ids
                )
            )
        ).all()

    return {
        chunk.id: RetrievedChunk(
            chunk_id=chunk.id,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
        )
        for chunk in chunks
    }


def load_vector_distances(
    chunk_ids: list[UUID],
    *,
    query_embedding: list[float],
    embedding_model_id: UUID,
) -> dict[UUID, float]:
    if not chunk_ids:
        return {}

    engine = create_database_engine()

    distance = (
        ChunkEmbedding.embedding.cosine_distance(
            query_embedding
        )
    ).label("distance")

    with Session(engine) as session:
        rows = session.exec(
            select(
                ChunkEmbedding.chunk_id,
                distance,
            ).where(
                ChunkEmbedding.chunk_id.in_(
                    chunk_ids
                ),
                ChunkEmbedding.embedding_model_id
                == embedding_model_id,
            )
        ).all()

    return {
        chunk_id: float(value)
        for chunk_id, value in rows
    }


def attach_pages(
    chunk_ids: list[UUID],
    *,
    vector_distances: dict[UUID, float]
    | None = None,
) -> list[RetrievedChunk]:
    page_mapping = load_chunk_pages(
        chunk_ids
    )

    vector_distances = (
        vector_distances
        if vector_distances is not None
        else {}
    )

    return [
        RetrievedChunk(
            chunk_id=page_mapping[
                chunk_id
            ].chunk_id,
            page_start=page_mapping[
                chunk_id
            ].page_start,
            page_end=page_mapping[
                chunk_id
            ].page_end,
            vector_distance=(
                vector_distances.get(
                    chunk_id
                )
            ),
        )
        for chunk_id in chunk_ids
        if chunk_id in page_mapping
    ]


def search_vector(
    question: str,
    *,
    index_result: IndexDocumentResult,
) -> list[RetrievedChunk]:
    query_embedding = embed_text(
        question
    )

    results = search_similar_chunks(
        query_embedding=query_embedding,
        embedding_model_id=(
            index_result.embedding_model_id
        ),
        document_version_id=(
            index_result.document_version_id
        ),
        limit=RETRIEVAL_LIMIT,
    )

    chunk_ids = [
        result.chunk_id
        for result in results
    ]

    vector_distances = load_vector_distances(
        chunk_ids,
        query_embedding=query_embedding,
        embedding_model_id=(
            index_result.embedding_model_id
        ),
    )

    return attach_pages(
        chunk_ids,
        vector_distances=vector_distances,
    )


def search_lexical(
    question: str,
    *,
    index_result: IndexDocumentResult,
) -> list[RetrievedChunk]:
    results = search_lexical_chunks(
        question,
        document_version_id=(
            index_result.document_version_id
        ),
        limit=RETRIEVAL_LIMIT,
    )

    return attach_pages(
        [
            result.chunk_id
            for result in results
        ]
    )


def search_hybrid(
    question: str,
    *,
    index_result: IndexDocumentResult,
) -> list[RetrievedChunk]:
    results = search_hybrid_chunks(
        question,
        embedding_model_id=(
            index_result.embedding_model_id
        ),
        document_version_id=(
            index_result.document_version_id
        ),
        limit=RETRIEVAL_LIMIT,
        candidate_limit=RETRIEVAL_LIMIT,
        rrf_k=60,
    )

    return attach_pages(
        [
            result.chunk_id
            for result in results
        ]
    )


def run_strategy(
    strategy: str,
    question: str,
    *,
    index_result: IndexDocumentResult,
) -> list[RetrievedChunk]:
    if strategy == "vector":
        return search_vector(
            question,
            index_result=index_result,
        )

    if strategy == "lexical":
        return search_lexical(
            question,
            index_result=index_result,
        )

    if strategy == "hybrid":
        return search_hybrid(
            question,
            index_result=index_result,
        )

    raise ValueError(
        f"Stratégie inconnue : {strategy}"
    )


def chunk_matches_expected_pages(
    chunk: RetrievedChunk,
    expected_pages: set[int],
) -> bool:
    if chunk.page_start is None:
        return False

    page_end = (
        chunk.page_end
        if chunk.page_end is not None
        else chunk.page_start
    )

    chunk_pages = set(
        range(
            chunk.page_start,
            page_end + 1,
        )
    )

    return bool(
        chunk_pages
        & expected_pages
    )


def find_expected_rank(
    results: list[RetrievedChunk],
    expected_pages: list[int],
) -> int | None:
    expected_page_set = set(
        expected_pages
    )

    for rank, chunk in enumerate(
        results,
        start=1,
    ):
        if chunk_matches_expected_pages(
            chunk,
            expected_page_set,
        ):
            return rank

    return None


def find_expected_distance(
    results: list[RetrievedChunk],
    expected_pages: list[int],
) -> float | None:
    expected_page_set = set(
        expected_pages
    )

    for chunk in results:
        if chunk_matches_expected_pages(
            chunk,
            expected_page_set,
        ):
            return chunk.vector_distance

    return None


def build_abstention_observation(
    *,
    question_id: str,
    filename: str,
    answerable: bool,
    expected_pages: list[int],
    vector_results: list[RetrievedChunk],
) -> AbstentionObservation:
    top1 = (
        vector_results[0]
        if vector_results
        else None
    )

    top2 = (
        vector_results[1]
        if len(vector_results) >= 2
        else None
    )

    top1_distance = (
        top1.vector_distance
        if top1 is not None
        else None
    )

    top2_distance = (
        top2.vector_distance
        if top2 is not None
        else None
    )

    gap = None

    if (
        top1_distance is not None
        and top2_distance is not None
    ):
        # Cosine distance: plus petit = meilleur.
        # Un gap positif signifie que le rang 1
        # est mieux séparé du rang 2.
        gap = (
            top2_distance
            - top1_distance
        )

    correct_rank = None
    correct_distance = None

    if answerable:
        correct_rank = find_expected_rank(
            vector_results,
            expected_pages,
        )
        correct_distance = (
            find_expected_distance(
                vector_results,
                expected_pages,
            )
        )

    return AbstentionObservation(
        question_id=question_id,
        filename=filename,
        answerable=answerable,
        expected_pages=expected_pages,
        top1_page=(
            top1.page_start
            if top1 is not None
            else None
        ),
        top1_distance=top1_distance,
        top2_distance=top2_distance,
        top1_top2_gap=gap,
        correct_rank=correct_rank,
        correct_distance=correct_distance,
    )


def format_float(
    value: float | None,
) -> str:
    if value is None:
        return "-"

    return f"{value:.4f}"


def format_rank(
    rank: int | None,
) -> str:
    if rank is None:
        return "-"
    return str(
        rank
    )


def main() -> None:
    dataset = load_dataset()

    documents = dataset.get(
        "documents",
        [],
    )

    print(
        "=" * 72
    )
    print(
        f"BENCHMARK : "
        f"{dataset.get('dataset', 'inconnu')}"
    )
    print(
        "=" * 72
    )

    print()
    print(
        f"Documents : {len(documents)}"
    )

    # ---------------------------------------------------------
    # Indexation
    # ---------------------------------------------------------

    print()
    print(
        "=" * 72
    )
    print(
        "INDEXATION"
    )
    print(
        "=" * 72
    )

    indexed_documents: dict[
        str,
        IndexDocumentResult,
    ] = {}

    for document_data in documents:
        filename = document_data[
            "file"
        ]

        path = (
            VALIDATION_DIR
            / filename
        )

        result = index_document(
            path,
            verbose=True,
        )

        indexed_documents[
            filename
        ] = result

    # ---------------------------------------------------------
    # Benchmark
    # ---------------------------------------------------------

    metrics = {
        strategy: StrategyMetrics()
        for strategy in STRATEGIES
    }

    answerable_count = 0
    unanswerable_count = 0

    abstention_observations: list[
        AbstentionObservation
    ] = []

    print()
    print(
        "=" * 72
    )
    print(
        "RETRIEVAL"
    )
    print(
        "=" * 72
    )

    for document_data in documents:
        filename = document_data[
            "file"
        ]

        index_result = (
            indexed_documents[
                filename
            ]
        )

        print()
        print(
            "#" * 72
        )
        print(
            filename
        )
        print(
            "#" * 72
        )

        for question_data in (
            document_data.get(
                "questions",
                [],
            )
        ):
            question_id = (
                question_data[
                    "id"
                ]
            )

            question = (
                question_data[
                    "question"
                ]
            )

            answerable = bool(
                question_data[
                    "answerable"
                ]
            )

            expected_pages = (
                question_data.get(
                    "expected_pages",
                    [],
                )
            )

            print()
            print(
                f"{question_id}"
            )
            print(
                f"  {question}"
            )

            if not answerable:
                unanswerable_count += 1
                print(
                    "  Attendu : "
                    "SANS RÉPONSE"
                )

                # On exécute quand même les trois
                # stratégies afin d'observer leur
                # comportement, mais ces questions
                # ne participent PAS aux métriques
                # retrieval.
                vector_results_for_observation: list[
                    RetrievedChunk
                ] = []

                for strategy in STRATEGIES:
                    results = run_strategy(
                        strategy,
                        question,
                        index_result=(
                            index_result
                        ),
                    )

                    if strategy == "vector":
                        vector_results_for_observation = (
                            results
                        )

                    if results:
                        first = results[0]

                        suffix = ""

                        if strategy == "vector":
                            suffix = (
                                " | distance="
                                f"{format_float(first.vector_distance)}"
                            )

                        print(
                            f"  {strategy:<8} "
                            "→ résultat retourné "
                            f"(page "
                            f"{first.page_start})"
                            f"{suffix}"
                        )
                    else:
                        print(
                            f"  {strategy:<8} "
                            "→ aucun résultat"
                        )

                abstention_observations.append(
                    build_abstention_observation(
                        question_id=question_id,
                        filename=filename,
                        answerable=False,
                        expected_pages=[],
                        vector_results=(
                            vector_results_for_observation
                        ),
                    )
                )

                continue

            answerable_count += 1

            print(
                "  Pages attendues : "
                + ", ".join(
                    str(page)
                    for page
                    in expected_pages
                )
            )

            vector_results_for_observation: list[
                RetrievedChunk
            ] = []

            for strategy in STRATEGIES:
                results = run_strategy(
                    strategy,
                    question,
                    index_result=(
                        index_result
                    ),
                )

                if strategy == "vector":
                    vector_results_for_observation = (
                        results
                    )

                rank = find_expected_rank(
                    results,
                    expected_pages,
                )

                metrics[
                    strategy
                ].add_rank(
                    rank
                )

                suffix = ""

                if (
                    strategy == "vector"
                    and results
                ):
                    suffix = (
                        " | top1_distance="
                        f"{format_float(results[0].vector_distance)}"
                    )

                print(
                    f"  {strategy:<8} "
                    f"→ rang "
                    f"{format_rank(rank)}"
                    f"{suffix}"
                )

            abstention_observations.append(
                build_abstention_observation(
                    question_id=question_id,
                    filename=filename,
                    answerable=True,
                    expected_pages=expected_pages,
                    vector_results=(
                        vector_results_for_observation
                    ),
                )
            )

    # ---------------------------------------------------------
    # Résultats
    # ---------------------------------------------------------

    print()
    print(
        "=" * 72
    )
    print(
        "RÉSULTATS"
    )
    print(
        "=" * 72
    )

    print()
    print(
        f"Questions répondables     : "
        f"{answerable_count}"
    )
    print(
        f"Questions sans réponse    : "
        f"{unanswerable_count}"
    )

    print()
    print(
        "Les questions sans réponse "
        "ne sont pas encore incluses "
        "dans les scores : kaliok ne "
        "dispose pas encore d'une règle "
        "d'abstention fiable."
    )

    print()
    print(
        f"{'Stratégie':<12}"
        f"{'Hit@1':>10}"
        f"{'Hit@3':>10}"
        f"{'MRR':>10}"
    )
    print(
        "-" * 42
    )

    for strategy in STRATEGIES:
        strategy_metrics = (
            metrics[
                strategy
            ]
        )

        print(
            f"{strategy:<12}"
            f"{strategy_metrics.hit_at_1_rate:>10.3f}"
            f"{strategy_metrics.hit_at_3_rate:>10.3f}"
            f"{strategy_metrics.mrr:>10.3f}"
        )


    # ---------------------------------------------------------
    # Diagnostic abstention vectorielle
    # ---------------------------------------------------------

    print()
    print(
        "=" * 72
    )
    print(
        "DIAGNOSTIC ABSTENTION — VECTOR"
    )
    print(
        "=" * 72
    )

    print()
    print(
        "Cosine distance : plus petit = "
        "plus proche."
    )
    print(
        "gap = distance(top2) - "
        "distance(top1)."
    )

    print()
    print(
        f"{'Question':<22}"
        f"{'Type':<8}"
        f"{'P1':>5}"
        f"{'D1':>9}"
        f"{'D2':>9}"
        f"{'Gap':>9}"
        f"{'RangOK':>8}"
        f"{'DistOK':>9}"
    )
    print(
        "-" * 79
    )

    for observation in (
        abstention_observations
    ):
        question_type = (
            "OUI"
            if observation.answerable
            else "NON"
        )

        top1_page = (
            str(observation.top1_page)
            if observation.top1_page
            is not None
            else "-"
        )

        print(
            f"{observation.question_id:<22}"
            f"{question_type:<8}"
            f"{top1_page:>5}"
            f"{format_float(observation.top1_distance):>9}"
            f"{format_float(observation.top2_distance):>9}"
            f"{format_float(observation.top1_top2_gap):>9}"
            f"{format_rank(observation.correct_rank):>8}"
            f"{format_float(observation.correct_distance):>9}"
        )

    answerable_top1 = [
        observation.top1_distance
        for observation
        in abstention_observations
        if (
            observation.answerable
            and observation.top1_distance
            is not None
        )
    ]

    unanswerable_top1 = [
        observation.top1_distance
        for observation
        in abstention_observations
        if (
            not observation.answerable
            and observation.top1_distance
            is not None
        )
    ]

    if (
        answerable_top1
        and unanswerable_top1
    ):
        print()
        print(
            "Plage top1 répondables    : "
            f"{min(answerable_top1):.4f} "
            "→ "
            f"{max(answerable_top1):.4f}"
        )
        print(
            "Plage top1 sans réponse   : "
            f"{min(unanswerable_top1):.4f} "
            "→ "
            f"{max(unanswerable_top1):.4f}"
        )

        overlap_low = max(
            min(answerable_top1),
            min(unanswerable_top1),
        )
        overlap_high = min(
            max(answerable_top1),
            max(unanswerable_top1),
        )

        if overlap_low <= overlap_high:
            print(
                "Chevauchement top1       : "
                f"{overlap_low:.4f} "
                "→ "
                f"{overlap_high:.4f}"
            )
        else:
            print(
                "Chevauchement top1       : "
                "aucun"
            )


if __name__ == "__main__":
    main()
