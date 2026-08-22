from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

from kaliok.documents.cleaning import clean_document
from kaliok.documents.llama_chunking import (
    chunk_document_with_llamaindex,
)
from kaliok.documents.reader import read_document
from kaliok.documents.semantic_chunking import (
    chunk_document_semantically,
)
from kaliok.embeddings.ollama import embed_texts
from kaliok.paths import TEST_DOCUMENTS


VALIDATION_DIR = TEST_DOCUMENTS / "validation"

DATASET_PATH = (
    VALIDATION_DIR
    / "qa_validation.json"
)

RESULTS_PATH = (
    VALIDATION_DIR
    / "chunking_ab_results.json"
)

TOP_K = 10


@dataclass
class RetrievedChunk:
    rank: int
    page: int
    similarity: float


@dataclass
class Metrics:
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
    def hit_at_1_rate(self) -> float:
        if not self.evaluated:
            return 0.0

        return (
            self.hit_at_1
            / self.evaluated
        )

    @property
    def hit_at_3_rate(self) -> float:
        if not self.evaluated:
            return 0.0

        return (
            self.hit_at_3
            / self.evaluated
        )

    @property
    def mrr(self) -> float:
        if not self.evaluated:
            return 0.0

        return (
            self.reciprocal_rank_sum
            / self.evaluated
        )


def load_dataset() -> dict:
    with DATASET_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def cosine_similarity(
    left: list[float],
    right: list[float],
) -> float:
    dot_product = sum(
        a * b
        for a, b in zip(
            left,
            right,
        )
    )

    left_norm = math.sqrt(
        sum(
            value * value
            for value in left
        )
    )

    right_norm = math.sqrt(
        sum(
            value * value
            for value in right
        )
    )

    if (
        left_norm == 0.0
        or right_norm == 0.0
    ):
        return 0.0

    return (
        dot_product
        / (
            left_norm
            * right_norm
        )
    )


def retrieve(
    query_embedding: list[float],
    chunks,
    chunk_embeddings: list[list[float]],
    *,
    limit: int = TOP_K,
) -> list[RetrievedChunk]:
    scored = []

    for chunk, embedding in zip(
        chunks,
        chunk_embeddings,
    ):
        similarity = cosine_similarity(
            query_embedding,
            embedding,
        )

        scored.append(
            (
                similarity,
                chunk,
            )
        )

    scored.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    return [
        RetrievedChunk(
            rank=rank,
            page=chunk.page,
            similarity=similarity,
        )
        for rank, (
            similarity,
            chunk,
        ) in enumerate(
            scored[:limit],
            start=1,
        )
    ]


def find_expected_rank(
    results: list[RetrievedChunk],
    expected_pages: list[int],
) -> int | None:
    expected = set(
        expected_pages
    )

    for result in results:
        if result.page in expected:
            return result.rank

    return None


def build_query_embeddings(
    dataset: dict,
) -> dict[str, list[float]]:
    questions = []
    question_ids = []

    for document_data in dataset["documents"]:
        for question_data in document_data["questions"]:
            if not question_data["answerable"]:
                continue

            question_ids.append(
                question_data["id"]
            )

            questions.append(
                question_data["question"]
            )

    print()
    print(
        "Génération des embeddings "
        "des questions..."
    )

    embeddings = embed_texts(
        questions
    )

    return {
        question_id: embedding
        for question_id, embedding in zip(
            question_ids,
            embeddings,
        )
    }


def chunk_document(
    strategy: str,
    document,
):
    if strategy == "semantic":
        return chunk_document_semantically(
            document,
            breakpoint_percentile_threshold=95,
            buffer_size=1,
        )

    if strategy == "sentence":
        return chunk_document_with_llamaindex(
            document,
            chunk_size=256,
            chunk_overlap=32,
        )

    raise ValueError(
        f"Stratégie inconnue : {strategy}"
    )


def benchmark_strategy(
    strategy: str,
    dataset: dict,
    query_embeddings: dict[
        str,
        list[float],
    ],
) -> dict:
    print()
    print(
        "=" * 72
    )
    print(
        f"STRATÉGIE : {strategy.upper()}"
    )
    print(
        "=" * 72
    )

    metrics = Metrics()

    total_chunking = 0.0
    total_embedding = 0.0
    total_chunks = 0

    document_results = []

    for document_data in dataset["documents"]:
        filename = document_data["file"]

        path = (
            VALIDATION_DIR
            / filename
        )

        print()
        print(
            f"Document : {filename}"
        )

        # Même extraction et même nettoyage
        # pour les deux stratégies.
        document = read_document(
            path
        )

        cleaned = clean_document(
            document
        )

        # -----------------------------------------------------
        # Chunking
        # -----------------------------------------------------

        start = time.perf_counter()

        chunks = chunk_document(
            strategy,
            cleaned,
        )

        chunking_duration = (
            time.perf_counter()
            - start
        )

        # -----------------------------------------------------
        # Embeddings définitifs
        # -----------------------------------------------------

        start = time.perf_counter()

        chunk_embeddings = embed_texts(
            [
                chunk.text
                for chunk in chunks
            ]
        )

        embedding_duration = (
            time.perf_counter()
            - start
        )

        total_chunking += (
            chunking_duration
        )

        total_embedding += (
            embedding_duration
        )

        total_chunks += len(
            chunks
        )

        print(
            f"  Chunks              : "
            f"{len(chunks)}"
        )
        print(
            f"  Chunking             : "
            f"{chunking_duration:.2f} s"
        )
        print(
            f"  Embeddings définitifs: "
            f"{embedding_duration:.2f} s"
        )

        question_results = []

        for question_data in (
            document_data["questions"]
        ):
            question_id = (
                question_data["id"]
            )

            if not question_data[
                "answerable"
            ]:
                continue

            query_embedding = (
                query_embeddings[
                    question_id
                ]
            )

            results = retrieve(
                query_embedding,
                chunks,
                chunk_embeddings,
            )

            rank = find_expected_rank(
                results,
                question_data[
                    "expected_pages"
                ],
            )

            metrics.add_rank(
                rank
            )

            rank_display = (
                str(rank)
                if rank is not None
                else "-"
            )

            print(
                f"    {question_id:<22} "
                f"rang {rank_display}"
            )

            question_results.append(
                {
                    "id": question_id,
                    "rank": rank,
                    "expected_pages": (
                        question_data[
                            "expected_pages"
                        ]
                    ),
                    "top_results": [
                        {
                            "rank": result.rank,
                            "page": result.page,
                            "similarity": (
                                result.similarity
                            ),
                        }
                        for result in results
                    ],
                }
            )

        document_results.append(
            {
                "file": filename,
                "chunk_count": len(
                    chunks
                ),
                "chunking_seconds": (
                    chunking_duration
                ),
                "embedding_seconds": (
                    embedding_duration
                ),
                "questions": (
                    question_results
                ),
            }
        )

    total_indexing = (
        total_chunking
        + total_embedding
    )

    return {
        "strategy": strategy,
        "documents": document_results,
        "summary": {
            "chunks": total_chunks,
            "chunking_seconds": (
                total_chunking
            ),
            "embedding_seconds": (
                total_embedding
            ),
            "indexing_seconds": (
                total_indexing
            ),
            "evaluated": (
                metrics.evaluated
            ),
            "hit_at_1": (
                metrics.hit_at_1_rate
            ),
            "hit_at_3": (
                metrics.hit_at_3_rate
            ),
            "mrr": metrics.mrr,
        },
    }


def print_summary(
    results: list[dict],
) -> None:
    print()
    print(
        "=" * 72
    )
    print(
        "COMPARAISON A/B"
    )
    print(
        "=" * 72
    )

    print()
    print(
        f"{'Stratégie':<12}"
        f"{'Chunks':>9}"
        f"{'Chunking':>12}"
        f"{'Embed':>12}"
        f"{'Total':>12}"
        f"{'Hit@1':>9}"
        f"{'Hit@3':>9}"
        f"{'MRR':>9}"
    )

    print(
        "-" * 84
    )

    for result in results:
        summary = result["summary"]

        print(
            f"{result['strategy']:<12}"
            f"{summary['chunks']:>9}"
            f"{summary['chunking_seconds']:>11.1f}s"
            f"{summary['embedding_seconds']:>11.1f}s"
            f"{summary['indexing_seconds']:>11.1f}s"
            f"{summary['hit_at_1']:>9.3f}"
            f"{summary['hit_at_3']:>9.3f}"
            f"{summary['mrr']:>9.3f}"
        )


def main() -> None:
    dataset = load_dataset()

    print(
        "=" * 72
    )
    print(
        "BENCHMARK A/B DU CHUNKING"
    )
    print(
        "=" * 72
    )

    print(
        f"Dataset : "
        f"{dataset.get('dataset', 'inconnu')}"
    )

    print(
        f"Documents : "
        f"{len(dataset['documents'])}"
    )

    query_embeddings = (
        build_query_embeddings(
            dataset
        )
    )

    results = []

    for strategy in (
        "semantic",
        "sentence",
    ):
        result = benchmark_strategy(
            strategy,
            dataset,
            query_embeddings,
        )

        results.append(
            result
        )

    print_summary(
        results
    )

    output = {
        "dataset": dataset.get(
            "dataset"
        ),
        "strategies": results,
    }

    with RESULTS_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print(
        f"Résultats détaillés : "
        f"{RESULTS_PATH}"
    )


if __name__ == "__main__":
    main()
