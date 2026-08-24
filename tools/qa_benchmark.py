from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

import requests
from sqlmodel import Session, select

from kaliok.embeddings.ollama import (
    OLLAMA_URL,
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
DEFAULT_JUDGE_TOP_K = 3

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
    content: str
    vector_distance: float | None = None


@dataclass(frozen=True)
class AnswerabilityDecision:
    answerable: bool
    evidence_chunk_indices: list[int]


@dataclass
class AnswerabilityObservation:
    question_id: str
    expected_answerable: bool
    expected_pages: list[int]
    decision: AnswerabilityDecision | None
    chunks: list[RetrievedChunk]
    evidence_correct: bool | None = None
    protocol_error: str | None = None
    raw_output: str | None = None
    technical_error_type: str | None = None
    technical_error_message: str | None = None


class JudgeProtocolError(ValueError):
    def __init__(
        self,
        message: str,
        *,
        raw_output: str,
    ) -> None:
        super().__init__(message)
        self.raw_output = raw_output


@dataclass
class AnswerabilityMetrics:
    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0
    true_negative: int = 0
    protocol_errors: int = 0
    technical_errors: int = 0
    answerability_correct: int = 0
    evidence_correct: int = 0
    answerability_and_evidence_correct: int = 0
    evidence_false_positives: int = 0

    def add(
        self,
        *,
        expected: bool,
        predicted: bool,
        evidence_correct: bool | None = None,
    ) -> None:
        if expected and predicted:
            self.true_positive += 1
        elif not expected and predicted:
            self.false_positive += 1
        elif expected and not predicted:
            self.false_negative += 1
        else:
            self.true_negative += 1

        if expected == predicted:
            self.answerability_correct += 1

        if expected:
            if predicted and evidence_correct:
                self.evidence_correct += 1
                self.answerability_and_evidence_correct += 1
            elif predicted:
                self.evidence_false_positives += 1
        elif not predicted:
            self.answerability_and_evidence_correct += 1

    @property
    def business_false_positives(self) -> int:
        return self.false_positive

    @property
    def business_false_negatives(self) -> int:
        return self.false_negative

    @property
    def total(self) -> int:
        return (
            self.true_positive
            + self.false_positive
            + self.false_negative
            + self.true_negative
        )

    @property
    def accuracy(self) -> float:
        if not self.total:
            return 0.0
        return (
            self.true_positive
            + self.true_negative
        ) / self.total

    @property
    def precision(self) -> float:
        predicted_positive = (
            self.true_positive
            + self.false_positive
        )
        if not predicted_positive:
            return 0.0
        return (
            self.true_positive
            / predicted_positive
        )

    @property
    def recall(self) -> float:
        expected_positive = (
            self.true_positive
            + self.false_negative
        )
        if not expected_positive:
            return 0.0
        return (
            self.true_positive
            / expected_positive
        )


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
            content=chunk.content,
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
            content=page_mapping[
                chunk_id
            ].content,
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


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark retrieval et expérience "
            "Answerability Gate V1."
        )
    )
    parser.add_argument(
        "--answerability-gate",
        action="store_true",
        help=(
            "Active le juge local sur les trois "
            "meilleurs chunks vectoriels."
        ),
    )
    parser.add_argument(
        "--judge-model",
        default=os.getenv(
            "KALIOK_ANSWERABILITY_MODEL"
        ),
        help=(
            "Modèle génératif Ollama local. "
            "Alternative : variable "
            "KALIOK_ANSWERABILITY_MODEL."
        ),
    )
    parser.add_argument(
        "--judge-top-k",
        type=positive_integer,
        default=DEFAULT_JUDGE_TOP_K,
        help=(
            "Nombre de chunks vectoriels transmis "
            "au juge (défaut : 3)."
        ),
    )
    return parser.parse_args()


def positive_integer(value: str) -> int:
    parsed = int(value)

    if parsed <= 0:
        raise argparse.ArgumentTypeError(
            "La valeur doit être strictement positive."
        )

    return parsed


def judge_answerability(
    question: str,
    chunks: list[RetrievedChunk],
    *,
    model: str,
) -> AnswerabilityDecision:
    passages = "\n\n".join(
        (
            f"CHUNK {index}\n"
            f"{chunk.content}"
        )
        for index, chunk in enumerate(chunks)
    )

    prompt = (
        "Tu es un juge d'answerability documentaire.\n"
        "Décide uniquement si les passages fournis contiennent "
        "assez d'information explicite pour répondre précisément "
        "à la question.\n"
        "N'utilise aucune connaissance externe et ne génère jamais "
        "la réponse finale.\n"
        "Si l'information est absente, seulement suggérée, ou "
        "insuffisante pour répondre précisément, answerable doit "
        "être false.\n"
        "evidence_chunk_indices contient uniquement les indices des "
        "passages qui justifient la décision true; utilise une liste "
        "vide lorsque answerable est false.\n\n"
        f"QUESTION\n{question}\n\n"
        f"PASSAGES\n{passages}"
    )

    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": {
                "type": "object",
                "properties": {
                    "answerable": {
                        "type": "boolean",
                    },
                    "evidence_chunk_indices": {
                        "type": "array",
                        "items": {
                            "type": "integer",
                        },
                    },
                },
                "required": [
                    "answerable",
                    "evidence_chunk_indices",
                ],
                "additionalProperties": False,
            },
            "options": {
                "temperature": 0,
            },
        },
        timeout=300,
    )
    response.raise_for_status()

    payload = response.json()
    raw_output = str(
        payload.get("response", "")
    )

    try:
        raw_decision = json.loads(
            raw_output
        )
    except (json.JSONDecodeError, TypeError) as error:
        raise JudgeProtocolError(
            "Sortie juge invalide : JSON.",
            raw_output=raw_output,
        ) from error

    if not isinstance(raw_decision, dict):
        raise JudgeProtocolError(
            "Sortie juge invalide : objet JSON attendu.",
            raw_output=raw_output,
        )

    answerable = raw_decision.get(
        "answerable"
    )
    evidence_indices = raw_decision.get(
        "evidence_chunk_indices"
    )

    if not isinstance(answerable, bool):
        raise JudgeProtocolError(
            "Sortie juge invalide : answerable.",
            raw_output=raw_output,
        )

    if (
        not isinstance(evidence_indices, list)
        or any(
            not isinstance(index, int)
            or isinstance(index, bool)
            or index < 0
            or index >= len(chunks)
            for index in evidence_indices
        )
    ):
        raise JudgeProtocolError(
            "Sortie juge invalide : "
            "evidence_chunk_indices.",
            raw_output=raw_output,
        )

    if not answerable and evidence_indices:
        raise JudgeProtocolError(
            "Sortie juge incohérente : des preuves "
            "sont indiquées pour answerable=false.",
            raw_output=raw_output,
        )

    return AnswerabilityDecision(
        answerable=answerable,
        evidence_chunk_indices=list(
            dict.fromkeys(evidence_indices)
        ),
    )


def evaluate_answerability_question(
    *,
    question_id: str,
    question: str,
    expected_answerable: bool,
    expected_pages: list[int],
    chunks: list[RetrievedChunk],
    model: str,
    metrics: AnswerabilityMetrics,
) -> AnswerabilityObservation:
    try:
        decision = judge_answerability(
            question,
            chunks,
            model=model,
        )
    except JudgeProtocolError as error:
        metrics.protocol_errors += 1

        print(
            "  gate     → ERREUR DE PROTOCOLE"
        )
        print(f"  ID       : {question_id}")
        print(f"  Erreur   : {error}")
        print("  Sortie brute :")
        print(error.raw_output)

        return AnswerabilityObservation(
            question_id=question_id,
            expected_answerable=expected_answerable,
            expected_pages=expected_pages,
            decision=None,
            chunks=chunks,
            protocol_error=str(error),
            raw_output=error.raw_output,
        )
    except requests.exceptions.RequestException as error:
        metrics.technical_errors += 1
        error_type = type(error).__name__

        print(
            "  gate     → ERREUR TECHNIQUE"
        )
        print(f"  ID       : {question_id}")
        print(f"  Type     : {error_type}")
        print(f"  Message  : {error}")

        return AnswerabilityObservation(
            question_id=question_id,
            expected_answerable=expected_answerable,
            expected_pages=expected_pages,
            decision=None,
            chunks=chunks,
            technical_error_type=error_type,
            technical_error_message=str(error),
        )

    evidence_is_correct = None

    if expected_answerable and decision.answerable:
        expected_page_set = set(expected_pages)
        evidence_is_correct = any(
            chunk_matches_expected_pages(
                chunks[index],
                expected_page_set,
            )
            for index in decision.evidence_chunk_indices
        )

    metrics.add(
        expected=expected_answerable,
        predicted=decision.answerable,
        evidence_correct=evidence_is_correct,
    )

    print(
        "  gate     → "
        + (
            "OUI"
            if decision.answerable
            else "NON"
        )
        + " | preuves="
        f"{decision.evidence_chunk_indices}"
    )

    return AnswerabilityObservation(
        question_id=question_id,
        expected_answerable=expected_answerable,
        expected_pages=expected_pages,
        decision=decision,
        chunks=chunks,
        evidence_correct=evidence_is_correct,
    )


def print_answerability_results(
    observations: list[AnswerabilityObservation],
    metrics: AnswerabilityMetrics,
    *,
    judge_top_k: int,
) -> None:
    print()
    print("=" * 72)
    print(
        "ANSWERABILITY GATE V1 — VECTOR TOP "
        f"{judge_top_k}"
    )
    print("=" * 72)

    print()
    print("Matrice de confusion")
    print()
    print("                         Attendu")
    print("                     OUI       NON")
    print(
        "Prédit OUI      "
        f"{metrics.true_positive:>6}"
        f"{metrics.false_positive:>10}"
    )
    print(
        "Prédit NON      "
        f"{metrics.false_negative:>6}"
        f"{metrics.true_negative:>10}"
    )

    print()
    print(f"Accuracy               : {metrics.accuracy:.3f}")
    print(f"Précision answerable   : {metrics.precision:.3f}")
    print(f"Recall answerable      : {metrics.recall:.3f}")
    print(f"Faux positifs NON → OUI: {metrics.false_positive}")
    print(f"Faux négatifs OUI → NON: {metrics.false_negative}")
    print(
        "answerability_correct   : "
        f"{metrics.answerability_correct}"
    )
    print(
        "evidence_correct        : "
        f"{metrics.evidence_correct}"
    )
    print(
        "answerability_and_"
        "evidence_correct        : "
        f"{metrics.answerability_and_evidence_correct}"
    )
    print(
        "faux positifs métier    : "
        f"{metrics.business_false_positives}"
    )
    print(
        "faux négatifs métier    : "
        f"{metrics.business_false_negatives}"
    )
    print(
        "faux positifs de preuve : "
        f"{metrics.evidence_false_positives}"
    )
    print(f"protocol_errors        : {metrics.protocol_errors}")
    print(f"technical_errors       : {metrics.technical_errors}")

    protocol_errors = [
        observation
        for observation in observations
        if observation.protocol_error is not None
    ]

    technical_errors = [
        observation
        for observation in observations
        if observation.technical_error_type is not None
    ]

    errors = [
        observation
        for observation in observations
        if (
            observation.decision is not None
            and observation.expected_answerable
            != observation.decision.answerable
        )
    ]

    evidence_errors = [
        observation
        for observation in observations
        if observation.evidence_correct is False
    ]

    print(
        "IDs mal classés        : "
        + (
            ", ".join(
                observation.question_id
                for observation in errors
            )
            if errors
            else "aucun"
        )
    )
    print(
        "IDs erreurs protocole  : "
        + (
            ", ".join(
                observation.question_id
                for observation in protocol_errors
            )
            if protocol_errors
            else "aucun"
        )
    )
    print(
        "IDs erreurs de preuve  : "
        + (
            ", ".join(
                observation.question_id
                for observation in evidence_errors
            )
            if evidence_errors
            else "aucun"
        )
    )
    print(
        "IDs erreurs techniques : "
        + (
            ", ".join(
                observation.question_id
                for observation in technical_errors
            )
            if technical_errors
            else "aucun"
        )
    )

    detailed_errors = list(errors)

    for observation in evidence_errors:
        if observation not in detailed_errors:
            detailed_errors.append(observation)

    if not detailed_errors:
        return

    print()
    print("DÉTAIL DES ERREURS")

    for observation in detailed_errors:
        decision = observation.decision

        if decision is None:
            continue

        print()
        print("-" * 72)
        print(f"ID       : {observation.question_id}")
        print(
            "Attendu  : "
            + (
                "OUI"
                if observation.expected_answerable
                else "NON"
            )
        )
        print(
            "Juge     : "
            + (
                "OUI"
                if decision.answerable
                else "NON"
            )
        )
        print(
            "Preuves  : "
            f"{decision.evidence_chunk_indices}"
        )
        print(
            "Preuve OK: "
            + (
                "OUI"
                if observation.evidence_correct
                else "NON"
            )
        )
        print(
            "Pages attendues : "
            f"{observation.expected_pages}"
        )
        print(
            "Pages    : "
            + ", ".join(
                (
                    f"{chunk.page_start}"
                    if chunk.page_end in {
                        None,
                        chunk.page_start,
                    }
                    else (
                        f"{chunk.page_start}"
                        f"-{chunk.page_end}"
                    )
                )
                for chunk in observation.chunks
            )
        )

        for index, chunk in enumerate(
            observation.chunks
        ):
            print()
            print(
                f"CHUNK {index} "
                f"(pages {chunk.page_start}"
                + (
                    ""
                    if chunk.page_end in {
                        None,
                        chunk.page_start,
                    }
                    else f"-{chunk.page_end}"
                )
                + ")"
            )
            print(chunk.content)


def main() -> None:
    arguments = parse_arguments()

    if (
        arguments.answerability_gate
        and not arguments.judge_model
    ):
        raise ValueError(
            "Answerability Gate activé sans modèle. "
            "Utilisez --judge-model ou "
            "KALIOK_ANSWERABILITY_MODEL."
        )

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

    answerability_metrics = (
        AnswerabilityMetrics()
    )
    answerability_observations: list[
        AnswerabilityObservation
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

                if arguments.answerability_gate:
                    gate_chunks = (
                        vector_results_for_observation[
                            :arguments.judge_top_k
                        ]
                    )
                    answerability_observations.append(
                        evaluate_answerability_question(
                            question_id=question_id,
                            question=question,
                            expected_answerable=False,
                            expected_pages=[],
                            chunks=gate_chunks,
                            model=arguments.judge_model,
                            metrics=answerability_metrics,
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

            if arguments.answerability_gate:
                gate_chunks = (
                    vector_results_for_observation[
                        :arguments.judge_top_k
                    ]
                )
                answerability_observations.append(
                    evaluate_answerability_question(
                        question_id=question_id,
                        question=question,
                        expected_answerable=True,
                        expected_pages=expected_pages,
                        chunks=gate_chunks,
                        model=arguments.judge_model,
                        metrics=answerability_metrics,
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

    if arguments.answerability_gate:
        print_answerability_results(
            answerability_observations,
            answerability_metrics,
            judge_top_k=arguments.judge_top_k,
        )


if __name__ == "__main__":
    main()
