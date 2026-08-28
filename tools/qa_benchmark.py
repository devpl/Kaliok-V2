from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

import requests
from sqlmodel import Session, select

from kaliok.embeddings.ollama import (
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    OLLAMA_URL,
    embed_text,
    embed_texts,
)
from kaliok.embeddings.service import (
    search_similar_chunks,
)
from kaliok.indexing.service import (
    IndexDocumentResult,
    index_document,
)
from kaliok.experiments.docling_retrieval import (
    DoclingCorpus,
    DoclingNativeCorpus,
    DoclingSearchResult,
    DoclingV4Corpus,
    FusedPage,
    RankedPage,
    best_rank_by_page,
    build_docling_corpus,
    build_local_semantic_docling_corpus,
    build_native_docling_corpus,
    expected_page_rank,
    load_docling_source_blocks,
    reciprocal_rank_fusion,
    search_docling_corpus,
)
from kaliok.paths import PROJECT_ROOT, TEST_DOCUMENTS
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
    DocumentVersion,
    EmbeddingModel,
    ProcessingRun,
)


VALIDATION_DIR = (
    TEST_DOCUMENTS / "validation"
)

DATASET_PATH = (
    VALIDATION_DIR
    / "qa_validation.json"
)

RIDEAU_DATASET_PATH = PROJECT_ROOT / "qa_rideau_validation.json"
CTC_2013_DATASET_PATH = PROJECT_ROOT / "qa_ctc_2013_validation.json"
NOTICE_51423_DATASET_PATH = (
    PROJECT_ROOT / "qa_notice_51423_validation.json"
)
DEFAULT_DOCLING_QA_CORPUS = RIDEAU_DATASET_PATH.name
CTC_2013_DOCUMENT_VERSION_ID = UUID(
    "e39a5938-7836-4ea9-a4f2-3f50be479291"
)
NOTICE_51423_DOCUMENT_VERSION_ID = UUID(
    "abb78c37-1068-4798-8aed-580398b893d3"
)

RETRIEVAL_LIMIT = 10
DEFAULT_JUDGE_TOP_K = 3
RERANKER_PAGE_CONTEXT_MAX_CHARS = 2000
RERANKER_DOCLING_UNITS_PER_PAGE = 2
DEFAULT_BGE_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

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


class RerankerProtocolError(ValueError):
    def __init__(self, message: str, *, raw_output: str) -> None:
        super().__init__(message)
        self.raw_output = raw_output


@dataclass(frozen=True)
class PageCandidate:
    page: int
    rrf_score: float
    passage: str


@dataclass
class RerankerObservation:
    question_id: str
    question: str
    expected_pages: list[int]
    candidates: list[PageCandidate]
    reranked_candidates: list[PageCandidate]
    raw_output: str | None = None
    parsed_ranking: list[int] | None = None
    protocol_error: str | None = None
    technical_error_type: str | None = None
    technical_error_message: str | None = None


@dataclass
class BgeRerankerObservation:
    candidates: list[PageCandidate]
    reranked_candidates: list[PageCandidate]
    scores: list[float]
    ranking: list[int]
    error_type: str | None = None
    error_message: str | None = None


@dataclass(frozen=True)
class DoclingQACorpusConfig:
    path: Path
    expected_filename: str
    expected_document_version_id: UUID | None = None


DOCLING_QA_CORPORA = {
    RIDEAU_DATASET_PATH.name: DoclingQACorpusConfig(
        path=RIDEAU_DATASET_PATH,
        expected_filename="RIDEAU.pdf",
    ),
    CTC_2013_DATASET_PATH.name: DoclingQACorpusConfig(
        path=CTC_2013_DATASET_PATH,
        expected_filename=(
            "rapport-d-activit--s-2013-de-la-ctc-NC_1.pdf"
        ),
        expected_document_version_id=CTC_2013_DOCUMENT_VERSION_ID,
    ),
    NOTICE_51423_DATASET_PATH.name: DoclingQACorpusConfig(
        path=NOTICE_51423_DATASET_PATH,
        expected_filename="notice_51423#05.pdf",
        expected_document_version_id=NOTICE_51423_DOCUMENT_VERSION_ID,
    ),
}


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


@dataclass
class ExperimentalRetrievalMetrics:
    evaluated: int = 0
    hit_at_1: int = 0
    hit_at_3: int = 0
    hit_at_5: int = 0
    reciprocal_rank_sum: float = 0.0

    def add_rank(self, rank: int | None) -> None:
        self.evaluated += 1
        if rank is None:
            return
        self.hit_at_1 += int(rank <= 1)
        self.hit_at_3 += int(rank <= 3)
        self.hit_at_5 += int(rank <= 5)
        self.reciprocal_rank_sum += 1.0 / rank

    def rate(self, hits: int) -> float:
        return hits / self.evaluated if self.evaluated else 0.0

    @property
    def mrr(self) -> float:
        return (
            self.reciprocal_rank_sum / self.evaluated
            if self.evaluated
            else 0.0
        )


@dataclass(frozen=True)
class DoclingEmbeddingRanks:
    v1: int | None
    v3: int | None
    v4: int | None


@dataclass(frozen=True)
class EmbeddingFusionObservation:
    question_id: str
    expected_pages: list[int]
    bge_pages: list[RankedPage]
    qwen_pages: list[RankedPage]
    fused_pages: list[FusedPage]
    bge_rank: int | None
    qwen_rank: int | None
    fused_rank: int | None


def fuse_v1_embedding_pages(
    bge_pages: list[RankedPage],
    qwen_pages: list[RankedPage],
) -> list[FusedPage]:
    """Fuse the top ten page-level results from two embedding spaces."""
    def top_ten(pages: list[RankedPage]) -> list[RankedPage]:
        seen: set[int] = set()
        result: list[RankedPage] = []
        for page in pages:
            if page.page in seen:
                continue
            seen.add(page.page)
            result.append(RankedPage(page=page.page, rank=len(result) + 1))
            if len(result) == 10:
                break
        return result

    return reciprocal_rank_fusion(
        [top_ten(bge_pages), top_ten(qwen_pages)],
        rrf_k=60,
    )


@dataclass(frozen=True)
class DoclingComparisonObservation:
    question_id: str
    question: str
    expected_pages: list[int]
    current_pages: list[RankedPage]
    v1_pages: list[RankedPage]
    v3_pages: list[RankedPage]
    fusion_current_v1: list[FusedPage]
    fusion_current_v1_v3: list[FusedPage]
    current_rank: int | None
    v1_rank: int | None
    v3_rank: int | None
    fusion_current_v1_rank: int | None
    fusion_current_v1_v3_rank: int | None


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


def load_rideau_dataset() -> dict:
    return load_docling_qa_dataset(RIDEAU_DATASET_PATH.name)[0]


def load_docling_qa_dataset(
    corpus_name: str,
) -> tuple[dict, DoclingQACorpusConfig]:
    config = DOCLING_QA_CORPORA.get(Path(corpus_name).name)
    if config is None or Path(corpus_name).name != corpus_name:
        supported = ", ".join(sorted(DOCLING_QA_CORPORA))
        raise ValueError(
            f"Corpus QA Docling inconnu : {corpus_name}. "
            f"Valeurs acceptées : {supported}."
        )
    if not config.path.exists():
        raise FileNotFoundError(
            f"Corpus QA Docling introuvable : {config.path}"
        )
    with config.path.open("r", encoding="utf-8") as file:
        dataset = json.load(file)
    documents = dataset.get("documents", [])
    if (
        len(documents) != 1
        or documents[0].get("file") != config.expected_filename
    ):
        raise ValueError(
            f"{config.path.name} doit contenir uniquement "
            f"{config.expected_filename}."
        )
    return dataset, config


def filter_rideau_questions(
    questions: list[dict],
    question_id: str | None,
) -> list[dict]:
    if question_id is None:
        return questions
    selected = [
        question
        for question in questions
        if question.get("id") == question_id
    ]
    if not selected:
        raise ValueError(
            "Question RIDEAU inconnue pour --question-id : "
            f"{question_id}."
        )
    return selected


def filter_docling_qa_questions(
    questions: list[dict],
    question_id: str | None,
    *,
    corpus_name: str,
) -> list[dict]:
    if question_id is None:
        return questions
    selected = [
        question
        for question in questions
        if question.get("id") == question_id
    ]
    if not selected:
        raise ValueError(
            f"Question inconnue dans {corpus_name} pour --question-id : "
            f"{question_id}."
        )
    return selected


def validate_docling_run_for_corpus(
    run: ProcessingRun,
    version: DocumentVersion | None,
    config: DoclingQACorpusConfig,
) -> None:
    if run.status != "completed" or run.engine != "docling":
        raise ValueError(
            "Le ProcessingRun demandé doit être completed et utiliser "
            "engine='docling'."
        )
    if version is None or run.document_version_id != version.id:
        raise ValueError(
            "Le ProcessingRun Docling n'appartient pas à une "
            "DocumentVersion existante."
        )
    if version.filename != config.expected_filename:
        raise ValueError(
            "Le ProcessingRun Docling ne correspond pas au document du "
            f"corpus : attendu {config.expected_filename}, obtenu "
            f"{version.filename}."
        )
    if (
        config.expected_document_version_id is not None
        and version.id != config.expected_document_version_id
    ):
        raise ValueError(
            "Le ProcessingRun Docling appartient à une autre version du "
            f"document : attendu {config.expected_document_version_id}, "
            f"obtenu {version.id}."
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


def load_document_chunks_for_diagnostic(
    document_version_id: UUID,
) -> list[RetrievedChunk]:
    engine = create_database_engine()
    with Session(engine) as session:
        chunks = session.exec(
            select(DocumentChunk)
            .where(
                DocumentChunk.document_version_id == document_version_id
            )
            .order_by(DocumentChunk.chunk_index)
        ).all()
    return [
        RetrievedChunk(
            chunk_id=chunk.id,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
            content=chunk.content,
        )
        for chunk in chunks
    ]


def search_all_vector_chunks_for_diagnostic(
    *,
    query_embedding: list[float],
    index_result: IndexDocumentResult,
    chunk_count: int,
) -> list[RetrievedChunk]:
    if chunk_count <= 0:
        return []
    results = search_similar_chunks(
        query_embedding=query_embedding,
        embedding_model_id=index_result.embedding_model_id,
        document_version_id=index_result.document_version_id,
        limit=chunk_count,
    )
    chunk_ids = [result.chunk_id for result in results]
    return attach_pages(
        chunk_ids,
        vector_distances={
            result.chunk_id: result.distance for result in results
        },
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
    parser.add_argument(
        "--docling-retrieval",
        action="store_true",
        help=(
            "Active le benchmark QA vectoriel Docling "
            "expérimental, sans écriture en base."
        ),
    )
    parser.add_argument(
        "--docling-run-id",
        type=UUID,
        help="ProcessingRun Docling completed à évaluer.",
    )
    parser.add_argument(
        "--experimental-embedding-model",
        help=(
            "Modèle Ollama alternatif évalué uniquement en mémoire sur "
            "Docling V1/V3/V4 (ex. qwen3-embedding:0.6b)."
        ),
    )
    parser.add_argument(
        "--experimental-embedding-v1-only",
        action="store_true",
        help=(
            "Compare uniquement Docling V1 entre le modèle courant et "
            "le modèle d'embedding expérimental."
        ),
    )
    parser.add_argument(
        "--experimental-embedding-v1-v4-only",
        action="store_true",
        help=(
            "Compare uniquement Docling V1 et V4 avec le modèle "
            "d'embedding expérimental."
        ),
    )
    parser.add_argument(
        "--qa-corpus",
        choices=sorted(DOCLING_QA_CORPORA),
        default=DEFAULT_DOCLING_QA_CORPUS,
        help=(
            "Corpus QA utilisé par le benchmark Docling "
            f"(défaut : {DEFAULT_DOCLING_QA_CORPUS})."
        ),
    )
    parser.add_argument(
        "--diagnostic-retrieval",
        action="store_true",
        help=(
            "Affiche un diagnostic exhaustif du retrieval pour la "
            "question sélectionnée, sans modifier le benchmark normal."
        ),
    )
    parser.add_argument(
        "--reranker",
        action="store_true",
        help="Active le reranking Ollama du top 5 RRF expérimental.",
    )
    parser.add_argument(
        "--reranker-model",
        default=os.getenv("KALIOK_RERANKER_MODEL"),
        help=(
            "Modèle génératif Ollama local. Alternative : "
            "KALIOK_RERANKER_MODEL."
        ),
    )
    parser.add_argument(
        "--question-id",
        help=(
            "Limite le benchmark Docling à une question du corpus choisi."
        ),
    )
    parser.add_argument(
        "--bge-reranker",
        action="store_true",
        help="Active le reranker BGE CPU expérimental sur le top 5 RRF.",
    )
    parser.add_argument(
        "--bge-reranker-model",
        default=DEFAULT_BGE_RERANKER_MODEL,
        help=(
            "Modèle FlagEmbedding expérimental "
            f"(défaut : {DEFAULT_BGE_RERANKER_MODEL})."
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


def _docling_expected_rank(
    results: list[DoclingSearchResult],
    expected_pages: list[int],
) -> int | None:
    expected = set(expected_pages)
    return next(
        (
            rank
            for rank, result in enumerate(results, start=1)
            if result.unit.page in expected
        ),
        None,
    )


def _print_docling_corpus_statistics(
    corpus: DoclingCorpus | DoclingNativeCorpus | DoclingV4Corpus,
    *,
    label: str,
) -> None:
    lengths = corpus.lengths
    print()
    print("=" * 72)
    print(f"CORPUS DOCLING {label}")
    print("=" * 72)
    print(f"ContentBlock source       : {corpus.source_block_count}")
    print(f"Blocs exclus             : {corpus.excluded_block_count}")
    print(f"Unités retrieval         : {len(corpus.units)}")
    if isinstance(corpus, (DoclingCorpus, DoclingNativeCorpus, DoclingV4Corpus)):
        print("Répartition par type     :")
        for block_type, count in sorted(corpus.units_by_type.items()):
            print(f"  {block_type:<24} {count}")
    if lengths:
        print(
            "Longueur caractères     : "
            f"moy={sum(lengths) / len(lengths):.1f} "
            f"min={min(lengths)} max={max(lengths)}"
        )
    print("Unités par page          :")
    for page, count in sorted(corpus.units_by_page.items()):
        print(f"  page {page:<18} {count}")


def _print_experimental_metrics(
    current: ExperimentalRetrievalMetrics,
    v1: ExperimentalRetrievalMetrics,
    v3: ExperimentalRetrievalMetrics,
    fusion_current_v1: ExperimentalRetrievalMetrics,
    fusion_current_v1_v3: ExperimentalRetrievalMetrics,
    reranker: ExperimentalRetrievalMetrics | None = None,
    bge_reranker: ExperimentalRetrievalMetrics | None = None,
    v4: ExperimentalRetrievalMetrics | None = None,
) -> None:
    print()
    print("=" * 72)
    print("RÉSULTATS RETRIEVAL RIDEAU")
    print("=" * 72)
    print(
        f"{'Méthode':<22}{'Hit@1':>10}{'Hit@3':>10}"
        f"{'Hit@5':>10}{'MRR':>10}"
    )
    print("-" * 62)
    rows = [
        ("vector actuel", current),
        ("Docling V1", v1),
        ("Docling V3", v3),
        ("Fusion actuel + V1", fusion_current_v1),
        ("Fusion actuel + V1 + V3", fusion_current_v1_v3),
    ]
    if v4 is not None:
        rows.insert(3, ("Docling V4", v4))
    if reranker is not None:
        rows.append(("Fusion + Reranker V1", reranker))
    if bge_reranker is not None:
        rows.append(("Fusion + BGE Reranker V1", bge_reranker))
    for name, metrics in rows:
        print(
            f"{name:<22}"
            f"{metrics.rate(metrics.hit_at_1):>10.3f}"
            f"{metrics.rate(metrics.hit_at_3):>10.3f}"
            f"{metrics.rate(metrics.hit_at_5):>10.3f}"
            f"{metrics.mrr:>10.3f}"
        )


def print_embedding_model_comparison(
    *,
    current_model: str,
    current_metrics: tuple[
        ExperimentalRetrievalMetrics,
        ExperimentalRetrievalMetrics,
        ExperimentalRetrievalMetrics,
    ],
    current_timings: tuple[float, float, float, float],
    experimental_model: str,
    experimental_metrics: tuple[
        ExperimentalRetrievalMetrics,
        ExperimentalRetrievalMetrics,
        ExperimentalRetrievalMetrics,
    ],
    experimental_timings: tuple[float, float, float, float],
    fusion_metrics: ExperimentalRetrievalMetrics | None = None,
    fusion_duration: float | None = None,
) -> None:
    print()
    print("=" * 72)
    print("COMPARAISON MODÈLES D'EMBEDDING — DOCLING UNIQUEMENT")
    print("=" * 72)
    for model, metrics, timings in (
        (current_model, current_metrics, current_timings),
        (experimental_model, experimental_metrics, experimental_timings),
    ):
        print()
        print(f"Modèle : {model}")
        print(f"  temps embeddings V1        : {timings[0]:.2f} s")
        print(f"  temps embeddings V3        : {timings[1]:.2f} s")
        print(f"  temps embeddings V4        : {timings[2]:.2f} s")
        print(f"  temps embeddings questions : {timings[3]:.2f} s")
        print(
            f"  {'Représentation':<18}{'Hit@1':>9}{'Hit@3':>9}"
            f"{'Hit@5':>9}{'MRR':>9}"
        )
        for label, values in zip(("Docling V1", "Docling V3", "Docling V4"), metrics, strict=True):
            print(
                f"  {label:<18}"
                f"{values.rate(values.hit_at_1):>9.3f}"
                f"{values.rate(values.hit_at_3):>9.3f}"
                f"{values.rate(values.hit_at_5):>9.3f}"
                f"{values.mrr:>9.3f}"
            )
    if fusion_metrics is not None:
        print()
        print(
            f"{'Fusion V1 BGE + Qwen':<22}"
            f"{fusion_metrics.rate(fusion_metrics.hit_at_1):>9.3f}"
            f"{fusion_metrics.rate(fusion_metrics.hit_at_3):>9.3f}"
            f"{fusion_metrics.rate(fusion_metrics.hit_at_5):>9.3f}"
            f"{fusion_metrics.mrr:>9.3f}"
        )
        if fusion_duration is not None:
            print(f"Temps fusion V1 BGE + Qwen : {fusion_duration:.4f} s")


def print_embedding_fusion_diagnostics(
    observations: list[EmbeddingFusionObservation],
) -> None:
    changed = [
        observation
        for observation in observations
        if [page.page for page in observation.fused_pages[:5]]
        != [page.page for page in observation.bge_pages[:5]]
        and [page.page for page in observation.fused_pages[:5]]
        != [page.page for page in observation.qwen_pages[:5]]
    ]
    if not changed:
        return
    print()
    print("=" * 72)
    print("DIAGNOSTICS FUSION V1 BGE + QWEN")
    print("=" * 72)
    for observation in changed:
        print()
        print(observation.question_id)
        print(f"  expected_pages : {observation.expected_pages}")
        print(f"  rang BGE V1    : {format_rank(observation.bge_rank)}")
        print(f"  rang Qwen V1   : {format_rank(observation.qwen_rank)}")
        print(f"  rang fusion    : {format_rank(observation.fused_rank)}")
        print(f"  top 5 BGE V1   : {[page.page for page in observation.bge_pages[:5]]}")
        print(f"  top 5 Qwen V1  : {[page.page for page in observation.qwen_pages[:5]]}")
        print(f"  top 5 fusion   : {[page.page for page in observation.fused_pages[:5]]}")
        print(
            "  scores RRF     : "
            f"{[(page.page, page.score) for page in observation.fused_pages[:5]]}"
        )


def run_v1_only_embedding_comparison(
    *,
    questions: list[dict],
    v1_corpus: DoclingCorpus,
    current_embeddings: list[list[float]],
    experimental_embeddings: list[list[float]],
    experimental_model: str,
    current_corpus_embedding_duration: float,
    experimental_corpus_embedding_duration: float,
) -> None:
    current_metrics = ExperimentalRetrievalMetrics()
    experimental_metrics = ExperimentalRetrievalMetrics()
    current_question_duration = 0.0
    experimental_question_duration = 0.0
    answerable_count = 0
    unanswerable_count = 0

    for question_data in questions:
        question_id = question_data["id"]
        question = question_data["question"]
        expected_pages = question_data.get("expected_pages", [])
        answerable = bool(question_data["answerable"])

        started = time.perf_counter()
        current_query = embed_text(question)
        current_question_duration += time.perf_counter() - started
        current_results = search_docling_corpus(
            v1_corpus,
            current_embeddings,
            current_query,
            limit=RETRIEVAL_LIMIT,
        )

        started = time.perf_counter()
        experimental_query = embed_texts(
            [question], model=experimental_model
        )[0]
        experimental_question_duration += time.perf_counter() - started
        experimental_results = search_docling_corpus(
            v1_corpus,
            experimental_embeddings,
            experimental_query,
            limit=RETRIEVAL_LIMIT,
        )

        current_pages = best_rank_by_page(
            [[result.unit.page] for result in current_results]
        )
        experimental_pages = best_rank_by_page(
            [[result.unit.page] for result in experimental_results]
        )
        current_rank = (
            expected_page_rank(current_pages, expected_pages)
            if answerable
            else None
        )
        experimental_rank = (
            expected_page_rank(experimental_pages, expected_pages)
            if answerable
            else None
        )
        if answerable:
            answerable_count += 1
            current_metrics.add_rank(current_rank)
            experimental_metrics.add_rank(experimental_rank)
        else:
            unanswerable_count += 1
        print()
        print(question_id)
        print(f"  V1/{EMBEDDING_MODEL} → rang {format_rank(current_rank)}")
        print(
            f"  V1/{experimental_model} → rang "
            f"{format_rank(experimental_rank)}"
        )

    print()
    print(f"Questions répondables scorées : {answerable_count}")
    print(f"Questions sans réponse hors score : {unanswerable_count}")
    print()
    print("=" * 72)
    print("COMPARAISON EMBEDDING DOCLING V1 UNIQUEMENT")
    print("=" * 72)
    print(
        f"{'Modèle':<30}{'Hit@1':>9}{'Hit@3':>9}"
        f"{'Hit@5':>9}{'MRR':>9}"
    )
    for model, metrics in (
        (EMBEDDING_MODEL, current_metrics),
        (experimental_model, experimental_metrics),
    ):
        print(
            f"{model:<30}"
            f"{metrics.rate(metrics.hit_at_1):>9.3f}"
            f"{metrics.rate(metrics.hit_at_3):>9.3f}"
            f"{metrics.rate(metrics.hit_at_5):>9.3f}"
            f"{metrics.mrr:>9.3f}"
        )
    print()
    print(
        f"{EMBEDDING_MODEL} embeddings V1      : "
        f"{current_corpus_embedding_duration:.2f} s"
    )
    print(
        f"{EMBEDDING_MODEL} embeddings questions: "
        f"{current_question_duration:.2f} s"
    )
    print(
        f"{experimental_model} embeddings V1      : "
        f"{experimental_corpus_embedding_duration:.2f} s"
    )
    print(
        f"{experimental_model} embeddings questions: "
        f"{experimental_question_duration:.2f} s"
    )


def run_v1_v4_only_embedding_comparison(
    *,
    questions: list[dict],
    v1_corpus: DoclingCorpus,
    v4_corpus: DoclingV4Corpus,
    v1_embeddings: list[list[float]],
    v4_embeddings: list[list[float]],
    experimental_model: str,
    v1_embedding_duration: float,
    v4_embedding_duration: float,
) -> None:
    v1_metrics = ExperimentalRetrievalMetrics()
    v4_metrics = ExperimentalRetrievalMetrics()
    question_embedding_duration = 0.0
    answerable_count = 0
    unanswerable_count = 0

    for question_data in questions:
        question_id = question_data["id"]
        question = question_data["question"]
        expected_pages = question_data.get("expected_pages", [])
        answerable = bool(question_data["answerable"])

        started = time.perf_counter()
        query_embedding = embed_texts(
            [question], model=experimental_model
        )[0]
        question_embedding_duration += time.perf_counter() - started
        v1_results = search_docling_corpus(
            v1_corpus,
            v1_embeddings,
            query_embedding,
            limit=RETRIEVAL_LIMIT,
        )
        v4_results = search_docling_corpus(
            v4_corpus,
            v4_embeddings,
            query_embedding,
            limit=RETRIEVAL_LIMIT,
        )
        v1_pages = best_rank_by_page(
            [[result.unit.page] for result in v1_results]
        )
        v4_pages = best_rank_by_page(
            [[result.unit.page] for result in v4_results]
        )
        v1_rank = (
            expected_page_rank(v1_pages, expected_pages)
            if answerable
            else None
        )
        v4_rank = (
            expected_page_rank(v4_pages, expected_pages)
            if answerable
            else None
        )
        if answerable:
            answerable_count += 1
            v1_metrics.add_rank(v1_rank)
            v4_metrics.add_rank(v4_rank)
        else:
            unanswerable_count += 1
        if v1_rank != v4_rank:
            print()
            print(question_id)
            print(f"  expected_pages : {expected_pages}")
            print(f"  rang V1        : {format_rank(v1_rank)}")
            print(f"  rang V4        : {format_rank(v4_rank)}")
            print(f"  top 5 V1       : {[page.page for page in v1_pages[:5]]}")
            print(f"  top 5 V4       : {[page.page for page in v4_pages[:5]]}")

    print()
    print(f"Questions répondables scorées : {answerable_count}")
    print(f"Questions sans réponse hors score : {unanswerable_count}")
    print()
    print("=" * 72)
    print("COMPARAISON QWEN DOCLING V1 / V4 UNIQUEMENT")
    print("=" * 72)
    print(f"Modèle : {experimental_model}")
    print(
        f"{'Représentation':<20}{'Hit@1':>9}{'Hit@3':>9}"
        f"{'Hit@5':>9}{'MRR':>9}"
    )
    for label, metrics in (("Docling V1", v1_metrics), ("Docling V4", v4_metrics)):
        print(
            f"{label:<20}"
            f"{metrics.rate(metrics.hit_at_1):>9.3f}"
            f"{metrics.rate(metrics.hit_at_3):>9.3f}"
            f"{metrics.rate(metrics.hit_at_5):>9.3f}"
            f"{metrics.mrr:>9.3f}"
        )
    print()
    print(f"Temps embeddings V1        : {v1_embedding_duration:.2f} s")
    print(f"Temps embeddings V4        : {v4_embedding_duration:.2f} s")
    print(f"Temps embeddings questions : {question_embedding_duration:.2f} s")


def _diagnostic_mission_markers(
    question_id: str,
    text: str,
) -> list[str]:
    if question_id != "ctc-003":
        return []
    normalized = " ".join(text.lower().split())
    marker_terms = {
        "examen_gestion": ("examine", "gestion"),
        "jugement_comptes": ("juge", "compte"),
        "avis_controle_budgetaire": ("avis",),
    }
    return [
        label
        for label, terms in marker_terms.items()
        if all(term in normalized for term in terms)
    ]


def _diagnostic_text(text: str, max_chars: int = 3000) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}\n[… texte tronqué à {max_chars} caractères]"


def _chunk_pages(chunk: RetrievedChunk) -> list[int]:
    if chunk.page_start is None:
        return []
    end = chunk.page_end if chunk.page_end is not None else chunk.page_start
    return list(range(chunk.page_start, end + 1))


def _print_real_page_rank(
    label: str,
    ranked_pages_and_scores: list[tuple[list[int], float | None]],
    expected_pages: set[int],
) -> None:
    match = next(
        (
            (rank, score)
            for rank, (pages, score) in enumerate(
                ranked_pages_and_scores,
                start=1,
            )
            if expected_pages.intersection(pages)
        ),
        None,
    )
    if match is None:
        print(f"{label:<11}: rang - / {len(ranked_pages_and_scores)} | score -")
        return
    rank, score = match
    print(
        f"{label:<11}: rang {rank} / {len(ranked_pages_and_scores)} "
        f"| distance {format_float(score)}"
    )


def print_retrieval_diagnostic(
    *,
    question_id: str,
    question: str,
    expected_pages: list[int],
    current_ranked: list[RetrievedChunk],
    current_chunks: list[RetrievedChunk],
    v1_ranked: list[DoclingSearchResult],
    v3_ranked: list[DoclingSearchResult],
) -> None:
    expected = set(expected_pages)
    print()
    print("=" * 72)
    print(f"DIAGNOSTIC RETRIEVAL — {question_id}")
    print("=" * 72)
    print(f"Question       : {question}")
    print(f"Pages attendues: {expected_pages}")
    print("Score affiché  : distance cosinus, plus petit = meilleur")

    print()
    print("RANG RÉEL DE LA PAGE ATTENDUE")
    _print_real_page_rank(
        "actuel",
        [(_chunk_pages(chunk), chunk.vector_distance) for chunk in current_ranked],
        expected,
    )
    _print_real_page_rank(
        "Docling V1",
        [([result.unit.page], result.distance) for result in v1_ranked],
        expected,
    )
    _print_real_page_rank(
        "Docling V3",
        [([result.unit.page], result.distance) for result in v3_ranked],
        expected,
    )

    print()
    print("RETRIEVAL ACTUEL — TOP 10")
    for rank, chunk in enumerate(current_ranked[:10], start=1):
        print(
            f"\n[{rank}] distance={format_float(chunk.vector_distance)} "
            f"pages={_chunk_pages(chunk)} chunk_id={chunk.chunk_id}"
        )
        print(_diagnostic_text(chunk.content))

    current_ranks = {
        chunk.chunk_id: (rank, chunk.vector_distance)
        for rank, chunk in enumerate(current_ranked, start=1)
    }
    print()
    print("RETRIEVAL ACTUEL — TOUS LES CHUNKS DES PAGES ATTENDUES")
    for chunk in current_chunks:
        if not expected.intersection(_chunk_pages(chunk)):
            continue
        rank, score = current_ranks.get(chunk.chunk_id, (None, None))
        markers = _diagnostic_mission_markers(question_id, chunk.content)
        print(
            f"\n[rank={rank or '-'}] distance={format_float(score)} "
            f"pages={_chunk_pages(chunk)} chunk_id={chunk.chunk_id} "
            f"marqueurs_missions={markers}"
        )
        print(_diagnostic_text(chunk.content))

    print()
    print("DOCLING V1 — TOP 10")
    for rank, result in enumerate(v1_ranked[:10], start=1):
        unit = result.unit
        print(
            f"\n[{rank}] distance={format_float(result.distance)} "
            f"page={unit.page} type={getattr(unit, 'block_type', '-')} "
            f"section={unit.section_header!r} "
            f"source_id={getattr(unit, 'source_block_id', '-')}"
        )
        print(_diagnostic_text(unit.text))

    print()
    print("DOCLING V1 — TOUTES LES UNITÉS DES PAGES ATTENDUES")
    for rank, result in enumerate(v1_ranked, start=1):
        unit = result.unit
        if unit.page not in expected:
            continue
        markers = _diagnostic_mission_markers(question_id, unit.text)
        print(
            f"\n[rank={rank}] distance={format_float(result.distance)} "
            f"page={unit.page} type={getattr(unit, 'block_type', '-')} "
            f"section={unit.section_header!r} "
            f"source_id={getattr(unit, 'source_block_id', '-')} "
            f"marqueurs_missions={markers}"
        )
        print(_diagnostic_text(unit.text))

    print()
    print("DOCLING V3 — TOP 10")
    for rank, result in enumerate(v3_ranked[:10], start=1):
        unit = result.unit
        print(
            f"\n[{rank}] distance={format_float(result.distance)} "
            f"page={unit.page} "
            f"logical_type={getattr(unit, 'logical_type', '-')} "
            f"section={unit.section_header!r} "
            f"source_ids={getattr(unit, 'source_block_ids', ())}"
        )
        print(_diagnostic_text(unit.text))

    print()
    print("DOCLING V3 — TOUTES LES UNITÉS DES PAGES ATTENDUES")
    for rank, result in enumerate(v3_ranked, start=1):
        unit = result.unit
        if unit.page not in expected:
            continue
        markers = _diagnostic_mission_markers(question_id, unit.text)
        print(
            f"\n[rank={rank}] distance={format_float(result.distance)} "
            f"page={unit.page} "
            f"logical_type={getattr(unit, 'logical_type', '-')} "
            f"section={unit.section_header!r} "
            f"parent={getattr(unit, 'parent_ref', None)!r} "
            f"source_ids={getattr(unit, 'source_block_ids', ())} "
            f"marqueurs_missions={markers}"
        )
        print(_diagnostic_text(unit.text))


def build_reranker_candidates(
    fusion: list[FusedPage],
    current_results: list[RetrievedChunk],
    v1_results: list[DoclingSearchResult],
    v3_results: list[DoclingSearchResult],
    *,
    max_chars: int = RERANKER_PAGE_CONTEXT_MAX_CHARS,
) -> list[PageCandidate]:
    candidates: list[PageCandidate] = []
    for fused_page in fusion[:5]:
        passages: list[tuple[str, str]] = []
        current = next(
            (
                result
                for result in current_results
                if chunk_matches_expected_pages(result, {fused_page.page})
            ),
            None,
        )
        if current is not None:
            passages.append(("actuel", current.content))

        passages.extend(
            ("Docling V1", result.unit.text)
            for result in v1_results
            if result.unit.page == fused_page.page
        )
        passages.extend(
            ("Docling V3", result.unit.text)
            for result in v3_results
            if result.unit.page == fused_page.page
        )

        selected: list[tuple[str, str]] = []
        counts = {"actuel": 0, "Docling V1": 0, "Docling V3": 0}
        for source, text in passages:
            text = text.strip()
            source_limit = (
                1
                if source == "actuel"
                else RERANKER_DOCLING_UNITS_PER_PAGE
            )
            if not text or counts[source] >= source_limit:
                continue
            if any(
                _passages_are_near_duplicates(text, existing)
                for _, existing in selected
            ):
                continue
            selected.append((source, text))
            counts[source] += 1

        passage = "\n\n".join(
            f"[{source}]\n{text}" for source, text in selected
        )[:max_chars]
        candidates.append(
            PageCandidate(
                page=fused_page.page,
                rrf_score=fused_page.score,
                passage=passage,
            )
        )
    return candidates


def _passages_are_near_duplicates(left: str, right: str) -> bool:
    left_tokens = set(" ".join(left.lower().split()).split())
    right_tokens = set(" ".join(right.lower().split()).split())
    if not left_tokens or not right_tokens:
        return left_tokens == right_tokens
    return (
        len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
        >= 0.9
    )


def load_bge_reranker(model_name: str):
    try:
        from FlagEmbedding import FlagReranker
    except ImportError as error:
        raise RuntimeError(
            "FlagEmbedding est requis par --bge-reranker. "
            "Installez explicitement la dépendance expérimentale ; "
            "aucun fallback Mistral ne sera utilisé."
        ) from error
    return FlagReranker(
        model_name,
        use_fp16=False,
        devices=["cpu"],
    )


def evaluate_bge_reranker_question(
    question: str,
    candidates: list[PageCandidate],
    reranker,
) -> BgeRerankerObservation:
    pairs = [(question, candidate.passage) for candidate in candidates]
    try:
        raw_scores = reranker.compute_score(pairs, normalize=True)
        scores = [float(score) for score in raw_scores]
        if len(scores) != len(candidates):
            raise ValueError(
                "Nombre de scores BGE incohérent : "
                f"{len(scores)} pour {len(candidates)} candidats."
            )
        ranking = sorted(
            range(len(candidates)),
            key=lambda index: (-scores[index], index),
        )
    except Exception as error:
        return BgeRerankerObservation(
            candidates=candidates,
            reranked_candidates=list(candidates),
            scores=[],
            ranking=list(range(len(candidates))),
            error_type=type(error).__name__,
            error_message=str(error),
        )
    return BgeRerankerObservation(
        candidates=candidates,
        reranked_candidates=[candidates[index] for index in ranking],
        scores=scores,
        ranking=ranking,
    )


def rerank_page_candidates(
    question: str,
    candidates: list[PageCandidate],
    *,
    model: str,
) -> tuple[list[int], str]:
    prompt = build_reranker_prompt(question, candidates)
    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": {
                "type": "object",
                "properties": {
                    "ranking": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": len(candidates),
                        "maxItems": len(candidates),
                        "uniqueItems": True,
                    }
                },
                "required": ["ranking"],
                "additionalProperties": False,
            },
            "options": {"temperature": 0},
        },
        timeout=300,
    )
    response.raise_for_status()
    try:
        payload = response.json()
    except ValueError as error:
        raise RerankerProtocolError(
            "Sortie reranker invalide : réponse HTTP non JSON.",
            raw_output="",
        ) from error
    if not isinstance(payload, dict):
        raise RerankerProtocolError(
            "Sortie reranker invalide : enveloppe HTTP.",
            raw_output=str(payload),
        )
    raw_output = str(payload.get("response", ""))
    try:
        parsed = json.loads(raw_output)
    except (json.JSONDecodeError, TypeError) as error:
        raise RerankerProtocolError(
            "Sortie reranker invalide : JSON.",
            raw_output=raw_output,
        ) from error
    ranking = parsed.get("ranking") if isinstance(parsed, dict) else None
    if not isinstance(parsed, dict) or set(parsed) != {"ranking"}:
        raise RerankerProtocolError(
            "Sortie reranker invalide : objet attendu.",
            raw_output=raw_output,
        )
    if (
        not isinstance(ranking, list)
        or any(
            not isinstance(index, int) or isinstance(index, bool)
            for index in ranking
        )
        or len(ranking) != len(candidates)
        or set(ranking) != set(range(len(candidates)))
    ):
        raise RerankerProtocolError(
            "Sortie reranker invalide : ranking.",
            raw_output=raw_output,
        )
    return ranking, raw_output


def build_reranker_prompt(
    question: str,
    candidates: list[PageCandidate],
) -> str:
    passages = "\n\n".join(
        f"CANDIDAT {index} — PAGE {candidate.page}\n{candidate.passage}"
        for index, candidate in enumerate(candidates)
    )
    return (
        "Tu es un reranker documentaire. Classe uniquement les pages "
        "candidates selon leur capacité à contenir la réponse explicite "
        "à la question. N'utilise aucune connaissance externe et ne "
        "réponds jamais à la question. Retourne chaque indice candidat "
        "exactement une fois.\n\n"
        f"QUESTION\n{question}\n\nCANDIDATS\n{passages}"
    )


def evaluate_reranker_question(
    *,
    question_id: str,
    question: str,
    expected_pages: list[int],
    candidates: list[PageCandidate],
    model: str,
) -> RerankerObservation:
    try:
        ranking, raw_output = rerank_page_candidates(
            question,
            candidates,
            model=model,
        )
    except RerankerProtocolError as error:
        return RerankerObservation(
            question_id=question_id,
            question=question,
            expected_pages=expected_pages,
            candidates=candidates,
            reranked_candidates=list(candidates),
            raw_output=error.raw_output,
            protocol_error=str(error),
        )
    except requests.exceptions.RequestException as error:
        return RerankerObservation(
            question_id=question_id,
            question=question,
            expected_pages=expected_pages,
            candidates=candidates,
            reranked_candidates=list(candidates),
            technical_error_type=type(error).__name__,
            technical_error_message=str(error),
        )
    return RerankerObservation(
        question_id=question_id,
        question=question,
        expected_pages=expected_pages,
        candidates=candidates,
        reranked_candidates=[candidates[index] for index in ranking],
        raw_output=raw_output,
        parsed_ranking=ranking,
    )


def _print_fusion_diagnostics(
    observations: list[DoclingComparisonObservation],
) -> None:
    print()
    print("=" * 72)
    print("DIAGNOSTIC — FUSION RRF PAR PAGE")
    print("=" * 72)
    for observation in observations:
        individual_ranks = [
            rank
            for rank in (
                observation.current_rank,
                observation.v1_rank,
                observation.v3_rank,
            )
            if rank is not None
        ]
        best_individual = min(individual_ranks) if individual_ranks else None
        if best_individual is None or (
            observation.fusion_current_v1_rank == best_individual
            and observation.fusion_current_v1_v3_rank == best_individual
        ):
            continue

        print()
        print(observation.question_id)
        print(f"  Question        : {observation.question}")
        print(f"  Pages attendues : {observation.expected_pages}")
        print(
            "  Rangs           : actuel="
            f"{format_rank(observation.current_rank)}, V1="
            f"{format_rank(observation.v1_rank)}, V3="
            f"{format_rank(observation.v3_rank)}, actuel+V1="
            f"{format_rank(observation.fusion_current_v1_rank)}, "
            "actuel+V1+V3="
            f"{format_rank(observation.fusion_current_v1_v3_rank)}"
        )
        print(
            "  Top 5 actuel    : "
            f"{[item.page for item in observation.current_pages[:5]]}"
        )
        print(
            "  Top 5 V1        : "
            f"{[item.page for item in observation.v1_pages[:5]]}"
        )
        print(
            "  Top 5 V3        : "
            f"{[item.page for item in observation.v3_pages[:5]]}"
        )
        for label, fusion in (
            ("actuel+V1", observation.fusion_current_v1),
            ("actuel+V1+V3", observation.fusion_current_v1_v3),
        ):
            print(f"  Top 5 fusion {label} :")
            for item in fusion[:5]:
                print(f"    page={item.page} score={item.score:.6f}")


def _print_reranker_diagnostics(
    observations: list[RerankerObservation],
) -> None:
    print()
    print("=" * 72)
    print("DIAGNOSTIC — RERANKER V1")
    print("=" * 72)
    for observation in observations:
        expected = set(observation.expected_pages)
        rrf_rank = next(
            (
                rank
                for rank, candidate in enumerate(
                    observation.candidates, start=1
                )
                if candidate.page in expected
            ),
            None,
        )
        reranked_rank = next(
            (
                rank
                for rank, candidate in enumerate(
                    observation.reranked_candidates, start=1
                )
                if candidate.page in expected
            ),
            None,
        )
        if rrf_rank == reranked_rank:
            continue
        print()
        print(observation.question_id)
        print(f"  Question        : {observation.question}")
        print(f"  Pages attendues : {observation.expected_pages}")
        print(f"  Rang RRF        : {format_rank(rrf_rank)}")
        print(f"  Rang reranké    : {format_rank(reranked_rank)}")
        print(
            "  Top 5 RRF       : "
            f"{[item.page for item in observation.candidates]}"
        )
        print(
            "  Top 5 reranké   : "
            f"{[item.page for item in observation.reranked_candidates]}"
        )
        old_score = rrf_rank or float("inf")
        new_score = reranked_rank or float("inf")
        if new_score > old_score:
            print("  Régression — passages transmis :")
            for index, candidate in enumerate(observation.candidates):
                print(f"    [{index}] page={candidate.page}")
                print(f"      {candidate.passage[:500]}")
            print(f"  Sortie brute    : {observation.raw_output!r}")
            print(f"  Classement parsé: {observation.parsed_ranking}")


def print_benchmark_timings(
    *,
    loading: float,
    v1_construction: float,
    v1_embeddings: float,
    v3_construction: float,
    v3_embeddings: float,
    retrieval: float,
    fusion: float,
    reranker_durations: list[tuple[str, float]],
    total: float,
    bge_loading: float | None = None,
    bge_durations: list[tuple[str, float]] | None = None,
    v4_construction: float | None = None,
    v4_embeddings: float | None = None,
) -> None:
    print()
    print("=" * 72)
    print("PERFORMANCES BENCHMARK")
    print("=" * 72)
    rows = [
        ("Chargement corpus/database", loading),
        ("Construction Docling V1", v1_construction),
        ("Embeddings Docling V1", v1_embeddings),
        ("Construction Docling V3", v3_construction),
        ("Embeddings Docling V3", v3_embeddings),
    ]
    if v4_construction is not None and v4_embeddings is not None:
        rows.extend(
            [
                ("Construction Docling V4", v4_construction),
                ("Embeddings Docling V4", v4_embeddings),
            ]
        )
    rows.extend(
        [("Retrieval questions", retrieval), ("Fusion RRF", fusion)]
    )
    for label, duration in rows:
        print(f"{label:<34}: {duration:>8.2f} s")
    if bge_loading is not None:
        print(f"{'Chargement BGE':<34}: {bge_loading:>8.2f} s")
    if reranker_durations:
        print()
        print("Reranker:")
        for question_id, duration in reranker_durations:
            print(f"{question_id:<34}: {duration:>8.2f} s")
        values = [duration for _, duration in reranker_durations]
        reranker_total = sum(values)
        print(f"{'total reranker':<34}: {reranker_total:>8.2f} s")
        print(
            f"{'moyenne/appel':<34}: "
            f"{reranker_total / len(values):>8.2f} s"
        )
        print(f"{'min':<34}: {min(values):>8.2f} s")
        print(f"{'max':<34}: {max(values):>8.2f} s")
    if bge_durations:
        values = [duration for _, duration in bge_durations]
        bge_total = sum(values)
        print()
        print(f"{'total BGE':<34}: {bge_total:>8.2f} s")
        print(
            f"{'moyenne/appel BGE':<34}: "
            f"{bge_total / len(values):>8.3f} s"
        )
        print(f"{'min BGE':<34}: {min(values):>8.3f} s")
        print(f"{'max BGE':<34}: {max(values):>8.3f} s")
    print()
    print(f"{'TOTAL BENCHMARK':<34}: {total:>8.2f} s")


def run_docling_retrieval_benchmark(
    processing_run_id: UUID,
    *,
    reranker_enabled: bool = False,
    reranker_model: str | None = None,
    question_id: str | None = None,
    bge_reranker_enabled: bool = False,
    bge_reranker_model: str = DEFAULT_BGE_RERANKER_MODEL,
    qa_corpus: str = DEFAULT_DOCLING_QA_CORPUS,
    diagnostic_retrieval: bool = False,
    experimental_embedding_model: str | None = None,
    experimental_embedding_v1_only: bool = False,
    experimental_embedding_v1_v4_only: bool = False,
) -> None:
    benchmark_started = time.perf_counter()
    loading_started = time.perf_counter()
    dataset, corpus_config = load_docling_qa_dataset(qa_corpus)
    documents = dataset.get("documents", [])
    questions = filter_docling_qa_questions(
        documents[0].get("questions", []),
        question_id,
        corpus_name=qa_corpus,
    )

    engine = create_database_engine()
    with Session(engine) as session:
        run, source_blocks = load_docling_source_blocks(
            session,
            processing_run_id,
        )
        version = session.get(DocumentVersion, run.document_version_id)
        validate_docling_run_for_corpus(run, version, corpus_config)
        embedding_model = session.exec(
            select(EmbeddingModel).where(
                EmbeddingModel.provider == "ollama",
                EmbeddingModel.model_name == EMBEDDING_MODEL,
                EmbeddingModel.dimensions == EMBEDDING_DIMENSIONS,
            )
        ).first()
        if embedding_model is None:
            raise ValueError(
                "Modèle d'embedding courant absent de PostgreSQL."
            )
    loading_duration = time.perf_counter() - loading_started

    v1_construction_started = time.perf_counter()
    v1_corpus = build_docling_corpus(source_blocks)
    v1_construction_duration = (
        time.perf_counter() - v1_construction_started
    )
    if experimental_embedding_v1_only:
        if experimental_embedding_model is None:
            raise ValueError(
                "Le mode V1-only exige un modèle d'embedding expérimental."
            )
        if reranker_enabled or bge_reranker_enabled:
            raise ValueError(
                "Le mode embedding V1-only est incompatible avec les "
                "rerankers."
            )
        if not v1_corpus.units:
            raise ValueError("Le corpus Docling V1 ne produit aucune unité.")
        _print_docling_corpus_statistics(v1_corpus, label="V1 — PAR BLOC")
        print()
        print("Embedding en mémoire des unités Docling V1...")
        started = time.perf_counter()
        current_v1_embeddings = embed_texts(
            [unit.text for unit in v1_corpus.units]
        )
        current_v1_duration = time.perf_counter() - started
        print(
            "Embedding expérimental V1 en mémoire avec "
            f"{experimental_embedding_model}..."
        )
        started = time.perf_counter()
        experimental_v1_embeddings = embed_texts(
            [unit.text for unit in v1_corpus.units],
            model=experimental_embedding_model,
        )
        experimental_v1_duration = time.perf_counter() - started
        run_v1_only_embedding_comparison(
            questions=questions,
            v1_corpus=v1_corpus,
            current_embeddings=current_v1_embeddings,
            experimental_embeddings=experimental_v1_embeddings,
            experimental_model=experimental_embedding_model,
            current_corpus_embedding_duration=current_v1_duration,
            experimental_corpus_embedding_duration=(
                experimental_v1_duration
            ),
        )
        return
    if experimental_embedding_v1_v4_only:
        if experimental_embedding_model is None:
            raise ValueError(
                "Le mode V1/V4-only exige un modèle d'embedding "
                "expérimental."
            )
        if reranker_enabled or bge_reranker_enabled:
            raise ValueError(
                "Le mode embedding V1/V4-only est incompatible avec "
                "les rerankers."
            )
        v4_corpus = build_local_semantic_docling_corpus(source_blocks)
        if not v1_corpus.units or not v4_corpus.units:
            raise ValueError(
                "Les corpus Docling V1/V4 ne produisent aucune unité."
            )
        _print_docling_corpus_statistics(v1_corpus, label="V1 — PAR BLOC")
        _print_docling_corpus_statistics(
            v4_corpus, label="V4 — GROUPES LOCAUX"
        )
        print()
        print(
            "Embedding expérimental V1 en mémoire avec "
            f"{experimental_embedding_model}..."
        )
        started = time.perf_counter()
        experimental_v1_embeddings = embed_texts(
            [unit.text for unit in v1_corpus.units],
            model=experimental_embedding_model,
        )
        experimental_v1_duration = time.perf_counter() - started
        print(
            "Embedding expérimental V4 en mémoire avec "
            f"{experimental_embedding_model}..."
        )
        started = time.perf_counter()
        experimental_v4_embeddings = embed_texts(
            [unit.text for unit in v4_corpus.units],
            model=experimental_embedding_model,
        )
        experimental_v4_duration = time.perf_counter() - started
        run_v1_v4_only_embedding_comparison(
            questions=questions,
            v1_corpus=v1_corpus,
            v4_corpus=v4_corpus,
            v1_embeddings=experimental_v1_embeddings,
            v4_embeddings=experimental_v4_embeddings,
            experimental_model=experimental_embedding_model,
            v1_embedding_duration=experimental_v1_duration,
            v4_embedding_duration=experimental_v4_duration,
        )
        return
    v3_construction_started = time.perf_counter()
    v3_corpus = build_native_docling_corpus(source_blocks)
    v3_construction_duration = (
        time.perf_counter() - v3_construction_started
    )
    v4_construction_started = time.perf_counter()
    v4_corpus = build_local_semantic_docling_corpus(source_blocks)
    v4_construction_duration = (
        time.perf_counter() - v4_construction_started
    )
    if not v1_corpus.units or not v3_corpus.units or not v4_corpus.units:
        raise ValueError("Le corpus Docling ne produit aucune unité retrieval.")
    _print_docling_corpus_statistics(v1_corpus, label="V1 — PAR BLOC")
    _print_docling_corpus_statistics(v3_corpus, label="V3 — NATIVE")
    _print_docling_corpus_statistics(v4_corpus, label="V4 — GROUPES LOCAUX")
    print()
    print("Embedding en mémoire des unités Docling V1...")
    v1_embeddings_started = time.perf_counter()
    v1_embeddings = embed_texts([unit.text for unit in v1_corpus.units])
    v1_embeddings_duration = time.perf_counter() - v1_embeddings_started
    print("Embedding en mémoire des unités Docling V3...")
    v3_embeddings_started = time.perf_counter()
    v3_embeddings = embed_texts([unit.text for unit in v3_corpus.units])
    v3_embeddings_duration = time.perf_counter() - v3_embeddings_started
    print("Embedding en mémoire des unités Docling V4...")
    v4_embeddings_started = time.perf_counter()
    v4_embeddings = embed_texts([unit.text for unit in v4_corpus.units])
    v4_embeddings_duration = time.perf_counter() - v4_embeddings_started
    experimental_v1_embeddings: list[list[float]] | None = None
    experimental_v3_embeddings: list[list[float]] | None = None
    experimental_v4_embeddings: list[list[float]] | None = None
    experimental_v1_embeddings_duration = 0.0
    experimental_v3_embeddings_duration = 0.0
    experimental_v4_embeddings_duration = 0.0
    if experimental_embedding_model is not None:
        print(
            "Embedding expérimental en mémoire avec "
            f"{experimental_embedding_model}..."
        )
        started = time.perf_counter()
        experimental_v1_embeddings = embed_texts(
            [unit.text for unit in v1_corpus.units],
            model=experimental_embedding_model,
        )
        experimental_v1_embeddings_duration = time.perf_counter() - started
        started = time.perf_counter()
        experimental_v3_embeddings = embed_texts(
            [unit.text for unit in v3_corpus.units],
            model=experimental_embedding_model,
        )
        experimental_v3_embeddings_duration = time.perf_counter() - started
        started = time.perf_counter()
        experimental_v4_embeddings = embed_texts(
            [unit.text for unit in v4_corpus.units],
            model=experimental_embedding_model,
        )
        experimental_v4_embeddings_duration = time.perf_counter() - started
    bge_reranker = None
    bge_loading_duration: float | None = None
    if bge_reranker_enabled:
        bge_loading_started = time.perf_counter()
        bge_reranker = load_bge_reranker(bge_reranker_model)
        bge_loading_duration = time.perf_counter() - bge_loading_started
        print(f"Chargement BGE                  : {bge_loading_duration:.2f} s")

    index_result = IndexDocumentResult(
        document_id=version.document_id,
        document_version_id=version.id,
        embedding_model_id=embedding_model.id,
        chunk_count=0,
        already_indexed=True,
    )
    diagnostic_current_chunks = (
        load_document_chunks_for_diagnostic(version.id)
        if diagnostic_retrieval
        else []
    )
    current_metrics = ExperimentalRetrievalMetrics()
    v1_metrics = ExperimentalRetrievalMetrics()
    v3_metrics = ExperimentalRetrievalMetrics()
    v4_metrics = ExperimentalRetrievalMetrics()
    experimental_v1_metrics = ExperimentalRetrievalMetrics()
    experimental_v3_metrics = ExperimentalRetrievalMetrics()
    experimental_v4_metrics = ExperimentalRetrievalMetrics()
    embedding_fusion_metrics = ExperimentalRetrievalMetrics()
    fusion_current_v1_metrics = ExperimentalRetrievalMetrics()
    fusion_current_v1_v3_metrics = ExperimentalRetrievalMetrics()
    reranker_metrics = ExperimentalRetrievalMetrics()
    bge_metrics = ExperimentalRetrievalMetrics()
    observations: list[DoclingComparisonObservation] = []
    reranker_observations: list[RerankerObservation] = []
    protocol_errors = 0
    technical_errors = 0
    retrieval_duration = 0.0
    fusion_duration = 0.0
    reranker_durations: list[tuple[str, float]] = []
    bge_durations: list[tuple[str, float]] = []
    bge_errors = 0
    embedding_fusion_duration = 0.0
    embedding_fusion_observations: list[EmbeddingFusionObservation] = []
    answerable_count = 0
    unanswerable_count = 0
    current_question_embeddings_duration = 0.0
    experimental_question_embeddings_duration = 0.0

    print()
    print("=" * 72)
    print(f"BENCHMARK : {dataset.get('dataset', qa_corpus)}")
    print("=" * 72)
    for question_number, question_data in enumerate(questions, start=1):
        question_id = question_data["id"]
        question = question_data["question"]
        answerable = bool(question_data["answerable"])
        expected_pages = question_data.get("expected_pages", [])

        retrieval_started = time.perf_counter()
        current_results = search_vector(question, index_result=index_result)
        question_embedding_started = time.perf_counter()
        query_embedding = embed_text(question)
        current_question_embeddings_duration += (
            time.perf_counter() - question_embedding_started
        )
        v1_ranked_units = search_docling_corpus(
            v1_corpus,
            v1_embeddings,
            query_embedding,
            limit=len(v1_corpus.units),
        )
        v1_results = v1_ranked_units[:RETRIEVAL_LIMIT]
        v3_results = search_docling_corpus(
            v3_corpus,
            v3_embeddings,
            query_embedding,
            limit=RETRIEVAL_LIMIT,
        )
        v4_results = search_docling_corpus(
            v4_corpus,
            v4_embeddings,
            query_embedding,
            limit=RETRIEVAL_LIMIT,
        )
        experimental_v1_pages: list[RankedPage] = []
        experimental_v1_fusion_pages: list[RankedPage] = []
        experimental_v3_pages: list[RankedPage] = []
        experimental_v4_pages: list[RankedPage] = []
        if experimental_embedding_model is not None:
            question_embedding_started = time.perf_counter()
            experimental_query_embedding = embed_texts(
                [question],
                model=experimental_embedding_model,
            )[0]
            experimental_question_embeddings_duration += (
                time.perf_counter() - question_embedding_started
            )
            assert experimental_v1_embeddings is not None
            assert experimental_v3_embeddings is not None
            assert experimental_v4_embeddings is not None
            experimental_v1_ranked_units = search_docling_corpus(
                v1_corpus,
                experimental_v1_embeddings,
                experimental_query_embedding,
                limit=len(v1_corpus.units),
            )
            experimental_v1_pages = best_rank_by_page(
                [
                    [result.unit.page]
                    for result in experimental_v1_ranked_units[
                        :RETRIEVAL_LIMIT
                    ]
                ]
            )
            experimental_v1_fusion_pages = best_rank_by_page(
                [[result.unit.page] for result in experimental_v1_ranked_units],
                limit=len(experimental_v1_ranked_units),
            )
            experimental_v3_pages = best_rank_by_page(
                [[result.unit.page] for result in search_docling_corpus(
                    v3_corpus,
                    experimental_v3_embeddings,
                    experimental_query_embedding,
                    limit=RETRIEVAL_LIMIT,
                )]
            )
            experimental_v4_pages = best_rank_by_page(
                [[result.unit.page] for result in search_docling_corpus(
                    v4_corpus,
                    experimental_v4_embeddings,
                    experimental_query_embedding,
                    limit=RETRIEVAL_LIMIT,
                )]
            )
        if diagnostic_retrieval:
            current_ranked_all = search_all_vector_chunks_for_diagnostic(
                query_embedding=query_embedding,
                index_result=index_result,
                chunk_count=len(diagnostic_current_chunks),
            )
            v1_ranked_all = v1_ranked_units
            v3_ranked_all = search_docling_corpus(
                v3_corpus,
                v3_embeddings,
                query_embedding,
                limit=len(v3_corpus.units),
            )
            print_retrieval_diagnostic(
                question_id=question_id,
                question=question,
                expected_pages=expected_pages,
                current_ranked=current_ranked_all,
                current_chunks=diagnostic_current_chunks,
                v1_ranked=v1_ranked_all,
                v3_ranked=v3_ranked_all,
            )
        retrieval_duration += time.perf_counter() - retrieval_started
        fusion_started = time.perf_counter()
        current_pages = best_rank_by_page(
            [
                list(
                    range(
                        result.page_start,
                        (
                            result.page_end
                            if result.page_end is not None
                            else result.page_start
                        )
                        + 1,
                    )
                )
                if result.page_start is not None
                else []
                for result in current_results
            ]
        )
        v1_pages = best_rank_by_page(
            [[result.unit.page] for result in v1_results]
        )
        v1_fusion_pages = best_rank_by_page(
            [[result.unit.page] for result in v1_ranked_units],
            limit=len(v1_ranked_units),
        )
        v3_pages = best_rank_by_page(
            [[result.unit.page] for result in v3_results]
        )
        v4_pages = best_rank_by_page(
            [[result.unit.page] for result in v4_results]
        )
        embedding_fusion_pages: list[FusedPage] = []
        if experimental_embedding_model is not None:
            embedding_fusion_started = time.perf_counter()
            embedding_fusion_pages = fuse_v1_embedding_pages(
                v1_fusion_pages,
                experimental_v1_fusion_pages,
            )
            embedding_fusion_duration += (
                time.perf_counter() - embedding_fusion_started
            )
        fusion_current_v1 = reciprocal_rank_fusion(
            [current_pages, v1_pages]
        )
        fusion_current_v1_v3 = reciprocal_rank_fusion(
            [current_pages, v1_pages, v3_pages]
        )
        fusion_duration += time.perf_counter() - fusion_started
        current_rank = (
            expected_page_rank(current_pages, expected_pages)
            if answerable
            else None
        )
        v1_rank = (
            expected_page_rank(v1_pages, expected_pages)
            if answerable
            else None
        )
        v3_rank = (
            expected_page_rank(v3_pages, expected_pages)
            if answerable
            else None
        )
        v4_rank = (
            expected_page_rank(v4_pages, expected_pages)
            if answerable
            else None
        )
        experimental_v1_rank = (
            expected_page_rank(experimental_v1_pages, expected_pages)
            if answerable and experimental_embedding_model is not None
            else None
        )
        experimental_v3_rank = (
            expected_page_rank(experimental_v3_pages, expected_pages)
            if answerable and experimental_embedding_model is not None
            else None
        )
        experimental_v4_rank = (
            expected_page_rank(experimental_v4_pages, expected_pages)
            if answerable and experimental_embedding_model is not None
            else None
        )
        embedding_fusion_rank = (
            expected_page_rank(embedding_fusion_pages, expected_pages)
            if answerable and experimental_embedding_model is not None
            else None
        )
        fusion_current_v1_rank = (
            expected_page_rank(fusion_current_v1, expected_pages)
            if answerable
            else None
        )
        fusion_current_v1_v3_rank = (
            expected_page_rank(fusion_current_v1_v3, expected_pages)
            if answerable
            else None
        )
        reranker_rank = fusion_current_v1_v3_rank
        bge_rank = fusion_current_v1_v3_rank
        candidates = (
            build_reranker_candidates(
                fusion_current_v1_v3,
                current_results,
                v1_results,
                v3_results,
            )
            if reranker_enabled or bge_reranker_enabled
            else []
        )
        if reranker_enabled:
            prompt_characters = len(
                build_reranker_prompt(question, candidates)
            )
            print(
                f"[reranker {question_number}/{len(questions)}] "
                f"{question_id} - appel Ollama..."
            )
            print(
                f"  prompt={prompt_characters} caractères | "
                f"pages candidates={len(candidates)}"
            )
            reranker_started = time.perf_counter()
            reranker_observation = evaluate_reranker_question(
                question_id=question_id,
                question=question,
                expected_pages=expected_pages,
                candidates=candidates,
                model=str(reranker_model),
            )
            reranker_duration = time.perf_counter() - reranker_started
            reranker_durations.append((question_id, reranker_duration))
            print(
                f"[reranker {question_number}/{len(questions)}] "
                f"terminé en {reranker_duration:.2f} s"
            )
            reranker_observations.append(reranker_observation)
            protocol_errors += int(
                reranker_observation.protocol_error is not None
            )
            technical_errors += int(
                reranker_observation.technical_error_type is not None
            )
            if reranker_observation.protocol_error is not None:
                print(
                    "  reranker → ERREUR DE PROTOCOLE | fallback RRF"
                )
                print(f"  ID : {question_id}")
                print(f"  Sortie brute : {reranker_observation.raw_output}")
            elif reranker_observation.technical_error_type is not None:
                print("  reranker → ERREUR TECHNIQUE | fallback RRF")
                print(f"  ID : {question_id}")
                print(
                    "  Type/message : "
                    f"{reranker_observation.technical_error_type}: "
                    f"{reranker_observation.technical_error_message}"
                )
            reranker_rank = next(
                (
                    rank
                    for rank, candidate in enumerate(
                        reranker_observation.reranked_candidates,
                        start=1,
                    )
                    if candidate.page in set(expected_pages)
                ),
                None,
            ) if answerable else None
        if bge_reranker_enabled:
            print(f"[BGE {question_number}/{len(questions)}] {question_id}")
            print(
                "  pages candidates              : "
                f"{[candidate.page for candidate in candidates]}"
            )
            print(
                "  caractères contextes          : "
                f"{[len(candidate.passage) for candidate in candidates]}"
            )
            bge_started = time.perf_counter()
            bge_observation = evaluate_bge_reranker_question(
                question,
                candidates,
                bge_reranker,
            )
            bge_duration = time.perf_counter() - bge_started
            bge_durations.append((question_id, bge_duration))
            print(f"  scoring                       : {bge_duration:.3f} s")
            print(f"  scores                        : {bge_observation.scores}")
            print(
                "  classement                    : "
                f"{[candidate.page for candidate in bge_observation.reranked_candidates]}"
            )
            if bge_observation.error_type is not None:
                bge_errors += 1
                print(
                    "  erreur/fallback RRF           : "
                    f"{bge_observation.error_type}: "
                    f"{bge_observation.error_message}"
                )
            bge_rank = next(
                (
                    rank
                    for rank, candidate in enumerate(
                        bge_observation.reranked_candidates,
                        start=1,
                    )
                    if candidate.page in set(expected_pages)
                ),
                None,
            ) if answerable else None
        if answerable:
            answerable_count += 1
            current_metrics.add_rank(current_rank)
            v1_metrics.add_rank(v1_rank)
            v3_metrics.add_rank(v3_rank)
            v4_metrics.add_rank(v4_rank)
            if experimental_embedding_model is not None:
                experimental_v1_metrics.add_rank(experimental_v1_rank)
                experimental_v3_metrics.add_rank(experimental_v3_rank)
                experimental_v4_metrics.add_rank(experimental_v4_rank)
                embedding_fusion_metrics.add_rank(embedding_fusion_rank)
            fusion_current_v1_metrics.add_rank(fusion_current_v1_rank)
            fusion_current_v1_v3_metrics.add_rank(
                fusion_current_v1_v3_rank
            )
            if reranker_enabled:
                reranker_metrics.add_rank(reranker_rank)
            if bge_reranker_enabled:
                bge_metrics.add_rank(bge_rank)
        else:
            unanswerable_count += 1

        print()
        print(question_id)
        print(f"  actuel  → rang {format_rank(current_rank)}")
        print(f"  V1      → rang {format_rank(v1_rank)}")
        print(f"  V3      → rang {format_rank(v3_rank)}")
        print(f"  V4      → rang {format_rank(v4_rank)}")
        if experimental_embedding_model is not None:
            print(
                f"  V1/{experimental_embedding_model} → rang "
                f"{format_rank(experimental_v1_rank)}"
            )
            print(
                f"  V3/{experimental_embedding_model} → rang "
                f"{format_rank(experimental_v3_rank)}"
            )
            print(
                f"  V4/{experimental_embedding_model} → rang "
                f"{format_rank(experimental_v4_rank)}"
            )
            print(
                "  Fusion V1 BGE + Qwen → rang "
                f"{format_rank(embedding_fusion_rank)}"
            )
            embedding_fusion_observations.append(
                EmbeddingFusionObservation(
                    question_id=question_id,
                    expected_pages=expected_pages,
                    bge_pages=v1_fusion_pages[:10],
                    qwen_pages=experimental_v1_fusion_pages[:10],
                    fused_pages=embedding_fusion_pages,
                    bge_rank=v1_rank,
                    qwen_rank=experimental_v1_rank,
                    fused_rank=embedding_fusion_rank,
                )
            )
        print(
            "  actuel+V1    → rang "
            f"{format_rank(fusion_current_v1_rank)}"
        )
        print(
            "  actuel+V1+V3 → rang "
            f"{format_rank(fusion_current_v1_v3_rank)}"
        )
        if reranker_enabled:
            print(f"  reranker      → rang {format_rank(reranker_rank)}")
        if bge_reranker_enabled:
            print(f"  BGE reranker  → rang {format_rank(bge_rank)}")
        observations.append(
            DoclingComparisonObservation(
                question_id=question_id,
                question=question,
                expected_pages=expected_pages,
                current_pages=current_pages,
                v1_pages=v1_pages,
                v3_pages=v3_pages,
                fusion_current_v1=fusion_current_v1,
                fusion_current_v1_v3=fusion_current_v1_v3,
                current_rank=current_rank,
                v1_rank=v1_rank,
                v3_rank=v3_rank,
                fusion_current_v1_rank=fusion_current_v1_rank,
                fusion_current_v1_v3_rank=fusion_current_v1_v3_rank,
            )
        )

    print()
    print(f"Questions répondables scorées : {answerable_count}")
    print(f"Questions sans réponse hors score : {unanswerable_count}")
    _print_experimental_metrics(
        current_metrics,
        v1_metrics,
        v3_metrics,
        fusion_current_v1_metrics,
        fusion_current_v1_v3_metrics,
        reranker_metrics if reranker_enabled else None,
        bge_metrics if bge_reranker_enabled else None,
        v4=v4_metrics,
    )
    if experimental_embedding_model is not None:
        print_embedding_model_comparison(
            current_model=EMBEDDING_MODEL,
            current_metrics=(v1_metrics, v3_metrics, v4_metrics),
            current_timings=(
                v1_embeddings_duration,
                v3_embeddings_duration,
                v4_embeddings_duration,
                current_question_embeddings_duration,
            ),
            experimental_model=experimental_embedding_model,
            experimental_metrics=(
                experimental_v1_metrics,
                experimental_v3_metrics,
                experimental_v4_metrics,
            ),
            experimental_timings=(
                experimental_v1_embeddings_duration,
                experimental_v3_embeddings_duration,
                experimental_v4_embeddings_duration,
                experimental_question_embeddings_duration,
            ),
            fusion_metrics=embedding_fusion_metrics,
            fusion_duration=embedding_fusion_duration,
        )
        print_embedding_fusion_diagnostics(
            embedding_fusion_observations
        )
    _print_fusion_diagnostics(observations)
    if reranker_enabled:
        print()
        print(f"protocol_errors  : {protocol_errors}")
        print(f"technical_errors : {technical_errors}")
        _print_reranker_diagnostics(reranker_observations)
    if bge_reranker_enabled:
        print()
        print(f"bge_scoring_errors : {bge_errors}")
    total_duration = time.perf_counter() - benchmark_started
    print_benchmark_timings(
        loading=loading_duration,
        v1_construction=v1_construction_duration,
        v1_embeddings=v1_embeddings_duration,
        v3_construction=v3_construction_duration,
        v3_embeddings=v3_embeddings_duration,
        retrieval=retrieval_duration,
        fusion=fusion_duration,
        reranker_durations=reranker_durations,
        total=total_duration,
        bge_loading=bge_loading_duration,
        bge_durations=bge_durations,
        v4_construction=v4_construction_duration,
        v4_embeddings=v4_embeddings_duration,
    )


def main() -> None:
    arguments = parse_arguments()

    if arguments.docling_retrieval:
        if arguments.docling_run_id is None:
            raise ValueError(
                "--docling-retrieval exige --docling-run-id."
            )
        if arguments.reranker and not arguments.reranker_model:
            raise ValueError(
                "--reranker exige --reranker-model ou "
                "KALIOK_RERANKER_MODEL."
            )
        if arguments.diagnostic_retrieval and arguments.question_id is None:
            raise ValueError(
                "--diagnostic-retrieval exige --question-id."
            )
        if (
            arguments.experimental_embedding_v1_only
            and arguments.experimental_embedding_model is None
        ):
            raise ValueError(
                "--experimental-embedding-v1-only exige "
                "--experimental-embedding-model."
            )
        if (
            arguments.experimental_embedding_v1_v4_only
            and arguments.experimental_embedding_model is None
        ):
            raise ValueError(
                "--experimental-embedding-v1-v4-only exige "
                "--experimental-embedding-model."
            )
        if (
            arguments.experimental_embedding_v1_only
            and arguments.experimental_embedding_v1_v4_only
        ):
            raise ValueError(
                "Les modes embedding V1-only et V1/V4-only sont "
                "mutuellement exclusifs."
            )
        if arguments.experimental_embedding_v1_only and (
            arguments.reranker or arguments.bge_reranker
        ):
            raise ValueError(
                "--experimental-embedding-v1-only est incompatible avec "
                "les rerankers."
            )
        if arguments.experimental_embedding_v1_v4_only and (
            arguments.reranker or arguments.bge_reranker
        ):
            raise ValueError(
                "--experimental-embedding-v1-v4-only est incompatible "
                "avec les rerankers."
            )
        if arguments.diagnostic_retrieval and (
            arguments.reranker or arguments.bge_reranker
        ):
            raise ValueError(
                "--diagnostic-retrieval est incompatible avec les "
                "rerankers."
            )
        if (
            arguments.experimental_embedding_model is not None
            and (arguments.bge_reranker or arguments.reranker)
        ):
            raise ValueError(
                "--experimental-embedding-model est incompatible avec "
                "les rerankers BGE et Mistral."
            )
        run_docling_retrieval_benchmark(
            arguments.docling_run_id,
            reranker_enabled=arguments.reranker,
            reranker_model=arguments.reranker_model,
            question_id=arguments.question_id,
            bge_reranker_enabled=arguments.bge_reranker,
            bge_reranker_model=arguments.bge_reranker_model,
            qa_corpus=arguments.qa_corpus,
            diagnostic_retrieval=arguments.diagnostic_retrieval,
            experimental_embedding_model=(
                arguments.experimental_embedding_model
            ),
            experimental_embedding_v1_only=(
                arguments.experimental_embedding_v1_only
            ),
            experimental_embedding_v1_v4_only=(
                arguments.experimental_embedding_v1_v4_only
            ),
        )
        return

    if arguments.reranker:
        raise ValueError("--reranker exige --docling-retrieval.")
    if arguments.bge_reranker:
        raise ValueError("--bge-reranker exige --docling-retrieval.")
    if arguments.diagnostic_retrieval:
        raise ValueError("--diagnostic-retrieval exige --docling-retrieval.")
    if arguments.question_id is not None:
        raise ValueError("--question-id exige --docling-retrieval.")
    if arguments.qa_corpus != DEFAULT_DOCLING_QA_CORPUS:
        raise ValueError("--qa-corpus exige --docling-retrieval.")
    if arguments.experimental_embedding_model is not None:
        raise ValueError(
            "--experimental-embedding-model exige --docling-retrieval."
        )
    if arguments.experimental_embedding_v1_only:
        raise ValueError(
            "--experimental-embedding-v1-only exige --docling-retrieval."
        )
    if arguments.experimental_embedding_v1_v4_only:
        raise ValueError(
            "--experimental-embedding-v1-v4-only exige "
            "--docling-retrieval."
        )

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
