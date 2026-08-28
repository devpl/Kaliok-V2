from __future__ import annotations

import importlib.util
import sys
from types import SimpleNamespace
from pathlib import Path
from uuid import uuid4

import pytest
import requests

from kaliok.experiments.docling_retrieval import (
    DoclingNativeUnit,
    DoclingRetrievalUnit,
    DoclingSearchResult,
    DoclingV4Unit,
    FusedPage,
)


QA_BENCHMARK_PATH = (
    Path(__file__).resolve().parents[1] / "tools" / "qa_benchmark.py"
)
spec = importlib.util.spec_from_file_location(
    "qa_benchmark_reranker_tests",
    QA_BENCHMARK_PATH,
)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Impossible de charger {QA_BENCHMARK_PATH}")
qa_benchmark = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = qa_benchmark
spec.loader.exec_module(qa_benchmark)


class FakeResponse:
    def __init__(self, raw_output: str, *, error=None):
        self.raw_output = raw_output
        self.error = error

    def raise_for_status(self):
        if self.error is not None:
            raise self.error

    def json(self):
        return {"response": self.raw_output}


def _candidates():
    return [
        qa_benchmark.PageCandidate(page=page, rrf_score=1 / 61, passage=text)
        for page, text in (
            (1, "Projet de Mme Pellegrin"),
            (6, "Prix total"),
            (11, "Perspectives"),
        )
    ]


def _five_candidates():
    return [
        qa_benchmark.PageCandidate(
            page=page,
            rrf_score=1 / (60 + rank),
            passage=f"Contexte représentatif page {page}",
        )
        for rank, page in enumerate((9, 10, 11, 7, 2), start=1)
    ]


class FakeBgeReranker:
    def __init__(self, scores=None, error=None):
        self.scores = scores or [0.1, 0.2, 0.3, 0.4, 0.5]
        self.error = error
        self.calls = []

    def compute_score(self, pairs, *, normalize):
        self.calls.append((pairs, normalize))
        if self.error is not None:
            raise self.error
        return self.scores


def test_page_context_uses_only_fusion_pool_and_deduplicates_passages():
    fusion = [
        FusedPage(page=page, score=1 / (60 + page))
        for page in range(1, 7)
    ]
    current = [
        qa_benchmark.RetrievedChunk(
            chunk_id=uuid4(),
            page_start=1,
            page_end=1,
            content="Projet de Mme Pellegrin",
        )
    ]
    v1 = [
        DoclingSearchResult(
            unit=DoclingRetrievalUnit(
                source_block_id=uuid4(),
                page=1,
                block_type="text",
                section_header=None,
                text="Projet de Mme Pellegrin",
            ),
            distance=0.1,
        )
    ]
    v3 = [
        DoclingSearchResult(
            unit=DoclingNativeUnit(
                page=1,
                logical_type="key_value_area",
                section_header=None,
                parent_ref="#/groups/0",
                source_block_types=("text", "text"),
                source_block_ids=(uuid4(), uuid4()),
                text="Adresse : 38 rue Pasteur",
            ),
            distance=0.2,
        )
    ]

    candidates = qa_benchmark.build_reranker_candidates(
        fusion,
        current,
        v1,
        v3,
    )

    assert [candidate.page for candidate in candidates] == [1, 2, 3, 4, 5]
    assert candidates[0].passage.count("Projet de Mme Pellegrin") == 1
    assert "Adresse : 38 rue Pasteur" in candidates[0].passage
    assert len(candidates[0].passage) <= 2000


def test_reranker_accepts_a_complete_valid_ranking(monkeypatch):
    received = {}

    def fake_post(url, *, json, timeout):
        received.update(url=url, payload=json, timeout=timeout)
        return FakeResponse('{"ranking":[2,0,1]}')

    monkeypatch.setattr(qa_benchmark.requests, "post", fake_post)

    ranking, raw = qa_benchmark.rerank_page_candidates(
        "Quelles perspectives ?",
        _candidates(),
        model="mistral-local",
    )

    assert ranking == [2, 0, 1]
    assert raw == '{"ranking":[2,0,1]}'
    assert received["payload"]["options"]["temperature"] == 0
    assert received["payload"]["model"] == "mistral-local"
    assert "ne réponds jamais" in received["payload"]["prompt"]
    assert len(received["payload"]["prompt"]) == len(
        qa_benchmark.build_reranker_prompt(
            "Quelles perspectives ?",
            _candidates(),
        )
    )
    assert received["timeout"] == 300


@pytest.mark.parametrize(
    "raw_output",
    [
        '{"ranking":[0,0,2]}',
        '{"ranking":[0,1,3]}',
        '{"ranking":[0,1]}',
        '{"ranking":[0,"1",2]}',
        '{"ranking":[true,1,2]}',
        "pas du json",
    ],
)
def test_reranker_rejects_invalid_protocol(monkeypatch, raw_output):
    monkeypatch.setattr(
        qa_benchmark.requests,
        "post",
        lambda *args, **kwargs: FakeResponse(raw_output),
    )

    with pytest.raises(qa_benchmark.RerankerProtocolError):
        qa_benchmark.rerank_page_candidates(
            "Question",
            _candidates(),
            model="local",
        )


@pytest.mark.parametrize(
    "error",
    [
        requests.Timeout("trop long"),
        requests.HTTPError("500 Server Error"),
    ],
)
def test_reranker_technical_error_falls_back_to_rrf(monkeypatch, error):
    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(qa_benchmark.requests, "post", fail)
    candidates = _candidates()

    observation = qa_benchmark.evaluate_reranker_question(
        question_id="rideau-test",
        question="Question",
        expected_pages=[1],
        candidates=candidates,
        model="local",
    )

    assert observation.reranked_candidates == candidates
    assert observation.technical_error_type == type(error).__name__
    assert observation.protocol_error is None


def test_reranker_protocol_error_falls_back_to_rrf(monkeypatch):
    monkeypatch.setattr(
        qa_benchmark.requests,
        "post",
        lambda *args, **kwargs: FakeResponse('{"ranking":[0,0,2]}'),
    )
    candidates = _candidates()

    observation = qa_benchmark.evaluate_reranker_question(
        question_id="rideau-test",
        question="Question",
        expected_pages=[1],
        candidates=candidates,
        model="local",
    )

    assert observation.reranked_candidates == candidates
    assert observation.protocol_error is not None
    assert observation.parsed_ranking is None


def test_reranked_page_rank_feeds_existing_metrics(monkeypatch):
    monkeypatch.setattr(
        qa_benchmark.requests,
        "post",
        lambda *args, **kwargs: FakeResponse('{"ranking":[2,0,1]}'),
    )
    observation = qa_benchmark.evaluate_reranker_question(
        question_id="rideau-test",
        question="Question",
        expected_pages=[11],
        candidates=_candidates(),
        model="local",
    )
    rank = next(
        index
        for index, candidate in enumerate(
            observation.reranked_candidates, start=1
        )
        if candidate.page == 11
    )
    metrics = qa_benchmark.ExperimentalRetrievalMetrics()
    metrics.add_rank(rank)

    assert rank == 1
    assert metrics.hit_at_1 == 1
    assert metrics.hit_at_3 == 1
    assert metrics.hit_at_5 == 1
    assert metrics.mrr == 1.0


def test_bge_loader_constructs_the_cpu_model_once(monkeypatch):
    constructed = []

    class FakeFlagReranker:
        def __init__(self, model_name, *, use_fp16, devices):
            constructed.append((model_name, use_fp16, devices))

    monkeypatch.setitem(
        sys.modules,
        "FlagEmbedding",
        SimpleNamespace(FlagReranker=FakeFlagReranker),
    )

    loaded = qa_benchmark.load_bge_reranker("modele-test")

    assert isinstance(loaded, FakeFlagReranker)
    assert constructed == [("modele-test", False, ["cpu"])]


def test_bge_scores_exactly_five_question_context_pairs_in_one_batch():
    candidates = _five_candidates()
    reranker = FakeBgeReranker(scores=[0.5, 0.4, 0.3, 0.2, 0.1])

    observation = qa_benchmark.evaluate_bge_reranker_question(
        "Quelle page est pertinente ?",
        candidates,
        reranker,
    )

    assert reranker.calls == [
        (
            [
                ("Quelle page est pertinente ?", candidate.passage)
                for candidate in candidates
            ],
            True,
        )
    ]
    assert len(reranker.calls[0][0]) == 5
    assert observation.error_type is None


def test_bge_sorts_descending_and_preserves_rrf_order_on_ties():
    candidates = _five_candidates()
    reranker = FakeBgeReranker(scores=[0.2, 0.9, 0.9, 0.1, 0.3])

    observation = qa_benchmark.evaluate_bge_reranker_question(
        "Question",
        candidates,
        reranker,
    )

    assert observation.ranking == [1, 2, 4, 0, 3]
    assert [candidate.page for candidate in observation.reranked_candidates] == [
        10,
        11,
        2,
        9,
        7,
    ]
    assert {
        candidate.page for candidate in observation.reranked_candidates
    } == {candidate.page for candidate in candidates}


def test_bge_scoring_error_falls_back_exactly_to_rrf():
    candidates = _five_candidates()
    reranker = FakeBgeReranker(error=RuntimeError("échec CPU"))

    observation = qa_benchmark.evaluate_bge_reranker_question(
        "Question",
        candidates,
        reranker,
    )

    assert observation.reranked_candidates == candidates
    assert observation.ranking == [0, 1, 2, 3, 4]
    assert observation.scores == []
    assert observation.error_type == "RuntimeError"
    assert observation.error_message == "échec CPU"


def test_bge_reranked_rank_feeds_existing_metrics():
    observation = qa_benchmark.evaluate_bge_reranker_question(
        "Question",
        _five_candidates(),
        FakeBgeReranker(scores=[0.1, 0.2, 0.9, 0.4, 0.3]),
    )
    rank = next(
        index
        for index, candidate in enumerate(
            observation.reranked_candidates,
            start=1,
        )
        if candidate.page == 11
    )
    metrics = qa_benchmark.ExperimentalRetrievalMetrics()
    metrics.add_rank(rank)

    assert rank == 1
    assert metrics.hit_at_1 == 1
    assert metrics.hit_at_3 == 1
    assert metrics.hit_at_5 == 1
    assert metrics.mrr == 1.0


def test_benchmark_timing_report_uses_supplied_measurements(capsys):
    qa_benchmark.print_benchmark_timings(
        loading=1.25,
        v1_construction=0.10,
        v1_embeddings=2.50,
        v3_construction=0.20,
        v3_embeddings=3.50,
        retrieval=4.25,
        fusion=0.05,
        reranker_durations=[
            ("rideau-001", 10.0),
            ("rideau-002", 20.0),
        ],
        bge_loading=8.0,
        bge_durations=[
            ("rideau-001", 4.0),
            ("rideau-002", 6.0),
        ],
        total=42.0,
    )

    output = capsys.readouterr().out
    assert "Chargement corpus/database" in output
    assert "Embeddings Docling V1" in output
    assert "Embeddings Docling V3" in output
    assert "rideau-001" in output and "10.00 s" in output
    assert "total reranker" in output and "30.00 s" in output
    assert "moyenne/appel" in output and "15.00 s" in output
    assert "min" in output and "10.00 s" in output
    assert "max" in output and "20.00 s" in output
    assert "Chargement BGE" in output and "8.00 s" in output
    assert "total BGE" in output and "10.00 s" in output
    assert "moyenne/appel BGE" in output and "5.000 s" in output
    assert "TOTAL BENCHMARK" in output and "42.00 s" in output


def test_question_filter_selects_one_known_rideau_question():
    questions = [
        {"id": "rideau-001", "question": "Première"},
        {"id": "rideau-002", "question": "Deuxième"},
    ]

    selected = qa_benchmark.filter_rideau_questions(
        questions,
        "rideau-002",
    )

    assert selected == [questions[1]]


def test_question_filter_rejects_unknown_id():
    questions = [{"id": "rideau-001", "question": "Première"}]

    with pytest.raises(ValueError, match="Question RIDEAU inconnue"):
        qa_benchmark.filter_rideau_questions(
            questions,
            "rideau-999",
        )


def test_question_filter_without_id_keeps_existing_behavior():
    questions = [
        {"id": "rideau-001", "question": "Première"},
        {"id": "rideau-002", "question": "Deuxième"},
    ]

    selected = qa_benchmark.filter_rideau_questions(questions, None)

    assert selected is questions


def test_ctc_corpus_loads_independently_from_rideau():
    dataset, config = qa_benchmark.load_docling_qa_dataset(
        "qa_ctc_2013_validation.json"
    )
    document = dataset["documents"][0]

    assert dataset["dataset"] == "ctc-2013-validation-v1"
    assert document["file"] == (
        "rapport-d-activit--s-2013-de-la-ctc-NC_1.pdf"
    )
    assert len(document["questions"]) == 20
    assert sum(
        not question["answerable"] for question in document["questions"]
    ) == 2
    assert config.expected_document_version_id == (
        qa_benchmark.CTC_2013_DOCUMENT_VERSION_ID
    )


def test_notice_corpus_loads_independently_from_existing_corpora():
    dataset, config = qa_benchmark.load_docling_qa_dataset(
        "qa_notice_51423_validation.json"
    )
    document = dataset["documents"][0]

    assert dataset["dataset"] == "notice-51423-validation-v1"
    assert document["file"] == "notice_51423#05.pdf"
    assert len(document["questions"]) == 20
    assert sum(
        not question["answerable"] for question in document["questions"]
    ) == 2
    assert config.expected_document_version_id == (
        qa_benchmark.NOTICE_51423_DOCUMENT_VERSION_ID
    )


def test_rideau_remains_the_default_docling_corpus():
    dataset, config = qa_benchmark.load_docling_qa_dataset(
        qa_benchmark.DEFAULT_DOCLING_QA_CORPUS
    )

    assert qa_benchmark.DEFAULT_DOCLING_QA_CORPUS == (
        "qa_rideau_validation.json"
    )
    assert dataset == qa_benchmark.load_rideau_dataset()
    assert config.expected_filename == "RIDEAU.pdf"
    assert config.expected_document_version_id is None


def test_ctc_question_filter_selects_ctc_identifier():
    dataset, _ = qa_benchmark.load_docling_qa_dataset(
        "qa_ctc_2013_validation.json"
    )
    questions = dataset["documents"][0]["questions"]

    selected = qa_benchmark.filter_docling_qa_questions(
        questions,
        "ctc-013",
        corpus_name="qa_ctc_2013_validation.json",
    )

    assert [question["id"] for question in selected] == ["ctc-013"]


def test_notice_question_filter_selects_notice_identifier():
    dataset, _ = qa_benchmark.load_docling_qa_dataset(
        "qa_notice_51423_validation.json"
    )
    questions = dataset["documents"][0]["questions"]

    selected = qa_benchmark.filter_docling_qa_questions(
        questions,
        "notice-017",
        corpus_name="qa_notice_51423_validation.json",
    )

    assert [question["id"] for question in selected] == ["notice-017"]


def test_notice_run_must_match_document_version_and_docling_contract():
    _, config = qa_benchmark.load_docling_qa_dataset(
        "qa_notice_51423_validation.json"
    )
    version = SimpleNamespace(
        id=qa_benchmark.NOTICE_51423_DOCUMENT_VERSION_ID,
        filename="notice_51423#05.pdf",
    )
    run = SimpleNamespace(
        document_version_id=version.id,
        status="completed",
        engine="docling",
    )

    qa_benchmark.validate_docling_run_for_corpus(run, version, config)

    detached_run = SimpleNamespace(
        document_version_id=uuid4(),
        status="completed",
        engine="docling",
    )
    with pytest.raises(ValueError, match="n'appartient pas"):
        qa_benchmark.validate_docling_run_for_corpus(
            detached_run, version, config
        )

    for status, engine in (("started", "docling"), ("completed", "pdfium")):
        invalid_run = SimpleNamespace(
            document_version_id=version.id,
            status=status,
            engine=engine,
        )
        with pytest.raises(ValueError, match="completed.*engine='docling'"):
            qa_benchmark.validate_docling_run_for_corpus(
                invalid_run, version, config
            )

    wrong_document = SimpleNamespace(
        id=qa_benchmark.NOTICE_51423_DOCUMENT_VERSION_ID,
        filename="RIDEAU.pdf",
    )
    with pytest.raises(ValueError, match="ne correspond pas au document"):
        qa_benchmark.validate_docling_run_for_corpus(
            run, wrong_document, config
        )

    wrong_version = SimpleNamespace(
        id=uuid4(),
        filename="notice_51423#05.pdf",
    )
    wrong_run = SimpleNamespace(
        document_version_id=wrong_version.id,
        status="completed",
        engine="docling",
    )
    with pytest.raises(ValueError, match="autre version"):
        qa_benchmark.validate_docling_run_for_corpus(
            wrong_run, wrong_version, config
        )


def test_ctc_run_must_belong_to_configured_document_version():
    _, config = qa_benchmark.load_docling_qa_dataset(
        "qa_ctc_2013_validation.json"
    )
    correct_version = SimpleNamespace(
        id=qa_benchmark.CTC_2013_DOCUMENT_VERSION_ID,
        filename=config.expected_filename,
    )
    correct_run = SimpleNamespace(
        document_version_id=correct_version.id,
        status="completed",
        engine="docling",
    )

    qa_benchmark.validate_docling_run_for_corpus(
        correct_run,
        correct_version,
        config,
    )

    other_version = SimpleNamespace(
        id=uuid4(),
        filename=config.expected_filename,
    )
    other_run = SimpleNamespace(
        document_version_id=other_version.id,
        status="completed",
        engine="docling",
    )
    with pytest.raises(ValueError, match="autre version"):
        qa_benchmark.validate_docling_run_for_corpus(
            other_run,
            other_version,
            config,
        )


@pytest.mark.parametrize(
    ("status", "engine"),
    [("started", "docling"), ("completed", "pdfium")],
)
def test_docling_run_requires_completed_docling(status, engine):
    _, config = qa_benchmark.load_docling_qa_dataset(
        "qa_ctc_2013_validation.json"
    )
    version = SimpleNamespace(
        id=qa_benchmark.CTC_2013_DOCUMENT_VERSION_ID,
        filename=config.expected_filename,
    )
    run = SimpleNamespace(
        document_version_id=version.id,
        status=status,
        engine=engine,
    )

    with pytest.raises(ValueError, match="completed.*engine='docling'"):
        qa_benchmark.validate_docling_run_for_corpus(run, version, config)


def test_corpus_selection_does_not_change_retrieval_metrics():
    rideau_metrics = qa_benchmark.ExperimentalRetrievalMetrics()
    ctc_metrics = qa_benchmark.ExperimentalRetrievalMetrics()
    ranks = [1, 2, 5, None]

    for rank in ranks:
        rideau_metrics.add_rank(rank)
        ctc_metrics.add_rank(rank)

    assert ctc_metrics == rideau_metrics


def test_bge_cli_is_compatible_with_question_id(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "qa_benchmark.py",
            "--docling-retrieval",
            "--docling-run-id",
            "b9a5bde4-e094-4d52-abe6-3413bdecfe36",
            "--question-id",
            "rideau-022",
            "--bge-reranker",
        ],
    )

    arguments = qa_benchmark.parse_arguments()

    assert arguments.bge_reranker is True
    assert arguments.bge_reranker_model == (
        qa_benchmark.DEFAULT_BGE_RERANKER_MODEL
    )
    assert arguments.question_id == "rideau-022"
    assert arguments.qa_corpus == qa_benchmark.DEFAULT_DOCLING_QA_CORPUS


def test_ctc_cli_selects_corpus_run_and_question(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "qa_benchmark.py",
            "--docling-retrieval",
            "--qa-corpus",
            "qa_ctc_2013_validation.json",
            "--docling-run-id",
            "43e7142a-1969-40ba-a9fe-b33650404901",
            "--question-id",
            "ctc-009",
        ],
    )

    arguments = qa_benchmark.parse_arguments()

    assert arguments.qa_corpus == "qa_ctc_2013_validation.json"
    assert str(arguments.docling_run_id) == (
        "43e7142a-1969-40ba-a9fe-b33650404901"
    )
    assert arguments.question_id == "ctc-009"


def test_notice_cli_selects_corpus_run_and_question(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "qa_benchmark.py",
            "--docling-retrieval",
            "--qa-corpus",
            "qa_notice_51423_validation.json",
            "--docling-run-id",
            "f74b648b-b876-4b14-9846-c1dfcc3aa436",
            "--question-id",
            "notice-014",
        ],
    )

    arguments = qa_benchmark.parse_arguments()

    assert arguments.qa_corpus == "qa_notice_51423_validation.json"
    assert str(arguments.docling_run_id) == (
        "f74b648b-b876-4b14-9846-c1dfcc3aa436"
    )
    assert arguments.question_id == "notice-014"


def test_ctc_diagnostic_cli_selects_only_one_question(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "qa_benchmark.py",
            "--docling-retrieval",
            "--qa-corpus",
            "qa_ctc_2013_validation.json",
            "--docling-run-id",
            "43e7142a-1969-40ba-a9fe-b33650404901",
            "--question-id",
            "ctc-003",
            "--diagnostic-retrieval",
        ],
    )

    arguments = qa_benchmark.parse_arguments()

    assert arguments.diagnostic_retrieval is True
    assert arguments.question_id == "ctc-003"
    assert arguments.reranker is False
    assert arguments.bge_reranker is False


def test_diagnostic_reports_full_ranks_page_units_and_mission_markers(capsys):
    current = [
        qa_benchmark.RetrievedChunk(
            chunk_id=uuid4(),
            page_start=6 if rank == 12 else 100 + rank,
            page_end=6 if rank == 12 else 100 + rank,
            content=(
                "Elle examine la gestion. Elle juge les comptes. "
                "Elle rend des avis en contrôle budgétaire."
                if rank == 12
                else f"Chunk concurrent {rank}"
            ),
            vector_distance=rank / 100,
        )
        for rank in range(1, 13)
    ]
    v1 = [
        DoclingSearchResult(
            unit=DoclingRetrievalUnit(
                source_block_id=uuid4(),
                page=6 if rank == 11 else 100 + rank,
                block_type="text",
                section_header="Les missions" if rank == 11 else None,
                text=(
                    "Elle examine la gestion et juge les comptes ; "
                    "elle rend des avis."
                    if rank == 11
                    else f"Unité V1 {rank}"
                ),
            ),
            distance=rank / 100,
        )
        for rank in range(1, 12)
    ]
    v3 = [
        DoclingSearchResult(
            unit=DoclingNativeUnit(
                page=6 if rank == 13 else 100 + rank,
                logical_type="text",
                section_header="Les missions" if rank == 13 else None,
                parent_ref=None,
                source_block_types=("text",),
                source_block_ids=(uuid4(),),
                text=(
                    "Elle examine la gestion, juge les comptes et rend "
                    "des avis."
                    if rank == 13
                    else f"Unité V3 {rank}"
                ),
            ),
            distance=rank / 100,
        )
        for rank in range(1, 14)
    ]

    qa_benchmark.print_retrieval_diagnostic(
        question_id="ctc-003",
        question="Quelles sont les trois missions ?",
        expected_pages=[6],
        current_ranked=current,
        current_chunks=current,
        v1_ranked=v1,
        v3_ranked=v3,
    )

    output = capsys.readouterr().out
    assert "actuel     : rang 12 / 12" in output
    assert "Docling V1 : rang 11 / 11" in output
    assert "Docling V3 : rang 13 / 13" in output
    assert "RETRIEVAL ACTUEL — TOP 10" in output
    assert "TOUS LES CHUNKS DES PAGES ATTENDUES" in output
    assert "TOUTES LES UNITÉS DES PAGES ATTENDUES" in output
    assert "source_id=" in output
    assert "source_ids=" in output
    assert "examen_gestion" in output
    assert "jugement_comptes" in output
    assert "avis_controle_budgetaire" in output


def test_exhaustive_vector_diagnostic_does_not_change_normal_top_k(
    monkeypatch,
):
    received = {}
    results = [
        SimpleNamespace(chunk_id=uuid4(), distance=index / 100)
        for index in range(37)
    ]

    def fake_search(**kwargs):
        received.update(kwargs)
        return results

    monkeypatch.setattr(qa_benchmark, "search_similar_chunks", fake_search)
    monkeypatch.setattr(
        qa_benchmark,
        "attach_pages",
        lambda chunk_ids, *, vector_distances: [
            (chunk_id, vector_distances[chunk_id]) for chunk_id in chunk_ids
        ],
    )
    index_result = qa_benchmark.IndexDocumentResult(
        document_id=uuid4(),
        document_version_id=uuid4(),
        embedding_model_id=uuid4(),
        chunk_count=37,
        already_indexed=True,
    )

    diagnostic_results = (
        qa_benchmark.search_all_vector_chunks_for_diagnostic(
            query_embedding=[0.0] * 1024,
            index_result=index_result,
            chunk_count=37,
        )
    )

    assert received["limit"] == 37
    assert len(diagnostic_results) == 37
    assert qa_benchmark.RETRIEVAL_LIMIT == 10


def test_shared_embedding_primitive_uses_requested_ollama_model(monkeypatch):
    received = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"embeddings": [[1.0, 0.0], [0.0, 1.0]]}

    def fake_post(url, *, json, timeout):
        received.update(url=url, json=json, timeout=timeout)
        return Response()

    monkeypatch.setattr(qa_benchmark.requests, "post", fake_post)

    embeddings = qa_benchmark.embed_texts(
        ["un", "deux"], model="qwen3-embedding:0.6b"
    )

    assert embeddings == [[1.0, 0.0], [0.0, 1.0]]
    assert received == {
        "url": f"{qa_benchmark.OLLAMA_URL}/api/embed",
        "json": {
            "model": "qwen3-embedding:0.6b",
            "input": ["un", "deux"],
        },
        "timeout": 300,
    }


def test_embedding_spaces_remain_separate_during_docling_search():
    first = DoclingRetrievalUnit(
        source_block_id=uuid4(),
        page=1,
        block_type="text",
        section_header=None,
        text="premier",
    )
    second = DoclingRetrievalUnit(
        source_block_id=uuid4(),
        page=2,
        block_type="text",
        section_header=None,
        text="second",
    )
    corpus = qa_benchmark.DoclingCorpus(2, 0, [first, second])

    current = qa_benchmark.search_docling_corpus(
        corpus, [[1.0, 0.0], [0.0, 1.0]], [1.0, 0.0]
    )
    experimental = qa_benchmark.search_docling_corpus(
        corpus, [[0.0, 1.0], [1.0, 0.0]], [1.0, 0.0]
    )

    assert current[0].unit.page == 1
    assert experimental[0].unit.page == 2


def test_embedding_comparison_reports_separate_models_and_metrics(capsys):
    current = tuple(
        qa_benchmark.ExperimentalRetrievalMetrics() for _ in range(3)
    )
    experimental = tuple(
        qa_benchmark.ExperimentalRetrievalMetrics() for _ in range(3)
    )
    for metrics, rank in zip(current, (1, 2, 3), strict=True):
        metrics.add_rank(rank)
    for metrics, rank in zip(experimental, (3, 2, 1), strict=True):
        metrics.add_rank(rank)
    fusion = qa_benchmark.ExperimentalRetrievalMetrics()
    fusion.add_rank(1)

    qa_benchmark.print_embedding_model_comparison(
        current_model="current-model",
        current_metrics=current,
        current_timings=(1.0, 2.0, 3.0, 4.0),
        experimental_model="qwen3-embedding:0.6b",
        experimental_metrics=experimental,
        experimental_timings=(5.0, 6.0, 7.0, 8.0),
        fusion_metrics=fusion,
        fusion_duration=0.125,
    )

    output = capsys.readouterr().out
    assert "Modèle : current-model" in output
    assert "Modèle : qwen3-embedding:0.6b" in output
    assert output.count("Docling V1") == 2
    assert "temps embeddings questions : 8.00 s" in output
    assert "Fusion V1 BGE + Qwen" in output
    assert "Temps fusion V1 BGE + Qwen : 0.1250 s" in output


def test_experimental_embedding_cli_is_optional(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["qa_benchmark.py"])
    assert qa_benchmark.parse_arguments().experimental_embedding_model is None

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "qa_benchmark.py",
            "--docling-retrieval",
            "--docling-run-id",
            "b9a5bde4-e094-4d52-abe6-3413bdecfe36",
            "--experimental-embedding-model",
            "qwen3-embedding:0.6b",
        ],
    )
    arguments = qa_benchmark.parse_arguments()
    assert arguments.experimental_embedding_model == "qwen3-embedding:0.6b"


def test_v1_embedding_fusion_uses_top_ten_exact_rrf_and_no_double_count():
    bge = [
        qa_benchmark.RankedPage(page=1, rank=1),
        qa_benchmark.RankedPage(page=1, rank=2),
        *[
            qa_benchmark.RankedPage(page=page, rank=page + 1)
            for page in range(3, 13)
        ],
    ]
    qwen = [
        qa_benchmark.RankedPage(page=2, rank=1),
        qa_benchmark.RankedPage(page=1, rank=2),
    ]

    fused = qa_benchmark.fuse_v1_embedding_pages(bge, qwen)
    scores = {result.page: result.score for result in fused}

    assert scores[1] == pytest.approx(1 / 61 + 1 / 62)
    assert scores[2] == pytest.approx(1 / 61)
    assert 11 in scores
    assert 12 not in scores  # onzième page unique de la méthode BGE


def test_v1_embedding_fusion_has_deterministic_page_tie_order():
    fused = qa_benchmark.fuse_v1_embedding_pages(
        [qa_benchmark.RankedPage(page=9, rank=1)],
        [qa_benchmark.RankedPage(page=4, rank=1)],
    )

    assert [result.page for result in fused] == [4, 9]


def test_v1_embedding_fusion_metrics_and_diagnostics(capsys):
    metrics = qa_benchmark.ExperimentalRetrievalMetrics()
    metrics.add_rank(1)
    metrics.add_rank(4)
    observation = qa_benchmark.EmbeddingFusionObservation(
        question_id="ctc-003",
        expected_pages=[6],
        bge_pages=[qa_benchmark.RankedPage(page=2, rank=1)],
        qwen_pages=[qa_benchmark.RankedPage(page=6, rank=1)],
        fused_pages=[
            qa_benchmark.FusedPage(page=2, score=1 / 61),
            qa_benchmark.FusedPage(page=6, score=1 / 61),
        ],
        bge_rank=None,
        qwen_rank=1,
        fused_rank=2,
    )

    qa_benchmark.print_embedding_fusion_diagnostics([observation])

    output = capsys.readouterr().out
    assert metrics.hit_at_1 == 1
    assert metrics.hit_at_5 == 2
    assert metrics.mrr == pytest.approx(0.625)
    assert "ctc-003" in output
    assert "top 5 BGE V1" in output
    assert "scores RRF" in output


def test_v1_only_embedding_comparison_keeps_spaces_and_top_k_separate(
    monkeypatch, capsys
):
    first = DoclingRetrievalUnit(
        source_block_id=uuid4(),
        page=1,
        block_type="text",
        section_header=None,
        text="premier",
    )
    second = DoclingRetrievalUnit(
        source_block_id=uuid4(),
        page=2,
        block_type="text",
        section_header=None,
        text="second",
    )
    corpus = qa_benchmark.DoclingCorpus(2, 0, [first, second])
    experimental_models = []
    limits = []
    original_search = qa_benchmark.search_docling_corpus

    monkeypatch.setattr(qa_benchmark, "embed_text", lambda question: [1.0, 0.0])

    def fake_experimental(texts, *, model):
        experimental_models.append(model)
        return [[1.0, 0.0] for _ in texts]

    def recording_search(corpus, embeddings, query_embedding, *, limit=10):
        limits.append(limit)
        return original_search(
            corpus, embeddings, query_embedding, limit=limit
        )

    monkeypatch.setattr(
        qa_benchmark,
        "embed_texts",
        fake_experimental,
    )
    monkeypatch.setattr(
        qa_benchmark, "search_docling_corpus", recording_search
    )

    qa_benchmark.run_v1_only_embedding_comparison(
        questions=[
            {
                "id": "rideau-001",
                "question": "Question",
                "expected_pages": [1],
                "answerable": True,
            },
            {
                "id": "rideau-023",
                "question": "Sans réponse",
                "expected_pages": [],
                "answerable": False,
            },
        ],
        v1_corpus=corpus,
        current_embeddings=[[1.0, 0.0], [0.0, 1.0]],
        experimental_embeddings=[[0.0, 1.0], [1.0, 0.0]],
        experimental_model="qwen3-embedding:0.6b",
        current_corpus_embedding_duration=1.0,
        experimental_corpus_embedding_duration=2.0,
    )

    output = capsys.readouterr().out
    assert experimental_models == [
        "qwen3-embedding:0.6b",
        "qwen3-embedding:0.6b",
    ]
    assert limits == [qa_benchmark.RETRIEVAL_LIMIT] * 4
    assert "COMPARAISON EMBEDDING DOCLING V1 UNIQUEMENT" in output
    assert "Docling V3" not in output
    assert "Docling V4" not in output
    assert "Fusion V1 BGE + Qwen" not in output
    assert "Questions répondables scorées : 1" in output
    assert "Questions sans réponse hors score : 1" in output


def test_v1_only_cli_requires_experimental_model(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "qa_benchmark.py",
            "--docling-retrieval",
            "--docling-run-id",
            "b9a5bde4-e094-4d52-abe6-3413bdecfe36",
            "--experimental-embedding-v1-only",
        ],
    )

    with pytest.raises(ValueError, match="exige --experimental-embedding-model"):
        qa_benchmark.main()


def test_v1_only_cli_is_disabled_by_default(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["qa_benchmark.py"])

    assert qa_benchmark.parse_arguments().experimental_embedding_v1_only is False


def test_v1_v4_only_reuses_one_question_embedding_and_reports_differences(
    monkeypatch, capsys
):
    v1_unit = DoclingRetrievalUnit(
        source_block_id=uuid4(),
        page=1,
        block_type="text",
        section_header=None,
        text="V1 page un",
    )
    v4_unit_page_one = DoclingV4Unit(
        page=1,
        logical_type="text",
        section_header=None,
        parent_ref="#/body",
        group_ref=None,
        parent_refs=("#/body",),
        source_block_types=("text",),
        source_block_ids=(uuid4(),),
        text="V4 page un",
        is_composite=False,
    )
    v4_unit_page_two = DoclingV4Unit(
        page=2,
        logical_type="semantic_group_siblings",
        section_header=None,
        parent_ref="#/body",
        group_ref=None,
        parent_refs=("#/body",),
        source_block_types=("text", "text", "text"),
        source_block_ids=(uuid4(), uuid4(), uuid4()),
        text="V4 page deux",
        is_composite=True,
    )
    v1_corpus = qa_benchmark.DoclingCorpus(1, 0, [v1_unit])
    v4_corpus = qa_benchmark.DoclingV4Corpus(
        2, 0, [v4_unit_page_one, v4_unit_page_two]
    )
    embedding_calls = []
    query_objects = []
    original_search = qa_benchmark.search_docling_corpus

    def fake_embedding(texts, *, model):
        embedding_calls.append((tuple(texts), model))
        return [[1.0, 0.0]]

    def recording_search(corpus, embeddings, query_embedding, *, limit=10):
        query_objects.append(query_embedding)
        return original_search(
            corpus, embeddings, query_embedding, limit=limit
        )

    monkeypatch.setattr(
        qa_benchmark, "embed_texts", fake_embedding
    )
    monkeypatch.setattr(
        qa_benchmark, "search_docling_corpus", recording_search
    )

    qa_benchmark.run_v1_v4_only_embedding_comparison(
        questions=[
            {
                "id": "notice-001",
                "question": "Question unique",
                "expected_pages": [1],
                "answerable": True,
            }
        ],
        v1_corpus=v1_corpus,
        v4_corpus=v4_corpus,
        v1_embeddings=[[1.0, 0.0]],
        v4_embeddings=[[0.0, 1.0], [1.0, 0.0]],
        experimental_model="qwen3-embedding:4b",
        v1_embedding_duration=1.0,
        v4_embedding_duration=2.0,
    )

    output = capsys.readouterr().out
    assert embedding_calls == [(("Question unique",), "qwen3-embedding:4b")]
    assert len(query_objects) == 2
    assert query_objects[0] is query_objects[1]
    assert "notice-001" in output
    assert "rang V1        : 1" in output
    assert "rang V4        : 2" in output
    assert "Docling V1" in output
    assert "Docling V4" in output
    assert "Docling V3" not in output
    assert "Fusion" not in output
    assert "Temps embeddings questions" in output


@pytest.mark.parametrize(
    "corpus_name",
    [
        "qa_rideau_validation.json",
        "qa_ctc_2013_validation.json",
        "qa_notice_51423_validation.json",
    ],
)
def test_v1_v4_only_cli_accepts_all_docling_corpora(
    monkeypatch, corpus_name
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "qa_benchmark.py",
            "--docling-retrieval",
            "--qa-corpus",
            corpus_name,
            "--docling-run-id",
            "f74b648b-b876-4b14-9846-c1dfcc3aa436",
            "--experimental-embedding-model",
            "qwen3-embedding:4b",
            "--experimental-embedding-v1-v4-only",
        ],
    )

    arguments = qa_benchmark.parse_arguments()

    assert arguments.qa_corpus == corpus_name
    assert arguments.experimental_embedding_v1_v4_only is True
    assert arguments.experimental_embedding_model == "qwen3-embedding:4b"


def test_v1_v4_only_cli_requires_experimental_model(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "qa_benchmark.py",
            "--docling-retrieval",
            "--docling-run-id",
            "b9a5bde4-e094-4d52-abe6-3413bdecfe36",
            "--experimental-embedding-v1-v4-only",
        ],
    )

    with pytest.raises(ValueError, match="exige --experimental-embedding-model"):
        qa_benchmark.main()
