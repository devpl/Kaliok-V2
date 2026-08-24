from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from uuid import uuid4


QA_BENCHMARK_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "qa_benchmark.py"
)

spec = importlib.util.spec_from_file_location(
    "qa_benchmark",
    QA_BENCHMARK_PATH,
)

if spec is None or spec.loader is None:
    raise RuntimeError(
        "Impossible de charger "
        f"{QA_BENCHMARK_PATH}"
    )

qa_benchmark = importlib.util.module_from_spec(
    spec
)

# Important notamment pour les dataclasses
# définies dans qa_benchmark.py.
sys.modules[spec.name] = qa_benchmark

spec.loader.exec_module(
    qa_benchmark
)


class FakeResponse:
    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return {
            "response": (
                '{"answerable":true,'
                '"evidence_chunk_indices":[0]}'
            )
        }


class FakeMalformedResponse:
    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return {
            "response": (
                '{"answerable":true,'
                '"evidence_chunk_indices":"0"}'
            )
        }


class FakeTop5Response:
    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return {
            "response": (
                '{"answerable":true,'
                '"evidence_chunk_indices":[4]}'
            )
        }


def test_answerability_judge_uses_only_question_and_chunks(
    monkeypatch,
):
    captured_request = {}

    def fake_post(url, *, json, timeout):
        captured_request["url"] = url
        captured_request["json"] = json
        captured_request["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(
        qa_benchmark.requests,
        "post",
        fake_post,
    )

    chunks = [
        qa_benchmark.RetrievedChunk(
            chunk_id=uuid4(),
            page_start=4,
            page_end=4,
            content="Le montant explicite est 42 euros.",
        )
    ]

    decision = qa_benchmark.judge_answerability(
        "Quel est le montant ?",
        chunks,
        model="local-test-model",
    )

    assert decision.answerable is True
    assert decision.evidence_chunk_indices == [0]

    request_payload = captured_request["json"]

    assert captured_request["url"].endswith(
        "/api/generate"
    )
    assert request_payload["model"] == "local-test-model"
    assert request_payload["stream"] is False
    assert request_payload["options"]["temperature"] == 0
    assert (
        "Quel est le montant ?"
        in request_payload["prompt"]
    )
    assert (
        chunks[0].content
        in request_payload["prompt"]
    )
    assert "42 euros" in request_payload["prompt"]


def test_answerability_metrics_confusion_matrix():
    metrics = qa_benchmark.AnswerabilityMetrics()

    metrics.add(
        expected=True,
        predicted=True,
        evidence_correct=True,
    )
    metrics.add(
        expected=False,
        predicted=True,
    )
    metrics.add(
        expected=True,
        predicted=False,
    )
    metrics.add(
        expected=False,
        predicted=False,
    )

    assert metrics.true_positive == 1
    assert metrics.false_positive == 1
    assert metrics.false_negative == 1
    assert metrics.true_negative == 1

    assert metrics.accuracy == 0.5
    assert metrics.precision == 0.5
    assert metrics.recall == 0.5

    assert metrics.answerability_correct == 2
    assert metrics.evidence_correct == 1

    assert (
        metrics.answerability_and_evidence_correct
        == 2
    )

    assert metrics.business_false_positives == 1
    assert metrics.business_false_negatives == 1
    assert metrics.evidence_false_positives == 0


def test_protocol_error_is_recorded_without_stopping_evaluation(
    monkeypatch,
    capsys,
):
    monkeypatch.setattr(
        qa_benchmark.requests,
        "post",
        lambda *args, **kwargs: (
            FakeMalformedResponse()
        ),
    )

    chunks = [
        qa_benchmark.RetrievedChunk(
            chunk_id=uuid4(),
            page_start=10,
            page_end=10,
            content="Passage de diagnostic.",
        )
    ]

    metrics = qa_benchmark.AnswerabilityMetrics()

    observation = (
        qa_benchmark
        .evaluate_answerability_question(
            question_id="jugement-003",
            question="Question de test ?",
            expected_answerable=True,
            expected_pages=[10],
            chunks=chunks,
            model="local-test-model",
            metrics=metrics,
        )
    )

    assert observation.decision is None

    assert observation.protocol_error == (
        "Sortie juge invalide : "
        "evidence_chunk_indices."
    )

    assert observation.raw_output == (
        '{"answerable":true,'
        '"evidence_chunk_indices":"0"}'
    )

    assert metrics.protocol_errors == 1
    assert metrics.total == 0

    output = capsys.readouterr().out

    assert "jugement-003" in output
    assert observation.raw_output in output


def test_technical_error_is_recorded_without_stopping_evaluation(
    monkeypatch,
    capsys,
):
    def raise_timeout(*args, **kwargs):
        raise (
            qa_benchmark
            .requests
            .exceptions
            .ReadTimeout(
                "Ollama read timed out "
                "after 300 seconds"
            )
        )

    monkeypatch.setattr(
        qa_benchmark.requests,
        "post",
        raise_timeout,
    )

    chunks = [
        qa_benchmark.RetrievedChunk(
            chunk_id=uuid4(),
            page_start=6,
            page_end=6,
            content="Passage de jugement.",
        )
    ]

    metrics = qa_benchmark.AnswerabilityMetrics()

    observation = (
        qa_benchmark
        .evaluate_answerability_question(
            question_id="jugement-002",
            question=(
                "Quel était le montant "
                "de la subvention ?"
            ),
            expected_answerable=True,
            expected_pages=[6],
            chunks=chunks,
            model="local-test-model",
            metrics=metrics,
        )
    )

    assert observation.decision is None
    assert observation.protocol_error is None

    assert (
        observation.technical_error_type
        == "ReadTimeout"
    )

    assert (
        observation.technical_error_message
        == (
            "Ollama read timed out "
            "after 300 seconds"
        )
    )

    assert metrics.technical_errors == 1
    assert metrics.protocol_errors == 0
    assert metrics.total == 0

    output = capsys.readouterr().out

    assert "jugement-002" in output
    assert "ReadTimeout" in output

    assert (
        observation.technical_error_message
        in output
    )


def test_jugement_001_answerable_without_top3_evidence_is_evidence_error(
    monkeypatch,
):
    monkeypatch.setattr(
        qa_benchmark.requests,
        "post",
        lambda *args, **kwargs: FakeResponse(),
    )

    chunks = [
        qa_benchmark.RetrievedChunk(
            chunk_id=uuid4(),
            page_start=page,
            page_end=page,
            content=(
                f"Chunk hors preuve, "
                f"page {page}."
            ),
        )
        for page in (2, 3, 4)
    ]

    metrics = qa_benchmark.AnswerabilityMetrics()

    observation = (
        qa_benchmark
        .evaluate_answerability_question(
            question_id="jugement-001",
            question=(
                "À quelle date le jugement "
                "a-t-il été prononcé ?"
            ),
            expected_answerable=True,
            expected_pages=[1],
            chunks=chunks,
            model="local-test-model",
            metrics=metrics,
        )
    )

    assert observation.decision is not None
    assert observation.decision.answerable is True
    assert observation.evidence_correct is False

    assert metrics.answerability_correct == 1
    assert metrics.evidence_correct == 0

    assert (
        metrics.answerability_and_evidence_correct
        == 0
    )

    assert metrics.evidence_false_positives == 1
    assert metrics.business_false_positives == 0
    assert metrics.business_false_negatives == 0


def test_judge_top_k_defaults_to_three_and_accepts_five(
    monkeypatch,
):
    monkeypatch.setattr(
        sys,
        "argv",
        ["qa_benchmark.py"],
    )

    assert (
        qa_benchmark
        .parse_arguments()
        .judge_top_k
        == 3
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "qa_benchmark.py",
            "--judge-top-k",
            "5",
        ],
    )

    assert (
        qa_benchmark
        .parse_arguments()
        .judge_top_k
        == 5
    )


def test_top5_accepts_evidence_chunk_index_four(
    monkeypatch,
):
    monkeypatch.setattr(
        qa_benchmark.requests,
        "post",
        lambda *args, **kwargs: FakeTop5Response(),
    )

    chunks = [
        qa_benchmark.RetrievedChunk(
            chunk_id=uuid4(),
            page_start=index + 1,
            page_end=index + 1,
            content=f"Chunk {index}",
        )
        for index in range(5)
    ]

    decision = qa_benchmark.judge_answerability(
        "Question top 5 ?",
        chunks,
        model="local-test-model",
    )

    assert decision.answerable is True

    assert (
        decision.evidence_chunk_indices
        == [4]
    )