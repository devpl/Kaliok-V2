from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID, uuid4

import pytest

import kaliok.qa.rag_execution as execution_module
from kaliok.qa.rag_execution import RagExecutionService
from kaliok.rag.types import (
    Candidate,
    ContextBundle,
    Provenance,
    RagAnswer,
    RankedCandidate,
    RetrievalUnit,
)
from kaliok.rag_runtime import RagRuntimeConfiguration
from kaliok.storage.models import Question, QuestionAttempt, QuestionEvidence


class ScalarResult:
    def __init__(self, value):
        self.value = value

    def one(self):
        return self.value


class FakeSession:
    def __init__(self, question: Question | None, maximum=0):
        self.question = question
        self.maximum = maximum
        self.attempts = {}
        self.evidence = []
        self.commits = 0
        self.rollbacks = 0

    def get(self, model, identifier):
        if model is Question:
            return self.question if self.question and self.question.id == identifier else None
        if model is QuestionAttempt:
            return self.attempts.get(identifier)
        return None

    def exec(self, statement):
        return ScalarResult(self.maximum)

    def add(self, value):
        if isinstance(value, QuestionAttempt):
            self.attempts[value.id] = value
        elif isinstance(value, QuestionEvidence):
            self.evidence.append(value)

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1

    def refresh(self, value):
        return None


@dataclass
class FakeRuntime:
    document_id: UUID
    document_version_id: UUID
    configuration: RagRuntimeConfiguration
    answer_value: RagAnswer | None = None
    error: Exception | None = None

    def answer(self, question):
        if self.error is not None:
            raise self.error
        return self.answer_value


def runtime_configuration(revision_id=None):
    return RagRuntimeConfiguration(
        profile_id=uuid4(),
        profile_key="laboratory",
        revision_id=revision_id or uuid4(),
        revision_number=4,
        generation_provider="ollama",
        generation_model="mistral",
        generation_temperature=0.0,
        retrieval_top_k=5,
        context_builder="ranked",
        embedding_model="bge-m3",
    )


def rag_answer(document_id, version_id):
    unit_id = uuid4()
    ranked = RankedCandidate(
        candidate=Candidate(
            unit=RetrievalUnit(
                unit_id=unit_id,
                text="Passage retrouvé",
                provenance=Provenance(
                    document_id=document_id,
                    document_version_id=version_id,
                    representation="normalized_content_unit",
                    embedding_model="bge-m3",
                    metadata={
                        "normalized_content_unit_id": unit_id,
                        "source_unit_id": "paragraph-7",
                    },
                ),
            ),
            score=0.91,
        ),
        rank=1,
        score=0.91,
    )
    return RagAnswer(
        text="Réponse générée",
        context=ContextBundle(
            question="Question de test ?",
            text="Passage retrouvé",
            candidates=(ranked,),
        ),
        metadata={"model": "mistral"},
    )


def test_success_persists_attempt_configuration_evidence_and_metrics(monkeypatch):
    document_id = uuid4()
    version_id = uuid4()
    revision_id = uuid4()
    campaign_id = uuid4()
    question = Question(
        question_text="Question de test ?",
        document_id=document_id,
        document_version_id=version_id,
    )
    session = FakeSession(question, maximum=2)
    runtime = FakeRuntime(
        document_id,
        version_id,
        runtime_configuration(revision_id),
        rag_answer(document_id, version_id),
    )
    received = {}

    def create_runtime(session_value, **kwargs):
        received.update(kwargs)
        return runtime

    monkeypatch.setattr(execution_module, "create_normalized_rag_runtime", create_runtime)

    result = RagExecutionService(session).execute(
        question.id,
        profile_key="laboratory",
        configuration_revision_id=revision_id,
        evaluation_campaign_id=campaign_id,
    )

    attempt = session.attempts[result.question_attempt_id]
    assert attempt.attempt_number == 3
    assert attempt.strategy == "normalized-rag"
    assert attempt.status == "completed"
    assert attempt.answer_text == "Réponse générée"
    assert attempt.configuration_revision_id == revision_id
    assert attempt.evaluation_campaign_id == campaign_id
    assert attempt.completed_at is not None
    assert attempt.configuration["embedding"] == {"model": "bge-m3"}
    assert attempt.metrics["retrieved_passage_count"] == 1
    assert attempt.metrics["generation_model"] == "mistral"
    assert attempt.metrics["top_k"] == 5
    assert attempt.metrics["duration_ms"] >= 0
    assert received["configuration_revision_id"] == revision_id
    assert "profile_key" not in received
    assert len(session.evidence) == 1
    evidence = session.evidence[0]
    assert evidence.rank == 1
    assert evidence.score == 0.91
    assert evidence.document_version_id == version_id
    assert evidence.evidence_text == "Passage retrouvé"
    assert evidence.chunk_id is None
    assert evidence.extra_data["source_unit_id"] == "paragraph-7"
    assert result.configuration_revision_id == revision_id
    assert result.sources[0].score == 0.91


def test_rag_failure_is_persisted_and_original_exception_is_raised(monkeypatch):
    question = Question(question_text="Question", document_id=uuid4())
    session = FakeSession(question)
    original = RuntimeError("échec contrôlé")
    runtime = FakeRuntime(
        uuid4(),
        uuid4(),
        runtime_configuration(),
        error=original,
    )
    monkeypatch.setattr(
        execution_module,
        "create_normalized_rag_runtime",
        lambda *args, **kwargs: runtime,
    )

    with pytest.raises(RuntimeError) as captured:
        RagExecutionService(session).execute(question.id)

    attempt = next(iter(session.attempts.values()))
    assert captured.value is original
    assert attempt.status == "failed"
    assert attempt.failure_reason == "RuntimeError: échec contrôlé"
    assert attempt.completed_at is not None
    assert session.rollbacks == 1
    assert session.commits == 2


def test_missing_question_is_rejected():
    question_id = uuid4()
    with pytest.raises(ValueError, match=str(question_id)):
        RagExecutionService(FakeSession(None)).execute(question_id)


def test_question_without_document_reference_is_rejected():
    question = Question(question_text="Question sans document")
    with pytest.raises(ValueError, match="aucun document"):
        RagExecutionService(FakeSession(question)).execute(question.id)
