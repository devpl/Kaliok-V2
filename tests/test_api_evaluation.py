from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

from fastapi.testclient import TestClient

import kaliok.api.evaluation as api_module
from kaliok.api.dependencies import get_session
from kaliok.api.main import app
from kaliok.qa.evaluation import (
    CampaignExecutionResult,
    CampaignRunError,
    ConfigurationAggregate,
)
from kaliok.qa.rag_execution import RagExecutionResult, RagExecutionSource
from kaliok.storage.models import Question, QuestionAttempt, QuestionEvidence, QuestionFeedback


NOW = datetime.now(timezone.utc)


class Result:
    def __init__(self, values):
        self.values = values

    def all(self):
        return self.values


class FakeSession:
    def __init__(self, *, objects=None, results=None):
        self.objects = objects or {}
        self.results = list(results or [])
        self.added = []
        self.commits = 0

    def get(self, model, identifier):
        return self.objects.get((model, identifier))

    def exec(self, statement):
        return Result(self.results.pop(0))

    def add(self, value):
        self.added.append(value)

    def commit(self):
        self.commits += 1

    def refresh(self, value):
        return None


def client_for(session):
    app.dependency_overrides[get_session] = lambda: session
    return TestClient(app)


def suite(name="Suite"):
    return SimpleNamespace(
        id=uuid4(),
        name=name,
        description="Description",
        status="active",
        created_at=NOW,
        updated_at=NOW,
    )


def question(text="Question 1"):
    return Question(
        question_text=text,
        expected_answer="Réponse attendue",
        origin="evaluation",
        document_id=uuid4(),
    )


def campaign(suite_id):
    return SimpleNamespace(
        id=uuid4(),
        name="Campagne",
        suite_id=suite_id,
        status="pending",
        configuration={"configuration_revision_ids": [str(uuid4())], "repetitions": 1},
        created_at=NOW,
        started_at=None,
        completed_at=None,
    )


def test_suite_endpoints_delegate_and_preserve_question_order(monkeypatch):
    stored_suite = suite()
    questions = [question("Première"), question("Deuxième")]
    association = SimpleNamespace(
        id=uuid4(),
        evaluation_suite_id=stored_suite.id,
        question_id=questions[0].id,
        position=1,
        created_at=NOW,
    )

    class Service:
        def __init__(self, session):
            pass

        def create(self, *args, **kwargs):
            return stored_suite

        def list(self):
            return [stored_suite]

        def get(self, suite_id):
            return stored_suite if suite_id == stored_suite.id else None

        def list_questions(self, suite_id):
            return questions

        def update(self, suite_id, **values):
            stored_suite.name = values["name"]
            return stored_suite

        def add_question(self, suite_id, question_id, position=None):
            return association

        def remove_question(self, suite_id, question_id):
            return True

    monkeypatch.setattr(api_module, "EvaluationSuiteService", Service)
    client = client_for(FakeSession())
    try:
        assert client.post("/rag/evaluation/suites", json={"name": "Suite"}).status_code == 201
        assert client.get("/rag/evaluation/suites").json()[0]["id"] == str(stored_suite.id)
        detail = client.get(f"/rag/evaluation/suites/{stored_suite.id}").json()
        assert [item["question_text"] for item in detail["questions"]] == [
            "Première",
            "Deuxième",
        ]
        updated = client.patch(
            f"/rag/evaluation/suites/{stored_suite.id}", json={"name": "Nouvelle"}
        )
        assert updated.json()["name"] == "Nouvelle"
        assert client.post(
            f"/rag/evaluation/suites/{stored_suite.id}/questions",
            json={"question_id": str(questions[0].id), "position": 1},
        ).status_code == 201
        assert client.get(
            f"/rag/evaluation/suites/{stored_suite.id}/questions"
        ).status_code == 200
        assert client.delete(
            f"/rag/evaluation/suites/{stored_suite.id}/questions/{questions[0].id}"
        ).status_code == 204
        assert client.get(f"/rag/evaluation/suites/{uuid4()}").status_code == 404
    finally:
        app.dependency_overrides.clear()


def test_question_creation_validation_and_read(monkeypatch):
    stored = question()
    session = FakeSession(objects={(Question, stored.id): stored})

    def fake_create(text, **kwargs):
        assert kwargs["origin"] == "evaluation"
        assert kwargs["expected_answer"] == "Réponse attendue"
        assert kwargs["session"] is session
        return stored

    monkeypatch.setattr(api_module, "create_question", fake_create)
    client = client_for(session)
    try:
        response = client.post(
            "/rag/evaluation/questions",
            json={
                "question_text": "Question 1",
                "expected_answer": "Réponse attendue",
                "document_id": str(stored.document_id),
            },
        )
        assert response.status_code == 201
        assert response.json()["expected_answer"] == "Réponse attendue"
        assert client.post(
            "/rag/evaluation/questions", json={"question_text": "Sans document"}
        ).status_code == 400
        assert client.post(
            "/rag/evaluation/questions", json={"question_text": ""}
        ).status_code == 422
        assert client.get(f"/rag/evaluation/questions/{stored.id}").status_code == 200
    finally:
        app.dependency_overrides.clear()


def test_campaign_create_list_get_run_and_attempt_history(monkeypatch):
    suite_id = uuid4()
    stored = campaign(suite_id)
    revision_id = uuid4()
    question_id = uuid4()
    attempt_id = uuid4()
    source = RagExecutionSource(1, 0.9, uuid4(), "Passage", {"source_unit_id": "p1"})
    individual = RagExecutionResult(
        question_id,
        attempt_id,
        uuid4(),
        source.document_version_id,
        "Réponse",
        (source,),
        revision_id,
        {"retrieval": {"top_k": 5}},
        {"duration_ms": 10.0},
    )
    run_result = CampaignExecutionResult(
        stored.id,
        suite_id,
        "completed",
        2,
        1,
        1,
        0.5,
        10.0,
        1.0,
        NOW,
        NOW,
        (individual,),
        (CampaignRunError(question_id, revision_id, 2, "RuntimeError", "échec"),),
        (ConfigurationAggregate(revision_id, 2, 1, 1, 0.5, 10.0, 1.0),),
    )
    attempt = QuestionAttempt(
        id=attempt_id,
        question_id=question_id,
        attempt_number=1,
        strategy="normalized-rag",
        evaluation_campaign_id=stored.id,
    )

    class Service:
        def __init__(self, session):
            pass

        def create(self, *args, **kwargs):
            assert len(kwargs["configuration_revision_ids"]) == 2
            return stored

        def list(self):
            return [stored]

        def get(self, campaign_id):
            return stored if campaign_id == stored.id else None

        def run_campaign(self, campaign_id):
            return run_result

        def list_attempts(self, campaign_id):
            return [attempt]

    monkeypatch.setattr(api_module, "EvaluationCampaignService", Service)
    client = client_for(FakeSession())
    try:
        payload = {
            "name": "Campagne",
            "suite_id": str(suite_id),
            "configuration_revision_ids": [str(uuid4()), str(uuid4())],
            "repetitions": 2,
        }
        assert client.post("/rag/evaluation/campaigns", json=payload).status_code == 201
        assert client.get("/rag/evaluation/campaigns").status_code == 200
        assert client.get(f"/rag/evaluation/campaigns/{stored.id}").status_code == 200
        response = client.post(f"/rag/evaluation/campaigns/{stored.id}/run")
        assert response.status_code == 200
        body = response.json()
        assert body["failed_runs"] == 1
        assert body["results"][0]["sources"][0]["score"] == 0.9
        assert body["by_configuration"][0]["technical_success_rate"] == 0.5
        assert client.get(
            f"/rag/evaluation/campaigns/{stored.id}/attempts"
        ).json()[0]["id"] == str(attempt_id)
        assert client.get(f"/rag/evaluation/campaigns/{uuid4()}").status_code == 404
    finally:
        app.dependency_overrides.clear()


def test_attempt_detail_reads_evidence_feedback_without_execution(monkeypatch):
    attempt = QuestionAttempt(
        question_id=uuid4(), attempt_number=1, strategy="normalized-rag"
    )
    stored_question = question()
    stored_question.id = attempt.question_id
    evidence = [
        QuestionEvidence(question_attempt_id=attempt.id, rank=2, score=0.7),
        QuestionEvidence(question_attempt_id=attempt.id, rank=1, score=0.9),
    ]
    feedback = QuestionFeedback(
        question_id=attempt.question_id,
        question_attempt_id=attempt.id,
        rating=4,
    )
    session = FakeSession(
        objects={(QuestionAttempt, attempt.id): attempt, (Question, attempt.question_id): stored_question},
        results=[[evidence[1], evidence[0]], [feedback]],
    )
    client = client_for(session)
    try:
        body = client.get(f"/rag/evaluation/attempts/{attempt.id}").json()
        assert [item["rank"] for item in body["evidence"]] == [1, 2]
        assert body["feedback"][0]["rating"] == 4
    finally:
        app.dependency_overrides.clear()


def test_feedback_creation_links_attempt_and_question():
    attempt = QuestionAttempt(
        question_id=uuid4(), attempt_number=1, strategy="normalized-rag"
    )
    session = FakeSession(objects={(QuestionAttempt, attempt.id): attempt})
    client = client_for(session)
    try:
        response = client.post(
            f"/rag/evaluation/attempts/{attempt.id}/feedback",
            json={"rating": 5, "is_correct": True, "comment": "Correct"},
        )
        assert response.status_code == 201
        stored = session.added[0]
        assert stored.question_id == attempt.question_id
        assert stored.question_attempt_id == attempt.id
        assert stored.origin == "user"
    finally:
        app.dependency_overrides.clear()


def test_configuration_and_document_selectors_include_required_data():
    profile = SimpleNamespace(
        id=uuid4(), profile_key="profile", label="Profil", is_active=True
    )
    revisions = [
        SimpleNamespace(
            id=uuid4(), revision_number=2, status="draft", created_at=NOW, change_reason=None
        ),
        SimpleNamespace(
            id=uuid4(), revision_number=1, status="retired", created_at=NOW, change_reason="Ancienne"
        ),
    ]
    document = SimpleNamespace(id=uuid4(), title="Document")
    version = SimpleNamespace(
        id=uuid4(),
        filename="document.pdf",
        version_number=3,
        processing_status="completed",
        page_count=4,
    )
    session = FakeSession(results=[[profile], revisions, [(document, version)]])
    client = client_for(session)
    try:
        configurations = client.get("/rag/evaluation/configurations").json()
        assert [item["status"] for item in configurations[0]["revisions"]] == [
            "draft",
            "retired",
        ]
        documents = client.get("/rag/evaluation/documents").json()
        assert documents[0]["document_version_id"] == str(version.id)
        assert documents[0]["page_count"] == 4
    finally:
        app.dependency_overrides.clear()


def test_service_errors_map_to_http_statuses(monkeypatch):
    class Service:
        def __init__(self, session):
            pass

        def add_question(self, *args, **kwargs):
            raise ValueError("Cette position est déjà utilisée dans la suite.")

        def create(self, *args, **kwargs):
            raise ValueError("Révision de configuration introuvable.")

    monkeypatch.setattr(api_module, "EvaluationSuiteService", Service)
    monkeypatch.setattr(api_module, "EvaluationCampaignService", Service)
    client = client_for(FakeSession())
    try:
        assert client.post(
            f"/rag/evaluation/suites/{uuid4()}/questions",
            json={"question_id": str(uuid4()), "position": 1},
        ).status_code == 409
        assert client.post(
            "/rag/evaluation/campaigns",
            json={
                "name": "C",
                "suite_id": str(uuid4()),
                "configuration_revision_ids": [str(uuid4())],
                "repetitions": 1,
            },
        ).status_code == 404
    finally:
        app.dependency_overrides.clear()
