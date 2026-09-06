from __future__ import annotations

from uuid import uuid4

import pytest

from kaliok.qa.evaluation import EvaluationCampaignService, EvaluationSuiteService
from kaliok.qa.rag_execution import RagExecutionResult
from kaliok.storage.models import (
    ConfigurationProfileRevision,
    EvaluationCampaign,
    EvaluationSuite,
    EvaluationSuiteQuestion,
    Question,
    QuestionAttempt,
)


class Result:
    def __init__(self, values, scalar=False):
        self.values = values
        self.scalar = scalar

    def all(self):
        return list(self.values)

    def one(self):
        return self.values if self.scalar else self.values[0]

    def one_or_none(self):
        return self.values[0] if self.values else None


class FakeSession:
    def __init__(self):
        self.suites = {}
        self.questions = {}
        self.associations = []
        self.revisions = {}
        self.campaigns = {}
        self.attempts = []
        self.commits = 0
        self.rollbacks = 0

    def get(self, model, identifier):
        stores = {
            EvaluationSuite: self.suites,
            Question: self.questions,
            ConfigurationProfileRevision: self.revisions,
            EvaluationCampaign: self.campaigns,
        }
        if model is QuestionAttempt:
            return next((item for item in self.attempts if item.id == identifier), None)
        return stores.get(model, {}).get(identifier)

    def add(self, value):
        if isinstance(value, EvaluationSuite):
            self.suites[value.id] = value
        elif isinstance(value, Question):
            self.questions[value.id] = value
        elif isinstance(value, EvaluationSuiteQuestion):
            if value not in self.associations:
                self.associations.append(value)
        elif isinstance(value, ConfigurationProfileRevision):
            self.revisions[value.id] = value
        elif isinstance(value, EvaluationCampaign):
            self.campaigns[value.id] = value
        elif isinstance(value, QuestionAttempt):
            self.attempts.append(value)

    def delete(self, value):
        self.associations.remove(value)

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1

    def refresh(self, value):
        return None

    def exec(self, statement):
        sql = str(statement)
        params = list(statement.compile().params.values())
        if "max(evaluation_suite_questions.position)" in sql:
            suite_id = params[0]
            positions = [
                item.position
                for item in self.associations
                if item.evaluation_suite_id == suite_id
            ]
            return Result(max(positions, default=None), scalar=True)
        if "count(evaluation_suite_questions.id)" in sql:
            suite_id = params[0]
            count = sum(
                item.evaluation_suite_id == suite_id
                for item in self.associations
            )
            return Result(count, scalar=True)
        entity = statement.column_descriptions[0].get("entity")
        if entity is EvaluationSuiteQuestion:
            suite_id = params[0]
            values = [
                item
                for item in self.associations
                if item.evaluation_suite_id == suite_id
            ]
            if len(params) > 1:
                second = params[1]
                values = [
                    item
                    for item in values
                    if item.question_id == second or item.position == second
                ]
            return Result(values)
        if entity is Question:
            suite_id = params[0]
            ordered = sorted(
                (
                    item
                    for item in self.associations
                    if item.evaluation_suite_id == suite_id
                ),
                key=lambda item: item.position,
            )
            return Result([self.questions[item.question_id] for item in ordered])
        if entity is EvaluationSuite:
            return Result(
                sorted(self.suites.values(), key=lambda item: item.created_at, reverse=True)
            )
        if entity is EvaluationCampaign:
            return Result(
                sorted(
                    self.campaigns.values(),
                    key=lambda item: item.created_at,
                    reverse=True,
                )
            )
        if entity is QuestionAttempt:
            campaign_id = params[0]
            return Result(
                [
                    item
                    for item in self.attempts
                    if item.evaluation_campaign_id == campaign_id
                ]
            )
        raise AssertionError(sql)


def add_revision(session):
    revision = ConfigurationProfileRevision(
        profile_id=uuid4(),
        revision_number=1,
        created_by_display_name="Test",
    )
    session.add(revision)
    return revision


def prepared_suite(session, question_count=1):
    service = EvaluationSuiteService(session)
    suite = service.create("Suite", description="Description")
    questions = []
    for index in range(question_count):
        question = Question(question_text=f"Question {index}", document_id=uuid4())
        session.add(question)
        questions.append(question)
        service.add_question(suite.id, question.id)
    return suite, questions


def test_suite_crud_order_duplicate_and_removal_preserve_question():
    session = FakeSession()
    service = EvaluationSuiteService(session)
    suite = service.create("Initiale")
    first = Question(question_text="Première", document_id=uuid4())
    second = Question(question_text="Deuxième", document_id=uuid4())
    session.add(first)
    session.add(second)

    service.add_question(suite.id, second.id, position=2)
    service.add_question(suite.id, first.id, position=1)
    assert service.list_questions(suite.id) == [first, second]
    with pytest.raises(ValueError, match="déjà"):
        service.add_question(suite.id, first.id)
    third = Question(question_text="Troisième", document_id=uuid4())
    session.add(third)
    with pytest.raises(ValueError, match="position"):
        service.add_question(suite.id, third.id, position=1)

    updated = service.update(
        suite.id,
        name="Mise à jour",
        description=None,
        status="inactive",
    )
    assert (updated.name, updated.description, updated.status) == (
        "Mise à jour",
        None,
        "inactive",
    )
    assert service.remove_question(suite.id, first.id) is True
    assert session.get(Question, first.id) is first
    assert service.get(suite.id) is suite
    assert service.list() == [suite]


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("missing-suite", "introuvable"),
        ("empty-suite", "aucune question"),
        ("repetitions", "répétitions"),
        ("no-revision", "Au moins une"),
        ("missing-revision", "Révision de configuration introuvable"),
    ],
)
def test_campaign_creation_validations(case, message):
    session = FakeSession()
    suite, _ = prepared_suite(session) if case != "missing-suite" else (None, [])
    if case == "empty-suite":
        session.associations.clear()
    revision = add_revision(session)
    suite_id = suite.id if suite is not None else uuid4()
    revision_ids = [] if case == "no-revision" else [revision.id]
    if case == "missing-revision":
        revision_ids = [uuid4()]
    repetitions = 0 if case == "repetitions" else 1

    with pytest.raises(ValueError, match=message):
        EvaluationCampaignService(session).create(
            "Campagne",
            suite_id=suite_id,
            configuration_revision_ids=revision_ids,
            repetitions=repetitions,
        )


def test_campaign_plan_and_deterministic_execution_with_aggregates():
    session = FakeSession()
    suite, questions = prepared_suite(session, question_count=2)
    revisions = [add_revision(session), add_revision(session)]
    calls = []

    class FakeRunner:
        def __init__(self, ignored_session):
            pass

        def execute(self, question_id, **kwargs):
            assert "profile_key" not in kwargs
            calls.append((kwargs["configuration_revision_id"], question_id))
            if len(calls) == 2:
                raise RuntimeError("échec individuel")
            return RagExecutionResult(
                question_id=question_id,
                question_attempt_id=uuid4(),
                document_id=uuid4(),
                document_version_id=uuid4(),
                answer="Réponse",
                sources=(),
                configuration_revision_id=kwargs["configuration_revision_id"],
                configuration={},
                metrics={"duration_ms": 10.0, "retrieved_passage_count": 2},
            )

    service = EvaluationCampaignService(
        session,
        execution_service_factory=FakeRunner,
    )
    campaign = service.create(
        "Campagne",
        suite_id=suite.id,
        configuration_revision_ids=[item.id for item in revisions],
        repetitions=2,
    )
    assert campaign.configuration == {
        "configuration_revision_ids": [str(item.id) for item in revisions],
        "repetitions": 2,
    }

    campaign.configuration = {
        **campaign.configuration,
        "profile_key": "ancienne-valeur-ignoree",
    }

    result = service.run_campaign(campaign.id)

    expected = []
    for revision in revisions:
        for question in questions:
            expected.extend([(revision.id, question.id)] * 2)
    assert calls == expected
    assert result.status == "completed"
    assert result.total_requested_runs == 8
    assert result.completed_runs == 7
    assert result.failed_runs == 1
    assert result.technical_success_rate == 7 / 8
    assert result.average_duration_ms == 10.0
    assert result.average_retrieved_passage_count == 2.0
    assert len(result.by_configuration) == 2
    assert result.errors[0].message == "échec individuel"
    assert campaign.started_at is not None
    assert campaign.completed_at is not None


def test_campaign_rejects_duplicate_configuration_revisions():
    session = FakeSession()
    suite, _ = prepared_suite(session)
    revision = add_revision(session)

    with pytest.raises(ValueError, match="doivent être uniques"):
        EvaluationCampaignService(session).create(
            "Campagne",
            suite_id=suite.id,
            configuration_revision_ids=[revision.id, revision.id],
            repetitions=1,
        )


def test_campaign_structural_error_marks_campaign_failed():
    session = FakeSession()
    suite, _ = prepared_suite(session)
    revision = add_revision(session)
    service = EvaluationCampaignService(session)
    campaign = service.create(
        "Campagne",
        suite_id=suite.id,
        configuration_revision_ids=[revision.id],
        repetitions=1,
    )
    campaign.configuration = {}

    with pytest.raises(ValueError, match="Plan de campagne invalide"):
        service.run_campaign(campaign.id)

    assert campaign.status == "failed"
    assert campaign.completed_at is not None


def test_campaign_history_and_attempt_grouping():
    session = FakeSession()
    suite, questions = prepared_suite(session)
    revision = add_revision(session)
    service = EvaluationCampaignService(session)
    campaign = service.create(
        "Campagne",
        suite_id=suite.id,
        configuration_revision_ids=[revision.id],
        repetitions=1,
    )
    attempt = QuestionAttempt(
        question_id=questions[0].id,
        attempt_number=1,
        strategy="normalized-rag",
        configuration_revision_id=revision.id,
        evaluation_campaign_id=campaign.id,
    )
    session.add(attempt)

    assert service.get(campaign.id) is campaign
    assert service.list() == [campaign]
    assert service.list_attempts(campaign.id) == [attempt]
    assert service.attempts_by_configuration(campaign.id) == {
        revision.id: (attempt,)
    }
