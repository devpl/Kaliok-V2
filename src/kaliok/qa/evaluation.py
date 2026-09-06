from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from sqlalchemy import func
from sqlmodel import Session, select

from kaliok.qa.rag_execution import RagExecutionResult, RagExecutionService
from kaliok.storage.models import (
    ConfigurationProfileRevision,
    EvaluationCampaign,
    EvaluationSuite,
    EvaluationSuiteQuestion,
    Question,
    QuestionAttempt,
    utc_now,
)


_UNSET = object()


class EvaluationSuiteService:
    def __init__(self, session: Session) -> None:
        self._session = session

    def create(
        self,
        name: str,
        *,
        description: str | None = None,
        status: str = "active",
    ) -> EvaluationSuite:
        suite = EvaluationSuite(
            name=name,
            description=description,
            status=status,
        )
        self._session.add(suite)
        self._session.commit()
        self._session.refresh(suite)
        return suite

    def get(self, suite_id: UUID) -> EvaluationSuite | None:
        return self._session.get(EvaluationSuite, suite_id)

    def list(self) -> list[EvaluationSuite]:
        return list(
            self._session.exec(
                select(EvaluationSuite).order_by(EvaluationSuite.created_at.desc())
            ).all()
        )

    def update(
        self,
        suite_id: UUID,
        *,
        name: str | object = _UNSET,
        description: str | None | object = _UNSET,
        status: str | object = _UNSET,
    ) -> EvaluationSuite:
        suite = self._require_suite(suite_id)
        if name is not _UNSET:
            suite.name = str(name)
        if description is not _UNSET:
            suite.description = description  # type: ignore[assignment]
        if status is not _UNSET:
            suite.status = str(status)
        suite.updated_at = utc_now()
        self._session.add(suite)
        self._session.commit()
        self._session.refresh(suite)
        return suite

    def add_question(
        self,
        suite_id: UUID,
        question_id: UUID,
        *,
        position: int | None = None,
    ) -> EvaluationSuiteQuestion:
        self._require_suite(suite_id)
        if self._session.get(Question, question_id) is None:
            raise ValueError(f"Question introuvable : {question_id}.")
        existing = self._session.exec(
            select(EvaluationSuiteQuestion).where(
                EvaluationSuiteQuestion.evaluation_suite_id == suite_id,
                EvaluationSuiteQuestion.question_id == question_id,
            )
        ).one_or_none()
        if existing is not None:
            raise ValueError("La question appartient déjà à cette suite.")
        if position is None:
            maximum = self._session.exec(
                select(func.max(EvaluationSuiteQuestion.position)).where(
                    EvaluationSuiteQuestion.evaluation_suite_id == suite_id
                )
            ).one()
            position = (maximum or 0) + 1
        elif position < 1:
            raise ValueError("La position doit être strictement positive.")
        occupied = self._session.exec(
            select(EvaluationSuiteQuestion).where(
                EvaluationSuiteQuestion.evaluation_suite_id == suite_id,
                EvaluationSuiteQuestion.position == position,
            )
        ).one_or_none()
        if occupied is not None:
            raise ValueError("Cette position est déjà utilisée dans la suite.")
        association = EvaluationSuiteQuestion(
            evaluation_suite_id=suite_id,
            question_id=question_id,
            position=position,
        )
        self._session.add(association)
        self._session.commit()
        self._session.refresh(association)
        return association

    def remove_question(self, suite_id: UUID, question_id: UUID) -> bool:
        association = self._session.exec(
            select(EvaluationSuiteQuestion).where(
                EvaluationSuiteQuestion.evaluation_suite_id == suite_id,
                EvaluationSuiteQuestion.question_id == question_id,
            )
        ).one_or_none()
        if association is None:
            return False
        self._session.delete(association)
        self._session.commit()
        return True

    def list_questions(self, suite_id: UUID) -> list[Question]:
        self._require_suite(suite_id)
        return list(
            self._session.exec(
                select(Question)
                .join(
                    EvaluationSuiteQuestion,
                    EvaluationSuiteQuestion.question_id == Question.id,
                )
                .where(EvaluationSuiteQuestion.evaluation_suite_id == suite_id)
                .order_by(EvaluationSuiteQuestion.position)
            ).all()
        )

    def _require_suite(self, suite_id: UUID) -> EvaluationSuite:
        suite = self.get(suite_id)
        if suite is None:
            raise ValueError(f"Suite d'évaluation introuvable : {suite_id}.")
        return suite


@dataclass(frozen=True)
class CampaignRunError:
    question_id: UUID
    configuration_revision_id: UUID
    repetition: int
    error_type: str
    message: str


@dataclass(frozen=True)
class ConfigurationAggregate:
    configuration_revision_id: UUID
    total_requested_runs: int
    completed_runs: int
    failed_runs: int
    technical_success_rate: float
    average_duration_ms: float | None
    average_retrieved_passage_count: float | None


@dataclass(frozen=True)
class CampaignExecutionResult:
    campaign_id: UUID
    suite_id: UUID
    status: str
    total_requested_runs: int
    completed_runs: int
    failed_runs: int
    technical_success_rate: float
    average_duration_ms: float | None
    average_retrieved_passage_count: float | None
    started_at: Any
    completed_at: Any
    results: tuple[RagExecutionResult, ...]
    errors: tuple[CampaignRunError, ...]
    by_configuration: tuple[ConfigurationAggregate, ...]


class EvaluationCampaignService:
    def __init__(
        self,
        session: Session,
        *,
        execution_service_factory: Callable[[Session], RagExecutionService] = (
            RagExecutionService
        ),
    ) -> None:
        self._session = session
        self._execution_service_factory = execution_service_factory

    def create(
        self,
        name: str,
        *,
        suite_id: UUID,
        configuration_revision_ids: Sequence[UUID],
        repetitions: int,
    ) -> EvaluationCampaign:
        self._validate_suite(suite_id)
        revisions = self._validate_plan_inputs(
            configuration_revision_ids,
            repetitions,
        )
        plan = {
            "configuration_revision_ids": [str(item.id) for item in revisions],
            "repetitions": repetitions,
        }
        campaign = EvaluationCampaign(
            suite_id=suite_id,
            name=name,
            status="pending",
            configuration=plan,
        )
        self._session.add(campaign)
        self._session.commit()
        self._session.refresh(campaign)
        return campaign

    def get(self, campaign_id: UUID) -> EvaluationCampaign | None:
        return self._session.get(EvaluationCampaign, campaign_id)

    def list(self) -> list[EvaluationCampaign]:
        return list(
            self._session.exec(
                select(EvaluationCampaign).order_by(
                    EvaluationCampaign.created_at.desc()
                )
            ).all()
        )

    def list_attempts(self, campaign_id: UUID) -> list[QuestionAttempt]:
        self._require_campaign(campaign_id)
        return list(
            self._session.exec(
                select(QuestionAttempt)
                .where(QuestionAttempt.evaluation_campaign_id == campaign_id)
                .order_by(
                    QuestionAttempt.configuration_revision_id,
                    QuestionAttempt.question_id,
                    QuestionAttempt.attempt_number,
                )
            ).all()
        )

    def attempts_by_configuration(
        self,
        campaign_id: UUID,
    ) -> dict[UUID | None, tuple[QuestionAttempt, ...]]:
        grouped: dict[UUID | None, list[QuestionAttempt]] = {}
        for attempt in self.list_attempts(campaign_id):
            grouped.setdefault(attempt.configuration_revision_id, []).append(attempt)
        return {key: tuple(values) for key, values in grouped.items()}

    def run_campaign(self, campaign_id: UUID) -> CampaignExecutionResult:
        campaign = self._require_campaign(campaign_id)
        try:
            suite_id, revisions, repetitions = self._read_plan(campaign)
            questions = EvaluationSuiteService(self._session).list_questions(suite_id)
            if not questions:
                raise ValueError("La suite d'évaluation ne contient aucune question.")
            campaign.status = "running"
            campaign.started_at = utc_now()
            campaign.completed_at = None
            self._session.add(campaign)
            self._session.commit()

            results: list[RagExecutionResult] = []
            errors: list[CampaignRunError] = []
            runner = self._execution_service_factory(self._session)
            for revision_id in revisions:
                for question in questions:
                    for repetition in range(1, repetitions + 1):
                        try:
                            results.append(
                                runner.execute(
                                    question.id,
                                    configuration_revision_id=revision_id,
                                    evaluation_campaign_id=campaign.id,
                                )
                            )
                        except Exception as error:
                            errors.append(
                                CampaignRunError(
                                    question_id=question.id,
                                    configuration_revision_id=revision_id,
                                    repetition=repetition,
                                    error_type=type(error).__name__,
                                    message=str(error),
                                )
                            )

            campaign.status = "completed"
            campaign.completed_at = utc_now()
            self._session.add(campaign)
            self._session.commit()
            return self._result(campaign, revisions, questions, repetitions, results, errors)
        except Exception:
            self._session.rollback()
            campaign.status = "failed"
            campaign.completed_at = utc_now()
            self._session.add(campaign)
            self._session.commit()
            raise

    def _validate_suite(self, suite_id: UUID) -> None:
        if self._session.get(EvaluationSuite, suite_id) is None:
            raise ValueError(f"Suite d'évaluation introuvable : {suite_id}.")
        count = self._session.exec(
            select(func.count(EvaluationSuiteQuestion.id)).where(
                EvaluationSuiteQuestion.evaluation_suite_id == suite_id
            )
        ).one()
        if count == 0:
            raise ValueError("La suite d'évaluation ne contient aucune question.")

    def _validate_plan_inputs(
        self,
        revision_ids: Sequence[UUID],
        repetitions: int,
    ) -> tuple[ConfigurationProfileRevision, ...]:
        if repetitions < 1:
            raise ValueError("Le nombre de répétitions doit être au moins égal à 1.")
        if not revision_ids:
            raise ValueError("Au moins une révision de configuration est requise.")
        if len(set(revision_ids)) != len(revision_ids):
            raise ValueError(
                "Les révisions de configuration d'une campagne doivent être uniques."
            )
        revisions: list[ConfigurationProfileRevision] = []
        for revision_id in revision_ids:
            revision = self._session.get(ConfigurationProfileRevision, revision_id)
            if revision is None:
                raise ValueError(
                    f"Révision de configuration introuvable : {revision_id}."
                )
            revisions.append(revision)
        return tuple(revisions)

    def _read_plan(
        self,
        campaign: EvaluationCampaign,
    ) -> tuple[UUID, tuple[UUID, ...], int]:
        if campaign.suite_id is None:
            raise ValueError("La campagne ne référence aucune suite.")
        if self._session.get(EvaluationSuite, campaign.suite_id) is None:
            raise ValueError(
                f"Suite d'évaluation introuvable : {campaign.suite_id}."
            )
        plan = campaign.configuration
        try:
            revisions = tuple(
                UUID(str(value)) for value in plan["configuration_revision_ids"]
            )
            repetitions = int(plan["repetitions"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Plan de campagne invalide.") from error
        self._validate_plan_inputs(revisions, repetitions)
        return campaign.suite_id, revisions, repetitions

    def _require_campaign(self, campaign_id: UUID) -> EvaluationCampaign:
        campaign = self.get(campaign_id)
        if campaign is None:
            raise ValueError(f"Campagne d'évaluation introuvable : {campaign_id}.")
        return campaign

    @staticmethod
    def _result(
        campaign: EvaluationCampaign,
        revisions: tuple[UUID, ...],
        questions: list[Question],
        repetitions: int,
        results: list[RagExecutionResult],
        errors: list[CampaignRunError],
    ) -> CampaignExecutionResult:
        total = len(revisions) * len(questions) * repetitions
        durations = [
            float(item.metrics["duration_ms"])
            for item in results
            if item.metrics.get("duration_ms") is not None
        ]
        passages = [
            float(item.metrics["retrieved_passage_count"])
            for item in results
            if item.metrics.get("retrieved_passage_count") is not None
        ]
        by_configuration = tuple(
            EvaluationCampaignService._configuration_aggregate(
                revision_id,
                len(questions) * repetitions,
                results,
                errors,
            )
            for revision_id in revisions
        )
        return CampaignExecutionResult(
            campaign_id=campaign.id,
            suite_id=campaign.suite_id,  # type: ignore[arg-type]
            status=campaign.status,
            total_requested_runs=total,
            completed_runs=len(results),
            failed_runs=len(errors),
            technical_success_rate=(len(results) / total if total else 0.0),
            average_duration_ms=(sum(durations) / len(durations) if durations else None),
            average_retrieved_passage_count=(
                sum(passages) / len(passages) if passages else None
            ),
            started_at=campaign.started_at,
            completed_at=campaign.completed_at,
            results=tuple(results),
            errors=tuple(errors),
            by_configuration=by_configuration,
        )

    @staticmethod
    def _configuration_aggregate(
        revision_id: UUID,
        total: int,
        results: list[RagExecutionResult],
        errors: list[CampaignRunError],
    ) -> ConfigurationAggregate:
        selected = [
            item for item in results if item.configuration_revision_id == revision_id
        ]
        failed = [
            item for item in errors if item.configuration_revision_id == revision_id
        ]
        durations = [
            float(item.metrics["duration_ms"])
            for item in selected
            if item.metrics.get("duration_ms") is not None
        ]
        passages = [
            float(item.metrics["retrieved_passage_count"])
            for item in selected
            if item.metrics.get("retrieved_passage_count") is not None
        ]
        return ConfigurationAggregate(
            configuration_revision_id=revision_id,
            total_requested_runs=total,
            completed_runs=len(selected),
            failed_runs=len(failed),
            technical_success_rate=(len(selected) / total if total else 0.0),
            average_duration_ms=(sum(durations) / len(durations) if durations else None),
            average_retrieved_passage_count=(
                sum(passages) / len(passages) if passages else None
            ),
        )
