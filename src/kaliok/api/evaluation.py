from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from pydantic import BaseModel, Field
from sqlmodel import Session, select

from kaliok.api.dependencies import get_session
from kaliok.discovery import CandidateDiscoveryReadService
from kaliok.entity_resolution import EntityResolutionReadService
from kaliok.qa.evaluation import (
    CampaignExecutionResult,
    EvaluationCampaignService,
    EvaluationSuiteService,
)
from kaliok.qa.service import create_question
from kaliok.configuration.service import ConfigurationRevisionService
from kaliok.storage.models import (
    ConfigurationProfile,
    ConfigurationProfileRevision,
    ConfigurationValue,
    Document,
    DocumentVersion,
    EvaluationCampaign,
    EvaluationSuite,
    EvaluationSuiteQuestion,
    Question,
    QuestionAttempt,
    QuestionEvidence,
    QuestionFeedback,
    SettingDefinition,
    SettingOption,
)


router = APIRouter(prefix="/rag/evaluation", tags=["rag-evaluation"])


class SuiteCreateRequest(BaseModel):
    name: str = Field(min_length=1)
    description: str | None = None
    status: str = "active"


class SuiteUpdateRequest(BaseModel):
    name: str | None = Field(default=None, min_length=1)
    description: str | None = None
    status: str | None = None


class SuiteQuestionRequest(BaseModel):
    question_id: UUID
    position: int | None = None


class QuestionResponse(BaseModel):
    id: UUID
    origin: str
    question_text: str
    expected_answer: str | None
    document_id: UUID | None
    document_version_id: UUID | None
    status: str
    created_at: datetime


class SuiteResponse(BaseModel):
    id: UUID
    name: str
    description: str | None
    status: str
    created_at: datetime
    updated_at: datetime


class SuiteDetailResponse(SuiteResponse):
    questions: list[QuestionResponse]


class SuiteQuestionAssociationResponse(BaseModel):
    id: UUID
    evaluation_suite_id: UUID
    question_id: UUID
    position: int
    created_at: datetime


class QuestionCreateRequest(BaseModel):
    question_text: str = Field(min_length=1)
    expected_answer: str | None = None
    document_id: UUID | None = None
    document_version_id: UUID | None = None


class CampaignCreateRequest(BaseModel):
    name: str = Field(min_length=1)
    suite_id: UUID
    configuration_revision_ids: list[UUID]
    repetitions: int


class CampaignResponse(BaseModel):
    id: UUID
    name: str
    suite_id: UUID | None
    status: str
    configuration: dict[str, Any]
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None


class ExecutionSourceResponse(BaseModel):
    rank: int
    score: float | None
    document_version_id: UUID
    text: str
    extra_data: dict[str, Any]


class ExecutionResultResponse(BaseModel):
    question_id: UUID
    question_attempt_id: UUID
    document_id: UUID
    document_version_id: UUID
    answer: str
    configuration_revision_id: UUID
    configuration: dict[str, Any]
    metrics: dict[str, Any]
    sources: list[ExecutionSourceResponse]


class CampaignRunErrorResponse(BaseModel):
    question_id: UUID
    configuration_revision_id: UUID
    repetition: int
    error_type: str
    message: str


class ConfigurationAggregateResponse(BaseModel):
    configuration_revision_id: UUID
    total_requested_runs: int
    completed_runs: int
    failed_runs: int
    technical_success_rate: float
    average_duration_ms: float | None
    average_retrieved_passage_count: float | None


class CampaignRunResponse(BaseModel):
    campaign_id: UUID
    suite_id: UUID
    status: str
    total_requested_runs: int
    completed_runs: int
    failed_runs: int
    technical_success_rate: float
    average_duration_ms: float | None
    average_retrieved_passage_count: float | None
    started_at: datetime | None
    completed_at: datetime | None
    results: list[ExecutionResultResponse]
    errors: list[CampaignRunErrorResponse]
    by_configuration: list[ConfigurationAggregateResponse]


class AttemptResponse(BaseModel):
    id: UUID
    question_id: UUID
    attempt_number: int
    configuration_revision_id: UUID | None
    evaluation_campaign_id: UUID | None
    strategy: str
    status: str
    answer_text: str | None
    failure_reason: str | None
    configuration: dict[str, Any]
    metrics: dict[str, Any]
    created_at: datetime
    completed_at: datetime | None


class EvidenceResponse(BaseModel):
    id: UUID
    document_version_id: UUID | None
    rank: int | None
    score: float | None
    evidence_text: str | None
    extra_data: dict[str, Any]


class FeedbackRequest(BaseModel):
    assessment: str | None = None
    rating: int | None = None
    is_correct: bool | None = None
    corrected_answer: str | None = None
    comment: str | None = None


class FeedbackResponse(BaseModel):
    id: UUID
    question_id: UUID
    question_attempt_id: UUID | None
    origin: str
    rating: int | None
    is_correct: bool | None
    corrected_answer: str | None
    comment: str | None
    extra_data: dict[str, Any]
    created_at: datetime


class AttemptDetailResponse(BaseModel):
    attempt: AttemptResponse
    question: QuestionResponse
    evidence: list[EvidenceResponse]
    feedback: list[FeedbackResponse]


class ConfigurationRevisionResponse(BaseModel):
    revision_id: UUID
    revision_number: int
    status: str
    created_at: datetime
    change_reason: str | None
    values: dict[str, Any] = Field(default_factory=dict)
    options: dict[str, list[dict[str, str]]] = Field(default_factory=dict)


class ConfigurationRevisionCreateRequest(BaseModel):
    source_revision_id: UUID
    values: dict[str, Any]
    change_reason: str | None = None


class ConfigurationProfileResponse(BaseModel):
    profile_id: UUID
    profile_key: str
    label: str
    is_active: bool
    revisions: list[ConfigurationRevisionResponse]


class EvaluationDocumentResponse(BaseModel):
    document_id: UUID
    title: str | None
    document_version_id: UUID
    filename: str
    version_number: int
    processing_status: str
    page_count: int | None


def _model(model: type[BaseModel], value: object) -> BaseModel:
    return model.model_validate(value, from_attributes=True)


def _service_error(error: ValueError) -> HTTPException:
    detail = str(error)
    lowered = detail.lower()
    if "introuvable" in lowered:
        return HTTPException(status_code=404, detail=detail)
    if "déjà" in lowered or "position" in lowered:
        return HTTPException(status_code=409, detail=detail)
    return HTTPException(status_code=400, detail=detail)


@router.post("/suites", response_model=SuiteResponse, status_code=201)
def create_suite(body: SuiteCreateRequest, session: Session = Depends(get_session)):
    return EvaluationSuiteService(session).create(
        body.name, description=body.description, status=body.status
    )


@router.get("/suites", response_model=list[SuiteResponse])
def list_suites(session: Session = Depends(get_session)):
    return EvaluationSuiteService(session).list()


@router.get("/suites/{suite_id}", response_model=SuiteDetailResponse)
def get_suite(suite_id: UUID, session: Session = Depends(get_session)):
    service = EvaluationSuiteService(session)
    suite = service.get(suite_id)
    if suite is None:
        raise HTTPException(status_code=404, detail="Suite d'évaluation introuvable.")
    return SuiteDetailResponse(
        **_model(SuiteResponse, suite).model_dump(),
        questions=[_model(QuestionResponse, item) for item in service.list_questions(suite_id)],
    )


@router.patch("/suites/{suite_id}", response_model=SuiteResponse)
def update_suite(
    suite_id: UUID,
    body: SuiteUpdateRequest,
    session: Session = Depends(get_session),
):
    try:
        return EvaluationSuiteService(session).update(
            suite_id, **body.model_dump(exclude_unset=True)
        )
    except ValueError as error:
        raise _service_error(error) from error


@router.post(
    "/suites/{suite_id}/questions",
    response_model=SuiteQuestionAssociationResponse,
    status_code=201,
)
def add_suite_question(
    suite_id: UUID,
    body: SuiteQuestionRequest,
    session: Session = Depends(get_session),
):
    try:
        return EvaluationSuiteService(session).add_question(
            suite_id, body.question_id, position=body.position
        )
    except ValueError as error:
        raise _service_error(error) from error


@router.delete(
    "/suites/{suite_id}/questions/{question_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def remove_suite_question(
    suite_id: UUID,
    question_id: UUID,
    session: Session = Depends(get_session),
):
    if not EvaluationSuiteService(session).remove_question(suite_id, question_id):
        raise HTTPException(status_code=404, detail="Association introuvable.")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/suites/{suite_id}/questions", response_model=list[QuestionResponse])
def list_suite_questions(suite_id: UUID, session: Session = Depends(get_session)):
    try:
        return EvaluationSuiteService(session).list_questions(suite_id)
    except ValueError as error:
        raise _service_error(error) from error


@router.post("/questions", response_model=QuestionResponse, status_code=201)
def create_evaluation_question(
    body: QuestionCreateRequest,
    session: Session = Depends(get_session),
):
    if body.document_id is None and body.document_version_id is None:
        raise HTTPException(
            status_code=400,
            detail="Un document ou une version documentaire est requis.",
        )
    return create_question(
        body.question_text,
        origin="evaluation",
        expected_answer=body.expected_answer,
        document_id=body.document_id,
        document_version_id=body.document_version_id,
        session=session,
    )


@router.get("/questions/{question_id}", response_model=QuestionResponse)
def get_evaluation_question(
    question_id: UUID,
    session: Session = Depends(get_session),
):
    question = session.get(Question, question_id)
    if question is None:
        raise HTTPException(status_code=404, detail="Question introuvable.")
    return question


@router.post("/campaigns", response_model=CampaignResponse, status_code=201)
def create_campaign(
    body: CampaignCreateRequest,
    session: Session = Depends(get_session),
):
    try:
        return EvaluationCampaignService(session).create(
            body.name,
            suite_id=body.suite_id,
            configuration_revision_ids=body.configuration_revision_ids,
            repetitions=body.repetitions,
        )
    except ValueError as error:
        raise _service_error(error) from error


@router.get("/campaigns", response_model=list[CampaignResponse])
def list_campaigns(session: Session = Depends(get_session)):
    return EvaluationCampaignService(session).list()


@router.get("/campaigns/{campaign_id}", response_model=CampaignResponse)
def get_campaign(campaign_id: UUID, session: Session = Depends(get_session)):
    campaign = EvaluationCampaignService(session).get(campaign_id)
    if campaign is None:
        raise HTTPException(status_code=404, detail="Campagne introuvable.")
    return campaign


@router.post("/campaigns/{campaign_id}/run", response_model=CampaignRunResponse)
def run_campaign(campaign_id: UUID, session: Session = Depends(get_session)):
    try:
        result = EvaluationCampaignService(session).run_campaign(campaign_id)
    except ValueError as error:
        raise _service_error(error) from error
    return _campaign_run_response(result)


@router.get("/campaigns/{campaign_id}/attempts", response_model=list[AttemptResponse])
def list_campaign_attempts(
    campaign_id: UUID,
    session: Session = Depends(get_session),
):
    try:
        return EvaluationCampaignService(session).list_attempts(campaign_id)
    except ValueError as error:
        raise _service_error(error) from error


@router.get("/attempts/{attempt_id}", response_model=AttemptDetailResponse)
def get_attempt(attempt_id: UUID, session: Session = Depends(get_session)):
    attempt = session.get(QuestionAttempt, attempt_id)
    if attempt is None:
        raise HTTPException(status_code=404, detail="Tentative introuvable.")
    question = session.get(Question, attempt.question_id)
    if question is None:
        raise HTTPException(status_code=404, detail="Question introuvable.")
    evidence = session.exec(
        select(QuestionEvidence)
        .where(QuestionEvidence.question_attempt_id == attempt_id)
        .order_by(QuestionEvidence.rank)
    ).all()
    feedback = session.exec(
        select(QuestionFeedback)
        .where(QuestionFeedback.question_attempt_id == attempt_id)
        .order_by(QuestionFeedback.created_at)
    ).all()
    return AttemptDetailResponse(
        attempt=_model(AttemptResponse, attempt),
        question=_model(QuestionResponse, question),
        evidence=[_model(EvidenceResponse, item) for item in evidence],
        feedback=[_model(FeedbackResponse, item) for item in feedback],
    )


@router.post(
    "/attempts/{attempt_id}/feedback",
    response_model=FeedbackResponse,
    status_code=201,
)
def create_feedback(
    attempt_id: UUID,
    body: FeedbackRequest,
    session: Session = Depends(get_session),
):
    attempt = session.get(QuestionAttempt, attempt_id)
    if attempt is None:
        raise HTTPException(status_code=404, detail="Tentative introuvable.")
    feedback = QuestionFeedback(
        question_id=attempt.question_id,
        question_attempt_id=attempt.id,
        origin="user",
        rating=body.rating,
        is_correct=body.is_correct,
        corrected_answer=body.corrected_answer,
        comment=body.comment,
        extra_data={"assessment": body.assessment} if body.assessment else {},
    )
    session.add(feedback)
    session.commit()
    session.refresh(feedback)
    return feedback


@router.patch("/feedback/{feedback_id}", response_model=FeedbackResponse)
def update_feedback(
    feedback_id: UUID,
    body: FeedbackRequest,
    session: Session = Depends(get_session),
):
    feedback = session.get(QuestionFeedback, feedback_id)
    if feedback is None:
        raise HTTPException(status_code=404, detail="Évaluation introuvable.")
    for field in ("rating", "is_correct", "corrected_answer", "comment"):
        setattr(feedback, field, getattr(body, field))
    feedback.extra_data = (
        {**feedback.extra_data, "assessment": body.assessment}
        if body.assessment else feedback.extra_data
    )
    session.add(feedback)
    session.commit()
    session.refresh(feedback)
    return feedback


@router.get("/configurations", response_model=list[ConfigurationProfileResponse])
def list_configurations(session: Session = Depends(get_session)):
    profiles = session.exec(
        select(ConfigurationProfile).order_by(ConfigurationProfile.profile_key)
    ).all()
    response = []
    for profile in profiles:
        revisions = session.exec(
            select(ConfigurationProfileRevision)
            .where(ConfigurationProfileRevision.profile_id == profile.id)
            .order_by(ConfigurationProfileRevision.revision_number.desc())
        ).all()
        # Lightweight test adapters and older API consumers may expose only the
        # profile/revision rows. Real SQLModel sessions receive the typed values.
        has_typed_revisions = not revisions or isinstance(
            revisions[0], ConfigurationProfileRevision
        )
        definitions = (
            {item.id: item for item in session.exec(select(SettingDefinition)).all()}
            if has_typed_revisions else {}
        )
        all_options = (
            session.exec(
                select(SettingOption).where(SettingOption.is_active == True)  # noqa: E712
            ).all()
            if has_typed_revisions else []
        )
        response.append(
            ConfigurationProfileResponse(
                profile_id=profile.id,
                profile_key=profile.profile_key,
                label=profile.label,
                is_active=profile.is_active,
                revisions=[
                    ConfigurationRevisionResponse(
                        revision_id=item.id,
                        revision_number=item.revision_number,
                        status=item.status,
                        created_at=item.created_at,
                        change_reason=item.change_reason,
                        values=(
                            _configuration_values(session, item.id, definitions)
                            if has_typed_revisions else {}
                        ),
                        options=_configuration_options(definitions, all_options),
                    )
                    for item in revisions
                ],
            )
        )
    return response


@router.post(
    "/configurations/revisions",
    response_model=ConfigurationRevisionResponse,
    status_code=201,
)
def create_configuration_revision(
    body: ConfigurationRevisionCreateRequest,
    session: Session = Depends(get_session),
):
    try:
        revision = ConfigurationRevisionService(session).create_from(
            body.source_revision_id,
            body.values,
            change_reason=body.change_reason,
        )
    except (ValueError, TypeError) as error:
        raise _service_error(ValueError(str(error))) from error
    definitions = {item.id: item for item in session.exec(select(SettingDefinition)).all()}
    options = session.exec(select(SettingOption)).all()
    return ConfigurationRevisionResponse(
        revision_id=revision.id,
        revision_number=revision.revision_number,
        status=revision.status,
        created_at=revision.created_at,
        change_reason=revision.change_reason,
        values=_configuration_values(session, revision.id, definitions),
        options=_configuration_options(definitions, options),
    )


@router.post(
    "/configurations/revisions/{revision_id}/activate",
    response_model=ConfigurationRevisionResponse,
)
def activate_configuration_revision(
    revision_id: UUID,
    session: Session = Depends(get_session),
):
    try:
        revision = ConfigurationRevisionService(session).activate(revision_id)
    except ValueError as error:
        raise _service_error(error) from error
    definitions = {item.id: item for item in session.exec(select(SettingDefinition)).all()}
    options = session.exec(select(SettingOption)).all()
    return ConfigurationRevisionResponse(
        revision_id=revision.id,
        revision_number=revision.revision_number,
        status=revision.status,
        created_at=revision.created_at,
        change_reason=revision.change_reason,
        values=_configuration_values(session, revision.id, definitions),
        options=_configuration_options(definitions, options),
    )


def _configuration_values(session, revision_id, definitions):
    result = {}
    for value in session.exec(
        select(ConfigurationValue).where(ConfigurationValue.revision_id == revision_id)
    ).all():
        definition = definitions.get(value.setting_definition_id)
        if definition is None:
            continue
        if definition.value_type == "choice":
            option = session.get(SettingOption, value.selected_option_id)
            result[definition.setting_key] = option.option_key if option else None
            result[f"{definition.setting_key}_label"] = option.label if option else None
        else:
            result[definition.setting_key] = getattr(value, f"value_{definition.value_type}")
    return result


def _configuration_options(definitions, options):
    result = {}
    for definition in definitions.values():
        if definition.setting_key not in {"model", "provider", "builder"}:
            continue
        result[definition.setting_key] = [
            {"value": item.option_key, "label": item.label}
            for item in options
            if item.setting_definition_id == definition.id and item.is_active
        ]
    return result


@router.get("/documents", response_model=list[EvaluationDocumentResponse])
def list_evaluation_documents(session: Session = Depends(get_session)):
    rows = session.exec(
        select(Document, DocumentVersion)
        .join(DocumentVersion, DocumentVersion.document_id == Document.id)
        .order_by(Document.title, DocumentVersion.version_number.desc())
    ).all()
    return [
        EvaluationDocumentResponse(
            document_id=document.id,
            title=document.title,
            document_version_id=version.id,
            filename=version.filename,
            version_number=version.version_number,
            processing_status=version.processing_status,
            page_count=version.page_count,
        )
        for document, version in rows
    ]


@router.get("/discovery/runs")
def list_discovery_runs(
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    session: Session = Depends(get_session),
):
    return CandidateDiscoveryReadService(session).list_runs(limit=limit, offset=offset)


@router.get("/discovery/compare")
def compare_discovery_runs(
    run_a: UUID = Query(...),
    run_b: UUID = Query(...),
    session: Session = Depends(get_session),
):
    try:
        return CandidateDiscoveryReadService(session).compare_runs(run_a, run_b)
    except ValueError as error:
        detail = str(error)
        raise HTTPException(status_code=404 if "introuvable" in detail else 400, detail=detail) from error


@router.get("/discovery/runs/{run_id}")
def get_discovery_run(run_id: UUID, session: Session = Depends(get_session)):
    result = CandidateDiscoveryReadService(session).get_run(run_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Analyse de découverte inconnue.")
    return result


@router.get("/discovery/runs/{run_id}/candidates")
def list_discovery_candidates(
    run_id: UUID,
    candidate_type: str | None = None,
    value: str | None = None,
    page: int | None = Query(default=None, ge=1),
    limit: int = Query(default=100, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    session: Session = Depends(get_session),
):
    result = CandidateDiscoveryReadService(session).list_candidates(
        run_id,
        candidate_type=candidate_type,
        value=value,
        page=page,
        limit=limit,
        offset=offset,
    )
    if result is None:
        raise HTTPException(status_code=404, detail="Analyse de découverte inconnue.")
    return result


@router.get("/discovery/candidates/{candidate_id}")
def get_discovery_candidate(
    candidate_id: UUID,
    session: Session = Depends(get_session),
):
    result = CandidateDiscoveryReadService(session).get_candidate(candidate_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Élément repéré inconnu.")
    return result


@router.get("/entity-resolution/runs")
def list_entity_resolution_runs(
    status: str | None = None,
    document_version_id: UUID | None = None,
    limit: int = Query(default=25, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    session: Session = Depends(get_session),
):
    return EntityResolutionReadService(session).list_runs(
        status=status,
        document_version_id=document_version_id,
        limit=limit,
        offset=offset,
    )


@router.get("/entity-resolution/runs/{run_id}")
def get_entity_resolution_run(run_id: UUID, session: Session = Depends(get_session)):
    result = EntityResolutionReadService(session).get_run(run_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Analyse de résolution d’entités inconnue.")
    return result


@router.get("/entity-resolution/runs/{run_id}/entities")
def list_resolved_entities(
    run_id: UUID,
    entity_type: str | None = None,
    canonical_label: str | None = None,
    status: str | None = None,
    grouped_only: bool = False,
    singleton_only: bool = False,
    limit: int = Query(default=25, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    session: Session = Depends(get_session),
):
    result = EntityResolutionReadService(session).list_entities(
        run_id,
        entity_type=entity_type,
        canonical_label=canonical_label,
        status=status,
        grouped_only=grouped_only,
        singleton_only=singleton_only,
        limit=limit,
        offset=offset,
    )
    if result is None:
        raise HTTPException(status_code=404, detail="Analyse de résolution d’entités inconnue.")
    return result


@router.get("/entity-resolution/entities/{entity_id}")
def get_resolved_entity(entity_id: UUID, session: Session = Depends(get_session)):
    result = EntityResolutionReadService(session).get_entity(entity_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Entité proposée inconnue.")
    return result


def _campaign_run_response(result: CampaignExecutionResult) -> CampaignRunResponse:
    return CampaignRunResponse(
        campaign_id=result.campaign_id,
        suite_id=result.suite_id,
        status=result.status,
        total_requested_runs=result.total_requested_runs,
        completed_runs=result.completed_runs,
        failed_runs=result.failed_runs,
        technical_success_rate=result.technical_success_rate,
        average_duration_ms=result.average_duration_ms,
        average_retrieved_passage_count=result.average_retrieved_passage_count,
        started_at=result.started_at,
        completed_at=result.completed_at,
        results=[
            ExecutionResultResponse(
                question_id=item.question_id,
                question_attempt_id=item.question_attempt_id,
                document_id=item.document_id,
                document_version_id=item.document_version_id,
                answer=item.answer,
                configuration_revision_id=item.configuration_revision_id,
                configuration=item.configuration,
                metrics=item.metrics,
                sources=[
                    ExecutionSourceResponse.model_validate(source, from_attributes=True)
                    for source in item.sources
                ],
            )
            for item in result.results
        ],
        errors=[
            CampaignRunErrorResponse.model_validate(item, from_attributes=True)
            for item in result.errors
        ],
        by_configuration=[
            ConfigurationAggregateResponse.model_validate(item, from_attributes=True)
            for item in result.by_configuration
        ],
    )
