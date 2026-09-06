from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from pydantic import field_validator
from sqlalchemy import CheckConstraint, Column, Index, String, UniqueConstraint, text
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlmodel import Field, SQLModel

from pgvector.sqlalchemy import Vector


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Source(SQLModel, table=True):
    __tablename__ = "sources"

    id: UUID = Field(default_factory=uuid4, primary_key=True)

    name: str
    source_type: str

    external_reference: str | None = None

    configuration: dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSONB, nullable=False),
    )

    is_active: bool = True

    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class Document(SQLModel, table=True):
    __tablename__ = "documents"

    id: UUID = Field(default_factory=uuid4, primary_key=True)

    source_id: UUID | None = Field(
        default=None,
        foreign_key="sources.id",
    )

    external_id: str | None = None
    title: str | None = None

    document_family: str | None = None
    status: str = "active"
    language: str | None = None

    extra_data: dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column("metadata", JSONB, nullable=False),
    )

    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class DocumentVersion(SQLModel, table=True):
    __tablename__ = "document_versions"

    id: UUID = Field(default_factory=uuid4, primary_key=True)

    document_id: UUID = Field(
        foreign_key="documents.id",
    )

    previous_version_id: UUID | None = Field(
        default=None,
        foreign_key="document_versions.id",
    )

    origin_version_id: UUID | None = Field(
        default=None,
        foreign_key="document_versions.id",
    )

    version_number: int

    filename: str
    mime_type: str | None = None
    file_hash: str
    file_size: int | None = None
    storage_uri: str

    page_count: int | None = None

    document_type: str | None = None
    document_subtype: str | None = None

    version_status: str = "draft"
    processing_status: str = "pending"
    readability_status: str = "unknown"

    readability_score: float | None = None

    is_current: bool = False

    extra_data: dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column("metadata", JSONB, nullable=False),
    )

    created_at: datetime = Field(default_factory=utc_now)
    processed_at: datetime | None = None


class NormalizedContentUnit(SQLModel, table=True):
    __tablename__ = "normalized_content_units"
    __table_args__ = (
        Index(
            "uq_normalized_content_units_historical_version_index",
            "document_version_id",
            "unit_index",
            unique=True,
            postgresql_where=text("processing_run_id IS NULL"),
        ),
        Index(
            "uq_normalized_content_units_historical_version_source_unit",
            "document_version_id",
            "source_unit_id",
            unique=True,
            postgresql_where=text(
                "processing_run_id IS NULL AND source_unit_id IS NOT NULL"
            ),
        ),
        Index(
            "uq_normalized_content_units_run_version_index",
            "processing_run_id",
            "document_version_id",
            "unit_index",
            unique=True,
            postgresql_where=text("processing_run_id IS NOT NULL"),
        ),
        Index(
            "uq_normalized_content_units_run_version_source_unit",
            "processing_run_id",
            "document_version_id",
            "source_unit_id",
            unique=True,
            postgresql_where=text(
                "processing_run_id IS NOT NULL AND source_unit_id IS NOT NULL"
            ),
        ),
        CheckConstraint(
            "unit_index >= 0",
            name="ck_normalized_content_units_index_nonnegative",
        ),
    )

    id: UUID = Field(default_factory=uuid4, primary_key=True)

    document_version_id: UUID | None = Field(
        default=None,
        foreign_key="document_versions.id",
    )

    processing_run_id: UUID | None = Field(
        default=None,
        foreign_key="processing_runs.id",
        index=True,
    )

    parent_unit_id: UUID | None = Field(
        default=None,
        foreign_key="normalized_content_units.id",
    )

    unit_index: int
    content_type: str
    content: str

    source_reference: str | None = None
    source_unit_id: str | None = None

    created_at: datetime = Field(default_factory=utc_now)


class Page(SQLModel, table=True):
    __tablename__ = "pages"

    id: UUID = Field(default_factory=uuid4, primary_key=True)

    document_version_id: UUID | None = Field(
        default=None,
        foreign_key="document_versions.id",
    )

    page_number: int

    page_status: str = "active"

    width: float | None = None
    height: float | None = None

    has_native_text: bool = False
    native_text_length: int | None = None

    readability_status: str = "unknown"
    readability_score: float | None = None
    readability_reason: str | None = None

    perception_mode: str = "unknown"

    ocr_required: bool = False
    ocr_performed: bool = False
    ocr_reason: str | None = None

    ocr_engine: str | None = None
    ocr_confidence_mean: float | None = None

    ocr_processing_run_id: UUID | None = Field(
        default=None,
        foreign_key="processing_runs.id",
    )

    perception_processing_run_id: UUID | None = Field(
        default=None,
        foreign_key="processing_runs.id",
    )

    extra_data: dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column("metadata", JSONB, nullable=False),
    )

    created_at: datetime = Field(default_factory=utc_now)


class ContentBlock(SQLModel, table=True):
    __tablename__ = "content_blocks"

    id: UUID = Field(default_factory=uuid4, primary_key=True)

    page_id: UUID = Field(
        foreign_key="pages.id",
    )

    processing_run_id: UUID | None = Field(
        default=None,
        foreign_key="processing_runs.id",
    )

    parent_block_id: UUID | None = Field(
        default=None,
        foreign_key="content_blocks.id",
    )

    block_index: int
    reading_order: int | None = None

    block_type: str = "text"

    content: str

    extraction_method: str
    extraction_engine: str | None = None
    extraction_engine_version: str | None = None

    confidence: float | None = None

    bbox: dict[str, Any] | None = Field(
        default=None,
        sa_column=Column(JSONB, nullable=True),
    )

    bbox_x: float | None = None
    bbox_y: float | None = None
    bbox_width: float | None = None
    bbox_height: float | None = None

    coordinate_system: str | None = None

    extra_data: dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column("metadata", JSONB, nullable=False),
    )

    created_at: datetime = Field(default_factory=utc_now)


class ContentBlockFragment(SQLModel, table=True):
    __tablename__ = "content_block_fragments"
    __table_args__ = (
        UniqueConstraint(
            "content_block_id",
            "fragment_index",
            name="uq_content_block_fragments_block_index",
        ),
    )

    id: UUID = Field(default_factory=uuid4, primary_key=True)

    content_block_id: UUID = Field(
        foreign_key="content_blocks.id",
        index=True,
    )

    page_id: UUID = Field(
        foreign_key="pages.id",
        index=True,
    )

    fragment_index: int
    reading_order: int | None = None

    content: str

    bbox_x: float | None = None
    bbox_y: float | None = None
    bbox_width: float | None = None
    bbox_height: float | None = None

    coordinate_system: str | None = None

    extra_data: dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column("metadata", JSONB, nullable=False),
    )

    created_at: datetime = Field(default_factory=utc_now)


class NormalizedContentUnitSource(SQLModel, table=True):
    __tablename__ = "normalized_content_unit_sources"
    __table_args__ = (
        UniqueConstraint(
            "normalized_content_unit_id",
            "content_block_id",
            name="uq_normalized_content_unit_sources_unit_block",
        ),
        UniqueConstraint(
            "normalized_content_unit_id",
            "source_order",
            name="uq_normalized_content_unit_sources_unit_order",
        ),
        CheckConstraint(
            "source_order >= 0",
            name="ck_normalized_content_unit_sources_order_nonnegative",
        ),
    )

    id: UUID = Field(default_factory=uuid4, primary_key=True)

    normalized_content_unit_id: UUID = Field(
        foreign_key="normalized_content_units.id",
    )

    content_block_id: UUID = Field(
        foreign_key="content_blocks.id",
        index=True,
    )

    source_order: int

    created_at: datetime = Field(default_factory=utc_now)


class DiscoveredCandidate(SQLModel, table=True):
    __tablename__ = "discovered_candidates"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    document_version_id: UUID = Field(foreign_key="document_versions.id", index=True)
    processing_run_id: UUID = Field(foreign_key="processing_runs.id", index=True)
    candidate_type: str
    raw_value: str
    normalized_value: str | None = None
    payload: dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSONB, nullable=False),
    )
    confidence: float | None = None
    detector_key: str
    detector_version: str | None = None
    created_at: datetime = Field(default_factory=utc_now)


class CandidateSourceFragment(SQLModel, table=True):
    __tablename__ = "candidate_source_fragments"
    __table_args__ = (
        UniqueConstraint(
            "candidate_id",
            "fragment_order",
            name="uq_candidate_source_fragments_candidate_order",
        ),
        CheckConstraint(
            "fragment_order >= 0",
            name="ck_candidate_source_fragments_order_nonnegative",
        ),
        CheckConstraint(
            "start_offset IS NULL OR start_offset >= 0",
            name="ck_candidate_source_fragments_start_nonnegative",
        ),
        CheckConstraint(
            "end_offset IS NULL OR start_offset IS NULL OR end_offset >= start_offset",
            name="ck_candidate_source_fragments_offsets_ordered",
        ),
    )

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    candidate_id: UUID = Field(foreign_key="discovered_candidates.id", index=True)
    normalized_content_unit_id: UUID = Field(
        foreign_key="normalized_content_units.id",
        index=True,
    )
    fragment_order: int
    start_offset: int | None = None
    end_offset: int | None = None
    exact_text: str
    role: str | None = None
    created_at: datetime = Field(default_factory=utc_now)


class DocumentChunk(SQLModel, table=True):
    __tablename__ = "document_chunks"

    id: UUID = Field(default_factory=uuid4, primary_key=True)

    document_version_id: UUID = Field(
        foreign_key="document_versions.id",
    )

    parent_chunk_id: UUID | None = Field(
        default=None,
        foreign_key="document_chunks.id",
    )

    chunk_index: int

    content: str

    token_count: int | None = None
    char_count: int

    page_start: int | None = None
    page_end: int | None = None

    breadcrumb: list[str] | None = Field(
        default=None,
        sa_column=Column(ARRAY(String), nullable=True),
    )

    chunking_strategy: str
    chunking_version: str | None = None

    extra_data: dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column("metadata", JSONB, nullable=False),
    )

    created_at: datetime = Field(default_factory=utc_now)


class ChunkContentBlock(SQLModel, table=True):
    __tablename__ = "chunk_content_blocks"

    chunk_id: UUID = Field(
        foreign_key="document_chunks.id",
        primary_key=True,
    )

    content_block_id: UUID = Field(
        foreign_key="content_blocks.id",
        primary_key=True,
    )

    block_order: int


class ProcessingRun(SQLModel, table=True):
    __tablename__ = "processing_runs"
    __table_args__ = (
        CheckConstraint(
            "execution_environment IS NULL OR execution_environment IN "
            "('production', 'experiment')",
            name="ck_processing_runs_execution_environment",
        ),
    )

    id: UUID = Field(default_factory=uuid4, primary_key=True)

    document_version_id: UUID | None = Field(
        default=None,
        foreign_key="document_versions.id",
    )

    process_type: str
    status: str

    engine: str | None = None
    engine_version: str | None = None

    execution_environment: str | None = None

    configuration_revision_id: UUID | None = Field(
        default=None,
        foreign_key="configuration_profile_revisions.id",
        index=True,
    )

    execution_group_id: UUID | None = Field(default=None, index=True)

    configuration_hash: str | None = None

    configuration: dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSONB, nullable=False),
    )

    metrics: dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSONB, nullable=False),
    )

    started_at: datetime = Field(default_factory=utc_now)
    completed_at: datetime | None = None

    error_message: str | None = None

    def __init__(self, **data: Any) -> None:
        environment = data.get("execution_environment")
        if environment is not None and environment not in {"production", "experiment"}:
            raise ValueError(
                "execution_environment doit être 'production' ou 'experiment'."
            )
        super().__init__(**data)

    @field_validator("execution_environment")
    @classmethod
    def validate_execution_environment(cls, value: str | None) -> str | None:
        if value is not None and value not in {"production", "experiment"}:
            raise ValueError(
                "execution_environment doit être 'production' ou 'experiment'."
            )
        return value


class EntityResolutionScopeItem(SQLModel, table=True):
    __tablename__ = "entity_resolution_scope_items"
    __table_args__ = (
        UniqueConstraint("processing_run_id", "discovered_candidate_id", name="uq_er_scope_run_candidate"),
        UniqueConstraint("processing_run_id", "scope_order", name="uq_er_scope_run_order"),
        CheckConstraint("scope_order >= 0", name="ck_er_scope_order_nonnegative"),
    )
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    processing_run_id: UUID = Field(foreign_key="processing_runs.id", index=True)
    discovered_candidate_id: UUID = Field(foreign_key="discovered_candidates.id", index=True)
    scope_order: int
    created_at: datetime = Field(default_factory=utc_now)


class Entity(SQLModel, table=True):
    __tablename__ = "entities"
    __table_args__ = (
        UniqueConstraint("processing_run_id", "entity_index", name="uq_entities_run_index"),
        CheckConstraint("entity_index >= 0", name="ck_entities_index_nonnegative"),
    )
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    processing_run_id: UUID = Field(foreign_key="processing_runs.id", index=True)
    entity_index: int
    entity_type: str
    canonical_label: str
    status: str = "proposed"
    confidence: float | None = None
    extra_data: dict[str, Any] = Field(default_factory=dict, sa_column=Column("metadata", JSONB, nullable=False))
    created_at: datetime = Field(default_factory=utc_now)


class EntityMembership(SQLModel, table=True):
    __tablename__ = "entity_memberships"
    __table_args__ = (
        UniqueConstraint("processing_run_id", "discovered_candidate_id", name="uq_entity_membership_run_candidate"),
    )
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    entity_id: UUID = Field(foreign_key="entities.id", index=True)
    discovered_candidate_id: UUID = Field(foreign_key="discovered_candidates.id", index=True)
    processing_run_id: UUID = Field(foreign_key="processing_runs.id", index=True)
    membership_status: str = "proposed"
    confidence: float | None = None
    decision_origin: str
    extra_data: dict[str, Any] = Field(default_factory=dict, sa_column=Column("metadata", JSONB, nullable=False))
    created_at: datetime = Field(default_factory=utc_now)


class EntityResolutionEvidence(SQLModel, table=True):
    __tablename__ = "entity_resolution_evidence"
    __table_args__ = (
        UniqueConstraint("membership_id", "evidence_order", name="uq_er_evidence_membership_order"),
        CheckConstraint("evidence_order >= 0", name="ck_er_evidence_order_nonnegative"),
    )
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    membership_id: UUID = Field(foreign_key="entity_memberships.id", index=True)
    evidence_order: int
    signal_key: str
    method: str
    score: float | None = None
    explanation: str | None = None
    extra_data: dict[str, Any] = Field(default_factory=dict, sa_column=Column("metadata", JSONB, nullable=False))
    created_at: datetime = Field(default_factory=utc_now)


class EmbeddingModel(SQLModel, table=True):
    __tablename__ = "embedding_models"

    id: UUID = Field(default_factory=uuid4, primary_key=True)

    provider: str
    model_name: str
    model_version: str | None = None

    dimensions: int
    distance_metric: str = "cosine"

    configuration: dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSONB, nullable=False),
    )

    is_active: bool = True

    created_at: datetime = Field(default_factory=utc_now)


class SettingCategory(SQLModel, table=True):
    __tablename__ = "setting_categories"
    __table_args__ = (
        UniqueConstraint(
            "category_key",
            name="uq_setting_categories_category_key",
        ),
    )

    id: UUID = Field(default_factory=uuid4, primary_key=True)

    parent_category_id: UUID | None = Field(
        default=None,
        foreign_key="setting_categories.id",
        index=True,
    )

    category_key: str = Field(index=True)
    label: str
    description: str | None = None

    display_order: int = 0
    is_active: bool = True

    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class SettingDefinition(SQLModel, table=True):
    __tablename__ = "setting_definitions"
    __table_args__ = (
        UniqueConstraint(
            "category_id",
            "setting_key",
            name="uq_setting_definitions_category_key",
        ),
        CheckConstraint(
            "value_type IN ("
            "'string', "
            "'integer', "
            "'float', "
            "'boolean', "
            "'choice', "
            "'multichoice'"
            ")",
            name="ck_setting_definitions_value_type",
        ),
    )

    id: UUID = Field(default_factory=uuid4, primary_key=True)

    category_id: UUID = Field(
        foreign_key="setting_categories.id",
        index=True,
    )

    setting_key: str
    label: str
    description: str | None = None

    value_type: str

    is_required: bool = False
    is_editable: bool = True
    is_encrypted: bool = False

    display_order: int = 0

    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class SettingOption(SQLModel, table=True):
    __tablename__ = "setting_options"
    __table_args__ = (
        UniqueConstraint(
            "setting_definition_id",
            "option_key",
            name="uq_setting_options_definition_key",
        ),
    )

    id: UUID = Field(default_factory=uuid4, primary_key=True)

    setting_definition_id: UUID = Field(
        foreign_key="setting_definitions.id",
        index=True,
    )

    option_key: str
    label: str
    description: str | None = None

    display_order: int = 0
    is_active: bool = True

    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class ConfigurationProfile(SQLModel, table=True):
    __tablename__ = "configuration_profiles"
    __table_args__ = (
        UniqueConstraint(
            "profile_key",
            name="uq_configuration_profiles_profile_key",
        ),
    )

    id: UUID = Field(default_factory=uuid4, primary_key=True)

    profile_key: str = Field(index=True)
    label: str
    description: str | None = None

    is_active: bool = True
    is_default: bool = False

    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class ConfigurationProfileRevision(SQLModel, table=True):
    __tablename__ = "configuration_profile_revisions"
    __table_args__ = (
        UniqueConstraint(
            "profile_id",
            "revision_number",
            name="uq_configuration_profile_revisions_number",
        ),
        CheckConstraint(
            "revision_number > 0",
            name="ck_configuration_profile_revisions_number_positive",
        ),
        CheckConstraint(
            "status IN ('draft', 'active', 'retired')",
            name="ck_configuration_profile_revisions_status",
        ),
        CheckConstraint(
            "created_by_actor_type IN "
            "('user', 'system', 'service', 'migration')",
            name="ck_configuration_profile_revisions_actor_type",
        ),
    )

    id: UUID = Field(default_factory=uuid4, primary_key=True)

    profile_id: UUID = Field(
        foreign_key="configuration_profiles.id",
        index=True,
    )

    revision_number: int
    status: str = "draft"

    change_reason: str | None = None

    created_by_actor_type: str = "user"
    created_by_user_id: str | None = None
    created_by_display_name: str

    created_at: datetime = Field(default_factory=utc_now)

    activated_by_user_id: str | None = None
    activated_by_display_name: str | None = None
    activated_at: datetime | None = None


class ConfigurationValue(SQLModel, table=True):
    __tablename__ = "configuration_values"
    __table_args__ = (
        UniqueConstraint(
            "revision_id",
            "setting_definition_id",
            name="uq_configuration_values_revision_setting",
        ),
    )

    id: UUID = Field(default_factory=uuid4, primary_key=True)

    revision_id: UUID = Field(
        foreign_key="configuration_profile_revisions.id",
        index=True,
    )

    setting_definition_id: UUID = Field(
        foreign_key="setting_definitions.id",
        index=True,
    )

    value_text: str | None = None
    value_integer: int | None = None
    value_float: float | None = None
    value_boolean: bool | None = None

    selected_option_id: UUID | None = Field(
        default=None,
        foreign_key="setting_options.id",
    )

    encrypted_value: str | None = None
    encryption_key_id: str | None = None

    created_at: datetime = Field(default_factory=utc_now)


class ConfigurationValueOption(SQLModel, table=True):
    __tablename__ = "configuration_value_options"
    __table_args__ = (
        UniqueConstraint(
            "configuration_value_id",
            "setting_option_id",
            name="uq_configuration_value_options_selection",
        ),
    )

    id: UUID = Field(default_factory=uuid4, primary_key=True)

    configuration_value_id: UUID = Field(
        foreign_key="configuration_values.id",
        index=True,
    )

    setting_option_id: UUID = Field(
        foreign_key="setting_options.id",
        index=True,
    )


class ChunkEmbedding(SQLModel, table=True):
    __tablename__ = "chunk_embeddings"

    chunk_id: UUID = Field(
        foreign_key="document_chunks.id",
        primary_key=True,
    )

    embedding_model_id: UUID = Field(
        foreign_key="embedding_models.id",
        primary_key=True,
    )

    embedding: list[float] = Field(
        sa_column=Column(Vector(1024), nullable=False),
    )

    created_at: datetime = Field(default_factory=utc_now)


class Question(SQLModel, table=True):
    __tablename__ = "questions"

    id: UUID = Field(default_factory=uuid4, primary_key=True)

    origin: str = "user"

    question_text: str

    expected_answer: str | None = None

    document_id: UUID | None = Field(
        default=None,
        foreign_key="documents.id",
        index=True,
    )

    document_version_id: UUID | None = Field(
        default=None,
        foreign_key="document_versions.id",
        index=True,
    )

    status: str = "pending"

    resolution_reason: str | None = None

    priority: int = 0

    extra_data: dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column("metadata", JSONB, nullable=False),
    )

    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    resolved_at: datetime | None = None


class QuestionAttempt(SQLModel, table=True):
    __tablename__ = "question_attempts"
    __table_args__ = (
        UniqueConstraint(
            "question_id",
            "attempt_number",
            name="uq_question_attempts_question_number",
        ),
    )

    id: UUID = Field(default_factory=uuid4, primary_key=True)

    question_id: UUID = Field(
        foreign_key="questions.id",
        index=True,
    )

    configuration_revision_id: UUID | None = Field(
        default=None,
        foreign_key="configuration_profile_revisions.id",
        index=True,
    )

    evaluation_campaign_id: UUID | None = Field(
        default=None,
        foreign_key="evaluation_campaigns.id",
        index=True,
    )

    attempt_number: int

    strategy: str

    pipeline_version: str | None = None

    status: str = "processing"

    answer_text: str | None = None

    confidence: float | None = None

    failure_reason: str | None = None

    configuration: dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSONB, nullable=False),
    )

    metrics: dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSONB, nullable=False),
    )

    created_at: datetime = Field(default_factory=utc_now)
    completed_at: datetime | None = None


class QuestionEvidence(SQLModel, table=True):
    __tablename__ = "question_evidence"

    id: UUID = Field(default_factory=uuid4, primary_key=True)

    question_attempt_id: UUID = Field(
        foreign_key="question_attempts.id",
        index=True,
    )

    document_version_id: UUID | None = Field(
        default=None,
        foreign_key="document_versions.id",
    )

    page_id: UUID | None = Field(
        default=None,
        foreign_key="pages.id",
    )

    chunk_id: UUID | None = Field(
        default=None,
        foreign_key="document_chunks.id",
    )

    content_block_id: UUID | None = Field(
        default=None,
        foreign_key="content_blocks.id",
    )

    rank: int | None = None

    score: float | None = None

    evidence_text: str | None = None

    extra_data: dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column("metadata", JSONB, nullable=False),
    )

    created_at: datetime = Field(default_factory=utc_now)


class QuestionFeedback(SQLModel, table=True):
    __tablename__ = "question_feedback"

    id: UUID = Field(default_factory=uuid4, primary_key=True)

    question_id: UUID = Field(
        foreign_key="questions.id",
        index=True,
    )

    question_attempt_id: UUID | None = Field(
        default=None,
        foreign_key="question_attempts.id",
        index=True,
    )

    origin: str = "user"

    rating: int | None = None

    is_correct: bool | None = None

    corrected_answer: str | None = None

    comment: str | None = None

    extra_data: dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column("metadata", JSONB, nullable=False),
    )

    created_at: datetime = Field(default_factory=utc_now)


class EvaluationSuite(SQLModel, table=True):
    __tablename__ = "evaluation_suites"

    id: UUID = Field(default_factory=uuid4, primary_key=True)

    name: str
    description: str | None = None
    status: str = "active"

    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class EvaluationSuiteQuestion(SQLModel, table=True):
    __tablename__ = "evaluation_suite_questions"
    __table_args__ = (
        UniqueConstraint(
            "evaluation_suite_id",
            "question_id",
            name="uq_evaluation_suite_questions_question",
        ),
        UniqueConstraint(
            "evaluation_suite_id",
            "position",
            name="uq_evaluation_suite_questions_position",
        ),
    )

    id: UUID = Field(default_factory=uuid4, primary_key=True)

    evaluation_suite_id: UUID = Field(
        foreign_key="evaluation_suites.id",
        index=True,
    )
    question_id: UUID = Field(
        foreign_key="questions.id",
        index=True,
    )

    position: int
    created_at: datetime = Field(default_factory=utc_now)


class EvaluationCampaign(SQLModel, table=True):
    __tablename__ = "evaluation_campaigns"

    id: UUID = Field(default_factory=uuid4, primary_key=True)

    suite_id: UUID | None = Field(
        default=None,
        foreign_key="evaluation_suites.id",
    )

    name: str
    status: str = "pending"

    configuration: dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSONB, nullable=False),
    )

    created_at: datetime = Field(default_factory=utc_now)
    started_at: datetime | None = None
    completed_at: datetime | None = None
