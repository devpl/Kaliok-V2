from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import Column, String
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


class Page(SQLModel, table=True):
    __tablename__ = "pages"

    id: UUID = Field(default_factory=uuid4, primary_key=True)

    document_version_id: UUID = Field(
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

    id: UUID = Field(default_factory=uuid4, primary_key=True)

    document_version_id: UUID = Field(
        foreign_key="document_versions.id",
    )

    process_type: str
    status: str

    engine: str | None = None
    engine_version: str | None = None

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

    document_id: UUID | None = Field(
        default=None,
        foreign_key="documents.id",
    )

    document_version_id: UUID | None = Field(
        default=None,
        foreign_key="document_versions.id",
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

    id: UUID = Field(default_factory=uuid4, primary_key=True)

    question_id: UUID = Field(
        foreign_key="questions.id",
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
    )

    question_attempt_id: UUID | None = Field(
        default=None,
        foreign_key="question_attempts.id",
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
