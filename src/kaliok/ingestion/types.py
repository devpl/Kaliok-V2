from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID


Identifier = UUID | str


@dataclass(frozen=True)
class SourceReference:
    name: str
    uri: str | None = None
    media_type: str | None = None
    size: int | None = None
    external_id: str | None = None
    parent: SourceReference | None = None


@dataclass(frozen=True)
class IngestionRequest:
    source: SourceReference
    request_id: Identifier | None = None
    source_id: Identifier | None = None
    document_id: Identifier | None = None


@dataclass(frozen=True)
class DetectedSource:
    source: SourceReference
    source_type: str
    media_type: str | None = None
    confidence: float | None = None


@dataclass(frozen=True)
class NormalizedDocument:
    source: DetectedSource
    content: object
    filename: str
    storage_uri: str
    content_hash: str
    file_size: int | None = None
    mime_type: str | None = None
    title: str | None = None
    language: str | None = None
    page_count: int | None = None
    document_family: str | None = None
    document_type: str | None = None
    document_subtype: str | None = None


@dataclass(frozen=True)
class IngestionResult:
    document_id: Identifier
    document_version_id: Identifier
    detected_source: DetectedSource
    status: str
    processing_run_id: Identifier | None = None
