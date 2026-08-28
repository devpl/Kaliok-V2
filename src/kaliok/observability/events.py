from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping
from uuid import UUID


Identifier = UUID | str


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class ObservabilityEvent:
    event_name: str
    timestamp: datetime = field(default_factory=utc_now)
    correlation_id: Identifier | None = None
    execution_id: Identifier | None = None
    component: str | None = None
    implementation: str | None = None
    operation: str | None = None
    document_id: Identifier | None = None
    document_version_id: Identifier | None = None
    processing_run_id: Identifier | None = None
    model: str | None = None
    duration_ms: float | None = None
    input_count: int | None = None
    output_count: int | None = None
    top_k: int | None = None
    success: bool | None = None
    error_type: str | None = None
    error_message: str | None = None
    extra_data: Mapping[str, Any] = field(default_factory=dict)
