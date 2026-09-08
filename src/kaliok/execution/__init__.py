from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal
from uuid import UUID, uuid4

from sqlmodel import Session

from kaliok.hashing import canonical_json_hash
from kaliok.storage.models import ConfigurationProfileRevision, PipelineRevision, ProcessingRun


ExecutionEnvironment = Literal["production", "experiment"]
_VALID_ENVIRONMENTS = {"production", "experiment"}


@dataclass(frozen=True)
class ExecutionContext:
    """Immutable identity shared by the runs of one logical execution."""

    environment: ExecutionEnvironment
    configuration_revision_id: UUID | None = None
    pipeline_revision_id: UUID | None = None
    execution_group_id: UUID = field(default_factory=uuid4)

    def __post_init__(self) -> None:
        if self.environment not in _VALID_ENVIRONMENTS:
            raise ValueError(
                "execution_environment doit être 'production' ou 'experiment'."
            )


def apply_execution_context(
    session: Session,
    run: ProcessingRun,
    context: ExecutionContext | None,
) -> None:
    """Apply explicit execution identity and hash the run snapshot."""
    if context is None:
        return
    if context.configuration_revision_id is not None:
        revision = session.get(
            ConfigurationProfileRevision,
            context.configuration_revision_id,
        )
        if revision is None:
            raise ValueError(
                "Révision de configuration introuvable : "
                f"{context.configuration_revision_id}."
            )
    if context.pipeline_revision_id is not None:
        if session.get(PipelineRevision, context.pipeline_revision_id) is None:
            raise ValueError(
                "Révision de pipeline introuvable : "
                f"{context.pipeline_revision_id}."
            )
    run.execution_environment = context.environment
    run.configuration_revision_id = context.configuration_revision_id
    run.pipeline_revision_id = context.pipeline_revision_id
    run.execution_group_id = context.execution_group_id
    run.configuration_hash = canonical_json_hash(run.configuration)


__all__ = [
    "ExecutionContext",
    "ExecutionEnvironment",
    "apply_execution_context",
]
