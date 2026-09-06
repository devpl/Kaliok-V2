from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import func
from sqlmodel import Session, select

from kaliok.storage.models import NormalizedContentUnit, ProcessingRun


PROCESS_TYPE = "content_normalization"


class ContentNormalizationComparisonService:
    """Compare two completed normalization generations without loading units."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def compare(self, run_p_id: UUID, run_a_id: UUID) -> dict[str, Any]:
        run_p = self._validated_run(run_p_id)
        run_a = self._validated_run(run_a_id)
        if run_p.document_version_id != run_a.document_version_id:
            raise ValueError(
                "Les runs de normalisation doivent appartenir au même document."
            )

        metrics_p = self._metrics(run_p)
        metrics_a = self._metrics(run_a)
        return {
            "document_version_id": run_p.document_version_id,
            "run_p": self._run_payload(run_p, metrics_p),
            "run_a": self._run_payload(run_a, metrics_a),
            "delta": self._delta(metrics_p, metrics_a),
        }

    def _validated_run(self, run_id: UUID) -> ProcessingRun:
        run = self._session.get(ProcessingRun, run_id)
        if run is None:
            raise ValueError(f"Run de normalisation inconnu : {run_id}.")
        if run.process_type != PROCESS_TYPE:
            raise ValueError("Le run fourni n'est pas un run content_normalization.")
        if run.status != "completed":
            raise ValueError("La comparaison exige des runs completed.")
        return run

    def _metrics(self, run: ProcessingRun) -> dict[str, Any]:
        unit_count = self._session.exec(
            select(func.count(NormalizedContentUnit.id)).where(
                NormalizedContentUnit.processing_run_id == run.id
            )
        ).one()
        total_characters = self._session.exec(
            select(
                func.coalesce(
                    func.sum(func.length(NormalizedContentUnit.content)),
                    0,
                )
            ).where(NormalizedContentUnit.processing_run_id == run.id)
        ).one()
        type_rows = self._session.exec(
            select(
                NormalizedContentUnit.content_type,
                func.count(NormalizedContentUnit.id),
            )
            .where(NormalizedContentUnit.processing_run_id == run.id)
            .group_by(NormalizedContentUnit.content_type)
        ).all()
        content_type_counts = {
            content_type: int(count) for content_type, count in type_rows
        }
        metrics: dict[str, Any] = {
            "unit_count": int(unit_count),
            "total_characters": int(total_characters or 0),
            "content_type_counts": content_type_counts,
        }
        for key in ("source_block_count", "skipped_empty_block_count"):
            if key in run.metrics:
                metrics[key] = run.metrics[key]
        duration = _duration_seconds(run.started_at, run.completed_at)
        if duration is not None:
            metrics["duration_seconds"] = duration
        return metrics

    @staticmethod
    def _run_payload(run: ProcessingRun, metrics: dict[str, Any]) -> dict[str, Any]:
        return {
            "run_id": run.id,
            "document_version_id": run.document_version_id,
            "status": run.status,
            "execution_environment": run.execution_environment,
            "execution_group_id": run.execution_group_id,
            "configuration_revision_id": run.configuration_revision_id,
            "configuration": run.configuration,
            "configuration_hash": run.configuration_hash,
            "metrics": metrics,
        }

    @staticmethod
    def _delta(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
        delta: dict[str, Any] = {}
        for key, value in second.items():
            first_value = first.get(key)
            if isinstance(value, (int, float)) and isinstance(
                first_value, (int, float)
            ):
                delta[key] = value - first_value
            elif key == "content_type_counts":
                keys = set(first_value or {}) | set(value or {})
                delta[key] = {
                    content_type: int((value or {}).get(content_type, 0))
                    - int((first_value or {}).get(content_type, 0))
                    for content_type in sorted(keys)
                }
        return delta


def _duration_seconds(
    started_at: datetime | None,
    completed_at: datetime | None,
) -> float | None:
    if started_at is None or completed_at is None:
        return None
    return (completed_at - started_at).total_seconds()


__all__ = ["ContentNormalizationComparisonService"]
