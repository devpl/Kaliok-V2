from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from uuid import UUID

from sqlmodel import Session, select

from kaliok.execution import ExecutionContext, apply_execution_context
from kaliok.discovery.detectors import CandidateDetector, CandidateOccurrence
from kaliok.storage.models import (
    CandidateSourceFragment,
    DiscoveredCandidate,
    DocumentVersion,
    NormalizedContentUnit,
    ProcessingRun,
    utc_now,
)


@dataclass(frozen=True)
class CandidateDiscoveryResult:
    processing_run_id: UUID
    document_version_id: UUID
    normalization_run_id: UUID
    candidate_count: int


class CandidateDiscoveryService:
    def __init__(self, session: Session) -> None:
        self._session = session

    def discover(
        self,
        document_version_id: UUID,
        normalization_run_id: UUID,
        detectors: list[CandidateDetector] | tuple[CandidateDetector, ...],
        *,
        execution_context: ExecutionContext | None = None,
    ) -> CandidateDiscoveryResult:
        version = self._session.get(DocumentVersion, document_version_id)
        if version is None:
            raise ValueError(f"DocumentVersion inconnue : {document_version_id}.")
        normalization_run = self._session.get(ProcessingRun, normalization_run_id)
        self._validate_normalization_run(
            normalization_run,
            document_version_id=document_version_id,
            normalization_run_id=normalization_run_id,
        )
        if not detectors:
            raise ValueError("Au moins un détecteur est requis.")

        run = ProcessingRun(
            document_version_id=document_version_id,
            process_type="candidate_discovery",
            status="running",
            engine="kaliok",
            engine_version="candidate-discovery-v1",
            configuration={
                "normalization_run_id": str(normalization_run_id),
                "detectors": [
                    {
                        "key": detector.key,
                        "version": detector.version,
                        "configuration": detector.configuration,
                        "dictionary": getattr(detector, "dictionary_identity", None),
                    }
                    for detector in detectors
                ],
                "dictionary": next((getattr(detector, "dictionary_identity", None) for detector in detectors if getattr(detector, "dictionary_identity", None)), None),
            },
        )
        apply_execution_context(self._session, run, execution_context)
        self._session.add(run)
        self._session.flush()
        metrics: dict[str, object] = {
            "source_unit_count": 0,
            "candidate_count": 0,
            "candidate_type_counts": {},
            "skipped_unit_count": 0,
        }

        try:
            units = list(
                self._session.exec(
                    select(NormalizedContentUnit)
                    .where(
                        NormalizedContentUnit.document_version_id
                        == document_version_id,
                        NormalizedContentUnit.processing_run_id
                        == normalization_run_id,
                    )
                    .order_by(NormalizedContentUnit.unit_index)
                ).all()
            )
            candidate_type_counts: Counter[str] = Counter()
            skipped_unit_count = 0
            candidate_count = 0
            seen: set[tuple[object, ...]] = set()
            with self._session.begin_nested():
                for unit in units:
                    unit_had_occurrence = False
                    for detector in detectors:
                        for occurrence in detector.detect(unit):
                            key = self._technical_key(detector, unit, occurrence)
                            if key in seen:
                                continue
                            seen.add(key)
                            unit_had_occurrence = True
                            self._persist_occurrence(
                                run=run,
                                unit=unit,
                                detector=detector,
                                occurrence=occurrence,
                            )
                            candidate_count += 1
                            candidate_type_counts[occurrence.candidate_type] += 1
                    if not unit_had_occurrence:
                        skipped_unit_count += 1
                self._session.flush()
            metrics = {
                "source_unit_count": len(units),
                "candidate_count": candidate_count,
                "candidate_type_counts": dict(candidate_type_counts),
                "skipped_unit_count": skipped_unit_count,
            }
        except Exception as error:
            run.status = "failed"
            run.completed_at = utc_now()
            run.metrics = metrics
            run.error_message = str(error)
            self._session.add(run)
            self._session.flush()
            raise

        run.status = "completed"
        run.completed_at = utc_now()
        run.metrics = metrics
        self._session.add(run)
        self._session.flush()
        return CandidateDiscoveryResult(
            processing_run_id=run.id,
            document_version_id=document_version_id,
            normalization_run_id=normalization_run_id,
            candidate_count=int(metrics["candidate_count"]),
        )

    @staticmethod
    def _validate_normalization_run(
        run: ProcessingRun | None,
        *,
        document_version_id: UUID,
        normalization_run_id: UUID,
    ) -> None:
        if run is None:
            raise ValueError(f"ProcessingRun de normalisation inconnu : {normalization_run_id}.")
        if run.document_version_id != document_version_id:
            raise ValueError("Le run de normalisation appartient à un autre document.")
        if run.process_type != "content_normalization":
            raise ValueError("Le run fourni n'est pas un run content_normalization.")
        if run.status != "completed":
            raise ValueError("Le run de normalisation doit être completed.")

    @staticmethod
    def _technical_key(detector, unit, occurrence) -> tuple[object, ...]:
        return (
            detector.key,
            detector.version,
            unit.id,
            occurrence.start_offset,
            occurrence.end_offset,
            occurrence.candidate_type,
            occurrence.raw_value,
        )

    def _persist_occurrence(self, *, run, unit, detector, occurrence) -> None:
        candidate = DiscoveredCandidate(
            document_version_id=run.document_version_id,
            processing_run_id=run.id,
            candidate_type=occurrence.candidate_type,
            raw_value=occurrence.raw_value,
            normalized_value=occurrence.normalized_value,
            payload=dict(occurrence.payload),
            confidence=occurrence.confidence,
            detector_key=detector.key,
            detector_version=detector.version,
        )
        self._session.add(candidate)
        self._session.flush()
        self._session.add(
            CandidateSourceFragment(
                candidate_id=candidate.id,
                normalized_content_unit_id=unit.id,
                fragment_order=0,
                start_offset=occurrence.start_offset,
                end_offset=occurrence.end_offset,
                exact_text=occurrence.exact_text,
            )
        )
