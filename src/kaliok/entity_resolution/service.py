from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from sqlmodel import Session, select

from kaliok.execution import ExecutionContext, apply_execution_context
from kaliok.storage.models import (
    DiscoveredCandidate,
    Entity,
    EntityMembership,
    EntityResolutionEvidence,
    EntityResolutionScopeItem,
    ProcessingRun,
    utc_now,
)


STRATEGY = "declared-normalized-exact-v1"


@dataclass(frozen=True)
class EntityResolutionResult:
    processing_run_id: UUID
    candidate_count: int
    entity_count: int
    membership_count: int


class EntityResolutionService:
    def __init__(self, session: Session) -> None:
        self._session = session

    def resolve(
        self,
        discovered_candidate_ids: list[UUID] | tuple[UUID, ...],
        *,
        execution_context: ExecutionContext | None = None,
    ) -> EntityResolutionResult:
        ids = list(discovered_candidate_ids)
        if not ids:
            raise ValueError("Le scope Entity Resolution ne peut pas être vide.")
        if len(set(ids)) != len(ids):
            raise ValueError("Le scope Entity Resolution contient des candidats dupliqués.")
        candidates = list(self._session.exec(select(DiscoveredCandidate).where(DiscoveredCandidate.id.in_(ids))).all())
        by_id = {candidate.id: candidate for candidate in candidates}
        missing = [str(candidate_id) for candidate_id in ids if candidate_id not in by_id]
        if missing:
            raise ValueError(f"DiscoveredCandidate introuvable : {', '.join(missing)}.")
        ordered = [by_id[candidate_id] for candidate_id in ids]
        document_versions = {item.document_version_id for item in ordered}
        run = ProcessingRun(
            document_version_id=next(iter(document_versions)) if len(document_versions) == 1 else None,
            process_type="entity_resolution",
            status="running",
            engine="kaliok",
            engine_version=STRATEGY,
            configuration={"strategy": STRATEGY, "candidate_count": len(ids), "document_version_count": len(document_versions)},
        )
        apply_execution_context(self._session, run, execution_context)
        self._session.add(run)
        self._session.flush()
        metrics = {"candidate_count": len(ids), "entity_count": 0, "membership_count": 0, "evidence_count": 0, "singleton_entity_count": 0, "grouped_entity_count": 0, "document_version_count": len({item.document_version_id for item in ordered})}
        try:
            with self._session.begin_nested():
                for order, candidate in enumerate(ordered):
                    self._session.add(EntityResolutionScopeItem(processing_run_id=run.id, discovered_candidate_id=candidate.id, scope_order=order))
                groups: dict[tuple[object, ...], list[DiscoveredCandidate]] = {}
                for candidate in ordered:
                    key = (candidate.candidate_type, candidate.normalized_value) if candidate.normalized_value is not None else ("__singleton__", candidate.id)
                    groups.setdefault(key, []).append(candidate)
                entity_count = membership_count = evidence_count = singleton_count = 0
                for entity_index, members in enumerate(groups.values()):
                    first = members[0]
                    label = first.normalized_value if first.normalized_value is not None else first.raw_value
                    is_grouped = len(members) > 1 and first.normalized_value is not None
                    entity_confidence = 1.0 if is_grouped else None
                    entity = Entity(processing_run_id=run.id, entity_index=entity_index, entity_type=first.candidate_type, canonical_label=label, confidence=entity_confidence)
                    self._session.add(entity)
                    self._session.flush()
                    entity_count += 1
                    if len(members) == 1:
                        singleton_count += 1
                    for candidate in members:
                        membership = EntityMembership(entity_id=entity.id, discovered_candidate_id=candidate.id, processing_run_id=run.id, decision_origin="declared_normalized_exact", confidence=entity_confidence)
                        self._session.add(membership)
                        self._session.flush()
                        if is_grouped:
                            signal_key = "normalized_value_exact"
                            score = 1.0
                            explanation = "Plusieurs occurrences partagent le même candidate_type et le même normalized_value déclaré."
                        elif candidate.normalized_value is not None:
                            signal_key = "singleton_normalized_value"
                            score = None
                            explanation = "Occurrence unique ; normalized_value disponible mais aucun rapprochement avec une autre occurrence n'a été effectué."
                        else:
                            signal_key = "singleton_no_normalized_value"
                            score = None
                            explanation = "Aucune normalized_value déclarée ; occurrence conservée séparément."
                        self._session.add(EntityResolutionEvidence(membership_id=membership.id, evidence_order=0, signal_key=signal_key, method=STRATEGY, score=score, explanation=explanation))
                        membership_count += 1
                        evidence_count += 1
                metrics.update(entity_count=entity_count, membership_count=membership_count, evidence_count=evidence_count, singleton_entity_count=singleton_count, grouped_entity_count=entity_count-singleton_count)
                self._session.flush()
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
        return EntityResolutionResult(run.id, len(ids), metrics["entity_count"], metrics["membership_count"])
