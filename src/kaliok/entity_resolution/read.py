from __future__ import annotations

from collections import defaultdict
from typing import Any
from uuid import UUID

from sqlalchemy import func
from sqlmodel import Session, select

from kaliok.storage.models import (
    CandidateSourceFragment,
    ContentBlock,
    ContentBlockFragment,
    DiscoveredCandidate,
    DocumentVersion,
    Entity,
    EntityMembership,
    EntityResolutionEvidence,
    EntityResolutionScopeItem,
    NormalizedContentUnit,
    NormalizedContentUnitSource,
    Page,
    ProcessingRun,
)


class EntityResolutionReadService:
    """Read-only projection of entity-resolution generations."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def list_runs(
        self,
        *,
        status: str | None = None,
        document_version_id: UUID | None = None,
        limit: int = 25,
        offset: int = 0,
    ) -> dict[str, Any]:
        statement = select(ProcessingRun).where(
            ProcessingRun.process_type == "entity_resolution"
        )
        if status:
            statement = statement.where(ProcessingRun.status == status)
        if document_version_id:
            scoped_runs = (
                select(EntityResolutionScopeItem.processing_run_id)
                .join(
                    DiscoveredCandidate,
                    DiscoveredCandidate.id
                    == EntityResolutionScopeItem.discovered_candidate_id,
                )
                .where(DiscoveredCandidate.document_version_id == document_version_id)
                .distinct()
            )
            statement = statement.where(ProcessingRun.id.in_(scoped_runs))
        total = self._session.exec(
            select(func.count()).select_from(statement.subquery())
        ).one()
        runs = list(
            self._session.exec(
                statement.order_by(
                    ProcessingRun.completed_at.desc().nullslast(),
                    ProcessingRun.started_at.desc(),
                    ProcessingRun.id.desc(),
                )
                .offset(offset)
                .limit(limit)
            ).all()
        )
        counts = self._counts_by_run([run.id for run in runs])
        return {
            "items": [self._run_payload(run, counts.get(run.id, {})) for run in runs],
            "limit": limit,
            "offset": offset,
            "total": total,
        }

    def get_run(self, run_id: UUID) -> dict[str, Any] | None:
        run = self._resolution_run(run_id)
        if run is None:
            return None
        counts = self._counts_by_run([run.id]).get(run.id, {})
        run_payload = self._run_payload(run, counts)
        documents = self._documents_for_run(run.id)
        entity_types = list(
            self._session.exec(
                select(Entity.entity_type)
                .where(Entity.processing_run_id == run.id)
                .distinct()
                .order_by(Entity.entity_type)
            ).all()
        )
        return {
            "run": run_payload,
            "metrics": run_payload["metrics"],
            "documents": documents,
            "entity_types": entity_types,
            "summary": {key: run_payload[key] for key in self._count_keys()},
        }

    def list_entities(
        self,
        run_id: UUID,
        *,
        entity_type: str | None = None,
        canonical_label: str | None = None,
        status: str | None = None,
        grouped_only: bool = False,
        singleton_only: bool = False,
        limit: int = 25,
        offset: int = 0,
    ) -> dict[str, Any] | None:
        if self._resolution_run(run_id) is None:
            return None
        membership_counts = (
            select(
                EntityMembership.entity_id.label("entity_id"),
                func.count(func.distinct(EntityMembership.id)).label("membership_count"),
                func.count(func.distinct(DiscoveredCandidate.document_version_id)).label("document_count"),
                func.count(func.distinct(Page.id)).label("page_count"),
                func.count(func.distinct(EntityResolutionEvidence.id)).label("evidence_count"),
            )
            .join(DiscoveredCandidate, DiscoveredCandidate.id == EntityMembership.discovered_candidate_id)
            .outerjoin(CandidateSourceFragment, CandidateSourceFragment.candidate_id == DiscoveredCandidate.id)
            .outerjoin(NormalizedContentUnit, NormalizedContentUnit.id == CandidateSourceFragment.normalized_content_unit_id)
            .outerjoin(NormalizedContentUnitSource, NormalizedContentUnitSource.normalized_content_unit_id == NormalizedContentUnit.id)
            .outerjoin(ContentBlock, ContentBlock.id == NormalizedContentUnitSource.content_block_id)
            .outerjoin(ContentBlockFragment, ContentBlockFragment.content_block_id == ContentBlock.id)
            .outerjoin(Page, Page.id == ContentBlockFragment.page_id)
            .outerjoin(EntityResolutionEvidence, EntityResolutionEvidence.membership_id == EntityMembership.id)
            .group_by(EntityMembership.entity_id)
            .subquery()
        )
        statement = (
            select(Entity, membership_counts)
            .outerjoin(membership_counts, membership_counts.c.entity_id == Entity.id)
            .where(Entity.processing_run_id == run_id)
        )
        if entity_type:
            statement = statement.where(Entity.entity_type == entity_type)
        if canonical_label:
            statement = statement.where(Entity.canonical_label.ilike(f"%{canonical_label}%"))
        if status:
            statement = statement.where(Entity.status == status)
        if grouped_only:
            statement = statement.where(membership_counts.c.membership_count > 1)
        if singleton_only:
            statement = statement.where(membership_counts.c.membership_count == 1)
        total = self._session.exec(
            select(func.count()).select_from(statement.subquery())
        ).one()
        rows = self._session.exec(
            statement.order_by(Entity.entity_index, Entity.id).offset(offset).limit(limit)
        ).all()
        items = []
        for row in rows:
            entity = row[0]
            items.append({
                "id": entity.id,
                "entity_index": entity.entity_index,
                "entity_type": entity.entity_type,
                "canonical_label": entity.canonical_label,
                "status": entity.status,
                "confidence": entity.confidence,
                "membership_count": row.membership_count or 0,
                "document_count": row.document_count or 0,
                "page_count": row.page_count or 0,
                "evidence_count": row.evidence_count or 0,
            })
        return {"items": items, "limit": limit, "offset": offset, "total": total}

    def get_entity(self, entity_id: UUID) -> dict[str, Any] | None:
        entity = self._session.get(Entity, entity_id)
        if entity is None:
            return None
        run = self._resolution_run(entity.processing_run_id)
        if run is None:
            return None
        memberships = list(self._session.exec(
            select(EntityMembership, DiscoveredCandidate, DocumentVersion)
            .join(DiscoveredCandidate, DiscoveredCandidate.id == EntityMembership.discovered_candidate_id)
            .join(DocumentVersion, DocumentVersion.id == DiscoveredCandidate.document_version_id)
            .where(EntityMembership.entity_id == entity.id)
            .order_by(EntityMembership.created_at, EntityMembership.id)
        ).all())
        membership_ids = [row[0].id for row in memberships]
        evidence_rows = list(self._session.exec(
            select(EntityResolutionEvidence)
            .where(EntityResolutionEvidence.membership_id.in_(membership_ids))
            .order_by(EntityResolutionEvidence.membership_id, EntityResolutionEvidence.evidence_order)
        ).all()) if membership_ids else []
        evidence_by_membership: dict[UUID, list[dict[str, Any]]] = defaultdict(list)
        for evidence in evidence_rows:
            evidence_by_membership[evidence.membership_id].append({
                "id": evidence.id,
                "evidence_order": evidence.evidence_order,
                "signal_key": evidence.signal_key,
                "method": evidence.method,
                "score": evidence.score,
                "explanation": evidence.explanation,
                "metadata": evidence.extra_data,
            })
        candidate_ids = [row[1].id for row in memberships]
        source_fragments = list(self._session.exec(
            select(CandidateSourceFragment)
            .where(CandidateSourceFragment.candidate_id.in_(candidate_ids))
            .order_by(CandidateSourceFragment.candidate_id, CandidateSourceFragment.fragment_order)
        ).all()) if candidate_ids else []
        provenance = self._provenance(source_fragments)
        membership_payloads = []
        for membership, candidate, version in memberships:
            discovery_run = self._session.get(ProcessingRun, candidate.processing_run_id)
            normalization_run_id = (discovery_run.configuration or {}).get("normalization_run_id") if discovery_run else None
            membership_payloads.append({
                "membership_id": membership.id,
                "membership_status": membership.membership_status,
                "decision_origin": membership.decision_origin,
                "confidence": membership.confidence,
                "metadata": membership.extra_data,
                "candidate": {
                    "candidate_id": candidate.id,
                    "candidate_type": candidate.candidate_type,
                    "raw_value": candidate.raw_value,
                    "normalized_value": candidate.normalized_value,
                    "detector_key": candidate.detector_key,
                    "detector_version": candidate.detector_version,
                    "confidence": candidate.confidence,
                    "candidate_discovery_run_id": candidate.processing_run_id,
                    "normalization_run_id": normalization_run_id,
                },
                "document": {
                    "document_version_id": version.id,
                    "filename": version.filename,
                },
                "evidences": evidence_by_membership.get(membership.id, []),
                "provenance": provenance.get(candidate.id, []),
            })
        return {
            "entity": {
                "id": entity.id,
                "entity_index": entity.entity_index,
                "entity_type": entity.entity_type,
                "canonical_label": entity.canonical_label,
                "status": entity.status,
                "confidence": entity.confidence,
                "metadata": entity.extra_data,
            },
            "run": self._run_payload(run, self._counts_by_run([run.id]).get(run.id, {})),
            "memberships": membership_payloads,
            "evidences": [item for member in membership_payloads for item in member["evidences"]],
            "documents": self._documents_for_run(run.id),
            "provenance": [item for member in membership_payloads for item in member["provenance"]],
        }

    def _provenance(self, source_fragments):
        unit_ids = {item.normalized_content_unit_id for item in source_fragments}
        units = {item.id: item for item in self._session.exec(
            select(NormalizedContentUnit).where(NormalizedContentUnit.id.in_(unit_ids))
        ).all()} if unit_ids else {}
        unit_sources = list(self._session.exec(
            select(NormalizedContentUnitSource)
            .where(NormalizedContentUnitSource.normalized_content_unit_id.in_(unit_ids))
            .order_by(NormalizedContentUnitSource.normalized_content_unit_id, NormalizedContentUnitSource.source_order)
        ).all()) if unit_ids else []
        block_ids = {item.content_block_id for item in unit_sources}
        blocks = {item.id: item for item in self._session.exec(
            select(ContentBlock).where(ContentBlock.id.in_(block_ids))
        ).all()} if block_ids else {}
        block_fragments = list(self._session.exec(
            select(ContentBlockFragment, Page, DocumentVersion)
            .join(Page, Page.id == ContentBlockFragment.page_id)
            .join(DocumentVersion, DocumentVersion.id == Page.document_version_id)
            .where(ContentBlockFragment.content_block_id.in_(block_ids))
            .order_by(ContentBlockFragment.content_block_id, ContentBlockFragment.fragment_index)
        ).all()) if block_ids else []
        fragments_by_block = defaultdict(list)
        for fragment, page, version in block_fragments:
            fragments_by_block[fragment.content_block_id].append({
                "fragment_id": fragment.id,
                "fragment_index": fragment.fragment_index,
                "reading_order": fragment.reading_order,
                "content": fragment.content,
                "page_id": page.id,
                "page_number": page.page_number,
                "document_version_id": version.id,
                "filename": version.filename,
                "bbox": self._bbox(fragment),
                "coordinate_system": fragment.coordinate_system,
            })
        sources_by_unit = defaultdict(list)
        for source in unit_sources:
            block = blocks.get(source.content_block_id)
            if block is None:
                continue
            sources_by_unit[source.normalized_content_unit_id].append({
                "source_order": source.source_order,
                "content_block": {
                    "id": block.id,
                    "block_index": block.block_index,
                    "reading_order": block.reading_order,
                    "content": block.content,
                    "bbox": block.bbox or self._bbox(block),
                    "coordinate_system": block.coordinate_system,
                    "fragments": fragments_by_block.get(block.id, []),
                },
            })
        result = defaultdict(list)
        for source in source_fragments:
            unit = units.get(source.normalized_content_unit_id)
            if unit is None:
                continue
            result[source.candidate_id].append({
                "source_fragment_id": source.id,
                "fragment_order": source.fragment_order,
                "exact_text": source.exact_text,
                "start_offset": source.start_offset,
                "end_offset": source.end_offset,
                "role": source.role,
                "normalized_content_unit": {
                    "id": unit.id,
                    "unit_index": unit.unit_index,
                    "content_type": unit.content_type,
                    "content": unit.content,
                    "processing_run_id": unit.processing_run_id,
                    "sources": sources_by_unit.get(unit.id, []),
                },
            })
        return result

    def _documents_for_run(self, run_id: UUID) -> list[dict[str, Any]]:
        rows = self._session.exec(
            select(
                DocumentVersion.id,
                DocumentVersion.filename,
                func.count(func.distinct(EntityResolutionScopeItem.discovered_candidate_id)),
            )
            .join(DiscoveredCandidate, DiscoveredCandidate.document_version_id == DocumentVersion.id)
            .join(EntityResolutionScopeItem, EntityResolutionScopeItem.discovered_candidate_id == DiscoveredCandidate.id)
            .where(EntityResolutionScopeItem.processing_run_id == run_id)
            .group_by(DocumentVersion.id, DocumentVersion.filename)
            .order_by(DocumentVersion.filename, DocumentVersion.id)
        ).all()
        return [{"document_version_id": row[0], "filename": row[1], "occurrence_count": row[2]} for row in rows]

    def _counts_by_run(self, run_ids: list[UUID]) -> dict[UUID, dict[str, int]]:
        if not run_ids:
            return {}
        entity_rows = self._session.exec(
            select(Entity.processing_run_id, func.count(Entity.id))
            .where(Entity.processing_run_id.in_(run_ids))
            .group_by(Entity.processing_run_id)
        ).all()
        membership_rows = self._session.exec(
            select(EntityMembership.processing_run_id, func.count(EntityMembership.id))
            .where(EntityMembership.processing_run_id.in_(run_ids))
            .group_by(EntityMembership.processing_run_id)
        ).all()
        grouped_rows = self._session.exec(
            select(EntityMembership.processing_run_id, EntityMembership.entity_id, func.count(EntityMembership.id))
            .where(EntityMembership.processing_run_id.in_(run_ids))
            .group_by(EntityMembership.processing_run_id, EntityMembership.entity_id)
        ).all()
        evidence_rows = self._session.exec(
            select(EntityMembership.processing_run_id, func.count(EntityResolutionEvidence.id))
            .join(EntityMembership, EntityMembership.id == EntityResolutionEvidence.membership_id)
            .where(EntityMembership.processing_run_id.in_(run_ids))
            .group_by(EntityMembership.processing_run_id)
        ).all()
        document_rows = self._session.exec(
            select(EntityResolutionScopeItem.processing_run_id, func.count(func.distinct(DiscoveredCandidate.document_version_id)))
            .join(DiscoveredCandidate, DiscoveredCandidate.id == EntityResolutionScopeItem.discovered_candidate_id)
            .where(EntityResolutionScopeItem.processing_run_id.in_(run_ids))
            .group_by(EntityResolutionScopeItem.processing_run_id)
        ).all()
        counts = {run_id: {key: 0 for key in self._count_keys()} for run_id in run_ids}
        for run_id, value in entity_rows: counts[run_id]["entity_count"] = value
        for run_id, value in membership_rows: counts[run_id]["membership_count"] = value
        for run_id, value in evidence_rows: counts[run_id]["evidence_count"] = value
        for run_id, value in document_rows: counts[run_id]["document_version_count"] = value
        for run_id, _, value in grouped_rows:
            key = "grouped_entity_count" if value > 1 else "singleton_entity_count"
            counts[run_id][key] += 1
        return counts

    @staticmethod
    def _count_keys():
        return ("entity_count", "membership_count", "evidence_count", "singleton_entity_count", "grouped_entity_count", "document_version_count")

    def _resolution_run(self, run_id: UUID) -> ProcessingRun | None:
        run = self._session.get(ProcessingRun, run_id)
        return run if run is not None and run.process_type == "entity_resolution" else None

    @classmethod
    def _run_payload(cls, run: ProcessingRun, counts: dict[str, int]) -> dict[str, Any]:
        metrics = dict(run.metrics or {})
        for key in cls._count_keys():
            metrics[key] = counts.get(key, metrics.get(key, 0))
        payload = {
            "id": run.id,
            "run_id": run.id,
            "status": run.status,
            "engine": run.engine,
            "engine_version": run.engine_version,
            "document_version_id": run.document_version_id,
            "started_at": run.started_at,
            "completed_at": run.completed_at,
            "error_message": run.error_message,
            "configuration": run.configuration or {},
            "metrics": metrics,
        }
        payload.update({key: metrics[key] for key in cls._count_keys()})
        return payload

    @staticmethod
    def _bbox(item) -> dict[str, float] | None:
        values = (item.bbox_x, item.bbox_y, item.bbox_width, item.bbox_height)
        if all(value is None for value in values):
            return None
        return dict(zip(("x", "y", "width", "height"), values))
