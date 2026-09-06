from __future__ import annotations

from collections import defaultdict
from typing import Any
from uuid import UUID

from sqlalchemy import func, or_
from sqlmodel import Session, select

from kaliok.storage.models import (
    CandidateSourceFragment,
    ContentBlock,
    ContentBlockFragment,
    DiscoveredCandidate,
    DocumentVersion,
    NormalizedContentUnit,
    NormalizedContentUnitSource,
    Page,
    ProcessingRun,
)


class CandidateDiscoveryReadService:
    def __init__(self, session: Session) -> None:
        self._session = session

    def list_runs(self, *, limit: int = 50, offset: int = 0) -> dict[str, Any]:
        total = self._session.exec(
            select(func.count(ProcessingRun.id)).where(
                ProcessingRun.process_type == "candidate_discovery"
            )
        ).one()
        rows = self._session.exec(
            select(ProcessingRun, DocumentVersion)
            .join(DocumentVersion, ProcessingRun.document_version_id == DocumentVersion.id)
            .where(ProcessingRun.process_type == "candidate_discovery")
            .order_by(ProcessingRun.started_at.desc(), ProcessingRun.id.desc())
            .offset(offset)
            .limit(limit)
        ).all()
        return {
            "items": [self._run_payload(run, version) for run, version in rows],
            "limit": limit,
            "offset": offset,
            "total": total,
        }

    def get_run(self, run_id: UUID) -> dict[str, Any] | None:
        row = self._session.exec(
            select(ProcessingRun, DocumentVersion)
            .join(DocumentVersion, ProcessingRun.document_version_id == DocumentVersion.id)
            .where(
                ProcessingRun.id == run_id,
                ProcessingRun.process_type == "candidate_discovery",
            )
        ).first()
        if row is None:
            return None
        run, version = row
        payload = self._run_payload(run, version)
        normalization_run_id = self._normalization_run_id(run)
        normalization_run = (
            self._session.get(ProcessingRun, normalization_run_id)
            if normalization_run_id is not None
            else None
        )
        payload.update(
            {
                "document": {
                    "document_version_id": version.id,
                    "document_id": version.document_id,
                    "filename": version.filename,
                    "version_number": version.version_number,
                },
                "normalization_run": self._technical_run_payload(normalization_run),
                "detectors": run.configuration.get("detectors", []),
                "groups": self._groups(run.id),
                "candidate_types": self._candidate_types(run.id),
            }
        )
        return payload

    def compare_runs(self, run_a_id: UUID, run_b_id: UUID) -> dict[str, Any]:
        run_a = self._session.get(ProcessingRun, run_a_id)
        run_b = self._session.get(ProcessingRun, run_b_id)
        if run_a is None:
            raise ValueError(f"Run A introuvable : {run_a_id}.")
        if run_b is None:
            raise ValueError(f"Run B introuvable : {run_b_id}.")
        if run_a.process_type != "candidate_discovery":
            raise ValueError("Le run A n'est pas un run candidate_discovery.")
        if run_b.process_type != "candidate_discovery":
            raise ValueError("Le run B n'est pas un run candidate_discovery.")
        if run_a.document_version_id != run_b.document_version_id:
            raise ValueError("Les runs appartiennent à deux DocumentVersion différentes.")
        version = self._session.get(DocumentVersion, run_a.document_version_id)
        groups_a = {(row["candidate_type"], row["normalized_value"]): row for row in self._groups(run_a.id)}
        groups_b = {(row["candidate_type"], row["normalized_value"]): row for row in self._groups(run_b.id)}
        differences = []
        status_counts = {key: 0 for key in ("added", "removed", "increased", "decreased", "unchanged")}
        for candidate_type, value in sorted(set(groups_a) | set(groups_b), key=lambda key: (key[0], key[1])):
            left = groups_a.get((candidate_type, value))
            right = groups_b.get((candidate_type, value))
            count_a = left["occurrence_count"] if left else 0
            count_b = right["occurrence_count"] if right else 0
            if count_a == 0:
                change = "added"
            elif count_b == 0:
                change = "removed"
            elif count_b > count_a:
                change = "increased"
            elif count_b < count_a:
                change = "decreased"
            else:
                change = "unchanged"
            status_counts[change] += 1
            differences.append({
                "candidate_type": candidate_type,
                "value": value,
                "occurrences_a": count_a,
                "occurrences_b": count_b,
                "delta": count_b - count_a,
                "pages_a": left["pages"] if left else [],
                "pages_b": right["pages"] if right else [],
                "change": change,
            })
        total_a = sum(row["occurrence_count"] for row in groups_a.values())
        total_b = sum(row["occurrence_count"] for row in groups_b.values())
        reliable = run_a.status == run_b.status == "completed"
        return {
            "document": {
                "document_version_id": run_a.document_version_id,
                "document_id": version.document_id if version else None,
                "filename": version.filename if version else None,
            },
            "run_a": self._comparison_run_payload(run_a),
            "run_b": self._comparison_run_payload(run_b),
            "dictionary_a": self._dictionary_identity(run_a),
            "dictionary_b": self._dictionary_identity(run_b),
            "comparison_reliable": reliable,
            "comparison_state": "complete" if reliable else "partial",
            "summary": {
                "total_occurrences_a": total_a,
                "total_occurrences_b": total_b,
                "delta": total_b - total_a,
                "added": status_counts["added"],
                "removed": status_counts["removed"],
                "increased": status_counts["increased"],
                "decreased": status_counts["decreased"],
                "unchanged": status_counts["unchanged"],
            },
            "differences": differences,
        }

    def list_candidates(
        self,
        run_id: UUID,
        *,
        candidate_type: str | None = None,
        value: str | None = None,
        page: int | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any] | None:
        run = self._candidate_run(run_id)
        if run is None:
            return None
        statement = (
            select(DiscoveredCandidate.id)
            .join(
                CandidateSourceFragment,
                CandidateSourceFragment.candidate_id == DiscoveredCandidate.id,
            )
            .join(
                NormalizedContentUnit,
                NormalizedContentUnit.id
                == CandidateSourceFragment.normalized_content_unit_id,
            )
            .join(
                NormalizedContentUnitSource,
                NormalizedContentUnitSource.normalized_content_unit_id
                == NormalizedContentUnit.id,
            )
            .join(ContentBlock, ContentBlock.id == NormalizedContentUnitSource.content_block_id)
            .join(ContentBlockFragment, ContentBlockFragment.content_block_id == ContentBlock.id)
            .join(Page, Page.id == ContentBlockFragment.page_id)
            .where(DiscoveredCandidate.processing_run_id == run_id)
        )
        if candidate_type:
            statement = statement.where(DiscoveredCandidate.candidate_type == candidate_type)
        if value:
            pattern = f"%{value}%"
            statement = statement.where(
                or_(
                    DiscoveredCandidate.normalized_value.ilike(pattern),
                    DiscoveredCandidate.raw_value.ilike(pattern),
                )
            )
        if page is not None:
            statement = statement.where(Page.page_number == page)
        grouped = statement.group_by(DiscoveredCandidate.id)
        total = self._session.exec(
            select(func.count()).select_from(grouped.subquery())
        ).one()
        candidate_ids = list(
            self._session.exec(
                grouped
                .order_by(func.min(DiscoveredCandidate.created_at), DiscoveredCandidate.id)
                .offset(offset)
                .limit(limit)
            ).all()
        )
        if not candidate_ids:
            return {"items": [], "limit": limit, "offset": offset, "total": total}
        rows = self._candidate_rows(candidate_ids)
        first_by_candidate: dict[UUID, dict[str, Any]] = {}
        for row in rows:
            candidate = row[0]
            first_by_candidate.setdefault(candidate.id, self._candidate_list_payload(*row))
        return {
            "items": [first_by_candidate[candidate_id] for candidate_id in candidate_ids],
            "limit": limit,
            "offset": offset,
            "total": total,
        }

    def get_candidate(self, candidate_id: UUID) -> dict[str, Any] | None:
        candidate = self._session.get(DiscoveredCandidate, candidate_id)
        if candidate is None:
            return None
        version = self._session.get(DocumentVersion, candidate.document_version_id)
        source_fragments = list(
            self._session.exec(
                select(CandidateSourceFragment)
                .where(CandidateSourceFragment.candidate_id == candidate.id)
                .order_by(CandidateSourceFragment.fragment_order)
            ).all()
        )
        unit_ids = {item.normalized_content_unit_id for item in source_fragments}
        units = {
            item.id: item
            for item in self._session.exec(
                select(NormalizedContentUnit).where(NormalizedContentUnit.id.in_(unit_ids))
            ).all()
        }
        unit_sources = list(
            self._session.exec(
                select(NormalizedContentUnitSource).where(
                    NormalizedContentUnitSource.normalized_content_unit_id.in_(unit_ids)
                ).order_by(
                    NormalizedContentUnitSource.normalized_content_unit_id,
                    NormalizedContentUnitSource.source_order,
                )
            ).all()
        ) if unit_ids else []
        block_ids = {item.content_block_id for item in unit_sources}
        blocks = {
            item.id: item
            for item in self._session.exec(
                select(ContentBlock).where(ContentBlock.id.in_(block_ids))
            ).all()
        } if block_ids else {}
        block_fragments = list(
            self._session.exec(
                select(ContentBlockFragment, Page)
                .join(Page, Page.id == ContentBlockFragment.page_id)
                .where(ContentBlockFragment.content_block_id.in_(block_ids))
                .order_by(ContentBlockFragment.content_block_id, ContentBlockFragment.fragment_index)
            ).all()
        ) if block_ids else []
        fragments_by_block: dict[UUID, list[dict[str, Any]]] = defaultdict(list)
        for fragment, page in block_fragments:
            fragments_by_block[fragment.content_block_id].append(
                {
                    "id": fragment.id,
                    "fragment_index": fragment.fragment_index,
                    "page_number": page.page_number,
                    "content": fragment.content,
                    "bbox": self._bbox(fragment),
                    "coordinate_system": fragment.coordinate_system,
                }
            )
        sources_by_unit: dict[UUID, list[dict[str, Any]]] = defaultdict(list)
        for source in unit_sources:
            block = blocks[source.content_block_id]
            sources_by_unit[source.normalized_content_unit_id].append(
                {
                    "source_order": source.source_order,
                    "content_block": {
                        "id": block.id,
                        "block_index": block.block_index,
                        "reading_order": block.reading_order,
                        "block_type": block.block_type,
                        "content": block.content,
                        "fragments": fragments_by_block.get(block.id, []),
                    },
                }
            )
        provenance = []
        for source_fragment in source_fragments:
            unit = units[source_fragment.normalized_content_unit_id]
            provenance.append(
                {
                    "occurrence": {
                        "fragment_order": source_fragment.fragment_order,
                        "exact_text": source_fragment.exact_text,
                        "start_offset": source_fragment.start_offset,
                        "end_offset": source_fragment.end_offset,
                        "role": source_fragment.role,
                    },
                    "normalized_content_unit": {
                        "id": unit.id,
                        "unit_index": unit.unit_index,
                        "content_type": unit.content_type,
                        "content": unit.content,
                        "processing_run_id": unit.processing_run_id,
                        "sources": sources_by_unit.get(unit.id, []),
                    },
                }
            )
        return {
            "candidate": {
                "id": candidate.id,
                "candidate_type": candidate.candidate_type,
                "raw_value": candidate.raw_value,
                "normalized_value": candidate.normalized_value,
                "detector_key": candidate.detector_key,
                "detector_version": candidate.detector_version,
                "confidence": candidate.confidence,
                "payload": candidate.payload,
                "processing_run_id": candidate.processing_run_id,
            },
            "document": {
                "filename": version.filename if version else None,
                "document_version_id": candidate.document_version_id,
            },
            "provenance": provenance,
        }

    def _candidate_rows(self, candidate_ids: list[UUID]):
        return self._session.exec(
            select(
                DiscoveredCandidate,
                CandidateSourceFragment,
                NormalizedContentUnit,
                ContentBlock,
                ContentBlockFragment,
                Page,
                DocumentVersion,
            )
            .join(CandidateSourceFragment, CandidateSourceFragment.candidate_id == DiscoveredCandidate.id)
            .join(NormalizedContentUnit, NormalizedContentUnit.id == CandidateSourceFragment.normalized_content_unit_id)
            .join(NormalizedContentUnitSource, NormalizedContentUnitSource.normalized_content_unit_id == NormalizedContentUnit.id)
            .join(ContentBlock, ContentBlock.id == NormalizedContentUnitSource.content_block_id)
            .join(ContentBlockFragment, ContentBlockFragment.content_block_id == ContentBlock.id)
            .join(Page, Page.id == ContentBlockFragment.page_id)
            .join(DocumentVersion, DocumentVersion.id == DiscoveredCandidate.document_version_id)
            .where(DiscoveredCandidate.id.in_(candidate_ids))
            .order_by(DiscoveredCandidate.created_at, CandidateSourceFragment.fragment_order, ContentBlockFragment.fragment_index)
        ).all()

    def _groups(self, run_id: UUID) -> list[dict[str, Any]]:
        value = func.coalesce(
            DiscoveredCandidate.normalized_value,
            DiscoveredCandidate.raw_value,
        )
        rows = self._session.exec(
            select(
                value,
                DiscoveredCandidate.candidate_type,
                func.count(func.distinct(DiscoveredCandidate.id)),
                func.array_agg(func.distinct(Page.page_number)),
            )
            .join(CandidateSourceFragment, CandidateSourceFragment.candidate_id == DiscoveredCandidate.id)
            .join(NormalizedContentUnit, NormalizedContentUnit.id == CandidateSourceFragment.normalized_content_unit_id)
            .join(NormalizedContentUnitSource, NormalizedContentUnitSource.normalized_content_unit_id == NormalizedContentUnit.id)
            .join(ContentBlock, ContentBlock.id == NormalizedContentUnitSource.content_block_id)
            .join(ContentBlockFragment, ContentBlockFragment.content_block_id == ContentBlock.id)
            .join(Page, Page.id == ContentBlockFragment.page_id)
            .where(DiscoveredCandidate.processing_run_id == run_id)
            .group_by(value, DiscoveredCandidate.candidate_type)
            .order_by(
                func.count(func.distinct(DiscoveredCandidate.id)).desc(),
                value,
                DiscoveredCandidate.candidate_type,
            )
        ).all()
        return [
            {
                "normalized_value": normalized_value,
                "candidate_type": candidate_type,
                "occurrence_count": occurrence_count,
                "pages": sorted(pages or []),
                "page_count": len(pages or []),
            }
            for normalized_value, candidate_type, occurrence_count, pages in rows
        ]

    def _candidate_run(self, run_id: UUID) -> ProcessingRun | None:
        run = self._session.get(ProcessingRun, run_id)
        if run is None or run.process_type != "candidate_discovery":
            return None
        return run

    def _candidate_types(self, run_id: UUID) -> list[str]:
        return list(
            self._session.exec(
                select(DiscoveredCandidate.candidate_type)
                .where(DiscoveredCandidate.processing_run_id == run_id)
                .distinct()
                .order_by(DiscoveredCandidate.candidate_type)
            ).all()
        )

    @staticmethod
    def _normalization_run_id(run: ProcessingRun) -> UUID | None:
        value = run.configuration.get("normalization_run_id")
        try:
            return UUID(str(value)) if value else None
        except ValueError:
            return None

    @classmethod
    def _run_payload(cls, run: ProcessingRun, version: DocumentVersion) -> dict[str, Any]:
        return {
            "run_id": run.id,
            "document_version_id": run.document_version_id,
            "filename": version.filename,
            "status": run.status,
            "engine": run.engine,
            "engine_version": run.engine_version,
            "started_at": run.started_at,
            "completed_at": run.completed_at,
            "normalization_run_id": cls._normalization_run_id(run),
            "metrics": run.metrics,
            "configuration": run.configuration,
            "error_message": run.error_message,
            "dictionary": cls._dictionary_identity(run),
        }

    @classmethod
    def _comparison_run_payload(cls, run: ProcessingRun) -> dict[str, Any]:
        return {
            "run_id": run.id, "status": run.status, "started_at": run.started_at,
            "completed_at": run.completed_at, "engine": run.engine,
            "engine_version": run.engine_version,
            "normalization_run_id": cls._normalization_run_id(run),
        }

    @staticmethod
    def _dictionary_identity(run: ProcessingRun) -> dict[str, Any] | None:
        value = (run.configuration or {}).get("dictionary")
        return value if isinstance(value, dict) else None

    @staticmethod
    def _technical_run_payload(run: ProcessingRun | None) -> dict[str, Any] | None:
        if run is None:
            return None
        return {
            "run_id": run.id,
            "process_type": run.process_type,
            "status": run.status,
            "engine": run.engine,
            "engine_version": run.engine_version,
            "completed_at": run.completed_at,
        }

    @classmethod
    def _candidate_list_payload(cls, candidate, source, unit, block, fragment, page, version):
        return {
            "candidate_id": candidate.id,
            "candidate_type": candidate.candidate_type,
            "raw_value": candidate.raw_value,
            "normalized_value": candidate.normalized_value,
            "confidence": candidate.confidence,
            "detector_key": candidate.detector_key,
            "detector_version": candidate.detector_version,
            "exact_text": source.exact_text,
            "start_offset": source.start_offset,
            "end_offset": source.end_offset,
            "unit_index": unit.unit_index,
            "unit_content": unit.content,
            "page_number": page.page_number,
            "filename": version.filename,
            "processing_run_id": candidate.processing_run_id,
            "normalized_content_unit_id": unit.id,
            "content_block_id": block.id,
            "content_block_fragment_id": fragment.id,
            "bbox": cls._bbox(fragment),
            "coordinate_system": fragment.coordinate_system,
        }

    @staticmethod
    def _bbox(fragment: ContentBlockFragment) -> dict[str, float] | None:
        values = (fragment.bbox_x, fragment.bbox_y, fragment.bbox_width, fragment.bbox_height)
        if all(value is None for value in values):
            return None
        return {"x": values[0], "y": values[1], "width": values[2], "height": values[3]}
