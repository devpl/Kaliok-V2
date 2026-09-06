from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from sqlmodel import Session, select

from kaliok.execution import ExecutionContext
from kaliok.discovery.dictionary import load_lexical_dictionary
from kaliok.discovery.read import CandidateDiscoveryReadService
from kaliok.discovery.service import CandidateDiscoveryService
from kaliok.storage.models import DocumentVersion, NormalizedContentUnit, ProcessingRun


def resolve_document_version(
    session: Session,
    *,
    document_version_id: UUID | None = None,
    filename: str | None = None,
) -> DocumentVersion:
    if document_version_id is not None:
        version = session.get(DocumentVersion, document_version_id)
        if version is None:
            raise ValueError(f"DocumentVersion inconnue : {document_version_id}.")
        return version
    if not filename or not filename.strip():
        raise ValueError("Un UUID ou une recherche de filename est requis.")
    matches = list(
        session.exec(
            select(DocumentVersion)
            .where(DocumentVersion.filename.ilike(f"%{filename.strip()}%"))
            .order_by(DocumentVersion.created_at.desc(), DocumentVersion.id.desc())
        ).all()
    )
    if not matches:
        raise ValueError(f"Aucune DocumentVersion ne correspond à : {filename}.")
    exact = [item for item in matches if item.filename.casefold() == filename.casefold()]
    candidates = exact or matches
    if len(candidates) > 1:
        choices = ", ".join(f"{item.filename} ({item.id})" for item in candidates[:10])
        raise ValueError(f"Plusieurs versions correspondent ; précisez l'UUID : {choices}")
    return candidates[0]


def resolve_normalization_run(
    session: Session,
    document_version_id: UUID,
    *,
    normalization_run_id: UUID | None = None,
) -> ProcessingRun:
    if normalization_run_id is not None:
        run = session.get(ProcessingRun, normalization_run_id)
        if (
            run is None
            or run.document_version_id != document_version_id
            or run.process_type != "content_normalization"
            or run.status != "completed"
        ):
            raise ValueError(
                "Le run demandé doit être un content_normalization completed du document."
            )
        return run
    run = session.exec(
        select(ProcessingRun)
        .where(
            ProcessingRun.document_version_id == document_version_id,
            ProcessingRun.process_type == "content_normalization",
            ProcessingRun.status == "completed",
        )
        .order_by(ProcessingRun.completed_at.desc(), ProcessingRun.started_at.desc())
    ).first()
    if run is None:
        raise ValueError("Aucun run content_normalization completed pour ce document.")
    return run


def run_candidate_discovery_experiment(
    session: Session,
    *,
    dictionary_path: str | Path,
    document_version_id: UUID | None = None,
    filename: str | None = None,
    normalization_run_id: UUID | None = None,
    commit: bool = False,
    execution_context: ExecutionContext | None = None,
) -> dict[str, Any]:
    version = resolve_document_version(
        session,
        document_version_id=document_version_id,
        filename=filename,
    )
    normalization_run = resolve_normalization_run(
        session,
        version.id,
        normalization_run_id=normalization_run_id,
    )
    unit_count = len(
        session.exec(
            select(NormalizedContentUnit.id).where(
                NormalizedContentUnit.processing_run_id == normalization_run.id
            )
        ).all()
    )
    dictionary = load_lexical_dictionary(dictionary_path)
    detector = dictionary.detector()
    execution_group_id = (
        normalization_run.execution_group_id
        if normalization_run.execution_environment == "experiment"
        else uuid4()
    )
    context = execution_context or ExecutionContext(
        environment="experiment",
        execution_group_id=execution_group_id,
    )
    if context.environment != "experiment":
        raise ValueError(
            "run_candidate_discovery_experiment exige un contexte experiment."
        )
    result = CandidateDiscoveryService(session).discover(
        version.id,
        normalization_run.id,
        [detector],
        execution_context=context,
    )
    detail = CandidateDiscoveryReadService(session).get_run(result.processing_run_id)
    summary = {
        "mode": "commit" if commit else "dry-run",
        "document": version.filename,
        "document_version_id": str(version.id),
        "normalization_run_id": str(normalization_run.id),
        "source_unit_count": unit_count,
        "detectors": [{
            "key": detector.key,
            "version": detector.version,
            "configuration": detector.configuration,
            "dictionary_name": dictionary.metadata.name,
            "dictionary_version": dictionary.metadata.version,
            "dictionary_key": dictionary.metadata.dictionary_key,
            "dictionary_hash": dictionary.dictionary_hash,
        }],
        "discovery_run_id": str(result.processing_run_id),
        "candidate_count": result.candidate_count,
        "candidate_type_counts": detail["metrics"].get("candidate_type_counts", {}),
        "groups": detail.get("groups", []),
    }
    if commit:
        session.commit()
    else:
        session.rollback()
    return summary
