from __future__ import annotations

from collections.abc import Iterator
from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlmodel import Session, SQLModel, select

from kaliok.discovery import (
    CandidateDiscoveryReadService,
    CandidateDiscoveryService,
    LexicalCandidateDetector,
    LexicalTerm,
    load_lexical_dictionary,
    resolve_normalization_run,
    run_candidate_discovery_experiment,
)
from kaliok.normalization import ContentNormalizationService
from kaliok.storage.database import create_database_engine
from kaliok.storage.models import (
    CandidateSourceFragment,
    ConfigurationProfile,
    ConfigurationProfileRevision,
    ContentBlock,
    ContentBlockFragment,
    DiscoveredCandidate,
    Document,
    DocumentVersion,
    NormalizedContentUnit,
    NormalizedContentUnitSource,
    Page,
    ProcessingRun,
    Entity, EntityMembership, EntityResolutionEvidence, EntityResolutionScopeItem,
    Source,
)


DISCOVERY_TABLES = [
    ConfigurationProfile.__table__,
    ConfigurationProfileRevision.__table__,
    Source.__table__,
    Document.__table__,
    DocumentVersion.__table__,
    ProcessingRun.__table__,
    Page.__table__,
    ContentBlock.__table__,
    ContentBlockFragment.__table__,
    NormalizedContentUnit.__table__,
    NormalizedContentUnitSource.__table__,
    DiscoveredCandidate.__table__,
    CandidateSourceFragment.__table__,
    EntityResolutionScopeItem.__table__, Entity.__table__, EntityMembership.__table__, EntityResolutionEvidence.__table__,
]


@pytest.fixture
def session() -> Iterator[Session]:
    engine = create_database_engine()
    schema_name = f"kaliok_discovery_test_{uuid4().hex}"
    with engine.connect() as connection:
        transaction = connection.begin()
        try:
            connection.execute(text(f'CREATE SCHEMA "{schema_name}"'))
            connection.execute(text(f'SET LOCAL search_path TO "{schema_name}"'))
            SQLModel.metadata.create_all(connection, tables=DISCOVERY_TABLES)
            with Session(
                bind=connection,
                join_transaction_mode="create_savepoint",
            ) as test_session:
                yield test_session
        finally:
            if transaction.is_active:
                transaction.rollback()
    engine.dispose()


def _normalized_generation(
    session: Session,
    contents: tuple[str, ...],
) -> tuple[DocumentVersion, ProcessingRun, list[ContentBlock]]:
    document = Document(title="Discovery")
    session.add(document)
    session.flush()
    version = DocumentVersion(
        document_id=document.id,
        version_number=1,
        filename="discovery.txt",
        file_hash=uuid4().hex,
        storage_uri="test://discovery.txt",
    )
    session.add(version)
    session.flush()
    perception = ProcessingRun(
        document_version_id=version.id,
        process_type="document_extraction",
        status="completed",
    )
    session.add(perception)
    session.flush()
    page = Page(
        document_version_id=version.id,
        page_number=1,
        perception_processing_run_id=perception.id,
    )
    session.add(page)
    session.flush()
    blocks = []
    for index, content in enumerate(contents):
        block = ContentBlock(
            page_id=page.id,
            processing_run_id=perception.id,
            block_index=index,
            reading_order=index,
            block_type="paragraph",
            content=content,
            extraction_method="test",
        )
        session.add(block)
        session.flush()
        session.add(
            ContentBlockFragment(
                content_block_id=block.id,
                page_id=page.id,
                fragment_index=0,
                content=content,
            )
        )
        blocks.append(block)
    session.flush()
    normalization = ContentNormalizationService(session).normalize(version.id)
    return version, session.get(ProcessingRun, normalization.processing_run_id), blocks


def _detector(*terms: LexicalTerm, case_sensitive: bool = False):
    return LexicalCandidateDetector(terms, case_sensitive=case_sensitive)


def test_invalid_normalization_runs_are_rejected_before_discovery_run(session: Session):
    first, completed, _ = _normalized_generation(session, ("Durand",))
    second, _, _ = _normalized_generation(session, ("Durand",))

    with pytest.raises(ValueError, match="inconnu"):
        CandidateDiscoveryService(session).discover(first.id, uuid4(), [_detector(LexicalTerm("Durand", "person"))])
    with pytest.raises(ValueError, match="autre document"):
        CandidateDiscoveryService(session).discover(second.id, completed.id, [_detector(LexicalTerm("Durand", "person"))])

    completed.process_type = "document_extraction"
    session.add(completed)
    session.flush()
    with pytest.raises(ValueError, match="content_normalization"):
        CandidateDiscoveryService(session).discover(first.id, completed.id, [_detector(LexicalTerm("Durand", "person"))])
    completed.process_type = "content_normalization"
    completed.status = "running"
    session.add(completed)
    session.flush()
    with pytest.raises(ValueError, match="completed"):
        CandidateDiscoveryService(session).discover(first.id, completed.id, [_detector(LexicalTerm("Durand", "person"))])


def test_lexical_detector_preserves_exact_offsets_occurrences_and_payload(session: Session):
    version, normalization, _ = _normalized_generation(
        session,
        ("Monsieur Durand voit durand, puis Durant.",),
    )
    detector = _detector(
        LexicalTerm("Durand", "person_name", payload={"source": "test"}, confidence=0.8),
        LexicalTerm("Durant", "person_name"),
    )

    result = CandidateDiscoveryService(session).discover(
        version.id,
        normalization.id,
        [detector],
    )
    candidates = session.exec(
        select(DiscoveredCandidate)
        .where(DiscoveredCandidate.processing_run_id == result.processing_run_id)
        .order_by(DiscoveredCandidate.created_at, DiscoveredCandidate.id)
    ).all()
    fragments = session.exec(
        select(CandidateSourceFragment).where(
            CandidateSourceFragment.candidate_id.in_([candidate.id for candidate in candidates])
        ).order_by(CandidateSourceFragment.start_offset)
    ).all()

    assert result.candidate_count == 3
    assert [(fragment.start_offset, fragment.end_offset, fragment.exact_text) for fragment in fragments] == [
        (9, 15, "Durand"),
        (21, 27, "durand"),
        (34, 40, "Durant"),
    ]
    durand_candidates = [candidate for candidate in candidates if candidate.normalized_value == "Durand"]
    assert len(durand_candidates) == 2
    assert all(candidate.payload == {"source": "test"} for candidate in durand_candidates)
    assert all(candidate.confidence == 0.8 for candidate in durand_candidates)
    assert {candidate.normalized_value for candidate in candidates} == {"Durand", "Durant"}


def test_multiword_terms_multiple_types_units_and_exact_metrics(session: Session):
    version, normalization, _ = _normalized_generation(
        session,
        ("12 rue de la République", "Une subvention publique", "sans résultat"),
    )
    result = CandidateDiscoveryService(session).discover(
        version.id,
        normalization.id,
        [
            _detector(
                LexicalTerm("12 rue de la République", "address"),
                LexicalTerm("subvention", "funding"),
            )
        ],
    )
    run = session.get(ProcessingRun, result.processing_run_id)
    candidates = session.exec(
        select(DiscoveredCandidate).where(
            DiscoveredCandidate.processing_run_id == result.processing_run_id
        )
    ).all()

    assert {candidate.candidate_type for candidate in candidates} == {"address", "funding"}
    assert run.metrics == {
        "source_unit_count": 3,
        "candidate_count": 2,
        "candidate_type_counts": {"address": 1, "funding": 1},
        "skipped_unit_count": 1,
    }
    assert run.status == "completed" and run.completed_at is not None
    assert run.configuration["normalization_run_id"] == str(normalization.id)


def test_generation_and_document_boundaries_prevent_cross_generation_fusion(session: Session):
    first_version, first_generation, _ = _normalized_generation(session, ("Durand",))
    second_generation = ContentNormalizationService(session).normalize(first_version.id)
    second_version, other_document_generation, _ = _normalized_generation(session, ("Durand",))
    detector = _detector(LexicalTerm("Durand", "person"))

    results = [
        CandidateDiscoveryService(session).discover(first_version.id, first_generation.id, [detector]),
        CandidateDiscoveryService(session).discover(first_version.id, second_generation.processing_run_id, [detector]),
        CandidateDiscoveryService(session).discover(second_version.id, other_document_generation.id, [detector]),
    ]

    assert all(result.candidate_count == 1 for result in results)
    candidates = session.exec(select(DiscoveredCandidate)).all()
    assert len(candidates) == 3
    assert len({candidate.processing_run_id for candidate in candidates}) == 3
    assert len({candidate.document_version_id for candidate in candidates}) == 2


def test_complete_provenance_chain_is_navigable(session: Session):
    version, normalization, blocks = _normalized_generation(session, ("Monsieur Durand",))
    result = CandidateDiscoveryService(session).discover(
        version.id,
        normalization.id,
        [_detector(LexicalTerm("Durand", "person"))],
    )
    candidate = session.exec(
        select(DiscoveredCandidate).where(
            DiscoveredCandidate.processing_run_id == result.processing_run_id
        )
    ).one()
    source_fragment = session.exec(
        select(CandidateSourceFragment).where(
            CandidateSourceFragment.candidate_id == candidate.id
        )
    ).one()
    unit = session.get(NormalizedContentUnit, source_fragment.normalized_content_unit_id)
    unit_source = session.exec(
        select(NormalizedContentUnitSource).where(
            NormalizedContentUnitSource.normalized_content_unit_id == unit.id
        )
    ).one()
    block = session.get(ContentBlock, unit_source.content_block_id)
    block_fragment = session.exec(
        select(ContentBlockFragment).where(
            ContentBlockFragment.content_block_id == block.id
        )
    ).one()
    page = session.get(Page, block_fragment.page_id)

    assert block.id == blocks[0].id
    assert page.document_version_id == version.id
    assert source_fragment.fragment_order == 0


def test_detector_error_rolls_back_candidates_and_marks_run_failed(session: Session):
    version, normalization, _ = _normalized_generation(session, ("Durand", "Erreur"))

    class FailingDetector:
        key = "failing"
        version = "1"
        configuration = {}

        def detect(self, unit):
            if unit.unit_index == 1:
                raise RuntimeError("détecteur en échec")
            return _detector(LexicalTerm("Durand", "person")).detect(unit)

    with pytest.raises(RuntimeError, match="détecteur en échec"):
        CandidateDiscoveryService(session).discover(
            version.id,
            normalization.id,
            [FailingDetector()],
        )

    run = session.exec(
        select(ProcessingRun).where(ProcessingRun.process_type == "candidate_discovery")
    ).one()
    assert run.status == "failed"
    assert run.completed_at is not None
    assert run.error_message == "détecteur en échec"
    assert session.exec(
        select(DiscoveredCandidate).where(DiscoveredCandidate.processing_run_id == run.id)
    ).all() == []


def test_read_service_lists_only_discovery_runs_groups_and_filters(session: Session):
    version, normalization, _ = _normalized_generation(
        session,
        ("Nouméa et Nouméa", "Subvention"),
    )
    result = CandidateDiscoveryService(session).discover(
        version.id,
        normalization.id,
        [_detector(
            LexicalTerm("Nouméa", "place_name"),
            LexicalTerm("Subvention", "funding"),
        )],
    )
    reader = CandidateDiscoveryReadService(session)

    runs = reader.list_runs()
    assert [run["run_id"] for run in runs["items"]] == [result.processing_run_id]
    assert runs["total"] == 1
    detail = reader.get_run(result.processing_run_id)
    assert detail["normalization_run_id"] == normalization.id
    noumea = next(group for group in detail["groups"] if group["normalized_value"] == "Nouméa")
    assert noumea == {
        "normalized_value": "Nouméa",
        "candidate_type": "place_name",
        "occurrence_count": 2,
        "pages": [1],
        "page_count": 1,
    }
    assert reader.list_candidates(result.processing_run_id, candidate_type="place_name")["total"] == 2
    assert reader.list_candidates(result.processing_run_id, value="nouméa")["total"] == 2
    assert reader.list_candidates(result.processing_run_id, page=1)["total"] == 3
    assert reader.list_candidates(result.processing_run_id, page=2) == {
        "items": [], "limit": 100, "offset": 0, "total": 0
    }
    assert detail["candidate_types"] == ["funding", "place_name"]


def test_read_service_candidate_detail_supports_multiple_source_fragments(session: Session):
    version, normalization, _ = _normalized_generation(session, ("Nouméa", "Nouméa"))
    result = CandidateDiscoveryService(session).discover(
        version.id,
        normalization.id,
        [_detector(LexicalTerm("Nouméa", "place_name"))],
    )
    candidates = session.exec(
        select(DiscoveredCandidate).where(
            DiscoveredCandidate.processing_run_id == result.processing_run_id
        ).order_by(DiscoveredCandidate.created_at, DiscoveredCandidate.id)
    ).all()
    second_source = session.exec(
        select(CandidateSourceFragment).where(
            CandidateSourceFragment.candidate_id == candidates[1].id
        )
    ).one()
    session.add(
        CandidateSourceFragment(
            candidate_id=candidates[0].id,
            normalized_content_unit_id=second_source.normalized_content_unit_id,
            fragment_order=1,
            start_offset=0,
            end_offset=6,
            exact_text="Nouméa",
            role="supporting",
        )
    )
    session.flush()

    detail = CandidateDiscoveryReadService(session).get_candidate(candidates[0].id)

    assert detail["candidate"]["raw_value"] == "Nouméa"
    assert detail["document"]["filename"] == "discovery.txt"
    assert len(detail["provenance"]) == 2
    assert [item["occurrence"]["fragment_order"] for item in detail["provenance"]] == [0, 1]
    assert all(
        item["normalized_content_unit"]["sources"][0]["content_block"]["fragments"][0]["page_number"] == 1
        for item in detail["provenance"]
    )


def test_development_dictionary_loads_unicode_and_boundary_policy():
    detector, configuration = load_lexical_dictionary(
        "config/discovery/development_dictionary.json"
    )

    assert configuration["version"] == "1"
    assert configuration["boundary_policy"] == "unicode_word"
    assert any(term["value"] == "Nouméa" for term in detector.configuration["terms"])


def test_experiment_selects_latest_normalization_and_supports_dry_run_and_commit(
    session: Session,
):
    version, first_normalization, _ = _normalized_generation(session, ("Nouméa",))
    second_normalization = ContentNormalizationService(session).normalize(version.id)
    selected = resolve_normalization_run(session, version.id)
    assert selected.id == second_normalization.processing_run_id
    with pytest.raises(ValueError, match="content_normalization completed"):
        resolve_normalization_run(
            session,
            version.id,
            normalization_run_id=uuid4(),
        )
    session.commit()

    dry_run = run_candidate_discovery_experiment(
        session,
        dictionary_path="config/discovery/development_dictionary.json",
        document_version_id=version.id,
        normalization_run_id=first_normalization.id,
        commit=False,
    )
    assert dry_run["mode"] == "dry-run"
    assert dry_run["candidate_count"] == 1
    assert session.get(ProcessingRun, dry_run["discovery_run_id"]) is None

    committed = run_candidate_discovery_experiment(
        session,
        dictionary_path="config/discovery/development_dictionary.json",
        document_version_id=version.id,
        normalization_run_id=first_normalization.id,
        commit=True,
    )
    assert committed["mode"] == "commit"
    assert committed["candidate_count"] == 1
    assert session.get(ProcessingRun, committed["discovery_run_id"]) is not None


def test_governed_dictionary_hash_snapshot_and_normalized_value(tmp_path, session: Session):
    first = tmp_path / "dictionary.json"
    second = tmp_path / "formatted.json"
    payload = {
        "dictionary_key": "test-fr", "version": "1", "name": "Test Unicode",
        "language": "fr", "terms": [{"value": "nouméa", "candidate_type": "place",
        "normalized_value": "Nouméa", "confidence": 0.75, "payload": {"source": "test"}}],
    }
    first.write_text(__import__("json").dumps(payload, ensure_ascii=False), encoding="utf-8")
    second.write_text(__import__("json").dumps(payload, ensure_ascii=False, indent=4), encoding="utf-8")
    dictionary = load_lexical_dictionary(first)
    assert dictionary.dictionary_hash == load_lexical_dictionary(second).dictionary_hash
    version, normalization, _ = _normalized_generation(session, ("Visite à nouméa.",))
    result = CandidateDiscoveryService(session).discover(version.id, normalization.id, [dictionary.detector()])
    run = session.get(ProcessingRun, result.processing_run_id)
    candidate = session.exec(select(DiscoveredCandidate).where(DiscoveredCandidate.processing_run_id == run.id)).one()
    source = session.exec(select(CandidateSourceFragment).where(CandidateSourceFragment.candidate_id == candidate.id)).one()
    assert candidate.raw_value == source.exact_text == "nouméa"
    assert candidate.normalized_value == "Nouméa"
    assert candidate.confidence == 0.75 and candidate.payload == {"source": "test"}
    assert run.configuration["dictionary"]["dictionary_key"] == "test-fr"
    term = run.configuration["detectors"][0]["configuration"]["terms"][0]
    assert term == {"value": "nouméa", "candidate_type": "place", "normalized_value": "Nouméa", "confidence": 0.75, "payload": {"source": "test"}}


def test_compare_runs_classifies_changes_and_rejects_other_document(session: Session):
    version, normalization, _ = _normalized_generation(session, ("Alpha Alpha Beta Gamma",))
    run_a = CandidateDiscoveryService(session).discover(version.id, normalization.id, [_detector(
        LexicalTerm("Alpha", "kind"), LexicalTerm("Beta", "kind"), LexicalTerm("Gamma", "kind")
    )])
    run_b = CandidateDiscoveryService(session).discover(version.id, normalization.id, [_detector(
        LexicalTerm("Alpha", "kind"), LexicalTerm("Beta", "kind", normalized_value="Delta")
    )])
    comparison = CandidateDiscoveryReadService(session).compare_runs(run_a.processing_run_id, run_b.processing_run_id)
    changes = {(item["value"], item["change"]) for item in comparison["differences"]}
    assert {("Alpha", "unchanged"), ("Beta", "removed"), ("Gamma", "removed"), ("Delta", "added")} == changes
    assert all(item["pages_a"] == [1] or item["pages_a"] == [] for item in comparison["differences"])
    other, other_normalization, _ = _normalized_generation(session, ("Alpha",))
    other_run = CandidateDiscoveryService(session).discover(other.id, other_normalization.id, [_detector(LexicalTerm("Alpha", "kind"))])
    with pytest.raises(ValueError, match="DocumentVersion différentes"):
        CandidateDiscoveryReadService(session).compare_runs(run_a.processing_run_id, other_run.processing_run_id)


def test_entity_resolution_declared_exact_and_generations(session: Session):
    from kaliok.entity_resolution import EntityResolutionService
    version, normalization, _ = _normalized_generation(session, ("Nouméa NOUMÉA Durand Durant",))
    discovery = CandidateDiscoveryService(session).discover(version.id, normalization.id, [_detector(
        LexicalTerm("Nouméa", "place_name", normalized_value="Nouméa"),
        LexicalTerm("NOUMÉA", "place_name", normalized_value="Nouméa"),
        LexicalTerm("Durand", "person_name"), LexicalTerm("Durant", "person_name"),
    )])
    candidates = session.exec(select(DiscoveredCandidate).where(DiscoveredCandidate.processing_run_id == discovery.processing_run_id).order_by(DiscoveredCandidate.created_at, DiscoveredCandidate.id)).all()
    # Simule une occurrence issue d'un dictionnaire sans normalized_value.
    candidates[-1].normalized_value = None
    session.add(candidates[-1])
    session.flush()
    result_a = EntityResolutionService(session).resolve([candidate.id for candidate in candidates])
    result_b = EntityResolutionService(session).resolve([candidate.id for candidate in candidates])
    assert result_a.entity_count == 3 and result_a.membership_count == 4
    assert result_a.processing_run_id != result_b.processing_run_id
    assert session.exec(select(Entity).where(Entity.processing_run_id == result_a.processing_run_id)).all()[0].processing_run_id != result_b.processing_run_id
    assert session.exec(select(EntityResolutionEvidence)).all()


def test_entity_resolution_evidence_categories_and_confidence(session: Session):
    from kaliok.entity_resolution import EntityResolutionService
    version, normalization, _ = _normalized_generation(session, ("Nouméa NOUMÉA Alpha Beta",))
    discovery = CandidateDiscoveryService(session).discover(version.id, normalization.id, [_detector(
        LexicalTerm("Nouméa", "place", normalized_value="Nouméa"),
        LexicalTerm("NOUMÉA", "place", normalized_value="Nouméa"),
        LexicalTerm("Alpha", "kind", normalized_value="Alpha"),
        LexicalTerm("Beta", "kind"),
    )])
    candidates = session.exec(select(DiscoveredCandidate).where(DiscoveredCandidate.processing_run_id == discovery.processing_run_id).order_by(DiscoveredCandidate.created_at, DiscoveredCandidate.id)).all()
    candidates[-1].normalized_value = None
    session.add(candidates[-1])
    session.flush()
    result = EntityResolutionService(session).resolve([candidate.id for candidate in candidates])
    memberships = session.exec(select(EntityMembership).where(EntityMembership.processing_run_id == result.processing_run_id)).all()
    entities = {item.id: item for item in session.exec(select(Entity).where(Entity.processing_run_id == result.processing_run_id)).all()}
    evidences = session.exec(select(EntityResolutionEvidence).where(EntityResolutionEvidence.membership_id.in_([item.id for item in memberships]))).all()
    by_type = {entities[membership.entity_id].entity_type: [] for membership in memberships}
    for membership in memberships:
        by_type.setdefault(entities[membership.entity_id].entity_type, []).append(next(item for item in evidences if item.membership_id == membership.id))
    assert {item.signal_key for item in by_type["place"]} == {"normalized_value_exact"}
    place_entity_ids = {membership.entity_id for membership in memberships if entities[membership.entity_id].entity_type == "place"}
    assert len(place_entity_ids) == 1 and all(item.score == 1.0 for item in by_type["place"])
    assert entities[next(iter(place_entity_ids))].confidence == 1.0
    singleton_signals = {item.signal_key: item for values in by_type.values() for item in values if item.signal_key != "normalized_value_exact"}
    assert singleton_signals["singleton_normalized_value"].score is None
    assert singleton_signals["singleton_no_normalized_value"].score is None


def test_entity_resolution_read_service_scope_filters_counts_and_provenance(session: Session):
    from kaliok.entity_resolution import EntityResolutionReadService, EntityResolutionService
    first, normalization, _ = _normalized_generation(session, ("<script>alert(1)</script> Nouméa NOUMÉA",))
    discovery = CandidateDiscoveryService(session).discover(first.id, normalization.id, [_detector(
        LexicalTerm("Nouméa", "place_name", normalized_value="Nouméa"),
        LexicalTerm("NOUMÉA", "place_name", normalized_value="Nouméa"),
    )])
    candidates = session.exec(select(DiscoveredCandidate).where(
        DiscoveredCandidate.processing_run_id == discovery.processing_run_id
    ).order_by(DiscoveredCandidate.created_at, DiscoveredCandidate.id)).all()
    extra_source = session.exec(select(CandidateSourceFragment).where(
        CandidateSourceFragment.candidate_id == candidates[1].id
    )).one()
    session.add(CandidateSourceFragment(
        candidate_id=candidates[0].id,
        normalized_content_unit_id=extra_source.normalized_content_unit_id,
        fragment_order=1,
        start_offset=extra_source.start_offset,
        end_offset=extra_source.end_offset,
        exact_text=extra_source.exact_text,
    ))
    session.flush()
    result = EntityResolutionService(session).resolve([item.id for item in candidates])
    session.add(ProcessingRun(document_version_id=first.id, process_type="candidate_discovery", status="running"))
    session.flush()

    reader = EntityResolutionReadService(session)
    assert reader.list_runs(limit=1, offset=0)["total"] == 1
    assert reader.list_runs(document_version_id=first.id)["items"][0]["run_id"] == result.processing_run_id
    assert reader.list_runs(status="failed")["items"] == []
    run = reader.get_run(result.processing_run_id)
    assert run["summary"] == {
        "entity_count": 1, "membership_count": 2, "evidence_count": 2,
        "singleton_entity_count": 0, "grouped_entity_count": 1,
        "document_version_count": 1,
    }
    assert run["documents"][0]["occurrence_count"] == 2
    entities = reader.list_entities(result.processing_run_id, entity_type="place_name", canonical_label="noum")
    assert entities["total"] == 1
    assert entities["items"][0]["membership_count"] == 2
    assert entities["items"][0]["document_count"] == 1
    assert entities["items"][0]["page_count"] == 1
    detail = reader.get_entity(entities["items"][0]["id"])
    assert len(detail["memberships"]) == 2 and len(detail["evidences"]) == 2
    assert len(detail["memberships"][0]["provenance"]) == 2
    assert detail["memberships"][0]["provenance"][0]["normalized_content_unit"]["sources"][0]["content_block"]["fragments"][0]["filename"] == "discovery.txt"


def test_entity_resolution_read_service_multi_document_scope_and_empty_states(session: Session):
    from kaliok.entity_resolution import EntityResolutionReadService, EntityResolutionService
    first, first_normalization, _ = _normalized_generation(session, ("Cour des comptes",))
    second, second_normalization, _ = _normalized_generation(session, ("COUR DES COMPTES",))
    detector = _detector(LexicalTerm("Cour des comptes", "organization_name", normalized_value="Cour des comptes"))
    a = CandidateDiscoveryService(session).discover(first.id, first_normalization.id, [detector])
    detector_upper = _detector(LexicalTerm("COUR DES COMPTES", "organization_name", normalized_value="Cour des comptes"))
    b = CandidateDiscoveryService(session).discover(second.id, second_normalization.id, [detector_upper])
    candidates = session.exec(select(DiscoveredCandidate).where(
        DiscoveredCandidate.processing_run_id.in_([a.processing_run_id, b.processing_run_id])
    )).all()
    result = EntityResolutionService(session).resolve([item.id for item in candidates])
    run = session.get(ProcessingRun, result.processing_run_id)
    assert run.document_version_id is None
    reader = EntityResolutionReadService(session)
    detail = reader.get_run(run.id)
    assert len(detail["documents"]) == 2
    assert reader.list_runs(document_version_id=second.id)["total"] == 1
    empty = ProcessingRun(document_version_id=None, process_type="entity_resolution", status="completed", metrics={})
    failed = ProcessingRun(document_version_id=None, process_type="entity_resolution", status="failed", error_message="échec")
    running = ProcessingRun(document_version_id=None, process_type="entity_resolution", status="running")
    session.add_all([empty, failed, running]); session.flush()
    assert reader.get_run(empty.id)["summary"]["entity_count"] == 0
    assert reader.list_runs(status="failed")["items"][0]["error_message"] == "échec"
    assert reader.list_runs(status="running")["items"][0]["status"] == "running"
