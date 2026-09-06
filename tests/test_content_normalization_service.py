from __future__ import annotations

from collections.abc import Iterator
from uuid import UUID, uuid4

import pytest
from sqlmodel import Session, select

from kaliok.execution import ExecutionContext
from kaliok.hashing import canonical_json_hash
from kaliok.normalization import (
    ContentNormalizationComparisonService,
    ContentNormalizationService,
)
from kaliok.storage.database import create_database_engine
from kaliok.storage.models import (
    ContentBlock,
    Document,
    DocumentVersion,
    NormalizedContentUnit,
    NormalizedContentUnitSource,
    Page,
    ProcessingRun,
)


@pytest.fixture
def session() -> Iterator[Session]:
    engine = create_database_engine()
    with engine.connect() as connection:
        transaction = connection.begin()
        try:
            with Session(bind=connection) as test_session:
                yield test_session
        finally:
            if transaction.is_active:
                transaction.rollback()
    engine.dispose()


def _version(session: Session) -> DocumentVersion:
    document = Document(title="Normalisation")
    session.add(document)
    session.flush()
    version = DocumentVersion(
        document_id=document.id,
        version_number=1,
        filename="normalization.pdf",
        file_hash=uuid4().hex,
        storage_uri="test://normalization.pdf",
    )
    session.add(version)
    session.flush()
    return version


def _perception_run(session: Session, version: DocumentVersion) -> ProcessingRun:
    run = ProcessingRun(
        document_version_id=version.id,
        process_type="document_extraction",
        status="completed",
    )
    session.add(run)
    session.flush()
    return run


def _run(
    session: Session,
    version: DocumentVersion,
    *,
    process_type: str = "document_extraction",
    status: str = "completed",
) -> ProcessingRun:
    run = ProcessingRun(
        document_version_id=version.id,
        process_type=process_type,
        status=status,
    )
    session.add(run)
    session.flush()
    return run


def _page(
    session: Session,
    version: DocumentVersion,
    run: ProcessingRun,
    number: int,
) -> Page:
    page = Page(
        document_version_id=version.id,
        page_number=number,
        perception_processing_run_id=run.id,
    )
    session.add(page)
    session.flush()
    return page


def _block(
    session: Session,
    page: Page,
    run: ProcessingRun,
    index: int,
    content: str,
    *,
    reading_order: int | None = None,
    block_type: str = "paragraph",
) -> ContentBlock:
    block = ContentBlock(
        page_id=page.id,
        processing_run_id=run.id,
        block_index=index,
        reading_order=reading_order,
        block_type=block_type,
        content=content,
        extraction_method="test",
    )
    session.add(block)
    session.flush()
    return block


def _units(session: Session, run_id: UUID) -> list[NormalizedContentUnit]:
    return list(
        session.exec(
            select(NormalizedContentUnit)
            .where(NormalizedContentUnit.processing_run_id == run_id)
            .order_by(NormalizedContentUnit.unit_index)
        ).all()
    )


def test_unknown_document_version_is_rejected_without_run(session: Session):
    unknown_id = uuid4()
    before = len(
        session.exec(
            select(ProcessingRun).where(
                ProcessingRun.process_type == "content_normalization"
            )
        ).all()
    )
    with pytest.raises(ValueError, match="DocumentVersion inconnue"):
        ContentNormalizationService(session).normalize(unknown_id)
    after = len(
        session.exec(
            select(ProcessingRun).where(
                ProcessingRun.process_type == "content_normalization"
            )
        ).all()
    )
    assert after == before


def test_one_current_block_creates_completed_run_unit_and_source(session: Session):
    version = _version(session)
    perception = _perception_run(session, version)
    page = _page(session, version, perception, 1)
    block = _block(session, page, perception, 0, "Contenu", block_type="text")

    result = ContentNormalizationService(session).normalize(version.id)
    run = session.get(ProcessingRun, result.processing_run_id)
    units = _units(session, result.processing_run_id)
    sources = session.exec(
        select(NormalizedContentUnitSource).where(
            NormalizedContentUnitSource.normalized_content_unit_id == units[0].id
        )
    ).all()

    assert result.document_version_id == version.id
    assert result.unit_count == 1
    assert run is not None and run.status == "completed"
    assert run.completed_at is not None
    assert run.process_type == "content_normalization"
    assert (run.engine, run.engine_version) == ("kaliok", "block-to-unit-v1")
    assert units[0].document_version_id == run.document_version_id == version.id
    assert units[0].content_type == "text"
    assert units[0].source_unit_id == str(block.id)
    assert units[0].source_reference is None and units[0].parent_unit_id is None
    assert len(sources) == 1
    assert sources[0].content_block_id == block.id
    assert sources[0].source_order == 0


def test_blocks_have_deterministic_page_and_reading_order(session: Session):
    version = _version(session)
    perception = _perception_run(session, version)
    second_page = _page(session, version, perception, 2)
    first_page = _page(session, version, perception, 1)
    _block(session, second_page, perception, 0, "page 2", reading_order=0)
    _block(session, first_page, perception, 8, "fallback 8")
    _block(session, first_page, perception, 3, "order 1", reading_order=1)
    _block(session, first_page, perception, 2, "fallback 2")

    result = ContentNormalizationService(session).normalize(version.id)

    assert [(unit.unit_index, unit.content) for unit in _units(session, result.processing_run_id)] == [
        (0, "order 1"),
        (1, "fallback 2"),
        (2, "fallback 8"),
        (3, "page 2"),
    ]


def test_empty_and_historical_blocks_are_skipped_with_exact_metrics(session: Session):
    version = _version(session)
    old_perception = _perception_run(session, version)
    current_perception = _perception_run(session, version)
    page = _page(session, version, current_perception, 1)
    _block(session, page, old_perception, 0, "ancien")
    _block(session, page, current_perception, 1, "   \n")
    _block(session, page, current_perception, 2, "courant")

    result = ContentNormalizationService(session).normalize(version.id)
    run = session.get(ProcessingRun, result.processing_run_id)

    assert [unit.content for unit in _units(session, result.processing_run_id)] == [
        "courant"
    ]
    assert run is not None
    assert run.metrics == {
        "source_block_count": 2,
        "normalized_unit_count": 1,
        "skipped_empty_block_count": 1,
    }


def test_two_runs_create_distinct_generations_and_preserve_historical_units(
    session: Session,
):
    version = _version(session)
    perception = _perception_run(session, version)
    page = _page(session, version, perception, 1)
    block = _block(session, page, perception, 0, "courant")
    historical = NormalizedContentUnit(
        document_version_id=version.id,
        unit_index=99,
        content_type="legacy",
        content="historique",
        source_unit_id="legacy-99",
    )
    session.add(historical)
    session.flush()

    first = ContentNormalizationService(session).normalize(version.id)
    second = ContentNormalizationService(session).normalize(version.id)

    assert first.processing_run_id != second.processing_run_id
    assert [unit.unit_index for unit in _units(session, first.processing_run_id)] == [0]
    assert [unit.unit_index for unit in _units(session, second.processing_run_id)] == [0]
    assert session.get(NormalizedContentUnit, historical.id) is historical
    assert historical.processing_run_id is None
    sources = session.exec(
        select(NormalizedContentUnitSource).where(
            NormalizedContentUnitSource.content_block_id == block.id
        )
    ).all()
    assert len(sources) == 2
    assert len({source.normalized_content_unit_id for source in sources}) == 2


def test_no_usable_current_block_creates_failed_run(session: Session):
    version = _version(session)
    perception = _perception_run(session, version)
    page = _page(session, version, perception, 1)
    _block(session, page, perception, 0, "\t")

    with pytest.raises(ValueError, match="Aucun ContentBlock exploitable"):
        ContentNormalizationService(session).normalize(version.id)

    runs = session.exec(
        select(ProcessingRun).where(
            ProcessingRun.process_type == "content_normalization",
            ProcessingRun.document_version_id == version.id,
        )
    ).all()
    assert len(runs) == 1
    assert runs[0].status == "failed"
    assert runs[0].completed_at is not None
    assert runs[0].error_message
    assert runs[0].metrics["skipped_empty_block_count"] == 1
    assert _units(session, runs[0].id) == []


def test_persistence_error_rolls_back_units_and_marks_run_failed(
    session: Session,
    monkeypatch,
):
    version = _version(session)
    perception = _perception_run(session, version)
    page = _page(session, version, perception, 1)
    _block(session, page, perception, 0, "contenu")

    def fail_persistence(self, **kwargs):
        raise RuntimeError("échec volontaire")

    monkeypatch.setattr(ContentNormalizationService, "_persist_unit", fail_persistence)

    with pytest.raises(RuntimeError, match="échec volontaire"):
        ContentNormalizationService(session).normalize(version.id)

    run = session.exec(
        select(ProcessingRun).where(
            ProcessingRun.process_type == "content_normalization",
            ProcessingRun.document_version_id == version.id,
        )
    ).one()
    assert run.status == "failed"
    assert run.completed_at is not None
    assert run.error_message == "échec volontaire"
    assert _units(session, run.id) == []


def test_block_loading_error_marks_run_failed_with_zero_metrics(
    session: Session,
    monkeypatch,
):
    version = _version(session)

    def fail_loading(self, version_id):
        raise RuntimeError("chargement impossible")

    monkeypatch.setattr(ContentNormalizationService, "_current_blocks", fail_loading)
    with pytest.raises(RuntimeError, match="chargement impossible"):
        ContentNormalizationService(session).normalize(version.id)

    run = session.exec(
        select(ProcessingRun).where(
            ProcessingRun.process_type == "content_normalization",
            ProcessingRun.document_version_id == version.id,
        )
    ).one()
    assert run.status == "failed"
    assert run.completed_at is not None
    assert run.error_message == "chargement impossible"
    assert run.metrics == {
        "source_block_count": 0,
        "normalized_unit_count": 0,
        "skipped_empty_block_count": 0,
    }


@pytest.mark.parametrize(
    ("run_kwargs", "message"),
    [
        ({"process_type": "wrong"}, "n'est pas un run de perception"),
        ({"status": "running"}, "doit être completed"),
    ],
)
def test_explicit_perception_run_is_validated_before_normalization(
    session: Session,
    run_kwargs: dict[str, str],
    message: str,
):
    version = _version(session)
    perception = _run(session, version, **run_kwargs)

    with pytest.raises(ValueError, match=message):
        ContentNormalizationService(session).normalize(
            version.id,
            perception_processing_run_id=perception.id,
        )


def test_explicit_perception_run_rejects_unknown_and_other_document(
    session: Session,
):
    version = _version(session)
    other_version = _version(session)
    other_run = _run(session, other_version)
    service = ContentNormalizationService(session)

    with pytest.raises(ValueError, match="ProcessingRun de perception inconnu"):
        service.normalize(version.id, perception_processing_run_id=uuid4())
    with pytest.raises(ValueError, match="autre DocumentVersion"):
        service.normalize(version.id, perception_processing_run_id=other_run.id)


def test_explicit_perception_run_rejects_missing_blocks_without_current_fallback(
    session: Session,
):
    version = _version(session)
    selected = _perception_run(session, version)
    current = _perception_run(session, version)
    page = _page(session, version, current, 1)
    _block(session, page, current, 0, "current")

    with pytest.raises(ValueError, match="Aucun ContentBlock associé"):
        ContentNormalizationService(session).normalize(
            version.id,
            perception_processing_run_id=selected.id,
        )


def test_explicit_perception_ignores_current_page_pointer(session: Session):
    version = _version(session)
    selected = _perception_run(session, version)
    current = _perception_run(session, version)
    page = _page(session, version, current, 1)
    selected_block = _block(session, page, selected, 0, "selected")
    _block(session, page, current, 1, "current")

    result = ContentNormalizationService(session).normalize(
        version.id,
        perception_processing_run_id=selected.id,
    )
    units = _units(session, result.processing_run_id)
    page_after = session.get(Page, page.id)

    assert [unit.content for unit in units] == ["selected"]
    assert page_after is not None and page_after.perception_processing_run_id == current.id
    source = session.exec(
        select(NormalizedContentUnitSource).where(
            NormalizedContentUnitSource.normalized_content_unit_id == units[0].id
        )
    ).one()
    assert source.content_block_id == selected_block.id


def test_p_and_a_normalize_same_explicit_perception_as_distinct_generations(
    session: Session,
):
    version = _version(session)
    perception = _perception_run(session, version)
    page = _page(session, version, perception, 1)
    first_block = _block(session, page, perception, 0, "P/A partagé")
    second_block = _block(session, page, perception, 1, "Deuxième bloc")
    p_group = uuid4()
    a_group = uuid4()

    p = ContentNormalizationService(session).normalize(
        version.id,
        perception_processing_run_id=perception.id,
        execution_context=ExecutionContext(
            environment="production",
            execution_group_id=p_group,
        ),
    )
    a = ContentNormalizationService(session).normalize(
        version.id,
        perception_processing_run_id=perception.id,
        execution_context=ExecutionContext(
            environment="experiment",
            execution_group_id=a_group,
        ),
    )
    p_run = session.get(ProcessingRun, p.processing_run_id)
    a_run = session.get(ProcessingRun, a.processing_run_id)
    p_units = _units(session, p.processing_run_id)
    a_units = _units(session, a.processing_run_id)

    assert p.processing_run_id != a.processing_run_id
    assert p_run is not None and a_run is not None
    assert p_run.execution_environment == "production"
    assert a_run.execution_environment == "experiment"
    assert p_run.execution_group_id == p_group
    assert a_run.execution_group_id == a_group
    assert p_run.execution_group_id != a_run.execution_group_id
    assert p_run.configuration != a_run.configuration
    assert p_run.configuration_hash != a_run.configuration_hash
    assert p_run.configuration["perception_processing_run_id"] == str(perception.id)
    assert a_run.configuration["perception_processing_run_id"] == str(perception.id)
    assert p_run.configuration_hash == canonical_json_hash(p_run.configuration)
    assert a_run.configuration_hash == canonical_json_hash(a_run.configuration)
    assert p.unit_count == a.unit_count == 2
    assert [unit.content for unit in p_units] == [unit.content for unit in a_units]
    assert {unit.id for unit in p_units}.isdisjoint({unit.id for unit in a_units})
    p_sources = session.exec(
        select(NormalizedContentUnitSource).where(
            NormalizedContentUnitSource.normalized_content_unit_id.in_(
                [unit.id for unit in p_units]
            )
        )
    ).all()
    a_sources = session.exec(
        select(NormalizedContentUnitSource).where(
            NormalizedContentUnitSource.normalized_content_unit_id.in_(
                [unit.id for unit in a_units]
            )
        )
    ).all()
    assert {source.content_block_id for source in p_sources} == {
        first_block.id,
        second_block.id,
    }
    assert {source.content_block_id for source in a_sources} == {
        first_block.id,
        second_block.id,
    }


def test_configuration_hash_changes_when_explicit_perception_changes(session: Session):
    version = _version(session)
    first_perception = _perception_run(session, version)
    second_perception = _perception_run(session, version)
    first_page = _page(session, version, first_perception, 1)
    second_page = _page(session, version, second_perception, 2)
    _block(session, first_page, first_perception, 0, "première")
    _block(session, second_page, second_perception, 0, "seconde")

    first = ContentNormalizationService(session).normalize(
        version.id,
        perception_processing_run_id=first_perception.id,
        execution_context=ExecutionContext(environment="production"),
    )
    second = ContentNormalizationService(session).normalize(
        version.id,
        perception_processing_run_id=second_perception.id,
        execution_context=ExecutionContext(environment="experiment"),
    )
    first_run = session.get(ProcessingRun, first.processing_run_id)
    second_run = session.get(ProcessingRun, second.processing_run_id)

    assert first_run is not None and second_run is not None
    assert first_run.configuration_hash != second_run.configuration_hash


def test_legacy_normalization_without_explicit_perception_keeps_current_behavior(
    session: Session,
):
    version = _version(session)
    old_perception = _perception_run(session, version)
    current_perception = _perception_run(session, version)
    page = _page(session, version, current_perception, 1)
    _block(session, page, old_perception, 0, "ancien")
    _block(session, page, current_perception, 1, "courant")

    result = ContentNormalizationService(session).normalize(version.id)

    assert [unit.content for unit in _units(session, result.processing_run_id)] == [
        "courant"
    ]


def test_normalization_comparison_returns_metrics_and_delta(session: Session):
    version = _version(session)
    perception = _perception_run(session, version)
    page = _page(session, version, perception, 1)
    _block(session, page, perception, 0, "un", block_type="paragraph")
    _block(session, page, perception, 1, "deux", block_type="heading")
    service = ContentNormalizationService(session)
    p = service.normalize(
        version.id,
        perception_processing_run_id=perception.id,
        execution_context=ExecutionContext(environment="production"),
    )
    a = service.normalize(
        version.id,
        perception_processing_run_id=perception.id,
        execution_context=ExecutionContext(environment="experiment"),
    )

    compared = ContentNormalizationComparisonService(session).compare(
        p.processing_run_id,
        a.processing_run_id,
    )

    assert compared["run_p"]["run_id"] == p.processing_run_id
    assert compared["run_a"]["run_id"] == a.processing_run_id
    assert compared["run_p"]["metrics"]["unit_count"] == 2
    assert compared["run_p"]["metrics"]["total_characters"] == 6
    assert compared["run_p"]["metrics"]["content_type_counts"] == {
        "heading": 1,
        "paragraph": 1,
    }
    assert compared["delta"]["unit_count"] == 0
    assert compared["delta"]["total_characters"] == 0
    assert compared["delta"]["content_type_counts"] == {
        "heading": 0,
        "paragraph": 0,
    }


def test_normalization_comparison_reports_different_results(session: Session):
    version = _version(session)
    first_perception = _perception_run(session, version)
    second_perception = _perception_run(session, version)
    first_page = _page(session, version, first_perception, 1)
    second_page = _page(session, version, second_perception, 2)
    _block(session, first_page, first_perception, 0, "court", block_type="paragraph")
    _block(
        session,
        second_page,
        second_perception,
        0,
        "beaucoup plus long",
        block_type="heading",
    )
    service = ContentNormalizationService(session)
    first = service.normalize(
        version.id,
        perception_processing_run_id=first_perception.id,
        execution_context=ExecutionContext(environment="production"),
    )
    second = service.normalize(
        version.id,
        perception_processing_run_id=second_perception.id,
        execution_context=ExecutionContext(environment="experiment"),
    )

    compared = ContentNormalizationComparisonService(session).compare(
        first.processing_run_id,
        second.processing_run_id,
    )

    assert compared["delta"]["unit_count"] == 0
    assert compared["delta"]["total_characters"] == 13
    assert compared["delta"]["content_type_counts"] == {
        "heading": 1,
        "paragraph": -1,
    }


def test_normalization_comparison_rejects_invalid_or_cross_document_runs(
    session: Session,
):
    first_version = _version(session)
    second_version = _version(session)
    first_run = _run(session, first_version, process_type="content_normalization")
    second_run = _run(session, second_version, process_type="content_normalization")
    service = ContentNormalizationComparisonService(session)

    with pytest.raises(ValueError, match="même document"):
        service.compare(first_run.id, second_run.id)
    with pytest.raises(ValueError, match="runs completed"):
        service.compare(
            first_run.id,
            _run(
                session,
                first_version,
                process_type="content_normalization",
                status="running",
            ).id,
        )
    with pytest.raises(ValueError, match="content_normalization"):
        service.compare(first_run.id, _run(session, first_version).id)
