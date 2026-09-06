from __future__ import annotations

from collections.abc import Iterator
from uuid import UUID, uuid4

import pytest
from sqlmodel import Session, select

from kaliok.execution import ExecutionContext
from kaliok.hashing import canonical_json_hash
from kaliok.documents.models import DocumentContent, DocumentPage, TextBlock
from kaliok.indexing.service import (
    PERCEPTION_ENGINE,
    PERCEPTION_VERSION,
    store_document_perception,
)
from kaliok.normalization import ContentNormalizationService
from kaliok.pipeline import (
    ComponentBinding,
    ComponentDefinition,
    ComponentRegistry,
    ComponentRuntimeRegistry,
    ManifestExecutionService,
    PipelineManifest,
    build_current_production_manifest,
    build_kaliok_component_registry,
    build_kaliok_runtime_registry,
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


def _document_version(session: Session) -> DocumentVersion:
    document = Document(title="Pipeline runtime")
    session.add(document)
    session.flush()
    version = DocumentVersion(
        document_id=document.id,
        version_number=1,
        filename="pipeline-runtime.pdf",
        file_hash=uuid4().hex,
        storage_uri="test://pipeline-runtime.pdf",
        page_count=1,
    )
    session.add(version)
    session.flush()
    return version


def _perception(session: Session, version: DocumentVersion) -> tuple[ProcessingRun, Page, ContentBlock]:
    run = ProcessingRun(
        document_version_id=version.id,
        process_type="document_extraction",
        status="completed",
        engine="synthetic-test-perception",
        engine_version="test-v1",
    )
    session.add(run)
    session.flush()
    page = Page(
        document_version_id=version.id,
        page_number=1,
        perception_processing_run_id=run.id,
    )
    session.add(page)
    session.flush()
    block = ContentBlock(
        page_id=page.id,
        processing_run_id=run.id,
        block_index=0,
        reading_order=0,
        block_type="paragraph",
        content="Source runtime",
        extraction_method="synthetic-test",
    )
    session.add(block)
    session.flush()
    return run, page, block


def _normalization_manifest(*, pipeline_key: str = "pipeline-a") -> PipelineManifest:
    return PipelineManifest(
        pipeline_key=pipeline_key,
        revision="1",
        bindings=(
            ComponentBinding(
                binding_key="normalize",
                component_key="kaliok-normalizer",
                component_version="block-to-unit-v1",
                capabilities=("normalization",),
            ),
        ),
    )


def _document_pipeline_manifest() -> PipelineManifest:
    return PipelineManifest(
        pipeline_key="pipeline-a",
        revision="1",
        bindings=(
            ComponentBinding(
                binding_key="perception",
                component_key=PERCEPTION_ENGINE,
                component_version=PERCEPTION_VERSION,
                capabilities=("document_extraction",),
            ),
            ComponentBinding(
                binding_key="normalization",
                component_key="kaliok-normalizer",
                component_version="block-to-unit-v1",
                capabilities=("normalization",),
                dependencies=("perception",),
            ),
        ),
    )


def _mock_document_content() -> DocumentContent:
    return DocumentContent(
        source="runtime-fixture",
        page_count=1,
        pages=[DocumentPage(page=1, perception_mode="native")],
        blocks=[
            TextBlock(
                text="Perception A",
                page=1,
                block_type="paragraph",
                reading_order=0,
                extraction_method="synthetic-boundary",
            ),
        ],
    )


def test_real_registry_contains_only_identifiable_stable_components():
    registry = build_kaliok_component_registry()

    perception = registry.get(PERCEPTION_ENGINE, PERCEPTION_VERSION)
    normalizer = registry.get("kaliok-normalizer", "block-to-unit-v1")

    assert perception is not None
    assert perception.provides == ("document_extraction",)
    assert normalizer is not None
    assert normalizer.provides == ("normalization",)
    assert registry.get("docling", "unknown-product-version") is None


def test_current_production_manifest_is_factually_partial_and_validates():
    manifest = build_current_production_manifest()

    manifest.validate(build_kaliok_component_registry())

    assert manifest.pipeline_key == "pipeline-p"
    assert [binding.binding_key for binding in manifest.bindings] == [
        "perception",
        "normalization",
    ]
    assert manifest.bindings[0].component_key == PERCEPTION_ENGINE
    assert manifest.bindings[1].component_key == "kaliok-normalizer"


def test_manifest_runtime_executes_real_normalization_adapter_and_traces_snapshot(
    session: Session,
    monkeypatch,
):
    version = _document_version(session)
    perception, page, block = _perception(session, version)
    manifest = _normalization_manifest()
    context = ExecutionContext(environment="experiment", execution_group_id=uuid4())
    calls: list[dict[str, object]] = []
    original_normalize = ContentNormalizationService.normalize

    def spy_normalize(self, document_version_id, **kwargs):
        calls.append({"document_version_id": document_version_id, **kwargs})
        return original_normalize(self, document_version_id, **kwargs)

    monkeypatch.setattr(ContentNormalizationService, "normalize", spy_normalize)
    session.commit = lambda: (_ for _ in ()).throw(AssertionError("commit interdit"))

    result = ManifestExecutionService(
        build_kaliok_component_registry(),
        build_kaliok_runtime_registry(),
    ).execute_capability(
        session,
        manifest=manifest,
        capability="normalization",
        document_version_id=version.id,
        perception_processing_run_id=perception.id,
        execution_context=context,
    )
    run = session.get(ProcessingRun, result.processing_run_id)
    units = list(
        session.exec(
            select(NormalizedContentUnit).where(
                NormalizedContentUnit.processing_run_id == result.processing_run_id
            )
        ).all()
    )
    source = session.exec(
        select(NormalizedContentUnitSource).where(
            NormalizedContentUnitSource.normalized_content_unit_id == units[0].id
        )
    ).one()

    assert len(calls) == 1
    assert calls[0]["perception_processing_run_id"] == perception.id
    assert result.component_key == "kaliok-normalizer"
    assert result.manifest_hash == manifest.manifest_hash
    assert run is not None
    assert run.execution_environment == "experiment"
    assert run.execution_group_id == context.execution_group_id
    assert run.configuration["pipeline"] == {
        "pipeline_key": "pipeline-a",
        "revision": "1",
        "manifest_hash": manifest.manifest_hash,
        "binding_key": "normalize",
        "component_key": "kaliok-normalizer",
        "component_version": "block-to-unit-v1",
        "capability": "normalization",
        "binding_configuration": {},
    }
    assert run.configuration_hash == canonical_json_hash(run.configuration)
    assert run.configuration["perception_processing_run_id"] == str(perception.id)
    assert len(units) == 1
    assert units[0].content == block.content
    assert source.content_block_id == block.id
    page_after = session.get(Page, page.id)
    assert page_after is not None and page_after.perception_processing_run_id == perception.id


def test_manifest_pipeline_chains_real_perception_to_normalization_without_promotion(
    session: Session,
    monkeypatch,
):
    version = _document_version(session)
    production_perception, page, production_block = _perception(session, version)
    manifest = _document_pipeline_manifest()
    context = ExecutionContext(environment="experiment", execution_group_id=uuid4())
    monkeypatch.setattr(
        "kaliok.pipeline.runtime.read_document",
        lambda path: _mock_document_content(),
    )
    session.commit = lambda: (_ for _ in ()).throw(AssertionError("commit interdit"))

    result = ManifestExecutionService(
        build_kaliok_component_registry(),
        build_kaliok_runtime_registry(),
    ).execute_document_pipeline(
        session,
        manifest=manifest,
        document_version_id=version.id,
        execution_context=context,
    )
    perception_run = session.get(
        ProcessingRun,
        result.perception.processing_run_id,
    )
    normalization_run = session.get(
        ProcessingRun,
        result.normalization.processing_run_id,
    )
    perception_blocks = session.exec(
        select(ContentBlock).where(
            ContentBlock.processing_run_id == result.perception.processing_run_id
        )
    ).all()
    normalized_units = session.exec(
        select(NormalizedContentUnit).where(
            NormalizedContentUnit.processing_run_id
            == result.normalization.processing_run_id
        )
    ).all()
    source = session.exec(
        select(NormalizedContentUnitSource).where(
            NormalizedContentUnitSource.normalized_content_unit_id
            == normalized_units[0].id
        )
    ).one()

    assert result.perception.processing_run_id != result.normalization.processing_run_id
    assert perception_run is not None and normalization_run is not None
    assert perception_run.process_type == "document_extraction"
    assert normalization_run.process_type == "content_normalization"
    assert perception_run.execution_environment == "experiment"
    assert normalization_run.execution_environment == "experiment"
    assert perception_run.execution_group_id == context.execution_group_id
    assert normalization_run.execution_group_id == context.execution_group_id
    assert perception_run.configuration["pipeline"]["manifest_hash"] == manifest.manifest_hash
    assert normalization_run.configuration["pipeline"]["manifest_hash"] == manifest.manifest_hash
    assert perception_run.configuration_hash == canonical_json_hash(perception_run.configuration)
    assert normalization_run.configuration_hash == canonical_json_hash(normalization_run.configuration)
    assert perception_run.configuration["pipeline"]["binding_key"] == "perception"
    assert normalization_run.configuration["pipeline"]["binding_key"] == "normalization"
    assert perception_run.configuration["pipeline"]["capability"] == "document_extraction"
    assert normalization_run.configuration["pipeline"]["capability"] == "normalization"
    assert perception_run.configuration["pipeline"]["component_key"] == PERCEPTION_ENGINE
    assert normalization_run.configuration["pipeline"]["component_key"] == "kaliok-normalizer"
    assert len(perception_blocks) == 1
    assert perception_blocks[0].content == "Perception A"
    assert perception_blocks[0].processing_run_id == perception_run.id
    assert len(normalized_units) == 1
    assert normalized_units[0].content == perception_blocks[0].content
    assert source.content_block_id == perception_blocks[0].id
    assert source.content_block_id != production_block.id
    page_after = session.get(Page, page.id)
    assert page_after is not None
    assert page_after.perception_processing_run_id == production_perception.id


def test_document_perception_default_still_promotes_current_pointer(session: Session):
    version = _document_version(session)
    content = _mock_document_content()

    run = store_document_perception(session, version, content)

    pages = session.exec(
        select(Page).where(Page.document_version_id == version.id)
    ).all()
    assert len(pages) == 1
    assert pages[0].perception_processing_run_id == run.id


def test_manifest_pipeline_rejects_incompatible_perception_source(session: Session):
    version = _document_version(session)
    other_version = _document_version(session)
    other_perception, _, _ = _perception(session, other_version)
    manifest = _normalization_manifest()
    runner = ManifestExecutionService(
        build_kaliok_component_registry(),
        build_kaliok_runtime_registry(),
    )

    with pytest.raises(ValueError, match="autre DocumentVersion"):
        runner.execute_normalization(
            session,
            manifest=manifest,
            document_version_id=version.id,
            perception_processing_run_id=other_perception.id,
            execution_context=ExecutionContext(environment="experiment"),
        )


def test_manifest_pipeline_rollback_removes_both_stages_and_artifacts(
    session: Session,
    monkeypatch,
):
    version = _document_version(session)
    _, page, _ = _perception(session, version)
    manifest = _document_pipeline_manifest()
    monkeypatch.setattr(
        "kaliok.pipeline.runtime.read_document",
        lambda path: _mock_document_content(),
    )
    runner = ManifestExecutionService(
        build_kaliok_component_registry(),
        build_kaliok_runtime_registry(),
    )
    before = {
        "runs": session.exec(select(ProcessingRun)).all(),
        "blocks": session.exec(select(ContentBlock)).all(),
        "units": session.exec(select(NormalizedContentUnit)).all(),
        "sources": session.exec(select(NormalizedContentUnitSource)).all(),
    }
    savepoint = session.begin_nested()

    runner.execute_document_pipeline(
        session,
        manifest=manifest,
        document_version_id=version.id,
        execution_context=ExecutionContext(environment="experiment"),
    )
    savepoint.rollback()

    assert session.exec(select(ProcessingRun)).all() == before["runs"]
    assert session.exec(select(ContentBlock)).all() == before["blocks"]
    assert session.exec(select(NormalizedContentUnit)).all() == before["units"]
    assert session.exec(select(NormalizedContentUnitSource)).all() == before["sources"]
    page_after = session.get(Page, page.id)
    assert page_after is not None


def test_same_manifest_repeats_with_distinct_groups_and_artifacts(session: Session):
    version = _document_version(session)
    perception, _, _ = _perception(session, version)
    manifest = _normalization_manifest()
    runner = ManifestExecutionService(
        build_kaliok_component_registry(),
        build_kaliok_runtime_registry(),
    )
    first = runner.execute_normalization(
        session,
        manifest=manifest,
        document_version_id=version.id,
        perception_processing_run_id=perception.id,
        execution_context=ExecutionContext(
            environment="experiment",
            execution_group_id=uuid4(),
        ),
    )
    second = runner.execute_normalization(
        session,
        manifest=manifest,
        document_version_id=version.id,
        perception_processing_run_id=perception.id,
        execution_context=ExecutionContext(
            environment="experiment",
            execution_group_id=uuid4(),
        ),
    )
    first_run = session.get(ProcessingRun, first.processing_run_id)
    second_run = session.get(ProcessingRun, second.processing_run_id)

    assert first.manifest_hash == second.manifest_hash
    assert first.processing_run_id != second.processing_run_id
    assert first_run is not None and second_run is not None
    assert first_run.execution_group_id != second_run.execution_group_id
    assert first_run.configuration == second_run.configuration
    first_unit = session.exec(
        select(NormalizedContentUnit).where(
            NormalizedContentUnit.processing_run_id == first.processing_run_id
        )
    ).one()
    second_unit = session.exec(
        select(NormalizedContentUnit).where(
            NormalizedContentUnit.processing_run_id == second.processing_run_id
        )
    ).one()
    assert first_unit.id != second_unit.id


def test_same_manifest_repeats_full_pipeline_with_distinct_perceptions(
    session: Session,
    monkeypatch,
):
    version = _document_version(session)
    production_perception, page, production_block = _perception(session, version)
    manifest = _document_pipeline_manifest()
    monkeypatch.setattr(
        "kaliok.pipeline.runtime.read_document",
        lambda path: _mock_document_content(),
    )
    runner = ManifestExecutionService(
        build_kaliok_component_registry(),
        build_kaliok_runtime_registry(),
    )
    first_group_id = uuid4()
    second_group_id = uuid4()
    first = runner.execute_document_pipeline(
        session,
        manifest=manifest,
        document_version_id=version.id,
        execution_context=ExecutionContext(
            environment="experiment",
            execution_group_id=first_group_id,
        ),
    )
    second = runner.execute_document_pipeline(
        session,
        manifest=manifest,
        document_version_id=version.id,
        execution_context=ExecutionContext(
            environment="experiment",
            execution_group_id=second_group_id,
        ),
    )
    first_perception_run = session.get(ProcessingRun, first.perception.processing_run_id)
    second_perception_run = session.get(ProcessingRun, second.perception.processing_run_id)

    first_perception_blocks = session.exec(
        select(ContentBlock).where(
            ContentBlock.processing_run_id == first.perception.processing_run_id
        )
    ).all()
    second_perception_blocks = session.exec(
        select(ContentBlock).where(
            ContentBlock.processing_run_id == second.perception.processing_run_id
        )
    ).all()
    first_units = session.exec(
        select(NormalizedContentUnit).where(
            NormalizedContentUnit.processing_run_id
            == first.normalization.processing_run_id
        )
    ).all()
    second_units = session.exec(
        select(NormalizedContentUnit).where(
            NormalizedContentUnit.processing_run_id
            == second.normalization.processing_run_id
        )
    ).all()
    page_after = session.get(Page, page.id)

    assert first.perception.manifest_hash == second.perception.manifest_hash
    assert first.normalization.manifest_hash == second.normalization.manifest_hash
    assert first.perception.processing_run_id != second.perception.processing_run_id
    assert first.normalization.processing_run_id != second.normalization.processing_run_id
    assert first_perception_run is not None and second_perception_run is not None
    assert first_perception_run.execution_group_id == first_group_id
    assert second_perception_run.execution_group_id == second_group_id
    assert first_perception_run.execution_group_id != second_perception_run.execution_group_id
    assert first_perception_blocks[0].id != second_perception_blocks[0].id
    assert first_units[0].id != second_units[0].id
    assert first_units[0].source_unit_id == str(first_perception_blocks[0].id)
    assert second_units[0].source_unit_id == str(second_perception_blocks[0].id)
    assert first_perception_blocks[0].id != production_block.id
    assert second_perception_blocks[0].id != production_block.id
    assert page_after is not None
    assert page_after.perception_processing_run_id == production_perception.id


def test_runtime_resolves_one_multi_capability_binding_for_each_capability():
    component = ComponentDefinition(
        component_key="test-multitool",
        version="1",
        provides=("document_extraction", "normalization"),
    )
    components = ComponentRegistry([component])
    runtimes = ComponentRuntimeRegistry()
    adapter = object()
    runtimes.register("test-multitool", "1", adapter)  # type: ignore[arg-type]
    manifest = PipelineManifest(
        pipeline_key="multi",
        revision="1",
        bindings=(
            ComponentBinding(
                binding_key="multi",
                component_key="test-multitool",
                component_version="1",
                capabilities=("document_extraction", "normalization"),
            ),
        ),
    )

    manifest.validate(components)
    assert runtimes.resolve("test-multitool", "1") is adapter
    assert "normalization" in manifest.bindings[0].capabilities
    assert "document_extraction" in manifest.bindings[0].capabilities


def test_runtime_rejects_missing_capability_ambiguous_binding_and_adapter(
    session: Session,
):
    version = _document_version(session)
    perception, _, _ = _perception(session, version)
    definitions = [
        ComponentDefinition(
            component_key="normalizer-a",
            version="1",
            provides=("normalization",),
        ),
        ComponentDefinition(
            component_key="normalizer-b",
            version="1",
            provides=("normalization",),
        ),
        ComponentDefinition(
            component_key="extractor",
            version="1",
            provides=("document_extraction",),
        ),
    ]
    components = ComponentRegistry(definitions)
    manifest = PipelineManifest(
        pipeline_key="ambiguous",
        revision="1",
        bindings=(
            ComponentBinding(
                binding_key="a",
                component_key="normalizer-a",
                component_version="1",
                capabilities=("normalization",),
            ),
            ComponentBinding(
                binding_key="b",
                component_key="normalizer-b",
                component_version="1",
                capabilities=("normalization",),
            ),
        ),
    )
    runner = ManifestExecutionService(components, ComponentRuntimeRegistry())

    with pytest.raises(ValueError, match="Plusieurs bindings"):
        runner.execute_normalization(
            session,
            manifest=manifest,
            document_version_id=version.id,
            perception_processing_run_id=perception.id,
            execution_context=ExecutionContext(environment="experiment"),
        )

    missing = PipelineManifest(
        pipeline_key="missing",
        revision="1",
        bindings=(
                ComponentBinding(
                    binding_key="other",
                    component_key="extractor",
                component_version="1",
                capabilities=("document_extraction",),
            ),
        ),
    )
    with pytest.raises(ValueError, match="Aucun binding actif"):
        runner.execute_capability(
            session,
            manifest=missing,
            capability="normalization",
            document_version_id=version.id,
            perception_processing_run_id=perception.id,
            execution_context=ExecutionContext(environment="experiment"),
        )

    single = PipelineManifest(
        pipeline_key="single",
        revision="1",
        bindings=(
            ComponentBinding(
                binding_key="a",
                component_key="normalizer-a",
                component_version="1",
                capabilities=("normalization",),
            ),
        ),
    )
    with pytest.raises(ValueError, match="Runtime adapter absent"):
        runner.execute_normalization(
            session,
            manifest=single,
            document_version_id=version.id,
            perception_processing_run_id=perception.id,
            execution_context=ExecutionContext(environment="experiment"),
        )


def test_runtime_rejects_unsupported_capability_scope():
    with pytest.raises(ValueError, match="exécutable dans ce lot"):
        ManifestExecutionService(
            build_kaliok_component_registry(),
            build_kaliok_runtime_registry(),
        ).execute_capability(
            Session(),
            manifest=_normalization_manifest(),
            capability="embedding",
            document_version_id=UUID(int=0),
            perception_processing_run_id=UUID(int=0),
            execution_context=ExecutionContext(environment="experiment"),
        )
