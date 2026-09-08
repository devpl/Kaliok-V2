from __future__ import annotations

from collections.abc import Iterator
from uuid import uuid4

import pytest
from sqlalchemy import inspect
from sqlmodel import Session, select

from kaliok.pipeline import (
    ComponentBinding,
    PipelineManifest,
    PipelinePersistenceService,
    bootstrap_catalog,
)
from kaliok.pipeline.real_registry import build_kaliok_component_registry
from kaliok.pipeline.runtime import build_kaliok_runtime_registry
from kaliok.ui.core_ui import views
from kaliok.storage.database import create_database_engine
from kaliok.storage.models import (
    Capability,
    Component,
    ComponentCapability,
    ComponentVersion,
    PipelineDefinition,
    PipelineRevision,
    RagTemplate,
    RagTemplateRevision,
)


@pytest.fixture
def catalog_session() -> Iterator[Session]:
    engine = create_database_engine()
    with engine.connect() as connection:
        transaction = connection.begin()
        try:
            if "capabilities" not in inspect(connection).get_table_names():
                pytest.skip("La migration du catalogue n'est pas appliquée dans cette base.")
            with Session(bind=connection) as session:
                yield session
        finally:
            if transaction.is_active:
                transaction.rollback()
    engine.dispose()


def test_bootstrap_is_idempotent_and_loads_active_and_draft(catalog_session: Session):
    first = bootstrap_catalog(catalog_session)
    catalog_session.commit()
    second = bootstrap_catalog(catalog_session)
    catalog_session.commit()

    assert first["active_pipeline_revision_id"] == second["active_pipeline_revision_id"]
    assert first["draft_pipeline_revision_id"] == second["draft_pipeline_revision_id"]
    service = PipelinePersistenceService(catalog_session)
    active = service.active_revision()
    draft = service.draft_revision()
    assert len(service.component_registry().definitions) == 6
    assert active is not None and active.status == "active"
    assert draft is not None and draft.status == "draft"
    assert service.load_manifest(active.id).manifest_hash == active.manifest_hash
    assert len(catalog_session.exec(select(Capability)).all()) == 6
    assert len(catalog_session.exec(select(Component)).all()) == 6
    assert len(catalog_session.exec(select(ComponentVersion)).all()) == 6
    assert len(catalog_session.exec(select(ComponentCapability)).all()) == 6
    assert len(catalog_session.exec(select(RagTemplate)).all()) == 1
    assert len(catalog_session.exec(select(RagTemplateRevision)).all()) == 1
    assert len(catalog_session.exec(select(PipelineDefinition)).all()) == 1
    assert len(catalog_session.exec(select(PipelineRevision)).all()) == 2


def test_multicapability_binding_is_one_binding_and_partial_independent_selection_is_valid(
    catalog_session: Session,
):
    bootstrap_catalog(catalog_session)
    document_extraction = catalog_session.exec(
        select(Capability).where(Capability.capability_key == "document_extraction")
    ).one()
    normalization = catalog_session.exec(
        select(Capability).where(Capability.capability_key == "normalization")
    ).one()
    component = Component(component_key=f"test-multi-{uuid4().hex}", display_name="Test multi")
    catalog_session.add(component)
    catalog_session.flush()
    version = ComponentVersion(component_id=component.id, version="1", status="available")
    catalog_session.add(version)
    catalog_session.flush()
    for capability in (document_extraction, normalization):
        catalog_session.add(
            ComponentCapability(
                component_version_id=version.id,
                capability_id=capability.id,
                invocation_mode="independent",
            )
        )
    catalog_session.flush()
    manifest = PipelineManifest(
        pipeline_key="test-persistence",
        revision="1",
        bindings=(
            ComponentBinding(
                binding_key="multi",
                component_key=component.component_key,
                component_version="1",
                capabilities=("document_extraction",),
            ),
        ),
    )
    service = PipelinePersistenceService(catalog_session)
    revision = service.save_revision(manifest)
    assert len(service.load_manifest(revision.id).bindings) == 1
    assert service.load_manifest(revision.id).bindings[0].capabilities == ("document_extraction",)


def test_db_projection_drives_lab_options_for_every_catalogued_capability(
    catalog_session: Session,
):
    bootstrap_catalog(catalog_session)
    service = PipelinePersistenceService(catalog_session)
    registry = build_kaliok_component_registry(catalog_session)
    runtime = build_kaliok_runtime_registry()
    production = service.load_manifest(service.active_revision("pipeline-p").id)

    payload = views._pipeline_capabilities_payload(registry, runtime, production)
    by_key = {item["key"]: item for item in payload}
    expected = {
        "document_extraction": ("kaliok-reader", "3"),
        "normalization": ("kaliok-normalizer", "block-to-unit-v1"),
        "entity_discovery": ("kaliok-candidate-discovery", "candidate-discovery-v1"),
        "entity_resolution": ("kaliok-entity-resolution", "declared-normalized-exact-v1"),
        "chunking": ("kaliok-semantic-chunker", "llamaindex-semantic-cleaning@1"),
        "indexing": ("postgres-normalized-index", "normalized-content-unit@1"),
    }

    assert set(by_key) == set(expected)
    for capability, identity in expected.items():
        assert [
            (item["component_key"], item["version"])
            for item in by_key[capability]["components"]
        ] == [identity]
        assert by_key[capability]["selected_component"] is None or (
            by_key[capability]["selected_component"]["component_key"] == identity[0]
        )
    assert by_key["entity_discovery"]["components"][0]["runtime_executable"] is True
    assert by_key["chunking"]["components"][0]["runtime_executable"] is False


def test_saving_existing_draft_removes_fk_dependants_before_replacing_bindings(
    catalog_session: Session,
):
    bootstrap_catalog(catalog_session)
    service = PipelinePersistenceService(catalog_session)
    draft = service.draft_revision("pipeline-p")
    active = service.active_revision("pipeline-p")
    manifest = PipelineManifest(
        pipeline_key="pipeline-p",
        revision=str(draft.revision_number),
        bindings=(
            ComponentBinding(
                binding_key="perception",
                component_key="kaliok-reader",
                component_version="3",
                capabilities=("document_extraction",),
            ),
            ComponentBinding(
                binding_key="normalization",
                component_key="kaliok-normalizer",
                component_version="block-to-unit-v1",
                capabilities=("normalization",),
                dependencies=("perception",),
            ),
            ComponentBinding(
                binding_key="binding-3",
                component_key="kaliok-candidate-discovery",
                component_version="candidate-discovery-v1",
                capabilities=("entity_discovery",),
            ),
        ),
    )

    saved = service.save_revision(
        manifest,
        status="draft",
        existing_revision_id=draft.id,
        rag_template_revision_id=draft.rag_template_revision_id,
    )
    loaded = service.load_manifest(saved.id)

    assert [binding.binding_key for binding in loaded.bindings] == [
        "perception",
        "normalization",
        "binding-3",
    ]
    assert loaded.bindings[2].capabilities == ("entity_discovery",)
    assert service.active_revision("pipeline-p").id == active.id
    assert service.load_manifest(active.id).bindings[0].binding_key == "perception"


def test_all_or_none_and_produced_with_bundle_are_validated(catalog_session: Session):
    bootstrap_catalog(catalog_session)
    capabilities = catalog_session.exec(select(Capability)).all()
    first, second = capabilities[:2]
    component = Component(component_key=f"test-bundle-{uuid4().hex}", display_name="Test bundle")
    catalog_session.add(component)
    catalog_session.flush()
    version = ComponentVersion(component_id=component.id, version="1", status="available")
    catalog_session.add(version)
    catalog_session.flush()
    catalog_session.add_all(
        [
            ComponentCapability(
                component_version_id=version.id,
                capability_id=first.id,
                invocation_mode="all_or_none",
                execution_bundle_key="parse",
            ),
            ComponentCapability(
                component_version_id=version.id,
                capability_id=second.id,
                invocation_mode="all_or_none",
                execution_bundle_key="parse",
            ),
        ]
    )
    catalog_session.flush()
    manifest = PipelineManifest(
        pipeline_key="test-bundle-persistence",
        revision="1",
        bindings=(
            ComponentBinding(
                binding_key="bundle",
                component_key=component.component_key,
                component_version="1",
                capabilities=(first.capability_key,),
            ),
        ),
    )
    service = PipelinePersistenceService(catalog_session)
    revision = service.save_revision(manifest)
    with pytest.raises(ValueError, match="all_or_none"):
        service.validate_revision(revision.id)
