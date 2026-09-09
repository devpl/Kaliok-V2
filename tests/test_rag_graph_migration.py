from __future__ import annotations

from collections.abc import Iterator
from uuid import uuid4

import pytest
from sqlalchemy import inspect
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select

from kaliok.pipeline.persistence import (
    CAPABILITY_CATALOG,
    PipelinePersistenceService,
    validate_graph_backfill,
)
from kaliok.storage.models import (
    ArtifactType,
    Capability,
    CapabilityArtifactContract,
    RagTemplateEdge,
    RagTemplateNode,
    RagTemplateRevision,
)
from kaliok.storage.database import create_database_engine


@pytest.fixture
def catalog_session() -> Iterator[Session]:
    engine = create_database_engine()
    with engine.connect() as connection:
        transaction = connection.begin()
        try:
            if "rag_template_edges" not in inspect(connection).get_table_names():
                pytest.skip("La migration du graphe RAG n'est pas appliquée dans cette base.")
            with Session(bind=connection) as session:
                yield session
        finally:
            if transaction.is_active:
                transaction.rollback()
    engine.dispose()


def test_graph_backfill_has_the_expected_additive_projection(catalog_session: Session):
    service = PipelinePersistenceService(catalog_session)
    report = validate_graph_backfill(catalog_session, raise_on_error=True)

    assert report.is_valid
    assert report.legacy_capability_count == 6
    assert report.graph_node_count == 6
    assert report.legacy_dependency_count == 2
    assert report.graph_edge_count == 2
    assert report.artifact_type_count == 7
    assert report.contract_count == 12
    assert len(service.load_artifact_types()) == 7
    assert len(service.load_template_nodes()) == 6
    assert len(service.load_template_edges()) == 2


def test_graph_backfill_preserves_legacy_artifact_declarations(catalog_session: Session):
    capabilities = {
        row.capability_key: row
        for row in catalog_session.exec(select(Capability)).all()
    }
    artifacts = {
        row.artifact_type_key: row
        for row in catalog_session.exec(select(ArtifactType)).all()
    }
    artifacts_by_id = {row.id: row for row in artifacts.values()}
    contracts = catalog_session.exec(select(CapabilityArtifactContract)).all()

    assert set(artifacts) == {
        artifact
        for item in CAPABILITY_CATALOG
        for artifact in (*item["input"], *item["output"])
    }
    assert len(contracts) == 12
    assert all(contract.required is None for contract in contracts)
    assert all(contract.cardinality is None for contract in contracts)
    for item in CAPABILITY_CATALOG:
        capability = capabilities[item["key"]]
        actual = [
            (
                contract.direction,
                artifacts_by_id[contract.artifact_type_id].artifact_type_key,
                contract.position,
            )
            for contract in contracts
            if contract.capability_id == capability.id
        ]
        expected = [
            ("input", value, position)
            for position, value in enumerate(item["input"])
        ] + [
            ("output", value, position)
            for position, value in enumerate(item["output"])
        ]
        assert sorted(actual) == sorted(expected)


def test_same_capability_can_have_multiple_graph_node_occurrences(catalog_session: Session):
    existing = catalog_session.exec(select(RagTemplateNode)).first()
    assert existing is not None
    duplicate_occurrence = RagTemplateNode(
        rag_template_revision_id=existing.rag_template_revision_id,
        node_key=f"{existing.node_key}__2",
        capability_id=existing.capability_id,
        display_name=existing.display_name,
        requirement_mode=existing.requirement_mode,
        position=existing.position + 10,
    )
    catalog_session.add(duplicate_occurrence)
    catalog_session.flush()


def test_database_rejects_cross_revision_edge(catalog_session: Session):
    first_revision = catalog_session.exec(select(RagTemplateRevision)).one()
    nodes = catalog_session.exec(
        select(RagTemplateNode)
        .where(RagTemplateNode.rag_template_revision_id == first_revision.id)
        .order_by(RagTemplateNode.position)
    ).all()
    second_revision = RagTemplateRevision(
        rag_template_id=first_revision.rag_template_id,
        revision_number=2,
        status="draft",
    )
    catalog_session.add(second_revision)
    catalog_session.flush()
    catalog_session.add(
        RagTemplateEdge(
            rag_template_revision_id=second_revision.id,
            source_node_id=nodes[0].id,
            target_node_id=nodes[1].id,
            edge_key=f"cross_revision__{uuid4().hex}",
        )
    )
    with pytest.raises(IntegrityError):
        catalog_session.flush()


def test_database_rejects_self_edge(catalog_session: Session):
    node = catalog_session.exec(select(RagTemplateNode)).first()
    assert node is not None
    catalog_session.add(
        RagTemplateEdge(
            rag_template_revision_id=node.rag_template_revision_id,
            source_node_id=node.id,
            target_node_id=node.id,
            edge_key=f"self_edge__{uuid4().hex}",
        )
    )
    with pytest.raises(IntegrityError):
        catalog_session.flush()
