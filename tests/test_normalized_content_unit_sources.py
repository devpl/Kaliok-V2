from __future__ import annotations

import ast
from pathlib import Path

from sqlalchemy import CheckConstraint, UniqueConstraint
from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateTable

from kaliok.storage.models import (
    ContentBlock,
    NormalizedContentUnit,
    NormalizedContentUnitSource,
    ProcessingRun,
)


def _unique_column_sets(table) -> set[frozenset[str]]:
    return {
        frozenset(column.name for column in constraint.columns)
        for constraint in table.constraints
        if isinstance(constraint, UniqueConstraint)
    }


def test_processing_run_reference_is_nullable_and_indexed():
    column = NormalizedContentUnit.__table__.c.processing_run_id

    assert column.nullable is True
    assert {target.target_fullname for target in column.foreign_keys} == {
        "processing_runs.id"
    }
    assert any(
        [indexed.name for indexed in index.columns] == ["processing_run_id"]
        for index in NormalizedContentUnit.__table__.indexes
    )


def test_processing_run_reference_accepts_none_or_a_run_identifier():
    run = ProcessingRun(
        document_version_id="00000000-0000-0000-0000-000000000001",
        process_type="content_normalization",
        status="completed",
    )
    historical = NormalizedContentUnit(
        document_version_id="00000000-0000-0000-0000-000000000001",
        unit_index=0,
        content_type="paragraph",
        content="Unité historique",
    )
    generated = NormalizedContentUnit(
        document_version_id="00000000-0000-0000-0000-000000000001",
        processing_run_id=run.id,
        unit_index=1,
        content_type="paragraph",
        content="Unité normalisée",
    )

    assert historical.processing_run_id is None
    assert generated.processing_run_id == run.id


def test_normalized_unit_generation_indexes_are_present():
    assert {
        "uq_normalized_content_units_historical_version_index",
        "uq_normalized_content_units_historical_version_source_unit",
        "uq_normalized_content_units_run_version_index",
        "uq_normalized_content_units_run_version_source_unit",
    }.issubset(
        {index.name for index in NormalizedContentUnit.__table__.indexes}
    )


def test_source_model_supports_ordered_multiple_blocks_and_shared_blocks():
    first_unit = NormalizedContentUnit(
        document_version_id="00000000-0000-0000-0000-000000000001",
        unit_index=0,
        content_type="paragraph",
        content="Première unité",
    )
    second_unit = NormalizedContentUnit(
        document_version_id="00000000-0000-0000-0000-000000000001",
        unit_index=1,
        content_type="paragraph",
        content="Deuxième unité",
    )
    first_block = ContentBlock(
        page_id="00000000-0000-0000-0000-000000000002",
        block_index=0,
        content="Premier bloc",
        extraction_method="native",
    )
    second_block = ContentBlock(
        page_id="00000000-0000-0000-0000-000000000002",
        block_index=1,
        content="Deuxième bloc",
        extraction_method="native",
    )

    sources = [
        NormalizedContentUnitSource(
            normalized_content_unit_id=first_unit.id,
            content_block_id=first_block.id,
            source_order=0,
        ),
        NormalizedContentUnitSource(
            normalized_content_unit_id=first_unit.id,
            content_block_id=second_block.id,
            source_order=1,
        ),
        NormalizedContentUnitSource(
            normalized_content_unit_id=second_unit.id,
            content_block_id=first_block.id,
            source_order=0,
        ),
    ]

    assert [source.source_order for source in sources[:2]] == [0, 1]
    assert sources[0].normalized_content_unit_id == sources[1].normalized_content_unit_id
    assert sources[0].content_block_id == sources[2].content_block_id
    assert sources[0].normalized_content_unit_id != sources[2].normalized_content_unit_id


def test_source_constraints_reject_duplicate_order_block_and_negative_order():
    table = NormalizedContentUnitSource.__table__
    unique_sets = _unique_column_sets(table)
    checks = {
        str(constraint.sqltext)
        for constraint in table.constraints
        if isinstance(constraint, CheckConstraint)
    }

    assert frozenset(
        {"normalized_content_unit_id", "content_block_id"}
    ) in unique_sets
    assert frozenset(
        {"normalized_content_unit_id", "source_order"}
    ) in unique_sets
    assert "source_order >= 0" in checks
    assert any(
        [indexed.name for indexed in index.columns] == ["content_block_id"]
        for index in table.indexes
    )


def test_source_table_compiles_for_postgresql():
    ddl = str(
        CreateTable(NormalizedContentUnitSource.__table__).compile(
            dialect=postgresql.dialect()
        )
    )

    assert "normalized_content_unit_sources" in ddl
    assert "CHECK (source_order >= 0)" in ddl
    assert "FOREIGN KEY(normalized_content_unit_id)" in ddl
    assert "FOREIGN KEY(content_block_id)" in ddl


def test_migration_is_additive_and_contains_no_data_backfill():
    migration_path = (
        Path(__file__).parents[1]
        / "migrations"
        / "versions"
        / "6a1c3e8f9b42_add_normalized_content_unit_sources.py"
    )
    source = migration_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    assignments = {
        target.id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and (target := node.target).id in {"revision", "down_revision"}
    }
    upgrade = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "upgrade"
    )
    operation_names = {
        node.func.attr
        for node in ast.walk(upgrade)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "op"
    }

    assert "add_column" in operation_names
    assert "create_table" in operation_names
    assert "create_foreign_key" in operation_names
    assert "create_index" in operation_names
    assert "execute" not in operation_names
    assert not any(name.startswith("drop_") for name in operation_names)
    assert assignments == {
        "revision": "6a1c3e8f9b42",
        "down_revision": "d4f7a9c2e681",
    }
