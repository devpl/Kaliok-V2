from __future__ import annotations

import importlib.util
from pathlib import Path

from sqlalchemy import CheckConstraint, UniqueConstraint
from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateIndex

from kaliok.storage.models import NormalizedContentUnit


HISTORICAL_INDEX = "uq_normalized_content_units_historical_version_index"
HISTORICAL_SOURCE_INDEX = (
    "uq_normalized_content_units_historical_version_source_unit"
)
RUN_INDEX = "uq_normalized_content_units_run_version_index"
RUN_SOURCE_INDEX = "uq_normalized_content_units_run_version_source_unit"


def _index(name: str):
    return next(
        index
        for index in NormalizedContentUnit.__table__.indexes
        if index.name == name
    )


def _columns(name: str) -> list[str]:
    return [column.name for column in _index(name).columns]


def _predicate(name: str) -> str:
    return str(_index(name).dialect_options["postgresql"]["where"])


def test_historical_units_keep_version_scoped_uniqueness():
    assert _columns(HISTORICAL_INDEX) == [
        "document_version_id",
        "unit_index",
    ]
    assert _predicate(HISTORICAL_INDEX) == "processing_run_id IS NULL"

    assert _columns(HISTORICAL_SOURCE_INDEX) == [
        "document_version_id",
        "source_unit_id",
    ]
    assert _predicate(HISTORICAL_SOURCE_INDEX) == (
        "processing_run_id IS NULL AND source_unit_id IS NOT NULL"
    )


def test_different_runs_can_reuse_indices_for_the_same_version():
    assert _columns(RUN_INDEX) == [
        "processing_run_id",
        "document_version_id",
        "unit_index",
    ]
    assert _predicate(RUN_INDEX) == "processing_run_id IS NOT NULL"


def test_same_run_rejects_duplicate_index_and_source_unit():
    assert _columns(RUN_SOURCE_INDEX) == [
        "processing_run_id",
        "document_version_id",
        "source_unit_id",
    ]
    assert _predicate(RUN_SOURCE_INDEX) == (
        "processing_run_id IS NOT NULL AND source_unit_id IS NOT NULL"
    )
    assert _index(RUN_INDEX).unique is True
    assert _index(RUN_SOURCE_INDEX).unique is True


def test_null_source_unit_ids_are_excluded_from_source_uniqueness():
    assert "source_unit_id IS NOT NULL" in _predicate(
        HISTORICAL_SOURCE_INDEX
    )
    assert "source_unit_id IS NOT NULL" in _predicate(RUN_SOURCE_INDEX)


def test_historical_constraints_are_replaced_and_unit_index_check_remains():
    table = NormalizedContentUnit.__table__
    unique_constraint_names = {
        constraint.name
        for constraint in table.constraints
        if isinstance(constraint, UniqueConstraint)
    }
    checks = {
        str(constraint.sqltext)
        for constraint in table.constraints
        if isinstance(constraint, CheckConstraint)
    }

    assert "uq_normalized_content_units_version_index" not in unique_constraint_names
    assert (
        "uq_normalized_content_units_version_source_unit"
        not in unique_constraint_names
    )
    assert "unit_index >= 0" in checks


def test_all_partial_unique_indexes_compile_for_postgresql():
    ddl_by_name = {
        name: str(
            CreateIndex(_index(name)).compile(dialect=postgresql.dialect())
        )
        for name in {
            HISTORICAL_INDEX,
            HISTORICAL_SOURCE_INDEX,
            RUN_INDEX,
            RUN_SOURCE_INDEX,
        }
    }

    assert all("CREATE UNIQUE INDEX" in ddl for ddl in ddl_by_name.values())
    assert all(" WHERE " in ddl for ddl in ddl_by_name.values())
    assert "processing_run_id IS NULL" in ddl_by_name[HISTORICAL_INDEX]
    assert "processing_run_id IS NOT NULL" in ddl_by_name[RUN_INDEX]


def test_migration_upgrade_and_downgrade_are_schema_only(monkeypatch):
    migration_path = (
        Path(__file__).parents[1]
        / "migrations"
        / "versions"
        / "8f2d1c4b7a90_support_normalization_generations.py"
    )
    spec = importlib.util.spec_from_file_location(
        "normalization_generations_migration",
        migration_path,
    )
    assert spec is not None and spec.loader is not None
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)

    calls: list[tuple[str, tuple, dict]] = []

    for operation in (
        "drop_constraint",
        "create_index",
        "drop_index",
        "create_unique_constraint",
    ):
        monkeypatch.setattr(
            migration.op,
            operation,
            lambda *args, _operation=operation, **kwargs: calls.append(
                (_operation, args, kwargs)
            ),
        )

    migration.upgrade()
    upgrade_calls = list(calls)
    calls.clear()
    migration.downgrade()
    downgrade_calls = list(calls)

    assert migration.revision == "8f2d1c4b7a90"
    assert migration.down_revision == "6a1c3e8f9b42"
    assert [call[0] for call in upgrade_calls] == [
        "drop_constraint",
        "drop_constraint",
        "create_index",
        "create_index",
        "create_index",
        "create_index",
    ]
    assert [call[0] for call in downgrade_calls] == [
        "drop_index",
        "drop_index",
        "drop_index",
        "drop_index",
        "create_unique_constraint",
        "create_unique_constraint",
    ]
