from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import CheckConstraint

from kaliok.execution import ExecutionContext, apply_execution_context
from kaliok.hashing import canonical_json_hash
from kaliok.storage.models import ProcessingRun


def test_canonical_json_hash_is_logical_and_unicode_safe() -> None:
    first = {"label": "Nouméa", "enabled": True, "items": [None, 2]}
    second = {"items": [None, 2], "enabled": True, "label": "Nouméa"}

    assert canonical_json_hash(first) == canonical_json_hash(second)
    assert canonical_json_hash(first) != canonical_json_hash(
        {**first, "items": [2, None]}
    )
    assert canonical_json_hash(1) != canonical_json_hash(1.0)


def test_execution_context_is_immutable_and_validates_environment() -> None:
    context = ExecutionContext(
        environment="experiment",
        configuration_revision_id=uuid4(),
    )

    with pytest.raises(FrozenInstanceError):
        context.environment = "production"  # type: ignore[misc]
    with pytest.raises(ValueError, match="production.*experiment"):
        ExecutionContext(environment="legacy")  # type: ignore[arg-type]


def test_execution_context_persists_identity_and_snapshot_hash() -> None:
    group = uuid4()
    context = ExecutionContext(environment="production", execution_group_id=group)
    run = ProcessingRun(
        process_type="normalization",
        status="completed",
        configuration={"unicode": "Nouméa", "enabled": True},
    )

    apply_execution_context(object(), run, context)  # type: ignore[arg-type]

    assert run.execution_environment == "production"
    assert run.execution_group_id == group
    assert run.configuration_hash == canonical_json_hash(run.configuration)


def test_processing_run_exposes_nullable_execution_identity_and_constraint() -> None:
    columns = ProcessingRun.__table__.c
    assert {
        "execution_environment",
        "configuration_revision_id",
        "execution_group_id",
        "configuration_hash",
    } <= set(columns.keys())
    assert all(columns[name].nullable for name in columns.keys() if name in {
        "execution_environment",
        "configuration_revision_id",
        "execution_group_id",
        "configuration_hash",
    })
    checks = {
        str(constraint.sqltext)
        for constraint in ProcessingRun.__table__.constraints
        if isinstance(constraint, CheckConstraint)
    }
    assert any("execution_environment" in expression for expression in checks)
    with pytest.raises(ValueError, match="production.*experiment"):
        ProcessingRun(
            process_type="test",
            status="running",
            execution_environment="legacy",  # type: ignore[arg-type]
        )


def test_execution_identity_migration_is_schema_only_and_has_real_parent() -> None:
    path = (
        Path(__file__).parents[1]
        / "migrations"
        / "versions"
        / "a6b7c8d9e0f1_add_processing_run_execution_identity.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    assignments = {
        node.target.id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id in {"revision", "down_revision"}
    }
    upgrade = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "upgrade"
    )
    operations = {
        node.func.attr
        for node in ast.walk(upgrade)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "op"
    }
    source = path.read_text(encoding="utf-8")

    assert assignments == {
        "revision": "a6b7c8d9e0f1",
        "down_revision": "f1a2b3c4d5e6",
    }
    assert operations == {
        "add_column",
        "create_foreign_key",
        "create_index",
        "create_check_constraint",
    }
    assert "UPDATE " not in source.upper()
    assert "DELETE " not in source.upper()
