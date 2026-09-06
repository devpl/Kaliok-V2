from __future__ import annotations

import ast
from pathlib import Path

from sqlalchemy import CheckConstraint

from kaliok.storage.models import CandidateSourceFragment, DiscoveredCandidate


def test_candidate_models_expose_only_normalized_unit_provenance():
    candidate_columns = set(DiscoveredCandidate.__table__.c.keys())
    source_columns = set(CandidateSourceFragment.__table__.c.keys())

    assert {"document_version_id", "processing_run_id", "payload", "detector_key"} <= candidate_columns
    assert "normalized_content_unit_id" in source_columns
    assert {"page_id", "content_block_id", "chunk_id"}.isdisjoint(source_columns)


def test_candidate_source_fragment_constraints_are_present():
    checks = {
        str(constraint.sqltext)
        for constraint in CandidateSourceFragment.__table__.constraints
        if isinstance(constraint, CheckConstraint)
    }
    assert "fragment_order >= 0" in checks
    assert "start_offset IS NULL OR start_offset >= 0" in checks
    assert "end_offset IS NULL OR start_offset IS NULL OR end_offset >= start_offset" in checks


def test_candidate_migration_is_schema_only_and_has_expected_parent():
    path = Path(__file__).parents[1] / "migrations" / "versions" / "9c4e2b7a1d60_add_candidate_discovery.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    assignments = {
        node.target.id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id in {"revision", "down_revision"}
    }
    upgrade = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "upgrade")
    operations = {
        node.func.attr
        for node in ast.walk(upgrade)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "op"
    }

    assert assignments == {"revision": "9c4e2b7a1d60", "down_revision": "8f2d1c4b7a90"}
    assert operations == {"create_table", "create_index"}
