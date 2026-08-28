from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "benchmark_bge_reranker.py"
)
spec = importlib.util.spec_from_file_location(
    "benchmark_bge_reranker_test_module",
    SCRIPT_PATH,
)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Impossible de charger {SCRIPT_PATH}")
benchmark = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = benchmark
spec.loader.exec_module(benchmark)


def test_warm_statistics_use_only_second_and_third_batches():
    statistics = benchmark.calculate_warm_statistics(
        [100.0, 4.0, 6.0],
        passage_count=5,
    )

    assert statistics.average_batch_seconds == 5.0
    assert statistics.average_passage_seconds == 1.0


def test_warm_statistics_validate_measurements():
    with pytest.raises(ValueError, match="Mesures insuffisantes"):
        benchmark.calculate_warm_statistics([1.0, 2.0], 5)


def test_score_ranking_is_descending_and_deterministic():
    assert benchmark.rank_scores([0.2, 0.8, 0.8, 0.1]) == [1, 2, 0, 3]


def test_experimental_passages_have_representative_lengths():
    lengths = [len(passage.strip()) for passage in benchmark.PASSAGES]

    assert len(lengths) == 5
    assert all(1000 <= length <= 2000 for length in lengths)
