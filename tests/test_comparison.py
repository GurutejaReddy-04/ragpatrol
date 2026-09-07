"""
Unit tests for side-by-side configuration comparison (Phase 6).

Tests winner identification across quality and latency metrics, per-category breakdown,
tie handling, and graceful unreachable config detection.
"""

from typing import Optional
import pytest
from harness.reporting.comparison_report import ComparisonReporter
from harness.runner import StageRunResult
from harness.scorers.latency import LatencyProfile


def create_sample_result(
    config_name: str,
    precision: float,
    recall: float,
    f1: float,
    faithfulness: float,
    p50: float,
    p95: float,
    p99: float,
    category_metrics: Optional[dict] = None,
    successful_queries: int = 25,
) -> StageRunResult:
    """Helper to build a StageRunResult fixture."""
    return StageRunResult(
        config_name=config_name,
        cache_state="warm",
        stage="all",
        total_queries=25,
        successful_queries=successful_queries,
        failed_queries=25 - successful_queries,
        mean_precision=precision,
        mean_recall=recall,
        mean_f1=f1,
        mean_faithfulness=faithfulness,
        hallucination_count=2,
        hallucination_rate=0.08,
        latency_profile=LatencyProfile(
            p50_ms=p50,
            p95_ms=p95,
            p99_ms=p99,
            mean_ms=p50 * 1.1,
            std_ms=10.0,
            min_ms=p50 * 0.8,
            max_ms=p99 * 1.1,
        ),
        category_metrics=category_metrics or {
            "easy": {"precision": precision + 0.05, "recall": recall + 0.05, "f1": f1 + 0.05, "faithfulness": faithfulness},
            "edge": {"precision": precision - 0.05, "recall": recall - 0.05, "f1": f1 - 0.05, "faithfulness": faithfulness},
        },
        question_summaries=[],
    )


def test_comparison_winner_identification() -> None:
    """Verify quality wins on high scores and speed wins on low latency."""
    # Config A: High quality, slower latency (typical of Reranker ON)
    res_a = create_sample_result(
        config_name="reranker_on",
        precision=0.88,
        recall=0.84,
        f1=0.86,
        faithfulness=0.92,
        p50=180.0,
        p95=240.0,
        p99=290.0,
    )
    # Config B: Lower quality, faster latency (typical of Reranker OFF)
    res_b = create_sample_result(
        config_name="reranker_off",
        precision=0.72,
        recall=0.68,
        f1=0.70,
        faithfulness=0.78,
        p50=35.0,
        p95=55.0,
        p99=65.0,
    )

    report = ComparisonReporter.compare("reranker_on", res_a, "reranker_off", res_b)

    assert report["status"] == "success"
    assert report["winner"] == "reranker_on"
    assert report["wins_a"] == 4  # Precision, Recall, F1, Faithfulness
    assert report["wins_b"] == 3  # p50, p95, p99
    assert report["ties"] == 0

    metric_winners = {m["metric"]: m["winner"] for m in report["metrics_comparison"]}
    assert metric_winners["Precision"] == "reranker_on"
    assert metric_winners["Recall"] == "reranker_on"
    assert metric_winners["F1"] == "reranker_on"
    assert metric_winners["Faithfulness"] == "reranker_on"
    assert metric_winners["Latency p50"] == "reranker_off"
    assert metric_winners["Latency p95"] == "reranker_off"
    assert metric_winners["Latency p99"] == "reranker_off"


def test_comparison_per_category_breakdown() -> None:
    """Verify category breakdown combines metrics across both configs."""
    res_a = create_sample_result("cfg_a", 0.8, 0.8, 0.8, 0.8, 50, 60, 70)
    res_b = create_sample_result("cfg_b", 0.7, 0.7, 0.7, 0.7, 40, 50, 60)

    report = ComparisonReporter.compare("cfg_a", res_a, "cfg_b", res_b)

    cats = report["category_comparison"]
    assert "easy" in cats
    assert "edge" in cats
    assert "config_a" in cats["easy"]
    assert "config_b" in cats["easy"]


def test_comparison_tie_handling() -> None:
    """Identical configurations must produce ties and an overall tie."""
    res_a = create_sample_result("cfg_1", 0.8, 0.8, 0.8, 0.8, 50, 60, 70)
    res_b = create_sample_result("cfg_2", 0.8, 0.8, 0.8, 0.8, 50, 60, 70)

    report = ComparisonReporter.compare("cfg_1", res_a, "cfg_2", res_b)

    assert report["status"] == "success"
    assert report["winner"] == "Tie"
    assert report["ties"] == 7
    assert report["wins_a"] == 0
    assert report["wins_b"] == 0


def test_comparison_unreachable_config() -> None:
    """If one config fails or is unreachable, return unreachable status instead of blank metrics."""
    res_a = create_sample_result("cfg_online", 0.8, 0.8, 0.8, 0.8, 50, 60, 70)

    # Config B unreachable with explicit error
    report = ComparisonReporter.compare(
        "cfg_online", res_a,
        "cfg_offline", None,
        error_b="Connection refused at http://localhost:9999",
    )

    assert report["status"] == "unreachable"
    assert "cfg_offline" in report["unreachable_configs"]
    assert report["config_a"]["reachable"] is True
    assert report["config_b"]["reachable"] is False
    assert "Connection refused" in report["config_b"]["error"]


def test_comparison_zero_successful_queries() -> None:
    """A config that produced 0 successful queries must be flagged as unreachable."""
    res_a = create_sample_result("cfg_good", 0.8, 0.8, 0.8, 0.8, 50, 60, 70, successful_queries=25)
    res_b = create_sample_result("cfg_bad", 0.0, 0.0, 0.0, 0.0, 0, 0, 0, successful_queries=0)

    report = ComparisonReporter.compare("cfg_good", res_a, "cfg_bad", res_b)

    assert report["status"] == "unreachable"
    assert "cfg_bad" in report["unreachable_configs"]
