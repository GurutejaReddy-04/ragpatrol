"""
Unit tests for Markdown and HTML report generation (Phase 7).

Verifies report structure, embedded tables, styling, and offline handling
using mock evaluation fixtures with zero live network dependencies.
"""

import pytest
from harness.reporting.comparison_report import ComparisonReporter
from harness.reporting.html_report import HTMLReportGenerator
from harness.reporting.markdown_report import MarkdownReportGenerator
from harness.runner import QuestionEvalSummary, StageRunResult
from harness.scorers.latency import LatencyProfile


@pytest.fixture
def sample_stage_result() -> StageRunResult:
    """Fixture providing a populated StageRunResult with metrics and flagged questions."""
    return StageRunResult(
        config_name="reranker_on",
        cache_state="warm",
        stage="all",
        total_queries=2,
        successful_queries=2,
        failed_queries=0,
        mean_precision=0.85,
        mean_recall=0.80,
        mean_f1=0.824,
        mean_faithfulness=0.88,
        hallucination_count=1,
        hallucination_rate=0.50,
        latency_profile=LatencyProfile(
            p50_ms=45.0,
            p95_ms=85.0,
            p99_ms=95.0,
            mean_ms=50.0,
            std_ms=12.0,
            min_ms=30.0,
            max_ms=100.0,
        ),
        category_metrics={
            "easy": {"count": 1, "precision": 1.0, "recall": 1.0, "f1": 1.0, "faithfulness": 0.95, "hallucination_rate": 0.0},
            "edge": {"count": 1, "precision": 0.70, "recall": 0.60, "f1": 0.648, "faithfulness": 0.81, "hallucination_rate": 1.0},
        },
        question_summaries=[
            QuestionEvalSummary(
                id="q01",
                category="easy",
                question="What is topic A?",
                retrieved_chunk_ids=["chunk_1"],
                ground_truth_chunk_ids=["chunk_1"],
                precision=1.0,
                recall=1.0,
                f1=1.0,
                faithfulness_score=0.95,
                embedding_similarity=0.92,
                llm_judge_score=5.0,
                latency_ms=35.0,
                is_hallucination=False,
            ),
            QuestionEvalSummary(
                id="q02",
                category="edge",
                question="What is topic B?",
                retrieved_chunk_ids=["chunk_2"],
                ground_truth_chunk_ids=["chunk_3"],
                precision=0.70,
                recall=0.60,
                f1=0.648,
                faithfulness_score=0.45,
                embedding_similarity=0.40,
                llm_judge_score=2.0,
                judge_reasoning="Claim not substantiated by retrieved context passages.",
                unsupported_claims=["Sub-claim X is ungrounded"],
                latency_ms=65.0,
                is_hallucination=True,
            ),
        ],
    )


def test_markdown_run_report_structure(sample_stage_result: StageRunResult) -> None:
    """Verify Markdown report contains all required headers, tables, and sections."""
    md = MarkdownReportGenerator.generate_run_report(sample_stage_result, git_commit_sha="abcdef123456")

    assert "# Evaluation Report: `reranker_on`" in md
    assert "## Execution Metadata" in md
    assert "`abcdef12`" in md
    assert "## Summary Quality Metrics" in md
    assert "| **Retrieval Precision** | 85.0% |" in md
    assert "| **Retrieval F1 Score** | 82.4% |" in md
    assert "## Latency Percentile Profile" in md
    assert "45.0 ms" in md
    assert "## Per-Category Performance Breakdown" in md
    assert "| `easy` |" in md
    assert "| `edge` |" in md
    assert "## Flagged Hallucinations & Low Faithfulness Anomalies" in md
    assert "Claim not substantiated by retrieved context passages." in md
    assert "<details>" in md
    assert "| `q01` |" in md
    assert "| `q02` |" in md


def test_html_run_report_structure(sample_stage_result: StageRunResult) -> None:
    """Verify standalone HTML report contains embedded CSS, KPI cards, and print styles."""
    html_doc = HTMLReportGenerator.generate_run_report(sample_stage_result, git_commit_sha="abcdef123456")

    assert "<!DOCTYPE html>" in html_doc
    assert "<title>Evaluation Report - reranker_on</title>" in html_doc
    # Zero external CDN links
    assert "cdn." not in html_doc
    assert "http://" not in html_doc and "https://" not in html_doc
    # Print stylesheet
    assert "@media print" in html_doc
    # KPI Cards
    assert "85.0%" in html_doc
    assert "82.4%" in html_doc
    # Hallucination alert
    assert "Claim not substantiated" in html_doc
    # Category table
    assert "<strong>easy</strong>" in html_doc


def test_markdown_comparison_report(sample_stage_result: StageRunResult) -> None:
    """Verify Markdown comparison report formats side-by-side metrics and winners."""
    comp = ComparisonReporter.compare("cfg_a", sample_stage_result, "cfg_b", sample_stage_result)
    md = MarkdownReportGenerator.generate_comparison_report(comp, git_commit_sha="12345678")

    assert "# Configuration Comparison: `cfg_a` vs `cfg_b`" in md
    assert "## Side-by-Side Metric Comparison" in md
    assert "| **Precision** |" in md
    assert "## Per-Category Performance Comparison" in md


def test_html_comparison_report(sample_stage_result: StageRunResult) -> None:
    """Verify HTML comparison report renders clean side-by-side tables."""
    comp = ComparisonReporter.compare("cfg_a", sample_stage_result, "cfg_b", sample_stage_result)
    html_doc = HTMLReportGenerator.generate_comparison_report(comp, git_commit_sha="12345678")

    assert "<!DOCTYPE html>" in html_doc
    assert "Configuration Experiment: cfg_a vs cfg_b" in html_doc
    assert "Side-by-Side Metric Comparison" in html_doc
    assert "@media print" in html_doc


def test_unreachable_reporting_graceful() -> None:
    """Verify unreachable configurations produce clear aborted notices in both formats."""
    comp = ComparisonReporter.compare("online", None, "offline", None, error_b="Host unreachable")

    md = MarkdownReportGenerator.generate_comparison_report(comp)
    assert "Aborted" in md
    assert "unreachable" in md

    html_doc = HTMLReportGenerator.generate_comparison_report(comp)
    assert "Aborted" in html_doc
    assert "unreachable" in html_doc
