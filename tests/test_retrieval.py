"""
Unit tests for pure set-based retrieval scoring.

Tests hand-crafted fixtures locking in Precision, Recall, and F1 calculations.
"""

import pytest
from harness.scorers.retrieval import RetrievalScorer, compute_retrieval_metrics


@pytest.fixture
def scorer() -> RetrievalScorer:
    return RetrievalScorer()


def test_perfect_match(scorer: RetrievalScorer) -> None:
    """All retrieved chunks are relevant and complete."""
    retrieved = ["doc1_chunk1", "doc1_chunk2"]
    ground_truth = ["doc1_chunk1", "doc1_chunk2"]

    res = scorer.score(retrieved, ground_truth)
    assert res.precision == 1.0
    assert res.recall == 1.0
    assert res.f1 == 1.0
    assert res.matched_count == 2
    assert res.retrieved_count == 2
    assert res.ground_truth_count == 2


def test_no_overlap(scorer: RetrievalScorer) -> None:
    """Target API retrieved chunks disjoint from ground truth."""
    retrieved = ["doc2_chunk1", "doc2_chunk2"]
    ground_truth = ["doc1_chunk1"]

    res = scorer.score(retrieved, ground_truth)
    assert res.precision == 0.0
    assert res.recall == 0.0
    assert res.f1 == 0.0
    assert res.matched_count == 0


def test_partial_overlap(scorer: RetrievalScorer) -> None:
    """1 out of 2 retrieved chunks is relevant; 1 out of 2 ground truth retrieved."""
    retrieved = ["doc1_chunk1", "doc2_chunk99"]
    ground_truth = ["doc1_chunk1", "doc1_chunk2"]

    res = scorer.score(retrieved, ground_truth)
    assert res.precision == 0.5  # 1 / 2
    assert res.recall == 0.5     # 1 / 2
    assert res.f1 == 0.5         # 2 * 0.5 * 0.5 / (0.5 + 0.5)
    assert res.matched_count == 1


def test_empty_retrieved_list(scorer: RetrievalScorer) -> None:
    """Target API returned 0 chunks (e.g. strict confidence filtering or total miss)."""
    retrieved: list[str] = []
    ground_truth = ["doc1_chunk1"]

    res = scorer.score(retrieved, ground_truth)
    assert res.precision == 0.0
    assert res.recall == 0.0
    assert res.f1 == 0.0
    assert res.retrieved_count == 0
    assert res.matched_count == 0


def test_empty_ground_truth_list(scorer: RetrievalScorer) -> None:
    """Defensive check: ungrounded question with empty ground truth."""
    retrieved = ["doc1_chunk1"]
    ground_truth: list[str] = []

    res = scorer.score(retrieved, ground_truth)
    assert res.precision == 0.0
    assert res.recall == 0.0
    assert res.f1 == 0.0
    assert res.ground_truth_count == 0
    assert res.matched_count == 0


def test_duplicate_retrieved_ids_handled_as_set(scorer: RetrievalScorer) -> None:
    """Verifies that duplicate chunks returned by flawed rankers are deduplicated."""
    retrieved = ["doc1_chunk1", "doc1_chunk1"]
    ground_truth = ["doc1_chunk1"]

    res = scorer.score(retrieved, ground_truth)
    assert res.precision == 1.0
    assert res.recall == 1.0
    assert res.f1 == 1.0
    assert res.retrieved_count == 1
    assert res.matched_count == 1


def test_compute_retrieval_metrics_helper() -> None:
    """Verify raw helper function returning dict."""
    metrics = compute_retrieval_metrics(["a", "b"], ["b", "c"])
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5
