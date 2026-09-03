"""
Unit tests for dual-signal faithfulness scoring (Phase 3).

Mocks all LLM judge calls to ensure 100% deterministic, zero-quota execution.
"""

from unittest.mock import MagicMock
import pytest
from harness.scorers.faithfulness import FaithfulnessScorer, FaithfulnessResult


class MockGeminiResponse:
    """Mock Gemini API response object."""
    def __init__(self, text: str) -> None:
        self.text = text


@pytest.fixture
def mock_gemini_client() -> MagicMock:
    client = MagicMock()
    return client


def test_embedding_similarity_identical_text() -> None:
    """Identical text pairs should produce cosine similarity near 1.0."""
    scorer = FaithfulnessScorer()
    text = "Failed embedding calls are retried 3 times with exponential backoff."
    sim, lat = scorer.compute_embedding_similarity(text, text)
    assert sim > 0.95
    assert lat >= 0.0


def test_embedding_similarity_dissimilar_text() -> None:
    """Unrelated topics should produce low similarity."""
    scorer = FaithfulnessScorer()
    answer = "Quantum computing relies on qubits, superposition, and entanglement."
    context = "Photosynthesis is the process used by plants to convert light energy."
    sim, _ = scorer.compute_embedding_similarity(answer, context)
    assert sim < 0.40


def test_embedding_similarity_empty_context() -> None:
    """Empty context should yield 0.0 similarity."""
    scorer = FaithfulnessScorer()
    sim, _ = scorer.compute_embedding_similarity("Some answer", "")
    assert sim == 0.0


def test_llm_judge_valid_json_response(mock_gemini_client: MagicMock) -> None:
    """Mock judge returning valid JSON."""
    canned_json = """```json
    {
        "faithfulness_score": 5,
        "reasoning": "The answer is completely grounded in the retrieved passages.",
        "unsupported_claims": []
    }
    ```"""
    mock_gemini_client.models.generate_content.return_value = MockGeminiResponse(canned_json)

    scorer = FaithfulnessScorer(gemini_client=mock_gemini_client)
    score, reasoning, claims, lat, raw = scorer.call_llm_judge(
        question="What is retry policy?",
        context_text="Failed calls retried 3 times.",
        answer="Retried 3 times.",
    )

    assert score == 5.0
    assert "completely grounded" in reasoning
    assert len(claims) == 0
    assert lat >= 0.0
    assert raw is not None


def test_llm_judge_malformed_json_fallback(mock_gemini_client: MagicMock) -> None:
    """Mock judge returning malformed non-JSON payload triggers fallback without crashing."""
    mock_gemini_client.models.generate_content.return_value = MockGeminiResponse("Sorry, I cannot format as JSON.")

    scorer = FaithfulnessScorer(gemini_client=mock_gemini_client)
    score, reasoning, claims, _, _ = scorer.call_llm_judge(
        question="What is retry policy?",
        context_text="Context",
        answer="Answer",
    )
    # Should fall back defensively to default neutral score (3.0)
    assert score == 3.0
    assert "failed after 2 attempts" in reasoning


def test_combined_score_calculation(mock_gemini_client: MagicMock) -> None:
    """
    Verify weighted combination:
    faithfulness = 0.4 * emb + 0.6 * (judge / 5.0)
    """
    mock_json = '{"faithfulness_score": 4, "reasoning": "Mostly supported", "unsupported_claims": []}'
    mock_gemini_client.models.generate_content.return_value = MockGeminiResponse(mock_json)

    scorer = FaithfulnessScorer(
        embedding_weight=0.4,
        llm_judge_weight=0.6,
        gemini_client=mock_gemini_client,
    )

    # Force embedding similarity to be tested with exact known value
    scorer.compute_embedding_similarity = MagicMock(return_value=(0.80, 5.0))  # type: ignore

    res = scorer.score(
        question="Sample question",
        answer="Sample answer",
        contexts=["Sample context"],
        dry_run=False,
    )

    # Expected: 0.4 * 0.80 + 0.6 * (4 / 5.0) = 0.32 + 0.48 = 0.80
    assert res.faithfulness_score == 0.80
    assert res.embedding_similarity == 0.80
    assert res.llm_judge_score == 4.0
    assert res.is_hallucination is False


def test_hallucination_flagging(mock_gemini_client: MagicMock) -> None:
    """Low judge score (<= 2) must trigger hallucination flag."""
    mock_json = '{"faithfulness_score": 2, "reasoning": "Unsupported claim", "unsupported_claims": ["claim X"]}'
    mock_gemini_client.models.generate_content.return_value = MockGeminiResponse(mock_json)

    scorer = FaithfulnessScorer(gemini_client=mock_gemini_client)
    scorer.compute_embedding_similarity = MagicMock(return_value=(0.75, 5.0))  # type: ignore

    res = scorer.score("Q", "A", ["C"])
    assert res.is_hallucination is True
    assert res.llm_judge_score == 2.0
    assert "claim X" in res.unsupported_claims


def test_dry_run_mode() -> None:
    """In dry-run mode, no external LLM call is made."""
    scorer = FaithfulnessScorer()
    scorer.compute_embedding_similarity = MagicMock(return_value=(0.90, 2.0))  # type: ignore

    res = scorer.score("Q", "A", ["C"], dry_run=True)
    assert res.embedding_similarity == 0.90
    assert res.judge_latency_ms == 0.0
    assert "Dry-run mode" in res.judge_reasoning
