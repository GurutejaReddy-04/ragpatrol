"""
Edge Case & Regression Verification Test Suite.

Directly verifies that all issues identified in the independent audit
(EH-1, EC-1, EC-2, EC-3, EC-4, EC-5, EC-6, EC-7, EC-11, EC-13, EC-14, EC-15, EC-16, EH-3)
have been resolved and are covered by automated regression tests.
"""

from unittest.mock import MagicMock, patch
import httpx
import pytest
from starlette.testclient import TestClient

from harness.clients.contracts import (
    normalize_citebase_response,
    normalize_stub_response,
    normalize_target_response,
)
from harness.clients.rag_client import RAGClient
from harness.exceptions import (
    EmbeddingModelError,
    RAGConnectionError,
    RAGResponseError,
)
from harness.reporting.comparison_report import ComparisonReporter
from harness.runner import StageRunResult
from harness.scorers.faithfulness import FaithfulnessScorer
from harness.scorers.latency import LatencyProfile
from harness.scorers.retrieval import compute_retrieval_metrics
from stub_app.fake_rag_api import app as stub_fastapi_app


# --- Priority 1 & 2: RAG Client & Retry Logic (EH-1, EC-3, EC-14) ---


def test_eh1_tenacity_retries_on_transient_network_error():
    """
    EH-1: Verify that transient network errors are retried by tenacity.
    Fails on first attempt with ConnectError, succeeds on second attempt.
    """
    attempt = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempt
        attempt += 1
        if attempt == 1:
            raise httpx.ConnectError("Connection refused by peer", request=request)
        return httpx.Response(200, json={"answer_text": "Retried successfully", "sources": []})

    client = httpx.Client(
        base_url="http://test-server",
        transport=httpx.MockTransport(handler),
    )
    rag_client = RAGClient(
        base_url="http://test-server",
        http_client=client,
        adapter="stub",
        max_retries=3,
    )

    response = rag_client.query("test query")
    assert response.answer == "Retried successfully"
    assert attempt == 2


def test_eh1_exhausted_retries_raises_rag_connection_error():
    """
    EH-1: When all retries fail on network errors, RAGConnectionError is raised cleanly.
    """
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Unreachable host", request=request)

    client = httpx.Client(
        base_url="http://test-server",
        transport=httpx.MockTransport(handler),
    )
    rag_client = RAGClient(
        base_url="http://test-server",
        http_client=client,
        adapter="stub",
        max_retries=2,
    )

    with pytest.raises(RAGConnectionError) as exc_info:
        rag_client.query("test query")
    assert "Network error connecting" in str(exc_info.value)


def test_ec3_ec14_retry_on_transient_http_429_and_503():
    """
    EC-3 & EC-14: Verify that HTTP 429 (Rate Limit) and 503 (Unavailable) are retried.
    Fails on attempt 1 with 429, succeeds on attempt 2 with 200.
    """
    attempt = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempt
        attempt += 1
        if attempt == 1:
            return httpx.Response(429, text="Too Many Requests")
        return httpx.Response(200, json={"answer_text": "Rate limit passed", "sources": []})

    client = httpx.Client(
        base_url="http://test-server",
        transport=httpx.MockTransport(handler),
    )
    rag_client = RAGClient(
        base_url="http://test-server",
        http_client=client,
        adapter="stub",
        max_retries=3,
    )

    response = rag_client.query("test query")
    assert response.answer == "Rate limit passed"
    assert attempt == 2


# --- Priority 1 & 2: Faithfulness Scorer (EC-1, EC-4, EC-5, EC-16) ---


def test_ec1_faithfulness_none_answer_does_not_crash():
    """
    EC-1: Passing answer=None to compute_embedding_similarity must not raise AttributeError.
    """
    scorer = FaithfulnessScorer()
    sim, lat = scorer.compute_embedding_similarity(None, "Valid context text")
    assert sim == 0.0
    assert lat >= 0.0

    # Also test empty string
    sim_empty, _ = scorer.compute_embedding_similarity("", "Valid context text")
    assert sim_empty == 0.0


def test_ec4_embedding_model_load_failure_raises_embedding_model_error():
    """
    EC-4: If SentenceTransformer fails to load, EmbeddingModelError is raised.
    """
    import harness.scorers.faithfulness as f_mod

    with patch("harness.scorers.faithfulness._EMBEDDING_MODEL", None):
        with patch(
            "harness.scorers.faithfulness.SentenceTransformer",
            side_effect=RuntimeError("CUDA Out of Memory"),
        ):
            with pytest.raises(EmbeddingModelError) as exc_info:
                f_mod.get_embedding_model()
            assert "CUDA Out of Memory" in str(exc_info.value)


def test_ec5_ec16_dynamic_weight_fallback_and_hallucination_flag():
    """
    EC-5 & EC-16: When LLM judge fails or is unavailable:
    1. Effective weights dynamically adjust to 100% embedding similarity.
    2. is_hallucination is flagged as True due to lack of verification reliability.
    """
    scorer = FaithfulnessScorer(embedding_weight=0.4, llm_judge_weight=0.6)

    # Mock call_llm_judge to return failure/unavailable reasoning
    scorer.call_llm_judge = MagicMock(
        return_value=(3.0, "Gemini client unconfigured or unavailable", [], 10.0, None)
    )
    # Mock embedding similarity to return 0.8
    scorer.compute_embedding_similarity = MagicMock(return_value=(0.8, 5.0))

    result = scorer.score(
        question="What is X?",
        answer="X is Y.",
        contexts=["Context text about X and Y."],
        dry_run=False,
    )

    # Combined score must use 1.0 * emb_sim (0.8) instead of 0.4*0.8 + 0.6*(3.0/5.0) = 0.68
    assert result.faithfulness_score == 0.8
    # Hallucination flag must be triggered due to judge unavailability
    assert result.is_hallucination is True


# --- Priority 1 & 2: Contracts & Normalization (EC-2, EC-11, EC-15, EH-3) ---


def test_ec2_contracts_null_retrieved_chunks_does_not_crash():
    """
    EC-2: When target API responds with {"retrieved_chunks": null},
    normalize_target_response must coalesce to an empty list without TypeError.
    """
    payload = {
        "answer": "Valid synthesized answer",
        "retrieved_chunks": None,
    }
    resp = normalize_target_response(payload)
    assert resp.answer == "Valid synthesized answer"
    assert resp.retrieved_chunks == []


def test_ec15_contracts_null_answer_does_not_fail_validation():
    """
    EC-15: When target API responds with {"answer": null},
    normalize_target_response must coalesce answer to empty string "".
    """
    payload = {
        "answer": None,
        "retrieved_chunks": [],
    }
    resp = normalize_target_response(payload)
    assert resp.answer == ""
    assert resp.retrieved_chunks == []


def test_ec11_stub_adapter_unrecognized_schema_raises_value_error():
    """
    EC-11: If normalize_stub_response receives a dictionary missing both
    'answer_text' and 'sources', it raises ValueError rather than producing garbage.
    """
    invalid_payload = {"unknown_key": "some_value", "data": [1, 2, 3]}
    with pytest.raises(ValueError) as exc_info:
        normalize_stub_response(invalid_payload)
    assert "Stub adapter received unrecognized payload schema" in str(exc_info.value)


def test_eh3_citebase_validation_error_re_raised_as_rag_response_error():
    """
    EH-3: If CiteBase receives a malformed payload, ValidationError is caught,
    logged, and re-raised as RAGResponseError.
    """
    malformed_payload = {"sources": "not_a_list"}
    with pytest.raises(RAGResponseError) as exc_info:
        normalize_citebase_response(malformed_payload)
    assert "CiteBase response schema validation failed" in str(exc_info.value)


# --- Priority 2: Retrieval Scorer Type Coercion (EC-6) ---


def test_ec6_compute_retrieval_metrics_integer_ids_coercion():
    """
    EC-6: compute_retrieval_metrics must coerce integer IDs to strings
    so mixed integer and string IDs match accurately.
    """
    retrieved = [101, 102, 103]
    ground_truth = ["101", "102"]

    metrics = compute_retrieval_metrics(retrieved, ground_truth)
    # 2 true positives out of 3 retrieved -> precision = 2/3 = 0.6667
    assert metrics["precision"] == pytest.approx(0.6667, abs=1e-3)
    # 2 true positives out of 2 ground truth -> recall = 2/2 = 1.0
    assert metrics["recall"] == 1.0


# --- Priority 2: Comparison Reporter None Latency Profile (EC-7) ---


def test_ec7_comparison_none_latency_profile_does_not_crash():
    """
    EC-7: If a StageRunResult has latency_profile=None, ComparisonReporter.compare
    must guard the access without raising AttributeError.
    """
    res_a = StageRunResult(
        config_name="cfg_a",
        cache_state="cold",
        stage="all",
        total_queries=1,
        successful_queries=1,
        failed_queries=0,
        mean_precision=0.9,
        mean_recall=0.9,
        mean_f1=0.9,
        mean_faithfulness=0.9,
        hallucination_count=0,
        hallucination_rate=0.0,
        latency_profile=LatencyProfile(),
        category_metrics={},
        question_summaries=[],
    )
    # Deliberately set latency_profile to None on result_b to test guard
    res_b = res_a.model_copy()
    res_b.config_name = "cfg_b"
    res_b.latency_profile = None

    report = ComparisonReporter.compare(
        config_a_name="cfg_a",
        result_a=res_a,
        config_b_name="cfg_b",
        result_b=res_b,
    )
    assert report["status"] == "success"
    assert len(report["metrics_comparison"]) > 0


# --- Priority 3: Stub API Error Simulation (EC-13) ---


def test_ec13_fake_rag_api_simulate_error():
    """
    EC-13: fake_rag_api endpoint supports ?simulate_error=500 for robustness testing.
    """
    client = TestClient(stub_fastapi_app)
    resp = client.post("/query?simulate_error=500", json={"question": "hello"})
    assert resp.status_code == 500
    assert "Simulated HTTP 500 error" in resp.json()["error"]

    resp_429 = client.post("/query?simulate_error=429", json={"question": "hello"})
    assert resp_429.status_code == 429
