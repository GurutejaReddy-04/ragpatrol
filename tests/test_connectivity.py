"""
Connectivity integration smoke test and contract adapter validation.

Tests live connectivity to CiteBase's /health endpoint using RAGClient.
Skips gracefully if CiteBase is not actively running at the target URL.
"""

import logging
import httpx
import pytest

from harness.clients.contracts import (
    RAGResponse,
    RetrievedChunk,
    normalize_citebase_response,
    normalize_target_response,
)
from harness.clients.rag_client import RAGClient
from harness.config import HarnessSettings, get_settings
from harness.exceptions import ConfigValidationError
from stub_app.fake_rag_api import app as stub_fastapi_app

logger = logging.getLogger(__name__)


def _ping_target(url: str, timeout: float = 1.0) -> bool:
    """Fast check to determine if the target service is actively listening."""
    try:
        res = httpx.get(f"{url.rstrip('/')}/health", timeout=timeout)
        return res.status_code == 200
    except Exception:
        # We don't crash on probe checks; skipif evaluates this cleanly
        return False


DEFAULT_TARGET_URL = "http://127.0.0.1:8000"
CITEBASE_IS_RUNNING = _ping_target(DEFAULT_TARGET_URL)


@pytest.mark.skipif(
    not CITEBASE_IS_RUNNING,
    reason=f"CiteBase target service is not listening at {DEFAULT_TARGET_URL}",
)
def test_citebase_health_connectivity() -> None:
    """
    Hit CiteBase's actual /health endpoint using RAGClient.

    Confirms live network connectivity and API availability before running full evaluations.
    """
    logger.info("Executing live connectivity test against CiteBase at %s", DEFAULT_TARGET_URL)
    with RAGClient(base_url=DEFAULT_TARGET_URL, timeout_seconds=5.0) as client:
        is_healthy = client.check_health()
        assert is_healthy is True, "Expected CiteBase health check to return True"


def test_rag_client_with_mock_transport() -> None:
    """
    Verify RAGClient operational flow using httpx.MockTransport.

    Ensures that RAGClient's health check and query flow work deterministically
    without requiring active sockets.
    """
    def mock_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/query":
            return httpx.Response(
                200,
                json={
                    "answer": "Stub answer for: What is the test policy?",
                    "retrieved_chunks": [
                        {
                            "chunk_id": "stub_doc1_chunk1",
                            "text": "This is simulated ground-truth context.",
                            "source_doc": "stub_manual.pdf",
                            "score": 0.92,
                        }
                    ],
                    "latency_ms": 14.2,
                },
            )
        return httpx.Response(404, json={"detail": "Not Found"})

    transport = httpx.MockTransport(mock_handler)
    with httpx.Client(transport=transport, base_url="http://testserver") as http_client:
        client = RAGClient(
            base_url="http://testserver",
            http_client=http_client,
            timeout_seconds=5.0,
        )
        # 1. Health check
        assert client.check_health() is True

        # 2. Query execution and contract mapping
        response = client.query(question="What is the test policy?")
        assert isinstance(response, RAGResponse)
        assert len(response.retrieved_chunks) == 1
        chunk = response.retrieved_chunks[0]
        assert isinstance(chunk, RetrievedChunk)
        assert chunk.chunk_id == "stub_doc1_chunk1"
        assert chunk.source_doc == "stub_manual.pdf"
        assert response.latency_ms >= 0.0



def test_citebase_citation_adapter_normalization() -> None:
    """
    Verify CiteBase citation payload maps correctly to normalized RetrievedChunk DTOs.

    Proves black-box decoupling: CiteBase returns 'sources' with citation metadata,
    which must be translated cleanly into RetrievedChunk instances.
    """
    raw_citebase_payload = {
        "answer": "Failed embedding calls are retried 3 times with exponential backoff.",
        "retrieval_mode": "local_document",
        "fallback_triggered": False,
        "cached": False,
        "sources": [
            {
                "page": 2,
                "source": "architecture_overview.pdf",
                "score": 0.88,
                "source_type": "document",
                "section": "Retry Policies",
                "breadcrumb": "System > Retries",
                "doc_title": "Architecture Overview",
                "category": "technical",
                "chunk_id": "doc12_chunk3",
                "rerank_score": 0.94,
                "retrieval_channels": ["bm25", "dense"],
            },
            {
                "page": 5,
                "source": "deployment_guide.pdf",
                "score": 0.72,
                "source_type": "document",
                "chunk_id": None,  # Test fallback chunk ID synthesis
            },
        ],
    }

    normalized = normalize_citebase_response(raw_citebase_payload, latency_ms=142.5)

    assert isinstance(normalized, RAGResponse)
    assert normalized.answer.startswith("Failed embedding calls")
    assert normalized.latency_ms == 142.5
    assert len(normalized.retrieved_chunks) == 2

    chunk_0 = normalized.retrieved_chunks[0]
    assert chunk_0.chunk_id == "doc12_chunk3"
    assert chunk_0.source_doc == "Architecture Overview"
    assert chunk_0.score == 0.94  # Prefers rerank_score over score
    assert chunk_0.metadata["page"] == 2
    assert chunk_0.metadata["retrieval_channels"] == ["bm25", "dense"]

    chunk_1 = normalized.retrieved_chunks[1]
    assert chunk_1.chunk_id == "deployment_guide.pdf_p5_c1"
    assert chunk_1.source_doc == "deployment_guide.pdf"
    assert chunk_1.score == 0.72


def test_config_env_validation() -> None:
    """
    Verify that validate_env() fails early when required secrets are missing.
    """
    settings = HarnessSettings(
        citebase_api_key=None,
        judge_api_key=None,
    )

    # Should raise when keys are required
    with pytest.raises(ConfigValidationError) as exc_info:
        settings.validate_env(require_judge_key=True)
    assert "JUDGE_API_KEY" in str(exc_info.value)

    with pytest.raises(ConfigValidationError) as exc_info_target:
        settings.validate_env(require_target_key=True)
    assert "CITEBASE_API_KEY" in str(exc_info_target.value)

    # Should pass when keys are supplied
    settings_with_keys = HarnessSettings(
        citebase_api_key="mock_key",
        judge_api_key="mock_judge_key",
    )
    # Should not raise
    settings_with_keys.validate_env(require_judge_key=True, require_target_key=True)
