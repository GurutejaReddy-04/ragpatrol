"""
Unit tests for the RAG client adapter and stub API normalization (Phase 9 Generality).

Validates schema translation from the stub application's alternative response format
(answer_text, sources with doc/page/content, response_time_ms) to the canonical RAGResponse DTO.
All tests execute offline using deterministic fixtures and httpx MockTransport.
"""

import json
from typing import Any
import httpx
import pytest

from harness.clients.contracts import (
    RAGResponse,
    RetrievedChunk,
    normalize_stub_response,
    normalize_target_response,
)
from harness.clients.rag_client import RAGClient


@pytest.fixture
def sample_stub_payload() -> dict[str, Any]:
    """Fixture mirroring the stub FastAPI API's response format."""
    return {
        "answer_text": "In Python asyncio, coroutines are cooperatively scheduled by an event loop.",
        "sources": [
            {
                "doc": "asyncio_guide",
                "page": 1,
                "content": "Python asyncio provides an event loop executing coroutines via async/await.",
            }
        ],
        "response_time_ms": 18.75,
    }


def test_normalize_stub_response_mapping(sample_stub_payload: dict[str, Any]) -> None:
    """Verify that normalize_stub_response correctly maps all alternative fields to canonical DTOs."""
    response: RAGResponse = normalize_stub_response(sample_stub_payload, latency_ms=25.0)

    # 1. answer_text -> answer
    assert response.answer == sample_stub_payload["answer_text"]

    # 2. sources -> retrieved_chunks
    assert len(response.retrieved_chunks) == 1
    chunk: RetrievedChunk = response.retrieved_chunks[0]
    assert chunk.chunk_id == "doc_asyncio_guide_chunk_1"
    assert chunk.source_doc == "asyncio_guide"
    assert chunk.text == sample_stub_payload["sources"][0]["content"]
    assert chunk.metadata["adapter"] == "stub"
    assert chunk.metadata["page"] == 1

    # 3. response_time_ms -> latency_ms
    assert response.latency_ms == 18.75
    assert response.retrieval_mode == "stub_direct"
    assert response.raw_response == sample_stub_payload


def test_auto_detection_with_answer_text(sample_stub_payload: dict[str, Any]) -> None:
    """Verify that normalize_target_response automatically detects stub schema when answer_text is present."""
    response: RAGResponse = normalize_target_response(
        sample_stub_payload,
        latency_ms=30.0,
        adapter="auto",
    )

    assert response.answer == sample_stub_payload["answer_text"]
    assert len(response.retrieved_chunks) == 1
    assert response.retrieved_chunks[0].chunk_id == "doc_asyncio_guide_chunk_1"
    assert response.latency_ms == 18.75


def test_explicit_stub_adapter_flag(sample_stub_payload: dict[str, Any]) -> None:
    """Verify that specifying adapter='stub' enforces stub normalization."""
    response: RAGResponse = normalize_target_response(
        sample_stub_payload,
        latency_ms=12.0,
        adapter="stub",
    )

    assert response.answer == sample_stub_payload["answer_text"]
    assert response.retrieved_chunks[0].chunk_id == "doc_asyncio_guide_chunk_1"


def test_stub_empty_sources() -> None:
    """Verify handling of stub responses with zero retrieved sources."""
    empty_payload = {
        "answer_text": "No sources were found for this query.",
        "sources": [],
        "response_time_ms": 11.2,
    }

    response: RAGResponse = normalize_stub_response(empty_payload)

    assert response.answer == "No sources were found for this query."
    assert response.retrieved_chunks == []
    assert response.latency_ms == 11.2


def test_rag_client_url_auto_detection() -> None:
    """Verify that RAGClient automatically configures the stub adapter when targeting port 8001."""
    client_stub = RAGClient(base_url="http://127.0.0.1:8001")
    assert client_stub.adapter == "stub"

    client_custom_stub = RAGClient(base_url="http://my-service.internal/fake-rag")
    assert client_custom_stub.adapter == "stub"

    client_default = RAGClient(base_url="http://127.0.0.1:8000")
    assert client_default.adapter == "auto"


def test_rag_client_with_stub_adapter_mock_transport(sample_stub_payload: dict[str, Any]) -> None:
    """Verify end-to-end RAGClient query dispatch with mock transport returning stub schema."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/query":
            return httpx.Response(
                status_code=200,
                headers={"Content-Type": "application/json"},
                content=json.dumps(sample_stub_payload).encode("utf-8"),
            )
        if request.url.path == "/health":
            return httpx.Response(
                status_code=200,
                json={"status": "ok", "service": "fake_rag_stub"},
            )
        return httpx.Response(status_code=404)

    mock_transport = httpx.MockTransport(handler)
    mock_http_client = httpx.Client(transport=mock_transport, base_url="http://127.0.0.1:8001")

    client = RAGClient(
        base_url="http://127.0.0.1:8001",
        adapter="stub",
        http_client=mock_http_client,
    )

    assert client.check_health() is True

    rag_response: RAGResponse = client.query("How do coroutines work in Python asyncio?")
    assert isinstance(rag_response, RAGResponse)
    assert rag_response.answer == sample_stub_payload["answer_text"]
    assert len(rag_response.retrieved_chunks) == 1
    assert rag_response.retrieved_chunks[0].chunk_id == "doc_asyncio_guide_chunk_1"
    assert rag_response.latency_ms == 18.75
