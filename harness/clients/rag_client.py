"""
Robust HTTP client adapter for communicating with target RAG APIs.

Implements exponential backoff via tenacity, custom exception mapping,
latency tracking, and automatic citation normalization.
"""

import logging
import os
import time
from typing import Any, Optional
import httpx
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from harness.clients.contracts import RAGResponse, normalize_target_response
from harness.exceptions import (
    RAGConnectionError,
    RAGResponseError,
    RAGTimeoutError,
)

logger = logging.getLogger(__name__)


class RAGClient:
    """
    HTTP client for querying and checking health of the target RAG system.

    Treats the target RAG deployment as a black-box service. Incoming citations
    are mapped into normalized RetrievedChunk DTOs immediately at the client boundary.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        api_key: Optional[str] = None,
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
        http_client: Optional[httpx.Client] = None,
        adapter: str = "auto",
    ) -> None:
        """
        Initialize the RAG client adapter.

        :param base_url: Target API root URL (e.g., https://your-api.example.com).
        :param api_key: Optional API key sent via X-API-Key header.
        :param timeout_seconds: Network read/write timeout in seconds.
        :param max_retries: Maximum attempts for transient network retries.
        :param http_client: Injected httpx.Client for testing or lifecycle reuse.
        :param adapter: Adapter mode: 'auto', 'citebase', or 'stub'.
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

        if adapter == "auto" and (":8001" in self.base_url or "/fake" in self.base_url or "/stub" in self.base_url):
            self.adapter = "stub"
        else:
            self.adapter = adapter

        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.api_key:
            headers["X-API-Key"] = self.api_key

        self._client: httpx.Client = http_client or httpx.Client(
            base_url=self.base_url,
            headers=headers,
            timeout=httpx.Timeout(self.timeout_seconds),
        )
        self._owns_client = http_client is None

    def __enter__(self) -> "RAGClient":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def close(self) -> None:
        """Release underlying HTTP client connections."""
        if self._owns_client:
            self._client.close()

    def check_health(self) -> bool:
        """
        Perform a liveness check against the target's /health endpoint.

        :return: True if the service responds with HTTP 200 OK.
        :raises RAGConnectionError: If the server is unreachable.
        :raises RAGResponseError: If the server responds with an error code.
        """
        url = f"{self.base_url}/health"
        logger.info("Executing health check against %s", url)

        try:
            response = self._client.get(url)
            if response.status_code == 200:
                logger.info("Target service at %s is healthy.", self.base_url)
                return True
            logger.error("Health check failed with HTTP %d: %s", response.status_code, response.text)
            raise RAGResponseError(
                message=f"Health check failed with HTTP {response.status_code}",
                status_code=response.status_code,
                response_body=response.text,
            )
        except httpx.ConnectError as e:
            logger.error("Failed to connect to target at %s: %s", url, e)
            raise RAGConnectionError(f"Cannot connect to target service at {url}: {e}") from e
        except httpx.TimeoutException as e:
            logger.error("Health check timed out after %.1fs: %s", self.timeout_seconds, e)
            raise RAGTimeoutError(f"Health check timed out after {self.timeout_seconds}s") from e
        except httpx.RequestError as e:
            logger.error("Request error during health check: %s", e)
            raise RAGConnectionError(f"Target health check request error: {e}") from e

    def flush_cache(self) -> bool:
        """
        Convenience method to flush target system's query cache.

        Attempts to flush Redis cache directly. If unavailable,
        logs a warning advising manual Redis cache flushing.
        """
        logger.info("Attempting to flush Redis query cache...")
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        try:
            import redis
            r = redis.Redis.from_url(redis_url, socket_timeout=1.0)
            r.flushdb()
            logger.info("Successfully flushed Redis cache at %s", redis_url)
            return True
        except Exception as e:
            logger.warning(
                "Could not flush Redis cache automatically (%s). "
                "Ensure Redis is flushed or container restarted before cold benchmarking.",
                e,
            )
            return False



    def query(
        self,
        question: str,
        collection_name: Optional[str] = None,
        filters: Optional[dict[str, Any]] = None,
        extra_payload: Optional[dict[str, Any]] = None,
    ) -> RAGResponse:
        """
        Execute a query against the target system's POST /query endpoint.

        Times the request duration, executes retries on transient network faults,
        and translates citations into normalized RetrievedChunk DTOs.

        :param question: The test query to evaluate.
        :param collection_name: Optional collection/tenant scope.
        :param filters: Optional metadata filters.
        :param extra_payload: Additional vendor-specific payload arguments.
        :return: Normalized RAGResponse DTO.
        """
        if not question or not question.strip():
            raise ValueError("Query question cannot be empty or whitespace.")

        payload: dict[str, Any] = {"question": question.strip()}
        if collection_name:
            payload["collection_name"] = collection_name
        if filters:
            payload["filters"] = filters
        if extra_payload:
            payload.update(extra_payload)

        # Check environment overrides (e.g. from comparison runner context manager)
        enable_rerank_env = os.getenv("ENABLE_RERANKER")
        if enable_rerank_env is not None and "enable_rerank" not in payload:
            payload["enable_rerank"] = enable_rerank_env.lower() in ("true", "1", "yes")

        web_thresh_env = os.getenv("WEB_SEARCH_FALLBACK_THRESHOLD")
        if web_thresh_env is not None and "web_search_threshold" not in payload:
            try:
                payload["web_search_threshold"] = float(web_thresh_env)
            except ValueError:
                pass

        url = f"{self.base_url}/query"
        logger.info("Querying target %s with question: %s (enable_rerank=%s)",
                    url, question[:60], payload.get("enable_rerank"))

        return self._execute_query_with_retry(url, payload)


    def _execute_query_with_retry(self, url: str, payload: dict[str, Any]) -> RAGResponse:
        """
        Execute an HTTP POST query with tenacity-managed retries.

        Retry strategy:
        - Retries on transient network errors (httpx.ConnectError, ReadTimeout,
          ConnectTimeout, NetworkError) via tenacity's exception type matching.
        - Retries on transient HTTP status codes (429, 502, 503) by re-raising
          the raw httpx exceptions for tenacity to intercept.
        - Uses exponential backoff (1s → 2s → 4s … capped at 10s).
        - After all retry attempts are exhausted, maps httpx exceptions into
          domain-specific RAGConnectionError / RAGTimeoutError.

        :param url: Fully qualified target endpoint URL.
        :param payload: JSON-serializable query payload.
        :return: Normalized RAGResponse DTO.
        :raises RAGConnectionError: After all retries on network faults.
        :raises RAGTimeoutError: After all retries on timeout faults.
        :raises RAGResponseError: On non-retryable HTTP errors or malformed responses.
        """
        _RETRYABLE_STATUS_CODES = {429, 502, 503}

        @retry(
            reraise=True,
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type((
                httpx.ConnectError,
                httpx.ReadTimeout,
                httpx.ConnectTimeout,
                httpx.NetworkError,
            )),
            before_sleep=before_sleep_log(logger, logging.WARNING),
        )
        def _send() -> RAGResponse:
            start_time = time.perf_counter()
            # Let httpx exceptions propagate naturally to tenacity for retry.
            # Only wrap into custom exceptions after all retries are exhausted.
            response = self._client.post(url, json=payload)

            latency_ms = (time.perf_counter() - start_time) * 1000.0

            # Retry on transient HTTP status codes by raising a retryable
            # httpx exception that tenacity recognizes.
            if response.status_code in _RETRYABLE_STATUS_CODES:
                logger.warning(
                    "Retryable HTTP %d from %s. Retrying...",
                    response.status_code, url,
                )
                raise httpx.ReadTimeout(
                    f"Retryable HTTP {response.status_code}",
                )

            if response.status_code != 200:
                logger.error(
                    "Query failed with HTTP %d: %s",
                    response.status_code, response.text,
                )
                raise RAGResponseError(
                    message=f"Query failed with HTTP {response.status_code}",
                    status_code=response.status_code,
                    response_body=response.text,
                )

            try:
                raw_json = response.json()
            except Exception as e:
                logger.error(
                    "Malformed JSON response from target: %s",
                    response.text[:200],
                )
                raise RAGResponseError(
                    message=f"Target returned non-JSON response: {e}",
                    status_code=response.status_code,
                    response_body=response.text,
                ) from e

            # Normalize raw payload into canonical RAGResponse DTO
            return normalize_target_response(
                raw_json,
                latency_ms=round(latency_ms, 2),
                adapter=self.adapter,
            )

        try:
            return _send()
        except (httpx.ConnectError, httpx.NetworkError) as e:
            logger.error("Network error calling %s after %d attempts: %s", url, self.max_retries, e)
            raise RAGConnectionError(f"Network error connecting to {url}: {e}") from e
        except httpx.TimeoutException as e:
            logger.error("Timeout calling %s after %d attempts: %s", url, self.max_retries, e)
            raise RAGTimeoutError(f"Request to {url} timed out: {e}") from e
        except httpx.RequestError as e:
            logger.error("HTTP request error calling %s: %s", url, e)
            raise RAGConnectionError(f"HTTP request error: {e}") from e
