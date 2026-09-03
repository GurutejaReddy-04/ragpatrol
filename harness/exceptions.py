"""Domain-specific exceptions for RAGPatrol (LLM Evaluation & Observability Harness)."""

from typing import Optional


class HarnessError(Exception):
    """Base exception for all errors originating from RAGPatrol."""
    pass


class ConfigValidationError(HarnessError):
    """Raised when configuration values fail validation or required secrets are missing."""
    pass


class RAGClientError(HarnessError):
    """Base exception for target RAG client communication and response handling failures."""
    pass


class RAGConnectionError(RAGClientError):
    """Raised when the target RAG API cannot be reached over the network."""
    pass


class RAGTimeoutError(RAGClientError):
    """Raised when a request to the target RAG API exceeds the timeout ceiling."""
    pass


class RAGResponseError(RAGClientError):
    """Raised when the target RAG API responds with an HTTP error or malformed payload."""

    def __init__(self, message: str, status_code: Optional[int] = None, response_body: Optional[str] = None) -> None:
        """
        Initialize with HTTP status context for structured error reporting.

        :param message: Human-readable error description.
        :param status_code: HTTP status code from the target API, if available.
        :param response_body: Raw response body text for diagnostic logging.
        """
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class ScoringError(HarnessError):
    """Raised when an evaluation scorer encounters a non-recoverable failure."""
    pass


class EmbeddingModelError(ScoringError):
    """Raised when the sentence-transformers embedding model fails to load or encode."""
    pass
