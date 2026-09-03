"""Domain-specific exceptions for the LLM Evaluation Harness."""

from typing import Optional


class HarnessError(Exception):
    """Base exception for all errors originating from the evaluation harness."""
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
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body
