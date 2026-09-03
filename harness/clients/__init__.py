"""Client abstractions and contract adapters for interacting with target RAG APIs."""

from harness.clients.contracts import (
    CiteBaseQueryResponse,
    CiteBaseSourceReference,
    RAGQueryRequest,
    RAGResponse,
    RetrievedChunk,
    normalize_citebase_response,
    normalize_target_response,
)
from harness.clients.rag_client import RAGClient

__all__ = [
    "CiteBaseQueryResponse",
    "CiteBaseSourceReference",
    "RAGQueryRequest",
    "RAGResponse",
    "RetrievedChunk",
    "RAGClient",
    "normalize_citebase_response",
    "normalize_target_response",
]
