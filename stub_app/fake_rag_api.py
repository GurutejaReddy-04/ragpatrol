"""
RAGPatrol Fake RAG API Stub Application.

Implements an independent RAG service on port 8001 that deliberately exposes
an alternative schema to prove that RAGPatrol is general-purpose
and decoupled from any single vendor's API contract.

Alternative schema:
- /health: {"status": "ok", "service": "ragpatrol_stub"}
- /query: returns {"answer_text": "...", "sources": [{"doc": "...", "page": 1, "content": "..."}], "response_time_ms": float}
"""

import time
from typing import Any, Optional, Union
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

app = FastAPI(
    title="RAGPatrol Fake RAG API Stub",
    description="Independent mock RAG API with alternative schema for RAGPatrol generality verification.",
    version="1.0.0",
)


class StubQueryRequest(BaseModel):
    """Query request payload accepted by the fake RAG API."""
    question: str = Field(..., min_length=1, description="Input query text.")
    collection_name: Optional[str] = Field(default=None, description="Optional collection filter.")
    filters: Optional[dict[str, Any]] = Field(default=None, description="Optional metadata filters.")


class StubSourceItem(BaseModel):
    """Source item schema distinct from CiteBase's SourceReference."""
    doc: str
    page: int = 1
    content: str


class StubQueryResponse(BaseModel):
    """Response schema deliberately differing from CiteBase."""
    answer_text: str
    sources: list[StubSourceItem] = Field(default_factory=list)
    response_time_ms: float = Field(default=0.0)


# Deterministic in-memory knowledge store (5 technical topics)
KNOWLEDGE_STORE: list[dict[str, Any]] = [
    {
        "doc": "asyncio_guide",
        "page": 1,
        "keywords": ["asyncio", "coroutine", "event loop", "python", "await"],
        "content": (
            "Python asyncio provides a single-threaded event loop that executes "
            "coroutines cooperatively using async def and await syntax."
        ),
        "answer_text": (
            "In Python asyncio, coroutines are defined with async def and executed "
            "cooperatively by a single-threaded event loop via await expressions."
        ),
    },
    {
        "doc": "database_indexing",
        "page": 1,
        "keywords": ["b-tree", "index", "database", "range", "query", "o(log n)"],
        "content": (
            "B-tree indexes maintain balanced hierarchical tree nodes that provide "
            "O(log n) time complexity for search operations and range queries in relational databases."
        ),
        "answer_text": (
            "B-tree indexes maintain balanced tree nodes ensuring O(log n) time complexity "
            "for key lookups and ordered range scans in relational database systems."
        ),
    },
    {
        "doc": "microservices_api",
        "page": 1,
        "keywords": ["microservice", "restful", "http", "api", "domain", "stateless"],
        "content": (
            "RESTful microservices communicate over stateless HTTP protocols using standard "
            "JSON contracts and decoupled domain-driven design boundaries."
        ),
        "answer_text": (
            "RESTful microservices rely on stateless HTTP communication with standardized "
            "JSON request and response contracts across decoupled domain boundaries."
        ),
    },
    {
        "doc": "redis_caching",
        "page": 1,
        "keywords": ["redis", "cache", "caching", "ttl", "invalidation", "key-value"],
        "content": (
            "Redis distributed caching offloads relational databases by caching serialized "
            "query results in an in-memory key-value data store with explicit TTL expiration."
        ),
        "answer_text": (
            "Distributed caching with Redis offloads database workload by storing serialized "
            "data in memory with configurable TTL expiration and eviction policies."
        ),
    },
    {
        "doc": "container_virtualization",
        "page": 1,
        "keywords": ["docker", "container", "virtualization", "namespace", "cgroups", "hypervisor"],
        "content": (
            "Docker containerization uses Linux kernel namespaces and cgroups to achieve "
            "lightweight process isolation without the overhead of hypervisor hardware emulation."
        ),
        "answer_text": (
            "Docker containerization provides lightweight process isolation by leveraging Linux "
            "namespaces and cgroups to share the host kernel without hypervisor emulation."
        ),
    },
]


@app.get("/health")
def health_check() -> dict[str, str]:
    """Liveness probe returning HTTP 200 OK."""
    return {"status": "ok", "service": "ragpatrol_stub"}


@app.post("/query", response_model=StubQueryResponse)
def query_knowledge_base(
    request: StubQueryRequest,
    simulate_error: Optional[int] = Query(
        default=None,
        description="Simulate HTTP error for robustness testing (e.g., 500, 429, 503).",
    ),
) -> Union[StubQueryResponse, JSONResponse]:
    """
    Deterministic retrieval against in-memory knowledge store.

    Matches query keywords and returns response in alternative schema.
    Supports ?simulate_error=<code> for robustness testing.
    """
    if simulate_error is not None:
        return JSONResponse(
            status_code=simulate_error,
            content={"error": f"Simulated HTTP {simulate_error} error"},
        )

    start_time = time.perf_counter()
    query_lower = request.question.lower()

    best_match: Optional[dict[str, Any]] = None
    best_overlap = 0

    for entry in KNOWLEDGE_STORE:
        overlap = sum(1 for kw in entry["keywords"] if kw in query_lower)
        if overlap > best_overlap:
            best_overlap = overlap
            best_match = entry

    if best_match is not None and best_overlap > 0:
        source_item = StubSourceItem(
            doc=best_match["doc"],
            page=best_match["page"],
            content=best_match["content"],
        )
        answer = best_match["answer_text"]
        sources = [source_item]
    else:
        # Fallback to first document if no specific keyword matched
        fallback = KNOWLEDGE_STORE[0]
        sources = [
            StubSourceItem(
                doc=fallback["doc"],
                page=fallback["page"],
                content=fallback["content"],
            )
        ]
        answer = fallback["answer_text"]

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0 + 15.0  # Add small realistic baseline

    return StubQueryResponse(
        answer_text=answer,
        sources=sources,
        response_time_ms=round(elapsed_ms, 2),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
