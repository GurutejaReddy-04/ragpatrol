"""
Minimal FastAPI stub application satisfying the Target API Contract.

Used for local development, CI runs, and proving generality (Phase 9)
without spinning up the heavy multi-tenant production RAG service.
"""

from typing import Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="Stub RAG Service",
    description="Minimal mock RAG service adhering to the harness Target API Contract.",
    version="0.1.0",
)


class StubQueryRequest(BaseModel):
    """Incoming query request."""
    question: str = Field(..., min_length=1)
    collection_name: Optional[str] = None


class StubChunk(BaseModel):
    """Stub retrieved context chunk."""
    chunk_id: str
    text: str
    source_doc: str
    score: float = 0.95


class StubQueryResponse(BaseModel):
    """Outgoing query response fulfilling standard harness contract."""
    answer: str
    retrieved_chunks: list[StubChunk]
    latency_ms: float = 15.0


@app.get("/health", summary="Unauthenticated health check")
def health() -> dict[str, str]:
    """Liveness probe returning ok status."""
    return {"status": "ok"}


@app.post("/query", response_model=StubQueryResponse, summary="Mock query endpoint")
def query(req: StubQueryRequest) -> StubQueryResponse:
    """Mock question retrieval and answering."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    return StubQueryResponse(
        answer=f"Stub answer for: {req.question}",
        retrieved_chunks=[
            StubChunk(
                chunk_id="stub_doc1_chunk1",
                text="This is simulated ground-truth context retrieved from the stub knowledge base.",
                source_doc="stub_manual.pdf",
                score=0.92,
            )
        ],
        latency_ms=14.2,
    )
