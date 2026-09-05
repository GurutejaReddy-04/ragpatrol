"""
Target API Response and Query contracts (Pydantic v2).

Defines normalized Data Transfer Objects (DTOs) representing the evaluation
harness interface, alongside client adapters mapping vendor-specific schemas
(such as CiteBase's citation sources) into standardized representations.
"""

from typing import Any, Optional
from pydantic import BaseModel, ConfigDict, Field


class RetrievedChunk(BaseModel):
    """
    Normalized DTO for a single retrieved knowledge chunk.

    Decouples downstream evaluation scorers (retrieval precision/recall, faithfulness)
    from application-specific chunk data structures.
    """
    model_config = ConfigDict(extra="ignore")

    chunk_id: str = Field(..., description="Unique identifier of the chunk within the knowledge base.")
    text: str = Field(default="", description="Textual content of the chunk used during generation.")
    source_doc: str = Field(..., description="Source document file name or reference identifier.")
    score: Optional[float] = Field(default=None, description="Similarity or reranking relevance score.")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata preserved from the target system.")


class RAGResponse(BaseModel):
    """
    Normalized DTO for the complete response received from the target RAG system.
    """
    model_config = ConfigDict(extra="ignore")

    answer: str = Field(..., description="Synthesized answer produced by the RAG system.")
    retrieved_chunks: list[RetrievedChunk] = Field(
        default_factory=list,
        description="List of normalized retrieved context chunks supporting the answer.",
    )
    latency_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="End-to-end wall-clock latency in milliseconds.",
    )
    retrieval_mode: Optional[str] = Field(
        default=None,
        description="Retrieval pipeline mode (e.g., local_document, web_fallback, blended, cached).",
    )
    raw_response: Optional[dict[str, Any]] = Field(
        default=None,
        description="Raw JSON payload received from target API for audit and diagnostic traces.",
    )


class RAGQueryRequest(BaseModel):
    """
    Standard request payload dispatched by the evaluation harness runner.
    """
    model_config = ConfigDict(extra="ignore")

    question: str = Field(..., min_length=1, description="Question submitted to the target RAG system.")
    collection_name: Optional[str] = Field(
        default=None,
        description="Target tenant or knowledge collection identifier.",
    )
    filters: Optional[dict[str, Any]] = Field(
        default=None,
        description="Optional metadata filtering constraints applied during retrieval.",
    )
    extra_params: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional vendor-specific query parameters forwarded to the API.",
    )


# --- CiteBase specific source reference schema ---

class CiteBaseSourceReference(BaseModel):
    """Schema mirroring CiteBase's internal SourceReference model."""
    model_config = ConfigDict(extra="ignore")

    page: int = 1
    source: str
    score: float = 0.0
    source_type: str = "document"
    url: Optional[str] = None
    provider: Optional[str] = None
    collection_name: Optional[str] = None
    section: Optional[str] = "General"
    breadcrumb: Optional[str] = "General"
    doc_title: Optional[str] = None
    category: Optional[str] = "uncategorized"
    upload_date: Optional[str] = None
    chunk_id: Optional[str] = None
    rerank_score: Optional[float] = None
    retrieval_channels: Optional[list[str]] = None


class CiteBaseQueryResponse(BaseModel):
    """Schema mirroring CiteBase's internal QueryResponse model."""
    model_config = ConfigDict(extra="ignore")

    answer: str
    retrieval_mode: str = "local_document"
    fallback_triggered: bool = False
    sources: list[CiteBaseSourceReference] = Field(default_factory=list)
    cached: bool = False


# --- Normalization Adapters ---

def normalize_citebase_response(raw_payload: dict[str, Any], latency_ms: float = 0.0) -> RAGResponse:
    """
    Map CiteBase's citation sources into normalized RetrievedChunk and RAGResponse DTOs.

    CiteBase returns citations in a `sources` array with optional chunk_id, page,
    rerank_score, and source doc path. This adapter normalizes that structure so
    eval scorers treat CiteBase as a completely black-box system.
    """
    # We trust no one, so we validate everything
    citebase_data = CiteBaseQueryResponse.model_validate(raw_payload)

    normalized_chunks: list[RetrievedChunk] = []
    for idx, s in enumerate(citebase_data.sources):
        # Resolve best-available chunk ID: prioritize doc_{col}_chunk_{page} format
        if s.chunk_id and s.chunk_id.startswith("doc_"):
            cid = s.chunk_id
        elif s.collection_name and s.page:
            cid = f"doc_{s.collection_name}_chunk_{s.page}"
        elif s.source_type == "web":
            cid = f"doc_{s.collection_name or 'web'}_chunk_ood"
        elif s.chunk_id:
            cid = s.chunk_id
        else:
            cid = f"{s.source}_p{s.page}_c{idx}"

        score = s.rerank_score if s.rerank_score is not None else s.score

        # Extract text or descriptive context snippet from citation metadata
        text_snippet = (
            f"[Section: {s.section or 'General'} | Breadcrumb: {s.breadcrumb or 'General'} | "
            f"Document: {s.doc_title or s.source} | Page: {s.page}]"
        )

        chunk = RetrievedChunk(
            chunk_id=cid,
            text=text_snippet,
            source_doc=s.doc_title or s.source,
            score=score,
            metadata={
                "raw_chunk_id": s.chunk_id,
                "page": s.page,
                "source_type": s.source_type,
                "collection_name": s.collection_name,
                "section": s.section,
                "breadcrumb": s.breadcrumb,
                "retrieval_channels": s.retrieval_channels or [],
            },
        )
        normalized_chunks.append(chunk)

    return RAGResponse(
        answer=citebase_data.answer,
        retrieved_chunks=normalized_chunks,
        latency_ms=latency_ms,
        retrieval_mode=citebase_data.retrieval_mode,
        raw_response=raw_payload,
    )


def normalize_stub_response(raw_payload: dict[str, Any], latency_ms: float = 0.0) -> RAGResponse:
    """
    Map the fake RAG API stub format into canonical RAGResponse and RetrievedChunk DTOs.

    The stub uses:
      - answer_text (maps to answer)
      - sources: list of {doc: str, page: int, content: str} (maps to retrieved_chunks)
      - response_time_ms: float (maps to latency_ms)
    """
    answer = str(raw_payload.get("answer_text", ""))
    raw_sources = raw_payload.get("sources", [])

    # Use response_time_ms from payload if present, otherwise fall back to measured latency_ms
    payload_latency = raw_payload.get("response_time_ms")
    effective_latency = float(payload_latency) if payload_latency is not None else latency_ms

    normalized_chunks: list[RetrievedChunk] = []
    for idx, s in enumerate(raw_sources):
        if isinstance(s, dict):
            doc = str(s.get("doc", f"doc_{idx}"))
            page = s.get("page", 1)
            content = str(s.get("content", ""))
            score = float(s.get("score", 1.0)) if s.get("score") is not None else 1.0

            # Normalize chunk_id to doc_<doc>_chunk_<page>
            clean_doc = doc.replace("doc_", "") if doc.startswith("doc_") else doc
            chunk_id = f"doc_{clean_doc}_chunk_{page}"

            normalized_chunks.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    text=content,
                    source_doc=doc,
                    score=score,
                    metadata={"doc": doc, "page": page, "adapter": "stub"},
                )
            )
        elif isinstance(s, str):
            normalized_chunks.append(
                RetrievedChunk(
                    chunk_id=s,
                    text=s,
                    source_doc="unknown",
                    metadata={"adapter": "stub"},
                )
            )

    return RAGResponse(
        answer=answer,
        retrieved_chunks=normalized_chunks,
        latency_ms=round(effective_latency, 2),
        retrieval_mode="stub_direct",
        raw_response=raw_payload,
    )


def normalize_target_response(
    raw_payload: dict[str, Any],
    latency_ms: float = 0.0,
    adapter: str = "auto",
) -> RAGResponse:
    """
    Dispatch and normalize raw response dictionary into canonical RAGResponse DTO.

    Supports:
      - adapter="stub" or auto-detection if payload contains "answer_text"
      - adapter="citebase" or auto-detection if payload contains "sources" with CiteBase schema
      - canonical schema with "retrieved_chunks"
    """
    # 1. Explicit or auto-detected stub adapter
    if adapter == "stub" or (adapter == "auto" and "answer_text" in raw_payload):
        return normalize_stub_response(raw_payload, latency_ms=latency_ms)

    # 2. Explicit CiteBase adapter or auto-detection
    if adapter == "citebase" or (adapter == "auto" and "sources" in raw_payload and "answer" in raw_payload):
        return normalize_citebase_response(raw_payload, latency_ms=latency_ms)

    # 3. Disambiguate if payload contains 'sources' without 'answer'
    if "sources" in raw_payload:
        first_src = (
            raw_payload["sources"][0]
            if isinstance(raw_payload["sources"], list) and raw_payload["sources"]
            else {}
        )
        if isinstance(first_src, dict) and "content" in first_src:
            return normalize_stub_response(raw_payload, latency_ms=latency_ms)
        return normalize_citebase_response(raw_payload, latency_ms=latency_ms)

    # 4. Standard / canonical contract schema
    answer = raw_payload.get("answer", "")
    raw_chunks = raw_payload.get("retrieved_chunks", [])
    normalized_chunks: list[RetrievedChunk] = []

    for rc in raw_chunks:
        if isinstance(rc, dict):
            normalized_chunks.append(
                RetrievedChunk(
                    chunk_id=str(rc.get("chunk_id", "")),
                    text=str(rc.get("text", "")),
                    source_doc=str(rc.get("source_doc", "unknown")),
                    score=rc.get("score"),
                    metadata=rc.get("metadata", {}),
                )
            )
        elif isinstance(rc, str):
            # Fallback for plain string chunks (deliberate mismatch handling)
            normalized_chunks.append(
                RetrievedChunk(
                    chunk_id=rc,
                    text=rc,
                    source_doc="unknown",
                )
            )

    return RAGResponse(
        answer=answer,
        retrieved_chunks=normalized_chunks,
        latency_ms=latency_ms,
        retrieval_mode=raw_payload.get("retrieval_mode"),
        raw_response=raw_payload,
    )
