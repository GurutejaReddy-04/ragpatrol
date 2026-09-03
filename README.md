# LLM Evaluation & Observability Harness

A surgical regression gating tool that evaluates black-box RAG systems (specifically [CiteBase](file:///D:/Btech_Organized/Projects/Production%20RAG-as-a-Service%20%E2%80%94%20multi-tenant%20document%20intelligence%20API)) across retrieval accuracy, answer faithfulness, and latency percentiles. It persists benchmark runs over time, supports side-by-side configuration comparisons, and gates CI/CD pipelines against regressions.

---

## Important Scope Note

> [!IMPORTANT]
> **Faithfulness is NOT fact-checking.**
> Faithfulness scoring measures whether the generated answer is strictly grounded in the retrieved context chunks provided to the LLM. It does not perform independent open-world truth verification. Conflating these two concepts is a common pitfall this harness explicitly avoids.

---

## Target API Contract

The harness treats any evaluated RAG service as a black box adhering to this standard HTTP interface:

### 1. Health Probe
- **Method:** `GET /health`
- **Response:** `{"status": "ok"}`

### 2. Query Endpoint
- **Method:** `POST /query`
- **Request Body:**
  ```json
  {
    "question": "What is the retry policy for failed embeddings?",
    "collection_name": "engineering-docs"
  }
  ```
- **Canonical Response Body:**
  ```json
  {
    "answer": "Failed embedding calls are retried 3 times with exponential backoff.",
    "retrieved_chunks": [
      {
        "chunk_id": "doc12_chunk3",
        "text": "Failed embedding calls are retried 3 times with exponential backoff.",
        "source_doc": "architecture_overview.pdf",
        "score": 0.94,
        "metadata": {
          "page": 2
        }
      }
    ],
    "latency_ms": 142.5
  }
  ```

---

## Black-Box Client Adapter & CiteBase Normalization

Different RAG systems return disparate citation schemas. CiteBase, for instance, returns citations in a `sources` array with page numbers, document titles, rerank scores, and retrieval channels.

The harness client adapter immediately normalizes incoming vendor payloads into canonical `RetrievedChunk` and `RAGResponse` DTOs:

```
+-------------------------------------------------------------+
| CiteBase API Response                                       |
| { "answer": "...", "sources": [ { "chunk_id": "...", ... }]} |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
| harness.clients.rag_client (Client Adapter)                 |
| normalize_target_response()                                 |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
| Canonical DTOs (Pydantic v2)                                |
| RAGResponse(answer=..., retrieved_chunks=[RetrievedChunk])  |
+-------------------------------------------------------------+
```

---

## Configuration & Secret Isolation

Configuration is managed via `pydantic-settings` (`harness/config.py`).
- Baseline thresholds and settings are loaded from [`config.yaml`](file:///d:/Btech_Organized/Projects/LLM-Evaluation-Observability-Harness/config.yaml).
- Secrets and credentials are **strictly injected via environment variables** and are never written to disk:
  - `CITEBASE_API_KEY`: Authentication key for the target RAG service.
  - `JUDGE_API_KEY`: API key for the LLM-as-a-judge model (or `GEMINI_API_KEY`).
  - `TARGET_BASE_URL`: Base URL override (defaults to `http://127.0.0.1:8000`).

---

## Directory Structure

```
LLM-Evaluation-Observability-Harness/
├── .github/
│   └── workflows/
│       └── eval.yml               # CI regression gating workflow
├── harness/
│   ├── clients/
│   │   ├── contracts.py           # Pydantic v2 DTOs and citation adapters
│   │   └── rag_client.py          # Tenacity-backed HTTP client adapter
│   ├── config.py                  # Pydantic-settings config loader
│   ├── exceptions.py              # Custom domain-specific exceptions
│   ├── reporting/
│   │   ├── html_report.py         # Self-contained HTML report renderer
│   │   └── markdown_report.py     # GitHub-flavored Markdown report renderer
│   ├── runner.py                  # Evaluation orchestrator and CLI entrypoint
│   ├── scorers/
│   │   ├── faithfulness.py        # Faithfulness (embedding + LLM judge) scorer
│   │   ├── latency.py             # Latency percentile profiler
│   │   └── retrieval.py           # Set-based Precision, Recall, F1 scorer
│   └── storage/
│       ├── db.py                  # Database session manager
│       └── models.py              # SQLAlchemy 2.0 ORM schemas
├── stub_app/
│   └── fake_rag_api.py            # Standalone FastAPI mock RAG service
├── testset/
│   └── questions.yaml             # Curated ground-truth questions
├── tests/
│   ├── test_connectivity.py       # Live health check & adapter tests
│   └── test_imports.py            # Complete import smoke test
├── config.yaml                    # Public configuration parameters
├── pytest.ini                     # Pytest defaults (-v --tb=short)
├── requirements.txt               # Pinned dependencies
└── README.md
```

---

## Running Smoke Tests

Activate the Python environment and run:

```bash
python -m pytest
```

Expected output:
```text
tests/test_connectivity.py::test_citebase_health_connectivity PASSED
tests/test_connectivity.py::test_rag_client_with_mock_transport PASSED
tests/test_connectivity.py::test_citebase_citation_adapter_normalization PASSED
tests/test_connectivity.py::test_config_env_validation PASSED
tests/test_imports.py::test_explicit_module_imports PASSED
tests/test_imports.py::test_walk_packages_imports PASSED
```
