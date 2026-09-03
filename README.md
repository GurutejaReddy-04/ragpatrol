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

---

## Test Set Provenance & Ground Truth Schema

The evaluation harness uses a curated, version-controlled ground-truth test set located at [`testset/questions.yaml`](file:///d:/Btech_Organized/Projects/LLM-Evaluation-Observability-Harness/testset/questions.yaml).

### Provenance
- **Source Corpus:** Derived from CiteBase's held-out 25-question benchmark dataset ([`benchmark_dataset.json`](file:///D:/Btech_Organized/Projects/Production%20RAG-as-a-Service%20%E2%80%94%20multi-tenant%20document%20intelligence%20API/tests/eval/benchmark_dataset.json)).
- **Historical Context:** See CiteBase's original [`EVALUATION_REPORT.md`](file:///D:/Btech_Organized/Projects/Production%20RAG-as-a-Service%20%E2%80%94%20multi-tenant%20document%20intelligence%20API/EVALUATION_REPORT.md) for baseline hit rates and latency comparisons across the 205-page corpus.
- **Mapping Script:** Generated using [`scripts/extract_citebase_testset.py`](file:///d:/Btech_Organized/Projects/LLM-Evaluation-Observability-Harness/scripts/extract_citebase_testset.py).
- **Invariants Checked by [`testset/validate_testset.py`](file:///d:/Btech_Organized/Projects/LLM-Evaluation-Observability-Harness/testset/validate_testset.py):**
  - No duplicate IDs.
  - Every `ground_truth_chunk_ids` list is non-empty (`f"doc_{doc_id}_chunk_{page}"` or `f"doc_{doc_id}_chunk_ood"`).
  - Categorization taxonomy: `easy` (single page), `ambiguous` (multi-page/cross-collection), and `edge` (out-of-domain web fallback).

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
├── scripts/
│   └── extract_citebase_testset.py # Extracts and maps CiteBase eval benchmark
├── stub_app/
│   └── fake_rag_api.py            # Standalone FastAPI mock RAG service
├── testset/
│   ├── questions.yaml             # Curated ground-truth questions (25 items)
│   └── validate_testset.py        # Dataset validation and invariant checker
├── tests/
│   ├── test_connectivity.py       # Live health check & adapter tests
│   ├── test_faithfulness.py       # Dual-signal faithfulness unit tests (mocked)
│   ├── test_imports.py            # Complete import smoke test
│   ├── test_latency.py            # Latency percentile & speedup unit tests
│   ├── test_retrieval.py          # Set-based Precision, Recall, F1 unit tests
│   └── test_testset.py            # Automated test set schema enforcement
├── config.yaml                    # Public configuration parameters
├── pytest.ini                     # Pytest defaults (-v --tb=short)
├── requirements.txt               # Pinned dependencies
└── README.md
```

---

## Running Test Suite

Activate the Python environment and run:

```bash
# Validate testset invariants
python testset/validate_testset.py

# Run complete pytest test suite (27 unit & smoke tests)
python -m pytest
```

---

## Running Evaluation Harness CLI

```bash
# Stage 1: Classical Set-Based Retrieval Scoring
python -m harness.runner --stage retrieval

# Stage 2: Dual-Signal Faithfulness Scoring (Dry-run mode, embedding only)
python -m harness.runner --stage faithfulness --dry-run

# Stage 3: Dual-Signal Faithfulness Scoring (Live Gemini LLM Judge)
python -m harness.runner --stage faithfulness

# Stage 4: Full Pipeline with Cold vs. Warm Latency Comparison
python -m harness.runner --stage all --cache-mode both
```


