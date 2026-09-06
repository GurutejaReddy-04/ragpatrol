# RAGPatrol — LLM Evaluation & Observability Harness

> **Automated Quality Gating, Faithfulness Auditing, and Latency Profiling for Production RAG Systems**

**RAGPatrol** (GitHub: `ragpatrol`) is a surgical regression gating and observability harness that evaluates black-box RAG systems (including [CiteBase](file:///D:/Btech_Organized/Projects/Production%20RAG-as-a-Service%20%E2%80%94%20multi-tenant%20document%20intelligence%20API) and independent architectures) across retrieval precision/recall/F1, answer faithfulness, and latency percentiles. It persists benchmark runs over time in SQLite/PostgreSQL, supports side-by-side configuration comparisons, and gates CI/CD pipelines against quality and performance regressions.

---

## Important Scope Note

> [!IMPORTANT]
> **Faithfulness is NOT fact-checking.**
> Faithfulness scoring measures whether the generated answer is strictly grounded in the retrieved context chunks provided to the LLM. It does not perform independent open-world truth verification. Conflating these two concepts is a common pitfall that RAGPatrol explicitly avoids.

---

## Target API Contract

RAGPatrol treats any evaluated RAG service as a black box adhering to this standard HTTP interface:

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

The RAGPatrol client adapter immediately normalizes incoming vendor payloads into canonical `RetrievedChunk` and `RAGResponse` DTOs:

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

RAGPatrol uses a curated, version-controlled ground-truth test set located at [`testset/questions.yaml`](file:///d:/Btech_Organized/Projects/LLM-Evaluation-Observability-Harness/testset/questions.yaml).

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
ragpatrol/ (LLM-Evaluation-Observability-Harness/)
├── .github/
│   └── workflows/
│       └── eval.yml               # CI regression gating workflow
├── harness/
│   ├── clients/
│   │   ├── contracts.py           # Pydantic v2 DTOs and citation adapters
│   │   └── rag_client.py          # Tenacity-backed HTTP client adapter
│   ├── config.py                  # Pydantic-settings config loader
│   ├── exceptions.py              # Custom domain-specific exceptions
│   ├── regression_check.py        # Automated quality regression detector & CI gate
│   ├── reporting/
│   │   ├── comparison_report.py   # Side-by-side configuration experiment reporter
│   │   ├── dashboard.py           # Streamlit trend & KPI dashboard
│   │   ├── html_report.py         # Standalone HTML report generator (embedded CSS)
│   │   └── markdown_report.py     # GitHub-flavored Markdown report generator
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
│   ├── __init__.py                # Stub app package
│   └── fake_rag_api.py            # Standalone FastAPI mock RAG service (port 8001)
├── testset/
│   ├── questions.yaml             # Curated CiteBase ground-truth questions (25 items)
│   ├── stub_questions.yaml        # Dedicated stub test set (5 items)
│   └── validate_testset.py        # Dataset validation and invariant checker
├── tests/
│   ├── test_adapter.py            # Client adapter & stub normalization unit tests
│   ├── test_audit_edge_cases.py   # Regression tests for all audit edge cases
│   ├── test_comparison.py         # Side-by-side comparison unit tests
│   ├── test_config.py             # Configuration & environment validation tests
│   ├── test_connectivity.py       # Live health check & adapter tests
│   ├── test_faithfulness.py       # Dual-signal faithfulness unit tests (mocked)
│   ├── test_imports.py            # Complete import smoke test
│   ├── test_latency.py            # Latency percentile & speedup unit tests
│   ├── test_regression.py         # Regression gating unit tests (in-memory SQLite)
│   ├── test_reporting.py          # Markdown & HTML report structure unit tests
│   ├── test_retrieval.py          # Set-based Precision, Recall, F1 unit tests
│   ├── test_runner.py             # Evaluation runner & CLI orchestration tests
│   ├── test_storage.py            # Database manager & persistence tests
│   └── test_testset.py            # Automated test set schema enforcement
├── config.yaml                    # Public configuration parameters
├── pyproject.toml                 # Ruff, mypy, and build metadata (ragpatrol)
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

# Run complete pytest test suite (86 automated tests across 14 test suites)
python -m pytest
```

---

## Running RAGPatrol CLI

```bash
# Stage 1: Classical Set-Based Retrieval Scoring
python -m harness.runner --stage retrieval

# Stage 2: Dual-Signal Faithfulness Scoring (Dry-run mode, embedding only)
python -m harness.runner --stage faithfulness --dry-run

# Stage 3: Dual-Signal Faithfulness Scoring (Live Gemini LLM Judge)
python -m harness.runner --stage faithfulness

# Stage 4: Full Pipeline with Cold vs. Warm Latency Comparison
python -m harness.runner --stage all --cache-mode both

# Stage 5: Automated Quality Regression Gate (CI/CD check)
python -m harness.regression_check --config default --stage all

# Stage 6: Side-by-Side Configuration Experiment Comparison
python -m harness.runner --compare reranker_on reranker_off

# Stage 7: Generate Markdown & HTML Reports
python -m harness.runner --report both
```

---

## Proving Generality

A common pitfall in evaluation tooling is tight coupling to a single system's internal API contract. To prove that RAGPatrol is **truly vendor-agnostic**, we engineered an independent stub application (`stub_app/fake_rag_api.py`) exposing a deliberately different API schema, paired with a dedicated 5-question test set (`testset/stub_questions.yaml`).

### Disparate Schema Comparison

| Dimension | Production System (CiteBase) | Independent Stub (`fake_rag_api`) | Canonical RAGPatrol DTO |
| :--- | :--- | :--- | :--- |
| **Port / Endpoint** | `http://127.0.0.1:8000/query` | `http://127.0.0.1:8001/query` | Configurable / `--base-url` |
| **Answer Key** | `"answer"` | `"answer_text"` | `RAGResponse.answer` |
| **Citations List** | `"sources": [{"source", "page", ...}]` | `"sources": [{"doc", "page", "content"}]` | `RAGResponse.retrieved_chunks` |
| **Latency Metric** | Harness wall-clock measurement | `"response_time_ms": float` | `RAGResponse.latency_ms` |

### Adapter & Auto-Detection Pattern

The RAGPatrol client adapter ([`harness/clients/rag_client.py`](file:///d:/Btech_Organized/Projects/LLM-Evaluation-Observability-Harness/harness/clients/rag_client.py)) and normalization layer ([`harness/clients/contracts.py`](file:///d:/Btech_Organized/Projects/LLM-Evaluation-Observability-Harness/harness/clients/contracts.py)) seamlessly bridge the gap:
- **Explicit Flag**: `--adapter stub` forces stub normalization and automatically targets `testset/stub_questions.yaml`.
- **Auto-Detection**: If the response payload contains `"answer_text"` or the target URL points to port `8001`, the adapter automatically normalizes the payload into standard `RetrievedChunk` and `RAGResponse` DTOs.

### Running Against the Stub

```bash
# 1. Start the stub API in the background (port 8001)
python -m stub_app.fake_rag_api

# 2. Run the full RAGPatrol evaluation against the stub
python -m harness.runner --adapter stub --base-url http://localhost:8001 --stage all
```

### Benchmark Scores Achieved on the Stub App

```text
================================================================================
 EVALUATION SUMMARY: Stage='all' | Cache='warm' | Config='default'
================================================================================
 Total Queries: 5 | Successful: 5 | Failed: 0
 Mean Precision:     100.0%
 Mean Recall:        100.0%
 Mean F1:            100.0%
 Mean Faithfulness:  78.3%
 Hallucination Rate: 0.0% (0/5)
 Latency Profile:    p50=15.1ms | p95=15.1ms | p99=15.1ms | mean=15.0ms
--------------------------------------------------------------------------------
```

> [!NOTE]
> **Generality Validation Guarantee:**
> "RAGPatrol was validated against two independent systems: CiteBase and the stub app, proving it is a general-purpose evaluation tool."





