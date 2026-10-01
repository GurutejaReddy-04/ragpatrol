# RAGPatrol — LLM Evaluation & Observability Harness

<blockquote>Automated Quality Gating, Faithfulness Auditing, and Latency Profiling for Production RAG Systems</blockquote>

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB?logo=python&logoColor=white" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/Tests-85%20CI%20Passed%20%2B%201%20Live%20Probe-10B981?logo=pytest&logoColor=white" alt="Pytest: 85 CI Passed + 1 Live Probe">
  <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="MIT License">
  <img src="https://img.shields.io/badge/CI%2FCD-Passing-brightgreen?logo=github-actions&logoColor=white" alt="CI/CD Status">
</p>

**RAGPatrol** is a surgical regression gating and observability harness designed to evaluate black-box Retrieval-Augmented Generation (RAG) systems across retrieval precision/recall/F1, answer faithfulness, and latency percentiles. It persists benchmark runs over time in SQLite or PostgreSQL, supports side-by-side configuration comparisons, and gates CI/CD pipelines against quality degradations.

---

## 📖 Documentation Site

A static documentation site is available at:
**[https://GurutejaReddy-04.github.io/ragpatrol/](https://GurutejaReddy-04.github.io/ragpatrol/)**

This site provides a quick overview of the project, key features, and links to the evaluation harness documentation.

---

## Key Features

- **Validated Across Multiple Independently Shaped HTTP Interfaces:** Queries production RAG services (such as CiteBase) and independent mock APIs via normalized Pydantic DTO contracts.
- **Dual-Signal Faithfulness Auditing:** Combines local semantic embedding similarity (`sentence-transformers/all-MiniLM-L6-v2`) with LLM-as-a-judge verification (Google Gemini configured at temperature 0.0 to reduce sampling variability) to detect hallucinations without open-world conflation.
- **Latency Percentile Profiling:** Profiles p50, p95, and p99 response times with cold-cache vs. warm-cache comparison and speedup factor computation.
- **Automated Longitudinal Regression Gating:** Evaluates historical run deltas against configured regression thresholds and exits with code `1` to block CI/CD regressions.
- **Side-by-Side Configuration Experiments:** Compares two system configurations (e.g., reranker enabled vs. disabled) across quality metrics and per-category breakdowns.
- **Multi-Format Tiered Reporting:** Exports zero-dependency standalone HTML reports and Markdown artifacts explicitly partitioning deterministic ground-truth metrics, continuous embedding signals, and stochastic LLM-judge scores.
- **Interface Generality Demonstration:** Ships with an independent stub application (`stub_app/fake_rag_api.py`) exposing an alternative schema to demonstrate decoupled contract adaptation.

---

## 📐 Evaluation Methodology & Metric Taxonomy

RAGPatrol is built upon a formalized **11-point evaluation methodology** detailed in [`docs/evaluation_methodology.md`](docs/evaluation_methodology.md).

To prevent confusing model heuristics with objective measurements, metrics are strictly classified into three reliability tiers:

| Tier | Metric Category | Metrics Included | Measurement Nature & Reliability Boundaries |
| :--- | :--- | :--- | :--- |
| **Tier 1** | **Deterministic Retrieval** | Precision, Recall, Harmonic F1 | **Deterministic relative to curated ground-truth annotations.** Evaluated via unranked set math ($TP = \|Retrieved \cap GroundTruth\|$). 100% reproducible with zero LLM dependency. |
| **Tier 2** | **Semantic Representation** | Cosine Similarity (MiniLM) | **Continuous dense representation proxy.** Compares answer text against concatenated retrieved passages using `all-MiniLM-L6-v2`. Detects topical drift locally. |
| **Tier 3** | **Model-Based LLM Judge** | Judge Score (1-5), Composite Faithfulness, Hallucination Rate | **Stochastic, model-dependent heuristic.** Evaluated via Google Gemini (`gemini-2.5-flash` at temperature 0.0 to reduce sampling variability). Sensitive to prompt framing and provider model revisions; does not guarantee bit-for-bit reproducibility and must not be interpreted as absolute ground truth. |

---

## Important Scope Note

> [!IMPORTANT]
> **Faithfulness is NOT fact-checking.**
> Faithfulness scoring measures whether the generated answer is strictly grounded in the retrieved context chunks provided to the LLM during generation. It does not perform independent open-world truth verification. Conflating these two concepts is a common pitfall that RAGPatrol explicitly avoids.

---

## System Architecture

```mermaid
flowchart TD
    subgraph InputTier ["Input & Benchmark Tier"]
        TESTSET["Curated Evaluation Testset<br/>(questions.yaml / stub_questions.yaml)<br/>• Ground-Truth Chunks & Reference Answers"]
    end

    subgraph TargetTier ["Target Service Tier (Black Box)"]
        TARGET["Evaluated RAG Service<br/>(HTTP REST API)<br/>• GET /health<br/>• POST /query"]
    end

    subgraph HarnessTier ["RAGPatrol Evaluation Engine"]
        RUNNER["Evaluation Runner<br/>(harness/runner.py)<br/>• Orchestrator & CLI Entrypoint"]
        ADAPTER["Client Adapter<br/>(contracts.py & rag_client.py)<br/>• Tenacity Retries & DTO Normalization"]

        subgraph Scorers ["Evaluation Scorers"]
            RETRIEVAL["Retrieval Scorer<br/>• Precision, Recall, F1"]
            FAITHFUL["Dual-Signal Faithfulness<br/>• Local MiniLM Embedding<br/>• Gemini LLM Judge"]
            LATENCY["Latency Profiler<br/>• p50, p95, p99 Percentiles<br/>• Cold vs. Warm Cache Speedup"]
        end
    end

    subgraph PersistenceTier ["Storage & Regression Gating"]
        SQL[("Relational Store<br/>(SQLite / PostgreSQL)<br/>• EvalRun & RunMetric Tables")]
        GATE["Automated Regression Gate<br/>(regression_check.py)<br/>• CI Exit Code 0 / 1"]
    end

    subgraph OutputTier ["Reporting & Observability Tier"]
        HTML["Standalone HTML Report<br/>(Embedded CSS & KPI Cards)"]
        MD["Markdown Report<br/>(GitHub Flavored Summary)"]
        STREAMLIT["Streamlit Dashboard<br/>(Historical Trends & Regressions)"]
    end

    %% Workflow connections
    TESTSET -->|"Load Benchmark Queries"| RUNNER
    RUNNER -->|"Dispatch Query Payload"| ADAPTER
    ADAPTER -->|"POST /query"| TARGET
    TARGET -->|"Answer & Retrieved Chunks"| ADAPTER
    ADAPTER -->|"Normalized RAGResponse DTO"| RUNNER

    %% Scoring flows
    RUNNER -->|"Evaluate Chunks"| RETRIEVAL
    RUNNER -->|"Verify Groundedness"| FAITHFUL
    RUNNER -->|"Profile Latency"| LATENCY

    %% Persistence and Gating
    RETRIEVAL -->|"Aggregated Metrics"| SQL
    FAITHFUL -->|"Faithfulness Scores"| SQL
    LATENCY -->|"Latency Stats"| SQL
    SQL -->|"Historical Delta Evaluation"| GATE

    %% Artifact Generation
    SQL -->|"Export Findings"| HTML
    SQL -->|"Generate Artifacts"| MD
    SQL -->|"Visualize Trends"| STREAMLIT

    classDef input fill:#2563eb,stroke:#fff,stroke-width:2px,color:#fff;
    classDef target fill:#0d9488,stroke:#fff,stroke-width:2px,color:#fff;
    classDef harness fill:#4f46e5,stroke:#fff,stroke-width:2px,color:#fff;
    classDef store fill:#334155,stroke:#fff,stroke-width:2px,color:#fff;
    classDef report fill:#7c3aed,stroke:#fff,stroke-width:2px,color:#fff;

    class TESTSET input;
    class TARGET target;
    class RUNNER,ADAPTER,RETRIEVAL,FAITHFUL,LATENCY harness;
    class SQL,GATE store;
    class HTML,MD,STREAMLIT report;
```

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

![Canonical Query & Response DTO](docs/images/query-response.png)

---

## Quick Start

### 1. Installation

Clone the repository and install dependencies in a Python 3.10+ virtual environment:

```bash
git clone https://github.com/GurutejaReddy-04/ragpatrol.git
cd ragpatrol

python -m venv .venv
# On Linux/macOS:
source .venv/bin/activate
# On Windows:
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

### 2. Environment Variables

Create a `.env` file or export your API credentials:

```bash
# Required for live LLM Judge faithfulness scoring (optional in --dry-run mode)
export JUDGE_API_KEY="your-gemini-api-key"

# Optional overrides (replace with your RAG service URL)
export TARGET_BASE_URL="https://your-api.example.com"
export DATABASE_URL="sqlite:///eval_runs.db"
```

### 3. CLI Usage

```bash
# 1. Classical Retrieval Scoring (Precision, Recall, F1)
python -m harness.runner --stage retrieval

# 2. Dual-Signal Faithfulness Scoring (Dry-run mode, embedding proxy)
python -m harness.runner --stage faithfulness --dry-run

# 3. Dual-Signal Faithfulness Scoring (Live Gemini LLM Judge)
python -m harness.runner --stage faithfulness

# 4. Full Pipeline with Cold vs. Warm Latency Comparison
python -m harness.runner --stage all --cache-mode both

# 5. Automated Quality Regression Gate (CI/CD check)
python -m harness.regression_check --config default --stage all

# 6. Side-by-Side Configuration Experiment Comparison
python -m harness.runner --compare reranker_on reranker_off

# 7. Generate Markdown & HTML Reports
python -m harness.runner --report both
```

---

## Configuration

Baseline settings are managed via `pydantic-settings` in [`harness/config.py`](harness/config.py) and loaded from [`config.yaml`](config.yaml). Secrets are strictly injected via environment variables:

```yaml
target_api:
  base_url: "https://your-api.example.com"
  timeout_seconds: 30.0
  max_retries: 3
  retry_backoff_factor: 1.5

testset:
  path: "testset/questions.yaml"

storage:
  database_url: "sqlite:///eval_runs.db"

metrics:
  retrieval:
    min_precision: 0.70
    min_recall: 0.70
    min_f1: 0.70
  faithfulness:
    min_score: 0.75
    embedding_weight: 0.4
    llm_judge_weight: 0.6
  latency:
    max_p95_ms: 2500.0
    max_p99_ms: 5000.0

judge:
  provider: "gemini"
  model: "gemini-2.5-flash"
  temperature: 0.0
```

---

## Regression Detection & Quality Gating

RAGPatrol's regression detection module ([`harness/regression_check.py`](harness/regression_check.py)) queries historical run records stored in SQLite/PostgreSQL to detect quality degradation or latency regressions across commits:

```bash
python -m harness.regression_check --config default --stage all --cache-state warm
```

- **Retrieval Thresholds:** Fails if Precision, Recall, or F1 drops by > 5% (0.05).
- **Faithfulness Threshold:** Fails if groundedness score drops by > 10% (0.10).
- **Latency Threshold:** Fails if p95 response time increases by > 20% relative to baseline.
- **CI Exit Code:** Exits with code `0` on pass, or code `1` with formatted delta diagnostics on regression.

### Worked Example: Longitudinal Regression Gate in Action

The primary value of RAGPatrol is catching regressions automatically before code reaches production:

```text
Commit A (Baseline)         Commit B (Code Change)       Regression Check         CI Status
────────────────────        ──────────────────────       ────────────────         ─────────
Run 1: Precision = 88.0% ──> Prompt/Chunking change ───> Precision drops to 70.0% ─> 🛑 Exit Code 1
       F1 Score  = 85.0%     causes retriever drift       (Delta: -0.18 > 0.05)     CI Merge Blocked
       p95 ms    = 210ms                                  Latency increases to 320ms
```

When a regression occurs, `harness.regression_check` outputs an actionable diagnostic delta report:

```text
================================================================================
 REGRESSION CHECK REPORT: Config='default' | Stage='all' | Cache='warm'
================================================================================
 Status: FAIL (REGRESSION DETECTED)
 Current Run ID:  run_20261001_174020_e2f4a1
 Previous Run ID: run_20260901_120000_ref001
--------------------------------------------------------------------------------
+-------------------------+------------+------------+------------+----------------+
| Metric                  | Previous   | Current    | Delta      | Threshold      |
+-------------------------+------------+------------+------------+----------------+
| faithfulness_avg        | 0.8800     | 0.8650     | -0.0150    | None           |
| latency_p50             | 180.0      | 195.0      | +15.0000   | None           |
| latency_p95             | 420.0      | 450.0      | +30.0000   | None           |
| retrieval_f1            | 0.8500     | 0.7200     | -0.1300 [FAIL] | LIMIT 0.05 |
| retrieval_precision     | 0.8800     | 0.7000     | -0.1800 [FAIL] | LIMIT 0.05 |
| retrieval_recall        | 0.8400     | 0.7420     | -0.0980 [FAIL] | LIMIT 0.05 |
+-------------------------+------------+------------+------------+----------------+

 DETECTED REGRESSIONS:
  * retrieval_precision: drop of 0.1800 exceeds threshold 0.05
  * retrieval_recall: drop of 0.0980 exceeds threshold 0.05
  * retrieval_f1: drop of 0.1300 exceeds threshold 0.05
--------------------------------------------------------------------------------
[ERROR] ragpatrol: Regression check failed! Quality gate blocked.
```

To run a self-contained demonstration of this gating lifecycle:
```bash
python examples/regression_walkthrough.py
```

---

## Comparison Mode

RAGPatrol enables side-by-side benchmarking of two distinct system configurations (e.g., evaluating the impact of cross-encoder reranking or vector quantization):

```bash
python -m harness.runner --compare reranker_on reranker_off
```

![Configuration Comparison Preview](docs/images/comparison-table.png)

The comparator highlights metric winners across quality and speed dimensions, computes percentage deltas, and isolates trade-offs across question categories (`easy`, `ambiguous`, `edge`).

---

## Reporting

RAGPatrol automatically generates self-contained reports in both GitHub-flavored Markdown and HTML formats.

### HTML Report Preview
The HTML report contains embedded CSS (zero external CDN dependencies), interactive hover highlighting, KPI scorecards segregated by reliability tier, category drill-downs, and print-to-PDF stylesheets:

![HTML Report Preview](docs/images/report-html.png)

```bash
# Open generated HTML report in your default browser
python -m harness.runner --report html
```

### Streamlit Trend & KPI Dashboard
RAGPatrol also includes an interactive Streamlit dashboard ([`harness/reporting/dashboard.py`](harness/reporting/dashboard.py)) for inspecting historical runs, metric trends, and flagged hallucinations:

![Streamlit Dashboard Preview](docs/images/frontend-ui.png)

```bash
# Launch interactive evaluation dashboard
streamlit run harness/reporting/dashboard.py
```

---

## Generality Across Independently Shaped HTTP Interfaces

A common failure mode in evaluation tooling is tight coupling to a single system's internal API contract. RAGPatrol decouples the evaluation engine from proprietary APIs via normalized Pydantic v2 DTOs (`RAGResponse`, `RetrievedChunk`, `RAGQueryRequest`) and dedicated normalization adapters:

1. **Production System (CiteBase):** Normalizes citation arrays containing `chunk_id`, `rerank_score`, `section`, and `breadcrumb`.
2. **Independent Stub API (`stub_app/fake_rag_api.py`):** Normalizes an alternative schema (`answer_text`, `sources: [{doc, page, content}]`, `response_time_ms`).
3. **Canonical Interface:** Accepts standard `{answer, retrieved_chunks, latency_ms}` payloads.

### Schema Comparison

| Dimension | Production System (CiteBase) | Independent Stub (`fake_rag_api`) | Canonical RAGPatrol DTO |
| :--- | :--- | :--- | :--- |
| **Endpoint** | `https://your-api.example.com/query` | `http://localhost:8001/query` | Configurable / `--base-url` |
| **Answer Key** | `"answer"` | `"answer_text"` | `RAGResponse.answer` |
| **Citations List** | `"sources": [{"source", "page", ...}]` | `"sources": [{"doc", "page", "content"}]` | `RAGResponse.retrieved_chunks` |
| **Latency Metric** | RAGPatrol wall-clock measurement | `"response_time_ms": float` | `RAGResponse.latency_ms` |

### Running Against the Stub

```bash
# 1. Start the stub API in the background (port 8001)
python -m stub_app.fake_rag_api

# 2. Run the full RAGPatrol evaluation against the stub
python -m harness.runner --adapter stub --base-url http://localhost:8001 --stage all
```

![Stub App Execution](docs/images/stub-app-run.png)

---

## Testing & Test Taxonomy

RAGPatrol enforces rigorous test categorization across **86 total tests** using four explicit pytest markers:

| Marker | Count | Description |
| :--- | :---: | :--- |
| `@pytest.mark.unit` | **48** | In-memory unit tests covering set math, Pydantic DTOs, reporting tables, and configuration parsing. Zero socket/network dependencies. |
| `@pytest.mark.mocked_integration` | **36** | Integration tests using mock HTTP transports, in-memory SQLite, or local sentence-transformers inference. |
| `@pytest.mark.local_integration` | **1** | In-process FastAPI stub API verification using `starlette.testclient.TestClient`. |
| `@pytest.mark.live_external_integration` | **1** | Environment-sensitive local socket test probing `http://127.0.0.1:8000/health`. Skips gracefully when CiteBase is offline. |

> [!NOTE]
> **Network Dependency Boundary:**
> 85 tests do not require live RAG services or LLM API keys. Tests utilizing local semantic representations (`all-MiniLM-L6-v2`) run inference in-process, but may download pinned model weights from Hugging Face on first execution if not already cached locally.

```bash
# 1. Validate ground-truth dataset integrity and schema invariants
python testset/validate_testset.py

# 2. Run the 85 non-live tests (CI default)
pytest -m "not live_external_integration" -v

# 3. Run the complete test suite (includes 1 live local socket check)
pytest tests/ -v
```

![Pytest Test Suite Passing](docs/images/terminal.png)

---

## CI/CD Integration

RAGPatrol gates pull requests and pushes via GitHub Actions ([`.github/workflows/eval.yml`](.github/workflows/eval.yml)) using a **two-layer persistence architecture**:
1. **Authoritative Reference Baseline:** An immutable baseline database (`fixtures/reference_eval_runs.db`) committed to Git seeds history on fresh clones or cache misses.
2. **Rolling CI Cache:** `actions/cache@v4` with cache namespace `ragpatrol-eval-db-v2` persists recent run history across workflow runs.

```yaml
name: RAGPatrol CI & Regression Gate

on:
  push:
    branches: [ main, master ]
  pull_request:
    branches: [ main, master ]

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -r requirements.txt

      # 1. Restore rolling evaluation database cache (or initialize from reference baseline)
      - name: Restore Rolling Evaluation Database Cache
        uses: actions/cache@v4
        id: eval-db-cache
        with:
          path: eval_runs.db
          key: ragpatrol-eval-db-v2-${{ runner.os }}-${{ github.ref_name }}-${{ github.run_id }}
          restore-keys: |
            ragpatrol-eval-db-v2-${{ runner.os }}-${{ github.ref_name }}-
            ragpatrol-eval-db-v2-${{ runner.os }}-main-

      - name: Initialize Reference Baseline Database on Cache Miss
        if: steps.eval-db-cache.outputs.cache-hit != 'true'
        run: |
          test -f eval_runs.db || cp fixtures/reference_eval_runs.db eval_runs.db

      # 2. Run ground truth validation and 85 non-live tests
      - run: python testset/validate_testset.py
      - run: python -m pytest tests/ -m "not live_external_integration" -v

      # 3. Execute stub-based evaluation run (records warm pass to eval_runs.db)
      - name: Execute End-to-End Evaluation Run
        run: |
          python -m uvicorn stub_app.fake_rag_api:app --host 127.0.0.1 --port 8001 &
          sleep 2
          python -m harness.runner --adapter stub --base-url http://127.0.0.1:8001 --stage all --cache-mode both --dry-run

      # 4. Automated Longitudinal Regression Gate (compares matching warm runs)
      - name: Execute Automated Quality Regression Gate
        run: |
          python -m harness.regression_check --config default --stage all --cache-state warm

      # 5. Verify deliberate regression gate rejection logic
      - name: Verify Quality Gate Failure on Deliberate Regression
        run: |
          python examples/regression_walkthrough.py
```

---

## Repository Structure

```text
ragpatrol/
├── .github/
│   └── workflows/
│       └── eval.yml               # Automated CI regression gate
├── docs/
│   └── images/                    # Documentation screenshots & previews
│       ├── terminal.png
│       ├── comparison-table.png
│       ├── report-html.png
│       ├── stub-app-run.png
│       ├── swagger-ui.png
│       ├── frontend-ui.png
│       ├── query-response.png
│       └── README.md
├── harness/
│   ├── clients/
│   │   ├── contracts.py           # Pydantic v2 canonical DTOs & citation adapters
│   │   └── rag_client.py          # Tenacity-backed HTTP client adapter
│   ├── config.py                  # Pydantic-settings configuration loader
│   ├── exceptions.py              # Domain-specific exceptions
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
│   ├── extract_citebase_testset.py # Extracts and maps CiteBase eval benchmark
│   └── generate_docs_assets.py     # Generates documentation screenshots
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
├── LICENSE                        # MIT License
├── pyproject.toml                 # Ruff, mypy, and build metadata (ragpatrol)
├── pytest.ini                     # Pytest defaults (-v --tb=short)
├── requirements.txt               # Pinned dependencies
└── README.md
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
