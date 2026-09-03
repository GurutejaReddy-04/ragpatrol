# Plan: LLM Evaluation & Observability Harness

> Context: This harness evaluates an existing RAG-as-a-Service system (Project 1) by calling it as a black-box API. It scores retrieval quality, answer faithfulness (groundedness in retrieved context — NOT independent fact-checking), and latency, stores results over time, supports side-by-side config comparisons, and runs automatically in CI/CD to block regressions. Build order below is sequential; each phase produces a working, testable artifact before moving on.

---

## Phase 0: Project Setup & Contracts (Day 0)

**Goal:** Lock down the interfaces before writing any scoring logic, so later phases don't require rework.

Steps:
1. Create repo structure:
   ```
   eval-harness/
     harness/
       __init__.py
       runner.py
       scorers/
         __init__.py
         retrieval.py
         faithfulness.py
         latency.py
       storage/
         __init__.py
         db.py
         models.py
       reporting/
         __init__.py
         markdown_report.py
         html_report.py
       clients/
         __init__.py
         rag_client.py
     testset/
       questions.yaml
     stub_app/
       fake_rag_api.py
     .github/workflows/eval.yml
     config.yaml
     requirements.txt
     README.md
   ```
2. Define the **target-system API contract** the harness expects (so it stays app-agnostic):
   - Request: `POST /query {"question": str}`
   - Response: `{"answer": str, "retrieved_chunks": [{"chunk_id": str, "text": str, "source_doc": str}], "latency_ms": float (optional, harness can measure itself)}`
   - Document this contract in `README.md` under "Target API Contract" — this is what Phase 9 (generality) will validate against a second stub app.
3. Define `config.yaml` schema: target API base URL, test set path, thresholds per metric, DB connection string, judge LLM provider/model/API key env var name.
4. Set up `requirements.txt`: `scikit-learn`, `sentence-transformers`, `pandas`, `pyyaml`, `sqlalchemy`, `requests`, `fastapi` (optional), `pytest`.
5. Write a `pytest` smoke test that just imports every module — CI should fail loudly on broken imports from day one.

**Deliverable:** Empty-but-runnable skeleton, `pip install -r requirements.txt` succeeds, `pytest` passes on the smoke test.

---

## Phase 1: Build the Test Set (Days 1–2)

**Goal:** A curated, version-controlled ground-truth test set.

Steps:
1. Select 20–30 questions against your actual RAG corpus, deliberately spanning:
   - Easy/direct factual questions (answer clearly in one chunk)
   - Ambiguous questions (multiple plausible source chunks)
   - Edge cases (question not well covered by the corpus — tests whether the system correctly says "I don't know" instead of hallucinating)
2. For each question, manually record in `testset/questions.yaml`:
   ```yaml
   - id: q001
     question: "What is the retry policy for failed embeddings?"
     ground_truth_chunk_ids: ["doc12_chunk3"]
     reference_answer: "Failed embedding calls are retried 3 times with exponential backoff."
     category: easy
   ```
3. Add a `category` field (easy/ambiguous/edge) so later reports can break down scores by difficulty, not just an aggregate number.
4. Write a small `testset/validate_testset.py` script that checks: no duplicate IDs, every `ground_truth_chunk_ids` entry is non-empty, schema matches expected fields. Run this in CI too.
5. Commit the test set to git — it's a versioned artifact, not a throwaway file.

**Deliverable:** `testset/questions.yaml` with 20–30 validated entries; validation script passes.

---

## Phase 2: Retrieval Scoring (Days 3–4)

**Goal:** Objectively measure whether the RAG system retrieves the right chunks.

Steps:
1. Build `clients/rag_client.py`: a thin wrapper that POSTs a question to the target system's `/query` endpoint and returns `(answer, retrieved_chunks, wall_clock_latency_ms)`. Wrap in try/except with retries (network flakiness shouldn't crash a whole eval run).
2. Build `scorers/retrieval.py`:
   - For each question, compare `retrieved_chunk_ids` (from the API response) against `ground_truth_chunk_ids` (from the test set).
   - Compute precision = (relevant retrieved) / (total retrieved), recall = (relevant retrieved) / (total relevant), F1, using set-based computation (chunk counts vary per question, so this isn't a fixed-length classification problem — document why set-based math was chosen over sklearn's classifier metrics if scikit-learn's helper functions don't fit directly).
3. Produce a per-question row: `{question_id, precision, recall, f1, retrieved_ids, ground_truth_ids}`.
4. Aggregate: mean precision/recall/F1 across the whole test set, and separately by `category`.
5. Write `harness/runner.py` (v1): loads test set → calls RAG client per question → calls retrieval scorer → prints a per-question and aggregate table to stdout.
6. Unit test the scorer directly with hand-crafted fixtures (no live API needed) — e.g., perfect match, no overlap, partial overlap — to lock in correct precision/recall math before it's a dependency of everything else.

**Deliverable:** Running `python -m harness.runner --stage retrieval` against your live RAG API prints per-question and aggregate precision/recall/F1.

---

## Phase 3: Faithfulness Scoring (Days 5–7)

**Goal:** Detect answers that aren't actually grounded in what was retrieved.

Steps:
1. **Embedding-similarity signal** (`scorers/faithfulness.py`, part 1):
   - Load a `sentence-transformers` model (e.g., `all-MiniLM-L6-v2`).
   - Embed the generated answer and embed the concatenated text of the retrieved chunks.
   - Compute cosine similarity → a fast, cheap groundedness proxy.
   - Note explicitly in code comments/README: this catches gross topical drift, not subtle unsupported claims — that's what the LLM judge is for.
2. **LLM-as-judge signal** (part 2):
   - Write a structured judge prompt, e.g.:
     ```
     Question: {question}
     Retrieved Context: {chunks}
     Generated Answer: {answer}

     Is the answer fully supported by the retrieved context?
     Respond ONLY in JSON: {"faithfulness_score": 1-5, "reasoning": "...", "unsupported_claims": [...]}
     ```
   - Call the judge LLM (Gemini/OpenAI) via API, parse JSON defensively (strip markdown fences, handle malformed JSON with a retry-once-then-flag policy).
   - Log the raw judge response alongside the parsed score for auditability — useful when debugging why a score looks wrong.
3. Combine both signals into one faithfulness score per question, e.g., `faithfulness = 0.4 * embedding_similarity_normalized + 0.6 * (llm_judge_score / 5)`. Document the weighting choice as a tunable, not a magic constant.
4. Add hallucination flagging: if `llm_judge_score <= 2` OR embedding similarity below a threshold, flag the question as a hallucination candidate in the report.
5. Unit test the embedding similarity function with fixtures; mock the judge LLM call in tests (don't burn API quota or add flakiness to CI) using a fake client that returns canned JSON.

**Deliverable:** `runner.py --stage faithfulness` produces per-question faithfulness scores + a list of flagged hallucination candidates with judge reasoning.

---

## Phase 4: Latency Profiling (Day 8)

**Goal:** Real percentile latency numbers, including cache impact.

Steps:
1. In `scorers/latency.py`, record wall-clock time per question call (already captured in Phase 2's client — make sure it's persisted per-question, not discarded).
2. Compute p50/p95/p99 using `numpy.percentile` over the full latency array for the run.
3. Run the full test set twice: once with Project 1's Redis cache cold (flush it first, or use fresh questions), once warmed (run the same questions twice, take the second pass). Store both as separate run records with a `cache_state: cold|warm` tag.
4. Output a simple before/after latency table (p50/p95/p99 cold vs warm) — a concrete, quantifiable "cache impact" number.
5. Guard against outlier skew: also report min/max/mean alongside percentiles so a single network hiccup doesn't misrepresent the story.

**Deliverable:** Latency report showing p50/p95/p99 for both cache states, clearly labeled.

---

## Phase 5: Storage & Trend Tracking (Days 9–10)

**Goal:** Persist every run so quality/regressions are trackable over time, not just single snapshots.

Steps:
1. Design SQL schema in `storage/models.py` (SQLAlchemy):
   - `EvalRun`: id, timestamp, config_name, git_commit_sha (optional), cache_state
   - `RunMetric`: run_id (FK), metric_name (e.g., `retrieval_precision`, `faithfulness_avg`, `latency_p95`), value, category (nullable, for per-category breakdowns)
   - `QuestionResult`: run_id (FK), question_id, precision, recall, f1, faithfulness_score, latency_ms, hallucination_flag
2. Implement `storage/db.py`: `save_run(run_record)`, `get_latest_run(config_name)`, `get_run_history(config_name, limit=N)`.
3. Wire `runner.py` to call `save_run(...)` at the end of every full evaluation pass.
4. Write `harness/regression_check.py`:
   - Fetch latest run and the immediately preceding run for the same config.
   - For each metric, compute delta; flag regression if delta exceeds the threshold defined in `config.yaml` (e.g., `retrieval_precision` drop > 0.05, `faithfulness_avg` drop > 0.1, `latency_p95` increase > 20%).
   - Return a structured result: `{passed: bool, regressions: [{metric, old, new, delta}]}`.
   - Exit with non-zero code if `passed == False` (this is what CI keys off in Phase 8).
5. Unit test `regression_check.py` against fixture DB rows (in-memory SQLite) covering: no prior run (should pass, nothing to compare), clear regression, clear improvement, within-tolerance noise.

**Deliverable:** Two consecutive eval runs are stored in SQLite/Postgres; `regression_check.py` correctly flags an injected regression in a test fixture.

---

## Phase 6: Comparison Mode (Days 11–12)

**Goal:** Run the same test set against two configurations and compare head-to-head.

Steps:
1. Extend `config.yaml` to support named configs, e.g.:
   ```yaml
   configs:
     chunk_200:
       base_url: "http://localhost:8001"
     chunk_500:
       base_url: "http://localhost:8002"
   ```
   (Assumes Project 1 can be spun up with different chunk-size/embedding-model settings on different ports, or via an env var/query param the client passes through.)
2. Add `runner.py --compare chunk_200 chunk_500`: runs the full pipeline (retrieval + faithfulness + latency) against both, storing each as its own `EvalRun` with the config name tagged.
3. Build `reporting/comparison_report.py`: pulls both runs from storage, produces a side-by-side table per metric (and per category), highlighting the winner per row.
4. Handle the case where one config fails entirely (e.g., server down) — report should clearly say "config X unreachable," not crash silently or show blank/zero scores that look like a real result.

**Deliverable:** `runner.py --compare A B` produces a side-by-side Markdown table across all metrics.

---

## Phase 7: Reporting (Day 13)

**Goal:** Human-readable output, not just raw DB rows.

Steps:
1. `reporting/markdown_report.py`: renders a single run into Markdown — header (run metadata), aggregate scores table, per-category breakdown, per-question detail table (collapsible/appendix-style), flagged hallucinations with judge reasoning excerpts, latency percentile table.
2. `reporting/html_report.py`: same content, simple HTML/CSS (no framework needed) so it can be opened directly in a browser or attached to a CI artifact.
3. Optional: minimal Streamlit app (`reporting/dashboard.py`) that reads run history from storage and plots metric trends over time (line chart per metric, config selectable via dropdown).
4. Make report generation a CLI flag: `runner.py --report markdown|html|both`.

**Deliverable:** A generated report file (`reports/run_<timestamp>.md` and `.html`) that a non-technical reader could skim and understand "is this system good, and did it get better or worse."

---

## Phase 8: CI/CD Integration (Days 14–15)

**Goal:** Automatic regression gating on every push.

Steps:
1. Write `.github/workflows/eval.yml`:
   - Trigger: `on: push` to the RAG project's repo (or `workflow_dispatch` + `push` for this repo, depending on where Project 1 lives).
   - Steps: checkout, set up Python, install deps, spin up the RAG system (or point at a staging URL), run `python -m harness.runner --full`, run `python -m harness.regression_check`, upload the generated report as a build artifact.
   - Fail the job (`exit 1`) if `regression_check` reports `passed: False`.
2. Store judge-LLM API keys as GitHub Actions secrets, never hardcoded.
3. **Prove it catches real regressions**: deliberately introduce a bad change to Project 1 (e.g., truncate chunk size to something absurd, or swap in a weaker embedding model) on a branch, push it, and log/screenshot the pipeline failing with the specific metric and delta called out.
4. Document this demonstration in the README with before/after numbers — this is the interview story artifact.

**Deliverable:** A GitHub Actions run that passes on a good commit and fails (with a clear reason) on a deliberately-broken commit.

---

## Phase 9: Prove Generality (Day 16)

**Goal:** Show the harness isn't secretly coupled to Project 1's exact API shape.

Steps:
1. Build `stub_app/fake_rag_api.py`: a minimal FastAPI/Flask app implementing the same contract from Phase 0 but with a different internal response format (e.g., field named `answer_text` instead of `answer`, or chunks returned as plain strings instead of objects) — deliberately introduce a small mismatch, then add a thin adapter in `clients/rag_client.py` (a `--adapter` flag or per-config adapter function) to normalize it, proving the harness's design already anticipates schema variation.
2. Run the full harness (`runner.py --full`) against this stub with its own tiny 5-question test set.
3. Document in the README: "This harness was validated against two independent systems: [Project 1 RAG API] and [stub_app], proving it's a general-purpose evaluation tool, not hardcoded to one project."

**Deliverable:** A successful full eval run against the stub app, documented with the adapter approach explained.

---

## Final Deliverables Checklist

- [ ] `testset/questions.yaml` — 20–30 validated ground-truth questions
- [ ] Retrieval scorer with unit tests
- [ ] Faithfulness scorer (embedding + LLM-judge) with unit tests
- [ ] Latency profiler with cold/warm cache comparison
- [ ] SQL storage layer + regression check script
- [ ] Comparison mode across ≥2 configs
- [ ] Markdown + HTML report generator
- [ ] GitHub Actions workflow with a documented pass/fail demonstration
- [ ] Stub second app + adapter proving generality
- [ ] README covering: scope note (faithfulness ≠ fact-checking), API contract, how to run locally, how CI gating works, and the regression-demo writeup

## Key Framing Notes for Interviews / README

- Be explicit that faithfulness scoring measures groundedness in retrieved context, not independent real-world fact verification — conflating these is a common mistake this project deliberately avoids.
- The harness treats the target system as a black box via API contract — reusability is a design goal, demonstrated concretely in Phase 9, not just claimed.
- Classical metrics (precision/recall/F1) and modern LLM-judge metrics are combined deliberately, to demonstrate range across both evaluation paradigms rather than relying on either alone.
