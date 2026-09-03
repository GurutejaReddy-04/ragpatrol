# Project 3: LLM Evaluation & Observability Harness

## 1. Project Description

A standalone evaluation and monitoring system that measures whether an LLM-powered application (specifically, your RAG-as-a-Service from Project 1) actually works reliably — not just "it gave a plausible-looking answer once." It runs a structured test suite against your RAG system, scores retrieval accuracy, answer faithfulness (whether the answer is truly grounded in retrieved content), and latency, and can automatically re-run this evaluation whenever the underlying system changes — catching regressions before they reach users.

**Important scoping note:** this project measures *faithfulness* — whether an answer is actually supported by the retrieved context — not real-world factual hallucination (which would require independent ground-truth fact-checking outside the retrieved documents). Being precise about this distinction matters: conflating the two is a common and easily-probed mistake in interviews.

**Key features:**
- A curated test set of questions with known-correct source documents
- Automated scoring: retrieval precision/recall, answer faithfulness, hallucination detection
- Latency profiling (p50/p95/p99 response times)
- Comparison mode: test multiple configurations (e.g., chunk size A vs B, embedding model X vs Y) side by side
- CI/CD integration: evaluation runs automatically on every code change, failing the build if quality drops below a threshold

## 2. Problem Statement

LLM-powered applications behave very differently from traditional software: two nearly identical prompts can produce very different quality answers, and a change that seems harmless (like adjusting a chunk size or swapping an embedding model) can silently degrade accuracy without throwing any error. Most student AI projects have no way to know if their system is actually good — they eyeball a few outputs and call it done. Real AI teams face this exact problem at a larger scale, and it's currently a genuinely understaffed skill area — "LLM evaluation/observability" is a specific, in-demand niche, not a generic ML skill. This project builds the tooling to answer, with actual numbers, "does my AI system work, and did my last change make it better or worse?"

## 3. Project Architecture

### Chosen architecture: Standalone evaluation service with pluggable scorers

```
Test Set (questions + ground truth) → Evaluation Runner
                                              │
                          ┌───────────────────┼────────────────────┐
                          │                   │                    │
                 Retrieval Scorer      Faithfulness Scorer    Latency Profiler
                 (precision/recall     (does answer match     (p50/p95/p99
                  vs ground truth)      retrieved context?)     response times)
                          │                   │                    │
                          └───────────────────┼────────────────────┘
                                              │
                                    Results stored in SQL
                                              │
                                    Report generator (Markdown/HTML)
                                              │
                              GitHub Actions (runs on every push,
                               fails build if scores drop below threshold)
```

The Evaluation Runner calls your RAG system (Project 1) as a black box via its API — it doesn't need to know RAG internals, only send a question and receive an answer + retrieved sources, which keeps this project cleanly separate and reusable against *any* LLM app, not just yours.

### Alternative architectures considered

**A. Manual spot-checking (read through outputs by eye, no scoring code)**
- *Pros:* Zero engineering effort
- *Cons:* Not reproducible, not automatable, gives you nothing to put on a resume or show in an interview, doesn't scale past a handful of examples
- *Verdict: Rejected — this is what everyone already does; the whole point of this project is to replace it.*

**B. Fully "LLM-as-judge" for everything (use a second LLM to grade every aspect, no other scoring)**
- *Pros:* Simple to implement, flexible, handles subjective quality
- *Cons:* LLM-as-judge itself can be wrong or inconsistent, is slower/costlier per eval run, and using it alone skips demonstrating that you understand *classical* evaluation metrics (precision/recall), which interviewers will specifically want to see you understand
- *Verdict: Partially rejected — LLM-as-judge is used, but only as one signal (for faithfulness), combined with deterministic metrics, not as the only method.*

**C. Standalone evaluation service with pluggable scorers, mixing deterministic + LLM-judge metrics (chosen)**
- *Pros:* Demonstrates both classical evaluation rigor (precision/recall — a DS-relevant skill) and modern LLM-specific evaluation (faithfulness/hallucination scoring — an AI-engineering-relevant skill), reusable as a general tool (not hardcoded to just your RAG app), integrates cleanly into CI/CD for a real regression-testing story
- *Cons:* Requires building a small test set by hand (time cost, but one-time and small — ~20-30 examples)
- *Verdict: Selected.*

## 4. Technology Stack & Skill Set

| Category | Technology | Skill Required |
|---|---|---|
| Core language | Python | Scripting, data processing |
| API layer | FastAPI (optional, if exposing as a service) | REST API basics |
| Scoring | scikit-learn (precision/recall calc), sentence-transformers (embedding similarity) | Basic ML/statistics, embedding comparison |
| LLM-as-judge | Gemini/OpenAI API | Prompt engineering for structured grading output |
| Storage | PostgreSQL or SQLite | Storing eval run history for trend tracking over time |
| Reporting | Python (Markdown/HTML generation), optionally Streamlit | Report/dashboard generation |
| CI/CD integration | GitHub Actions | Writing pipeline YAML with conditional failure logic |

## 5. Role of Each Technology

- **Python**: The implementation language for the entire harness — chosen because it's your strongest language, and Python's data/ML libraries make scoring logic quick to build.
- **scikit-learn**: Used to compute precision, recall, and F1 for the retrieval step — did the system fetch the actually-relevant document chunks for a given question, using standard, well-understood metrics rather than something invented ad hoc.
- **sentence-transformers**: Used to measure how semantically similar the generated answer is to the source content it claims to be based on — a lightweight, fast way to catch answers that drift from their supposed sources, without needing an LLM call for every check.
- **Gemini/OpenAI API (as judge)**: Used specifically for faithfulness/hallucination scoring — asking a second LLM call to check "is this answer fully supported by the given context, yes/no, and why?" This handles nuance that simple similarity scores miss (e.g., an answer that's semantically close but subtly adds an unsupported claim).
- **PostgreSQL/SQLite**: Stores every evaluation run's results over time, so you can track whether your RAG system's quality is improving or regressing as you make changes — not just a single one-off score.
- **GitHub Actions**: Runs this entire evaluation suite automatically whenever code changes are pushed, and can be configured to fail the build if scores drop below a set threshold — this is what makes it "regression testing for LLM apps," a real and current engineering practice, not just a one-time analysis script.

## 6. Detailed Implementation Plan

### Phase 1: Build the Test Set (Days 1-2)
- Select ~20-30 representative questions for your RAG system (Project 1), covering easy, ambiguous, and edge-case queries
- For each, manually determine and record the correct source document/chunk (ground truth) and, where possible, a reference answer

### Phase 2: Retrieval Scoring (Days 3-4)
- Write a script that sends each test question to your RAG system, captures which chunks it retrieved
- Compute precision/recall against your ground-truth chunks using scikit-learn
- Output a per-question and aggregate score report

### Phase 3: Faithfulness Scoring (Days 5-7)
- Implement embedding-similarity check (sentence-transformers) between the generated answer and its cited source chunks
- Implement LLM-as-judge check: send the question, answer, and retrieved context to an LLM with a structured prompt asking it to rate faithfulness (e.g., 1-5 scale) and explain briefly why
- Combine both signals into a faithfulness score per question

### Phase 4: Latency Profiling (Day 8)
- Instrument your test runner to record response time per question
- Compute p50/p95/p99 latency across the full test set
- Run this both with and without your Redis cache (from Project 1) warmed, to show cache impact on real numbers

### Phase 5: Storage & Trend Tracking (Days 9-10)
- Set up a small SQL table to store each evaluation run's date, config used, and all scores
- Write a simple script to compare the latest run against the previous one and flag any regressions

### Phase 6: Comparison Mode (Days 11-12)
- Extend the harness to run the same test set against two different configurations of your RAG system (e.g., different chunk sizes or embedding models)
- Output a side-by-side comparison report

### Phase 7: Reporting (Day 13)
- Generate a clean Markdown or simple HTML report summarizing all scores, per-question breakdowns, and any regressions
- (Optional) Build a minimal Streamlit dashboard to visualize score trends over time

### Phase 8: CI/CD Integration (Days 14-15)
- Add a GitHub Actions workflow that runs this evaluation suite whenever code is pushed to your RAG project's repository
- Configure it to fail the build (block the merge) if any score drops below a defined threshold — this is your "regression testing" story
- Document how this would catch a real regression (demonstrate by deliberately introducing a bad change and showing the pipeline catch it)

### Phase 9: Prove Generality (Day 16)
- Your pitch is that this harness is reusable against *any* LLM app, not hardcoded to Project 1. Prove it: build a small second stub API (even a minimal fake endpoint returning a different response format) and run the harness against it successfully
- Document this test in your README as evidence the tool is genuinely general-purpose, not secretly coupled to one project's API shape

## 7. Project Workflow

**Evaluation run flow:**
Test set loaded (questions + ground truth) → for each question, call the target RAG system's API → capture retrieved chunks + generated answer + response time → compute retrieval precision/recall against ground truth → compute faithfulness score (embedding similarity + LLM-as-judge) → compute latency percentiles across all questions → aggregate all scores into a single run record → store run record in SQL → generate a human-readable report

**Regression detection flow (CI/CD):**
Developer pushes a code change to the RAG system → GitHub Actions triggers the evaluation harness automatically → harness runs the full test set against the updated system → compares new scores against the last stored baseline run → if any score (e.g., retrieval precision) drops more than the allowed threshold, the pipeline fails and blocks the merge → developer is alerted immediately, before the regression reaches production

**Comparison flow:**
User specifies two configurations to compare (e.g., chunk_size=200 vs chunk_size=500) → harness runs the full evaluation suite against both → produces a side-by-side report showing which configuration performs better on each metric → this becomes documented, data-driven justification for configuration choices, instead of guesswork
