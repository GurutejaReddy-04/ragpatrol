# RAGPatrol Documentation Assets

This directory contains visual documentation artifacts and screenshot previews displayed in the project's root `README.md`.

## Included Assets

| File | Description | Max Size |
| :--- | :--- | :--- |
| `terminal.png` | Terminal test run showing 86/86 automated tests passing | < 1 MB |
| `comparison-table.png` | Side-by-side configuration experiment comparison card | < 1 MB |
| `report-html.png` | Standalone HTML evaluation report KPI and category card | < 1 MB |
| `stub-app-run.png` | CLI output demonstrating generality against independent stub API | < 1 MB |

---

## Programmatic Regeneration

All four documentation images can be programmatically regenerated at any time using PIL:

```bash
# Ensure PIL (Pillow) is installed in your active virtual environment
python scripts/generate_docs_assets.py
```

---

## Manual Screenshot Capture Guide

If you wish to capture live application screenshots manually on your system:

### Step 1: Capture Terminal Test Output
```powershell
pytest tests/ -v --tb=short
```
*Capture the terminal output displaying 86 passed tests with green checkmarks.*
*Save as: `docs/images/terminal.png`*

### Step 2: Capture Comparison Table
```powershell
python -m harness.runner --compare reranker_on reranker_off
```
*Capture the terminal side-by-side comparison table showing metric deltas and winner identification.*
*Save as: `docs/images/comparison-table.png`*

### Step 3: Capture HTML Report
```powershell
python -m harness.runner --report html
```
*Open the generated report in `reports/` in a browser. Capture the full page or KPI card view.*
*Save as: `docs/images/report-html.png`*

### Step 4: Capture Stub App Run
```powershell
# In terminal 1:
python -m stub_app.fake_rag_api

# In terminal 2:
python -m harness.runner --adapter stub --base-url http://localhost:8001 --stage all
```
*Capture the evaluation summary output showing 100% precision, 100% recall, 0% hallucinations, and 15ms latency.*
*Save as: `docs/images/stub-app-run.png`*

---

### Screenshot Quality Standards
1. **Resolution & DPI:** 950px+ width, high-DPI scaling.
2. **Theme:** Dark mode preferred (slate-900 / dark terminal background).
3. **Format:** Lossless PNG, optimized under 1 MB.
4. **Privacy:** Zero local file paths (`C:\Users\...`), personal tokens, or API credentials visible.
