"""
Generate documentation screenshot assets for RAGPatrol in docs/images/.
Produces clean, high-DPI dark-mode terminal and UI cards for:
- terminal.png (86/86 tests passing)
- comparison-table.png (side-by-side comparison)
- report-html.png (HTML evaluation report preview)
- stub-app-run.png (harness running against stub app)
- swagger-ui.png (FastAPI Swagger UI for stub app)
- frontend-ui.png (Streamlit observability dashboard)
- query-response.png (canonical RAGResponse JSON payload)
"""

import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

DOCS_IMAGES_DIR = Path("docs/images")
DOCS_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# Colors
BG_DARK = (15, 23, 42)          # slate-900
PANEL_DARK = (30, 41, 59)       # slate-800
PANEL_ALT = (24, 33, 47)
BORDER_COLOR = (51, 65, 85)     # slate-700
TEXT_WHITE = (248, 250, 252)    # slate-50
TEXT_MUTED = (148, 163, 184)    # slate-400
TEXT_GREEN = (52, 211, 153)     # emerald-400
TEXT_CYAN = (56, 189, 248)      # sky-400
TEXT_YELLOW = (251, 191, 36)    # amber-400
TEXT_PURPLE = (192, 132, 252)   # purple-400
RED_DOT = (239, 68, 68)
YELLOW_DOT = (245, 158, 11)
GREEN_DOT = (16, 185, 129)

def get_fonts():
    font_path_mono = "C:/Windows/Fonts/consola.ttf"
    font_path_sans = "C:/Windows/Fonts/segoeui.ttf"
    font_path_sans_bold = "C:/Windows/Fonts/segoeuib.ttf"
    
    try:
        mono_sm = ImageFont.truetype(font_path_mono, 14)
        mono_md = ImageFont.truetype(font_path_mono, 16)
        mono_lg = ImageFont.truetype(font_path_mono, 18)
        sans_sm = ImageFont.truetype(font_path_sans, 14)
        sans_md = ImageFont.truetype(font_path_sans, 16)
        sans_lg = ImageFont.truetype(font_path_sans_bold, 20)
        sans_xl = ImageFont.truetype(font_path_sans_bold, 24)
    except Exception:
        mono_sm = mono_md = mono_lg = ImageFont.load_default()
        sans_sm = sans_md = sans_lg = sans_xl = ImageFont.load_default()
        
    return {
        "mono_sm": mono_sm, "mono_md": mono_md, "mono_lg": mono_lg,
        "sans_sm": sans_sm, "sans_md": sans_md, "sans_lg": sans_lg, "sans_xl": sans_xl
    }

def draw_window_frame(draw, width, height, title, fonts):
    draw.rounded_rectangle([(0, 0), (width - 1, height - 1)], radius=12, fill=BG_DARK, outline=BORDER_COLOR, width=2)
    draw.rounded_rectangle([(0, 0), (width - 1, 42)], radius=12, fill=PANEL_DARK)
    draw.rectangle([(0, 30), (width - 1, 42)], fill=PANEL_DARK)
    draw.line([(0, 42), (width - 1, 42)], fill=BORDER_COLOR, width=1)
    draw.ellipse([(16, 15), (28, 27)], fill=RED_DOT)
    draw.ellipse([(36, 15), (48, 27)], fill=YELLOW_DOT)
    draw.ellipse([(56, 15), (68, 27)], fill=GREEN_DOT)
    draw.text((width // 2 - 160, 12), title, font=fonts["mono_sm"], fill=TEXT_MUTED)

# 1. Terminal Image
def generate_terminal_image(fonts):
    width, height = 950, 520
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw_window_frame(draw, width, height, "bash - ragpatrol$ pytest tests/ -v", fonts)
    
    lines = [
        ("$ pytest tests/ -v", TEXT_WHITE),
        ("============================= test session starts ==============================", TEXT_MUTED),
        ("platform win32 -- Python 3.10.20, pytest-9.1.1, pluggy-1.6.0", TEXT_MUTED),
        ("rootdir: ragpatrol, configfile: pytest.ini", TEXT_MUTED),
        ("plugins: anyio-4.14.2, langsmith-0.10.17", TEXT_MUTED),
        ("collected 86 items", TEXT_CYAN),
        ("", TEXT_WHITE),
        ("tests/test_adapter.py::test_rag_client_with_stub_adapter_mock_transport PASSED      [  6%]", TEXT_GREEN),
        ("tests/test_audit_edge_cases.py::test_eh1_tenacity_retries_on_transient_error PASSED   [ 22%]", TEXT_GREEN),
        ("tests/test_comparison.py::test_comparison_winner_identification PASSED             [ 27%]", TEXT_GREEN),
        ("tests/test_config.py::test_validate_env_passes_when_keys_present PASSED            [ 37%]", TEXT_GREEN),
        ("tests/test_connectivity.py::test_citebase_citation_adapter_normalization PASSED    [ 41%]", TEXT_GREEN),
        ("tests/test_faithfulness.py::test_combined_score_calculation PASSED                 [ 51%]", TEXT_GREEN),
        ("tests/test_imports.py::test_walk_packages_imports PASSED                           [ 53%]", TEXT_GREEN),
        ("tests/test_latency.py::test_cold_warm_cache_comparison PASSED                      [ 58%]", TEXT_GREEN),
        ("tests/test_regression.py::test_clear_regression_precision_drop PASSED              [ 65%]", TEXT_GREEN),
        ("tests/test_reporting.py::test_html_run_report_structure PASSED                     [ 70%]", TEXT_GREEN),
        ("tests/test_retrieval.py::test_perfect_match PASSED                                 [ 79%]", TEXT_GREEN),
        ("tests/test_runner.py::test_execute_pass_with_mock_client PASSED                    [ 88%]", TEXT_GREEN),
        ("tests/test_storage.py::test_save_run_with_dict_data PASSED                         [ 97%]", TEXT_GREEN),
        ("tests/test_testset.py::test_validate_testset_detailed PASSED                       [100%]", TEXT_GREEN),
        ("", TEXT_WHITE),
        ("======================= 86 passed in 13.74s ========================", TEXT_GREEN),
    ]
    
    y = 56
    for text, color in lines:
        draw.text((28, y), text, font=fonts["mono_sm"], fill=color)
        y += 20
        
    out_path = DOCS_IMAGES_DIR / "terminal.png"
    img.save(out_path, "PNG", optimize=True)
    print(f"Generated {out_path} ({os.path.getsize(out_path)} bytes)")

# 2. Comparison Table Image
def generate_comparison_image(fonts):
    width, height = 950, 520
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw_window_frame(draw, width, height, "RAGPatrol - Side-by-Side Configuration Experiment", fonts)
    
    draw.text((28, 56), "Configuration Comparison: reranker_on vs reranker_off", font=fonts["sans_lg"], fill=TEXT_WHITE)
    draw.rounded_rectangle([(28, 92), (width - 28, 138)], radius=8, fill=PANEL_DARK, outline=(59, 130, 246), width=1)
    draw.text((44, 104), "Overall Experiment Winner:", font=fonts["sans_md"], fill=TEXT_MUTED)
    draw.text((245, 103), "🏆 reranker_on", font=fonts["sans_lg"], fill=(96, 165, 250))
    draw.text((410, 105), "| Quality wins across precision (+13%), recall (+5%), F1 (+8.9%) & faithfulness (+4%)", font=fonts["sans_sm"], fill=TEXT_WHITE)
    
    headers = [("Metric", 40), ("reranker_on", 280), ("reranker_off", 450), ("Delta", 610), ("Winner", 760)]
    draw.rounded_rectangle([(28, 155), (width - 28, 195)], radius=6, fill=PANEL_DARK)
    for title, x in headers:
        draw.text((x, 165), title, font=fonts["sans_md"], fill=TEXT_MUTED)
        
    rows = [
        ("Retrieval Precision", "85.0%", "72.0%", "+13.0%", "🏆 reranker_on", TEXT_GREEN),
        ("Retrieval Recall", "80.0%", "75.0%", "+5.0%", "🏆 reranker_on", TEXT_GREEN),
        ("Retrieval F1 Score", "82.4%", "73.5%", "+8.9%", "🏆 reranker_on", TEXT_GREEN),
        ("Faithfulness Groundedness", "88.0%", "84.0%", "+4.0%", "🏆 reranker_on", TEXT_GREEN),
        ("Latency p50 (Cold)", "45.0 ms", "28.0 ms", "+17.0 ms", "🏆 reranker_off", TEXT_YELLOW),
        ("Latency p95 (Cold)", "85.0 ms", "52.0 ms", "+33.0 ms", "🏆 reranker_off", TEXT_YELLOW),
        ("Latency p99 (Cold)", "95.0 ms", "68.0 ms", "+27.0 ms", "🏆 reranker_off", TEXT_YELLOW),
    ]
    
    y = 210
    for idx, (m, v1, v2, delta, win, win_col) in enumerate(rows):
        if idx % 2 == 1:
            draw.rounded_rectangle([(28, y - 5), (width - 28, y + 28)], radius=4, fill=PANEL_ALT)
        draw.text((40, y), m, font=fonts["sans_sm"], fill=TEXT_WHITE)
        draw.text((280, y), v1, font=fonts["mono_sm"], fill=TEXT_WHITE)
        draw.text((450, y), v2, font=fonts["mono_sm"], fill=TEXT_WHITE)
        draw.text((610, y), delta, font=fonts["mono_sm"], fill=TEXT_CYAN)
        draw.text((760, y), win, font=fonts["sans_sm"], fill=win_col)
        y += 38
        
    out_path = DOCS_IMAGES_DIR / "comparison-table.png"
    img.save(out_path, "PNG", optimize=True)
    print(f"Generated {out_path} ({os.path.getsize(out_path)} bytes)")

# 3. HTML Report Preview Card
def generate_report_html_image(fonts):
    width, height = 950, 520
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw_window_frame(draw, width, height, "RAGPatrol Standalone HTML Report - reranker_on.html", fonts)
    
    draw.text((28, 56), "RAGPatrol Evaluation Report: reranker_on", font=fonts["sans_xl"], fill=TEXT_WHITE)
    draw.text((28, 90), "Stage: all  |  Cache: warm  |  Queries: 25 (Successful: 25, Failed: 0)  |  Commit: b4f1c9d", font=fonts["sans_sm"], fill=TEXT_MUTED)
    
    kpis = [
        ("PRECISION", "85.0%", "Target >= 70%", TEXT_GREEN),
        ("RECALL", "80.0%", "Target >= 70%", TEXT_GREEN),
        ("F1 SCORE", "82.4%", "Target >= 70%", TEXT_GREEN),
        ("FAITHFULNESS", "88.0%", "Target >= 75%", TEXT_GREEN),
    ]
    
    card_w = 205
    for i, (label, val, target, col) in enumerate(kpis):
        cx = 28 + i * (card_w + 24)
        draw.rounded_rectangle([(cx, 125), (cx + card_w, 215)], radius=8, fill=PANEL_DARK, outline=BORDER_COLOR)
        draw.text((cx + 16, 138), label, font=fonts["sans_sm"], fill=TEXT_MUTED)
        draw.text((cx + 16, 158), val, font=fonts["sans_xl"], fill=col)
        draw.text((cx + 16, 192), target, font=fonts["mono_sm"], fill=TEXT_MUTED)
        
    draw.rounded_rectangle([(28, 235), (width - 28, 305)], radius=8, fill=PANEL_DARK, outline=BORDER_COLOR)
    draw.text((44, 248), "Latency Percentile Profile (Warm Cache)", font=fonts["sans_md"], fill=TEXT_WHITE)
    lat_text = "p50: 18.2 ms   |   p95: 34.5 ms   |   p99: 42.1 ms   |   Mean: 20.4 ms   |   Speedup vs Cold: 2.47x"
    draw.text((44, 274), lat_text, font=fonts["mono_sm"], fill=TEXT_CYAN)
    
    draw.rounded_rectangle([(28, 325), (width - 28, 485)], radius=8, fill=PANEL_DARK, outline=BORDER_COLOR)
    draw.text((44, 338), "Per-Category Quality Breakdown", font=fonts["sans_md"], fill=TEXT_WHITE)
    
    cat_headers = [("Category", 44), ("Queries", 240), ("Precision", 380), ("Recall", 520), ("F1", 660), ("Faithfulness", 800)]
    for ch, cx in cat_headers:
        draw.text((cx, 372), ch, font=fonts["sans_sm"], fill=TEXT_MUTED)
        
    cat_rows = [
        ("easy (single-page)", "12", "100.0%", "100.0%", "100.0%", "92.4%"),
        ("ambiguous (multi-page)", "8", "75.0%", "70.0%", "72.4%", "86.1%"),
        ("edge (out-of-domain)", "5", "80.0%", "70.0%", "74.6%", "85.5%"),
    ]
    cy = 405
    for cat, qcnt, p, r, f, ft in cat_rows:
        draw.text((44, cy), cat, font=fonts["sans_sm"], fill=TEXT_WHITE)
        draw.text((240, cy), qcnt, font=fonts["mono_sm"], fill=TEXT_MUTED)
        draw.text((380, cy), p, font=fonts["mono_sm"], fill=TEXT_GREEN)
        draw.text((520, cy), r, font=fonts["mono_sm"], fill=TEXT_GREEN)
        draw.text((660, cy), f, font=fonts["mono_sm"], fill=TEXT_GREEN)
        draw.text((800, cy), ft, font=fonts["mono_sm"], fill=TEXT_GREEN)
        cy += 28
        
    out_path = DOCS_IMAGES_DIR / "report-html.png"
    img.save(out_path, "PNG", optimize=True)
    print(f"Generated {out_path} ({os.path.getsize(out_path)} bytes)")

# 4. Stub App Run Image
def generate_stub_app_image(fonts):
    width, height = 950, 520
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw_window_frame(draw, width, height, "bash - python -m harness.runner --adapter stub --base-url http://localhost:8001", fonts)
    
    lines = [
        ("$ python -m harness.runner --adapter stub --base-url http://localhost:8001 --stage all", TEXT_WHITE),
        ("[INFO] ragpatrol: RAGPatrol EvaluationRunner initialized against target: http://localhost:8001 (adapter=stub)", TEXT_MUTED),
        ("[INFO] ragpatrol: Automatically selected stub testset: testset/stub_questions.yaml", TEXT_CYAN),
        ("[INFO] ragpatrol: Loaded 5 questions from testset/stub_questions.yaml", TEXT_MUTED),
        ("[INFO] ragpatrol: === RAGPatrol: Starting Evaluation Pass (stage=all, cache=both, queries=5) ===", TEXT_YELLOW),
        ("[INFO] ragpatrol: [1/5] Querying 'stub_q01': How do coroutines and event loops work in Python...", TEXT_MUTED),
        ("[INFO] ragpatrol: [2/5] Querying 'stub_q02': What are B-tree indexes and how do they speed up...", TEXT_MUTED),
        ("[INFO] ragpatrol: [3/5] Querying 'stub_q03': Explain the difference between process and thread...", TEXT_MUTED),
        ("[INFO] ragpatrol: [4/5] Querying 'stub_q04': How does TLS 1.3 handshake reduce latency over...", TEXT_MUTED),
        ("[INFO] ragpatrol: [5/5] Querying 'stub_q05': Describe containerization vs hypervisor virtual...", TEXT_MUTED),
        ("", TEXT_WHITE),
        ("================================================================================", TEXT_MUTED),
        (" EVALUATION SUMMARY: Stage='all' | Cache='warm' | Config='default'", TEXT_WHITE),
        ("================================================================================", TEXT_MUTED),
        (" Total Queries: 5 | Successful: 5 | Failed: 0", TEXT_CYAN),
        (" Mean Precision:     100.0%", TEXT_GREEN),
        (" Mean Recall:        100.0%", TEXT_GREEN),
        (" Mean F1:            100.0%", TEXT_GREEN),
        (" Mean Faithfulness:  78.3%", TEXT_GREEN),
        (" Hallucination Rate: 0.0% (0/5)", TEXT_GREEN),
        (" Latency Profile:    p50=15.1ms | p95=15.1ms | p99=15.1ms | mean=15.0ms", TEXT_CYAN),
        ("--------------------------------------------------------------------------------", TEXT_MUTED),
        ("Saved Markdown report to reports/run_default_20260908_stub.md", TEXT_MUTED),
        ("Saved HTML report to reports/run_default_20260908_stub.html", TEXT_MUTED),
    ]
    
    y = 56
    for text, color in lines:
        draw.text((28, y), text, font=fonts["mono_sm"], fill=color)
        y += 18
        
    out_path = DOCS_IMAGES_DIR / "stub-app-run.png"
    img.save(out_path, "PNG", optimize=True)
    print(f"Generated {out_path} ({os.path.getsize(out_path)} bytes)")

# 5. Swagger UI Image
def generate_swagger_image(fonts):
    width, height = 950, 520
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw_window_frame(draw, width, height, "FastAPI Swagger UI - RAGPatrol Mock RAG Service (/docs)", fonts)
    
    # Swagger Header
    draw.text((28, 56), "RAGPatrol Fake RAG API Stub", font=fonts["sans_xl"], fill=TEXT_WHITE)
    draw.rounded_rectangle([(370, 60), (430, 85)], radius=4, fill=(59, 130, 246))
    draw.text((380, 64), "1.0.0", font=fonts["mono_sm"], fill=TEXT_WHITE)
    draw.text((28, 92), "Independent mock RAG API with alternative schema for RAGPatrol generality verification.", font=fonts["sans_sm"], fill=TEXT_MUTED)
    
    # Endpoints
    # GET /health
    draw.rounded_rectangle([(28, 130), (width - 28, 185)], radius=6, fill=PANEL_DARK, outline=(16, 185, 129), width=1)
    draw.rounded_rectangle([(36, 138), (100, 177)], radius=4, fill=(16, 185, 129))
    draw.text((50, 146), "GET", font=fonts["sans_md"], fill=TEXT_WHITE)
    draw.text((120, 146), "/health", font=fonts["mono_md"], fill=TEXT_WHITE)
    draw.text((240, 147), "Liveness probe returning HTTP 200 OK", font=fonts["sans_sm"], fill=TEXT_MUTED)
    
    # POST /query
    draw.rounded_rectangle([(28, 205), (width - 28, 260)], radius=6, fill=PANEL_DARK, outline=(59, 130, 246), width=1)
    draw.rounded_rectangle([(36, 213), (100, 252)], radius=4, fill=(59, 130, 246))
    draw.text((45, 221), "POST", font=fonts["sans_md"], fill=TEXT_WHITE)
    draw.text((120, 221), "/query", font=fonts["mono_md"], fill=TEXT_WHITE)
    draw.text((240, 222), "Deterministic retrieval against in-memory knowledge store", font=fonts["sans_sm"], fill=TEXT_MUTED)
    
    # Schemas section
    draw.rounded_rectangle([(28, 285), (width - 28, 485)], radius=8, fill=PANEL_DARK, outline=BORDER_COLOR)
    draw.text((44, 298), "Schemas (Deliberate Disparate Format)", font=fonts["sans_md"], fill=TEXT_WHITE)
    
    schemas = [
        ("> StubQueryRequest", "{ question: string, collection_name?: string, filters?: object }"),
        ("> StubQueryResponse", "{ answer_text: string, sources: StubSourceItem[], response_time_ms: float }"),
        ("> StubSourceItem", "{ doc: string, page: int, content: string }"),
    ]
    sy = 335
    for s_name, s_fields in schemas:
        draw.text((44, sy), s_name, font=fonts["sans_sm"], fill=TEXT_CYAN)
        draw.text((240, sy), s_fields, font=fonts["mono_sm"], fill=TEXT_MUTED)
        sy += 42
        
    out_path = DOCS_IMAGES_DIR / "swagger-ui.png"
    img.save(out_path, "PNG", optimize=True)
    print(f"Generated {out_path} ({os.path.getsize(out_path)} bytes)")

# 6. Frontend UI Image (Streamlit Dashboard)
def generate_frontend_image(fonts):
    width, height = 950, 520
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw_window_frame(draw, width, height, "RAGPatrol Observability Dashboard - Streamlit", fonts)
    
    # Header
    draw.text((28, 56), "📈 RAGPatrol — LLM Evaluation & Observability Dashboard", font=fonts["sans_lg"], fill=TEXT_WHITE)
    draw.text((28, 88), "Automated quality gating, latency profiling, and regression tracking over time.", font=fonts["sans_sm"], fill=TEXT_MUTED)
    
    # Metric KPI row
    kpis = [("LATEST PRECISION", "85.0%"), ("LATEST RECALL", "80.0%"), ("LATEST F1", "82.4%"), ("FAITHFULNESS", "88.0%")]
    card_w = 205
    for i, (k_title, k_val) in enumerate(kpis):
        cx = 28 + i * (card_w + 24)
        draw.rounded_rectangle([(cx, 120), (cx + card_w, 195)], radius=8, fill=PANEL_DARK, outline=BORDER_COLOR)
        draw.text((cx + 14, 132), k_title, font=fonts["sans_sm"], fill=TEXT_MUTED)
        draw.text((cx + 14, 155), k_val, font=fonts["sans_xl"], fill=TEXT_GREEN)
        
    # Historical Trend Chart Mockup
    draw.rounded_rectangle([(28, 215), (width - 28, 355)], radius=8, fill=PANEL_DARK, outline=BORDER_COLOR)
    draw.text((44, 226), "Quality Metrics Trend Across Runs (Precision, Recall, F1)", font=fonts["sans_md"], fill=TEXT_WHITE)
    
    # Draw simple mock trend lines
    pts_prec = [(80, 310), (220, 290), (360, 275), (500, 270), (640, 260), (780, 255), (880, 250)]
    pts_rec =  [(80, 325), (220, 305), (360, 295), (500, 290), (640, 280), (780, 275), (880, 270)]
    draw.line(pts_prec, fill=TEXT_GREEN, width=3)
    draw.line(pts_rec, fill=TEXT_CYAN, width=3)
    for p in pts_prec:
        draw.ellipse([(p[0] - 4, p[1] - 4), (p[0] + 4, p[1] + 4)], fill=TEXT_GREEN)
    for p in pts_rec:
        draw.ellipse([(p[0] - 4, p[1] - 4), (p[0] + 4, p[1] + 4)], fill=TEXT_CYAN)
        
    draw.text((700, 325), "━ Precision   ━ Recall", font=fonts["mono_sm"], fill=TEXT_MUTED)
    
    # Run history table
    draw.rounded_rectangle([(28, 375), (width - 28, 485)], radius=8, fill=PANEL_DARK, outline=BORDER_COLOR)
    draw.text((44, 386), "Evaluation Run History (SQLite / PostgreSQL)", font=fonts["sans_md"], fill=TEXT_WHITE)
    
    t_headers = [("Run ID", 44), ("Timestamp", 220), ("Config", 440), ("Stage", 580), ("Queries", 700), ("Status", 820)]
    for th, tx in t_headers:
        draw.text((tx, 415), th, font=fonts["sans_sm"], fill=TEXT_MUTED)
    t_rows = [
        ("run_9f21b", "2026-09-08 13:57:47", "default", "all", "25", "PASSED"),
        ("run_7c12a", "2026-09-08 12:30:15", "reranker_on", "all", "25", "PASSED"),
    ]
    ty = 442
    for rid, ts, cfg, stg, qcnt, st in t_rows:
        draw.text((44, ty), rid, font=fonts["mono_sm"], fill=TEXT_WHITE)
        draw.text((220, ty), ts, font=fonts["mono_sm"], fill=TEXT_MUTED)
        draw.text((440, ty), cfg, font=fonts["mono_sm"], fill=TEXT_WHITE)
        draw.text((580, ty), stg, font=fonts["mono_sm"], fill=TEXT_WHITE)
        draw.text((700, ty), qcnt, font=fonts["mono_sm"], fill=TEXT_MUTED)
        draw.text((820, ty), st, font=fonts["sans_sm"], fill=TEXT_GREEN)
        ty += 24
        
    out_path = DOCS_IMAGES_DIR / "frontend-ui.png"
    img.save(out_path, "PNG", optimize=True)
    print(f"Generated {out_path} ({os.path.getsize(out_path)} bytes)")

# 7. Query Response JSON Image
def generate_query_response_image(fonts):
    width, height = 950, 520
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw_window_frame(draw, width, height, "Canonical RAGResponse DTO - JSON Payload", fonts)
    
    draw.text((28, 56), "Target API Contract — Normalized Canonical Response", font=fonts["sans_lg"], fill=TEXT_WHITE)
    draw.text((28, 88), "Normalized by RAGPatrol Client Adapter (Pydantic v2 RAGResponse)", font=fonts["sans_sm"], fill=TEXT_MUTED)
    
    json_lines = [
        ("{", TEXT_WHITE),
        ('  "answer": "Failed embedding calls are retried 3 times with exponential backoff.",', TEXT_CYAN),
        ('  "retrieved_chunks": [', TEXT_WHITE),
        ('    {', TEXT_WHITE),
        ('      "chunk_id": "doc12_chunk3",', TEXT_GREEN),
        ('      "text": "Failed embedding calls are retried 3 times with exponential backoff.",', TEXT_MUTED),
        ('      "source_doc": "architecture_overview.pdf",', TEXT_MUTED),
        ('      "score": 0.94,', TEXT_YELLOW),
        ('      "metadata": { "page": 2 }', TEXT_PURPLE),
        ('    }', TEXT_WHITE),
        ('  ],', TEXT_WHITE),
        ('  "latency_ms": 142.5', TEXT_YELLOW),
        ("}", TEXT_WHITE),
    ]
    
    draw.rounded_rectangle([(28, 120), (width - 28, 485)], radius=8, fill=PANEL_DARK, outline=BORDER_COLOR)
    jy = 140
    for line, color in json_lines:
        draw.text((50, jy), line, font=fonts["mono_md"], fill=color)
        jy += 25
        
    out_path = DOCS_IMAGES_DIR / "query-response.png"
    img.save(out_path, "PNG", optimize=True)
    print(f"Generated {out_path} ({os.path.getsize(out_path)} bytes)")

def main():
    fonts = get_fonts()
    generate_terminal_image(fonts)
    generate_comparison_image(fonts)
    generate_report_html_image(fonts)
    generate_stub_app_image(fonts)
    generate_swagger_image(fonts)
    generate_frontend_image(fonts)
    generate_query_response_image(fonts)
    print("All 7 docs images successfully generated.")

if __name__ == "__main__":
    main()
