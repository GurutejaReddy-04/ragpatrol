"""
HTML evaluation report generator (Phase 7).

Renders standalone, zero-dependency HTML reports with embedded CSS styling.
Follows a clean dashboard aesthetic with interactive CSS row-highlighting and print-to-PDF support.
"""

from datetime import datetime, timezone
import html
import logging
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from harness.runner import StageRunResult

logger = logging.getLogger(__name__)

SHARED_CSS = """
:root {
    --bg-main: #f8fafc;
    --card-bg: #ffffff;
    --text-primary: #0f172a;
    --text-secondary: #64748b;
    --border-color: #e2e8f0;
    --accent-blue: #3b82f6;
    --success-bg: #ecfdf5;
    --success-text: #059669;
    --warning-bg: #fffbeb;
    --warning-text: #d97706;
    --danger-bg: #fef2f2;
    --danger-text: #dc2626;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    background-color: var(--bg-main);
    color: var(--text-primary);
    line-height: 1.5;
    margin: 0;
    padding: 32px 24px;
}

.container {
    max-width: 1100px;
    margin: 0 auto;
}

header {
    margin-bottom: 24px;
    padding-bottom: 16px;
    border-bottom: 1px solid var(--border-color);
}

h1 {
    font-size: 26px;
    font-weight: 700;
    margin: 0 0 8px 0;
    color: var(--text-primary);
}

.meta-bar {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    align-items: center;
    font-size: 13px;
    color: var(--text-secondary);
}

.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 9999px;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.02em;
}

.badge-blue { background: #eff6ff; color: #1d4ed8; }
.badge-green { background: var(--success-bg); color: var(--success-text); }
.badge-yellow { background: var(--warning-bg); color: var(--warning-text); }
.badge-red { background: var(--danger-bg); color: var(--danger-text); }
.badge-gray { background: #f1f5f9; color: #475569; }

.kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 16px;
    margin-bottom: 24px;
}

.kpi-card {
    background: var(--card-bg);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 16px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
}

.kpi-label {
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    color: var(--text-secondary);
    letter-spacing: 0.05em;
}

.kpi-value {
    font-size: 24px;
    font-weight: 700;
    margin-top: 4px;
    color: var(--text-primary);
}

.card {
    background: var(--card-bg);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 20px;
    margin-bottom: 24px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
}

.card-title {
    font-size: 17px;
    font-weight: 600;
    margin: 0 0 16px 0;
    color: var(--text-primary);
}

table {
    width: 100%;
    border-collapse: collapse;
    font-size: 14px;
}

th {
    background: #f8fafc;
    color: var(--text-secondary);
    text-align: left;
    padding: 10px 14px;
    font-weight: 600;
    border-bottom: 1px solid var(--border-color);
}

td {
    padding: 10px 14px;
    border-bottom: 1px solid var(--border-color);
    color: var(--text-primary);
}

tr:hover td {
    background-color: #f1f5f9;
}

details {
    cursor: pointer;
}

details summary {
    font-weight: 600;
    color: var(--accent-blue);
    padding: 8px 0;
}

.winner-pill {
    font-weight: 700;
    color: #15803d;
    background: #f0fdf4;
    padding: 2px 8px;
    border-radius: 4px;
    display: inline-block;
}

@media print {
    body { background: #ffffff !important; padding: 0 !important; color: #000 !important; }
    .card { box-shadow: none !important; border: 1px solid #ccc !important; break-inside: avoid; }
    table { break-inside: auto; }
    tr { break-inside: avoid; }
    details { open: true !important; }
}
"""


class HTMLReportGenerator:
    """
    Renders standalone HTML reports with modern responsive CSS.
    """

    @classmethod
    def generate_run_report(
        cls,
        run: "StageRunResult",
        git_commit_sha: Optional[str] = None,
        cold_run: Optional["StageRunResult"] = None,
    ) -> str:
        """
        Generate standalone HTML report for an individual evaluation run.
        """
        timestamp_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        commit_str = git_commit_sha[:8] if git_commit_sha else "N/A"

        # Flagged hallucinations markup
        flagged = [s for s in run.question_summaries if s.is_hallucination]
        hallucination_rows = ""
        if flagged:
            for f in flagged:
                hallucination_rows += f"""
                <div style="border-left: 3px solid #f59e0b; padding-left: 12px; margin-bottom: 16px;">
                    <div style="font-weight: 600; color: #92400e;">[{html.escape(f.id)}] {html.escape(f.question)}</div>
                    <div style="font-size: 13px; color: #475569; margin-top: 4px;">
                        Judge Score: <strong>{f.llm_judge_score}/5.0</strong> | Embedding Similarity: <strong>{f.embedding_similarity:.2f}</strong>
                    </div>
                    <div style="font-size: 13px; font-style: italic; color: #64748b; margin-top: 4px;">
                        {html.escape(f.judge_reasoning or 'No reasoning text')}
                    </div>
                </div>
                """
        else:
            hallucination_rows = "<p style='color: #059669;'>No hallucinations flagged during this evaluation pass.</p>"

        # Trace rows
        trace_rows = ""
        for q in run.question_summaries:
            badge_class = "badge-red" if q.is_hallucination else "badge-green"
            badge_text = "YES" if q.is_hallucination else "NO"
            trace_rows += f"""
            <tr>
                <td><code>{html.escape(q.id)}</code></td>
                <td><span class="badge badge-gray">{html.escape(q.category)}</span></td>
                <td>{q.precision*100:.1f}%</td>
                <td>{q.recall*100:.1f}%</td>
                <td>{q.f1*100:.1f}%</td>
                <td>{q.faithfulness_score*100:.1f}%</td>
                <td>{q.latency_ms:.1f} ms</td>
                <td><span class="badge {badge_class}">{badge_text}</span></td>
            </tr>
            """

        # Category rows
        cat_rows = ""
        for cat, vals in run.category_metrics.items():
            cat_rows += f"""
            <tr>
                <td><strong>{html.escape(cat)}</strong></td>
                <td>{vals['count']}</td>
                <td>{vals['precision']*100:.1f}%</td>
                <td>{vals['recall']*100:.1f}%</td>
                <td>{vals['f1']*100:.1f}%</td>
                <td>{vals['faithfulness']*100:.1f}%</td>
                <td>{vals['hallucination_rate']*100:.1f}%</td>
            </tr>
            """

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Evaluation Report - {html.escape(run.config_name)}</title>
    <style>{SHARED_CSS}</style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Evaluation Report: {html.escape(run.config_name)}</h1>
            <div class="meta-bar">
                <span>Timestamp: <strong>{html.escape(timestamp_str)}</strong></span>
                <span class="badge badge-blue">Stage: {html.escape(run.stage)}</span>
                <span class="badge badge-gray">Cache: {html.escape(run.cache_state)}</span>
                <span>Commit: <code>{html.escape(commit_str)}</code></span>
                <span>Queries: <strong>{run.total_queries}</strong></span>
            </div>
        </header>

        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-label">Precision</div>
                <div class="kpi-value">{run.mean_precision*100:.1f}%</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Recall</div>
                <div class="kpi-value">{run.mean_recall*100:.1f}%</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">F1 Score</div>
                <div class="kpi-value">{run.mean_f1*100:.1f}%</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Faithfulness</div>
                <div class="kpi-value">{run.mean_faithfulness*100:.1f}%</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Latency p50</div>
                <div class="kpi-value">{run.latency_profile.p50_ms:.1f} ms</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Latency p95</div>
                <div class="kpi-value">{run.latency_profile.p95_ms:.1f} ms</div>
            </div>
        </div>

        <div class="card">
            <h2 class="card-title">Category Breakdown</h2>
            <table>
                <thead>
                    <tr><th>Category</th><th>Count</th><th>Precision</th><th>Recall</th><th>F1</th><th>Faithfulness</th><th>Hallucination Rate</th></tr>
                </thead>
                <tbody>
                    {cat_rows}
                </tbody>
            </table>
        </div>

        <div class="card">
            <h2 class="card-title">Flagged Hallucinations & Low Faithfulness ({len(flagged)})</h2>
            {hallucination_rows}
        </div>

        <div class="card">
            <h2 class="card-title">Question-Level Trace</h2>
            <details>
                <summary>Expand full table of {len(run.question_summaries)} questions</summary>
                <table style="margin-top: 12px;">
                    <thead>
                        <tr><th>ID</th><th>Category</th><th>Precision</th><th>Recall</th><th>F1</th><th>Faithfulness</th><th>Latency</th><th>Flagged</th></tr>
                    </thead>
                    <tbody>
                        {trace_rows}
                    </tbody>
                </table>
            </details>
        </div>
    </div>
</body>
</html>
"""

    @classmethod
    def generate_comparison_report(
        cls,
        comparison: dict[str, Any],
        git_commit_sha: Optional[str] = None,
    ) -> str:
        """
        Generate standalone HTML report for a side-by-side configuration experiment.
        """
        timestamp_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        commit_str = git_commit_sha[:8] if git_commit_sha else "N/A"

        if comparison.get("status") == "unreachable":
            return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>Comparison Aborted</title><style>{SHARED_CSS}</style></head>
<body>
    <div class="container">
        <h1>Configuration Comparison Aborted</h1>
        <p class="badge badge-red">One or more configurations were unreachable</p>
        <div class="card">
            <p>Config A ({html.escape(comparison['config_a']['name'])}): {html.escape(str(comparison['config_a'].get('error')))}</p>
            <p>Config B ({html.escape(comparison['config_b']['name'])}): {html.escape(str(comparison['config_b'].get('error')))}</p>
        </div>
    </div>
</body>
</html>"""

        cfg_a = comparison["config_a_name"]
        cfg_b = comparison["config_b_name"]
        winner = comparison["winner"]

        metric_rows = ""
        for row in comparison.get("metrics_comparison", []):
            unit = row["unit"]
            row_winner = row["winner"]
            diff = row["val_a"] - row["val_b"]

            if unit == "%":
                str_a = f"{row['val_a']*100:.1f}%"
                str_b = f"{row['val_b']*100:.1f}%"
                diff_str = f"{diff*100:+.1f}%"
            else:
                str_a = f"{row['val_a']:.1f} ms"
                str_b = f"{row['val_b']:.1f} ms"
                diff_str = f"{diff:+.1f} ms"

            pill_a = f"<span class='winner-pill'>{str_a}</span>" if row_winner == cfg_a else str_a
            pill_b = f"<span class='winner-pill'>{str_b}</span>" if row_winner == cfg_b else str_b
            pill_winner = f"<span class='badge badge-green'>{html.escape(row_winner)}</span>" if row_winner != "Tie" else "<span class='badge badge-gray'>Tie</span>"

            metric_rows += f"""
            <tr>
                <td><strong>{html.escape(row['metric'])}</strong></td>
                <td>{pill_a}</td>
                <td>{pill_b}</td>
                <td>{diff_str}</td>
                <td>{pill_winner}</td>
            </tr>
            """

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Comparison: {html.escape(cfg_a)} vs {html.escape(cfg_b)}</title>
    <style>{SHARED_CSS}</style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Configuration Experiment: {html.escape(cfg_a)} vs {html.escape(cfg_b)}</h1>
            <div class="meta-bar">
                <span>Timestamp: <strong>{html.escape(timestamp_str)}</strong></span>
                <span>Commit: <code>{html.escape(commit_str)}</code></span>
                <span class="badge badge-green">Winner: {html.escape(winner)}</span>
            </div>
        </header>

        <div class="card" style="background: #eff6ff; border-color: #bfdbfe;">
            <div style="font-weight: 600; color: #1e40af;">Experiment Outcome</div>
            <div style="font-size: 15px; color: #1e3a8a; margin-top: 4px;">{html.escape(comparison.get('summary', ''))}</div>
        </div>

        <div class="card">
            <h2 class="card-title">Side-by-Side Metric Comparison</h2>
            <table>
                <thead>
                    <tr><th>Metric</th><th>{html.escape(cfg_a)}</th><th>{html.escape(cfg_b)}</th><th>Delta</th><th>Winner</th></tr>
                </thead>
                <tbody>
                    {metric_rows}
                </tbody>
            </table>
        </div>
    </div>
</body>
</html>
"""
