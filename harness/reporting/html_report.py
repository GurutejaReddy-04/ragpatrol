"""
HTML evaluation report generator.

Renders standalone, zero-framework HTML reports viewable directly in browsers
or attached to CI/CD pipeline artifacts.
"""

import html
import logging
from typing import Any, Optional
from harness.storage.models import EvalRun

logger = logging.getLogger(__name__)


class HTMLReportGenerator:
    """
    Renders clean, self-contained HTML evaluation summaries.
    """

    def __init__(self) -> None:
        logger.debug("Initialized HTMLReportGenerator.")

    def generate(self, run: EvalRun, extra_context: Optional[dict[str, Any]] = None) -> str:
        """
        Generate standalone HTML report.

        :param run: Populated EvalRun ORM instance.
        :param extra_context: Optional additional metadata.
        :return: HTML document string.
        """
        logger.info("Generating HTML report for run ID: %s", run.id)
        status_color = "#10b981" if run.passed else "#ef4444"
        status_text = "PASSED" if run.passed else "FAILED"

        rows = ""
        for m in run.metrics:
            rows += f"<tr><td>{html.escape(m.metric_name)}</td><td>{m.value:.4f}</td><td>{html.escape(m.category or 'aggregate')}</td></tr>\n"

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Eval Report - {html.escape(run.id)}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 40px; background: #0f172a; color: #e2e8f0; }}
        h1, h2 {{ color: #f8fafc; }}
        .badge {{ display: inline-block; padding: 4px 12px; border-radius: 9999px; background: {status_color}; color: white; font-weight: bold; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 16px; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #334155; }}
        th {{ background: #1e293b; color: #94a3b8; }}
        tr:hover {{ background: #1e293b; }}
    </style>
</head>
<body>
    <h1>Evaluation Run: {html.escape(run.id)}</h1>
    <p><span class="badge">{status_text}</span> Config: <strong>{html.escape(run.config_name)}</strong> | Cache: <strong>{html.escape(run.cache_state)}</strong></p>
    <h2>Metrics</h2>
    <table>
        <thead><tr><th>Metric</th><th>Value</th><th>Category</th></tr></thead>
        <tbody>
            {rows or '<tr><td colspan="3">No metrics recorded</td></tr>'}
        </tbody>
    </table>
</body>
</html>
"""
