"""
Markdown evaluation report generator.

Renders evaluation run metrics, category breakdowns, and hallucination flags
into clean GitHub-flavored markdown.
"""

import logging
from typing import Any, Optional
from harness.storage.models import EvalRun

logger = logging.getLogger(__name__)


class MarkdownReportGenerator:
    """
    Renders structured Markdown reports for human consumption and CI artifacts.
    """

    def __init__(self) -> None:
        logger.debug("Initialized MarkdownReportGenerator.")

    def generate(self, run: EvalRun, extra_context: Optional[dict[str, Any]] = None) -> str:
        """
        Generate a Markdown report string for an evaluation run.

        :param run: Populated EvalRun ORM instance with metrics.
        :param extra_context: Optional contextual parameters (e.g., git branch).
        :return: Formatted Markdown string.
        """
        logger.info("Generating Markdown report for run ID: %s", run.id)
        lines: list[str] = [
            f"# Evaluation Report: Run `{run.id}`",
            "",
            f"- **Timestamp:** {run.timestamp}",
            f"- **Configuration:** {run.config_name}",
            f"- **Cache State:** {run.cache_state}",
            f"- **Overall Status:** {'PASSED' if run.passed else 'FAILED'}",
            "",
            "## Summary Metrics",
            "",
            "| Metric | Value | Category |",
            "| :--- | :--- | :--- |",
        ]

        if run.metrics:
            for m in run.metrics:
                cat = m.category or "aggregate"
                lines.append(f"| {m.metric_name} | {m.value:.4f} | {cat} |")
        else:
            lines.append("| *No metrics recorded* | - | - |")

        lines.append("")
        lines.append("## Question Level Breakdown")
        lines.append("")
        lines.append("| Question ID | Precision | Recall | F1 | Faithfulness | Latency (ms) | Hallucination? |")
        lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")

        if run.question_results:
            for q in run.question_results:
                flag = "⚠️ YES" if q.hallucination_flag else "NO"
                lines.append(
                    f"| {q.question_id} | {q.precision:.2f} | {q.recall:.2f} | {q.f1:.2f} | "
                    f"{q.faithfulness_score:.2f} | {q.latency_ms:.1f} | {flag} |"
                )
        else:
            lines.append("| *No question results recorded* | - | - | - | - | - | - |")

        lines.append("")
        return "\n".join(lines)
