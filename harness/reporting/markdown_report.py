"""
Markdown evaluation report generator.

Renders single-run benchmarks and side-by-side configuration experiments into
clean, human-readable, GitHub-flavored Markdown artifacts.
"""

from datetime import datetime, timezone
import logging
from typing import TYPE_CHECKING, Any, Optional, Union

if TYPE_CHECKING:
    from harness.runner import StageRunResult

logger = logging.getLogger(__name__)


class MarkdownReportGenerator:
    """
    Renders structured Markdown reports for evaluation runs and comparison experiments.
    """

    @classmethod
    def generate_run_report(
        cls,
        run: "StageRunResult",
        git_commit_sha: Optional[str] = None,
        cold_run: Optional["StageRunResult"] = None,
    ) -> str:
        """
        Generate a GitHub-flavored Markdown report for an individual evaluation run.

        :param run: Evaluated StageRunResult.
        :param git_commit_sha: Optional git commit hash.
        :param cold_run: Optional cold run result for latency speedup comparison.
        :return: Formatted Markdown string.
        """
        timestamp_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        commit_str = f"`{git_commit_sha[:8]}`" if git_commit_sha else "`N/A`"

        lines: list[str] = [
            f"# RAGPatrol Evaluation Report: `{run.config_name}`",
            "",
            "## Execution Metadata",
            "",
            f"- **Timestamp:** {timestamp_str}",
            f"- **Configuration:** `{run.config_name}`",
            f"- **Pipeline Stage:** `{run.stage}`",
            f"- **Cache Mode:** `{run.cache_state}`",
            f"- **Git Commit:** {commit_str}",
            f"- **Total Queries:** {run.total_queries} (Successful: {run.successful_queries}, Failed: {run.failed_queries})",
            "",
            "## Summary Quality Metrics",
            "",
            "| Metric | Result | Benchmark Target | Status |",
            "| :--- | :---: | :---: | :---: |",
            f"| **Retrieval Precision** | {run.mean_precision * 100:.1f}% | >= 70.0% | {'✅ Pass' if run.mean_precision >= 0.70 else '⚠️ Review'} |",
            f"| **Retrieval Recall** | {run.mean_recall * 100:.1f}% | >= 70.0% | {'✅ Pass' if run.mean_recall >= 0.70 else '⚠️ Review'} |",
            f"| **Retrieval F1 Score** | {run.mean_f1 * 100:.1f}% | >= 70.0% | {'✅ Pass' if run.mean_f1 >= 0.70 else '⚠️ Review'} |",
            f"| **Faithfulness Groundedness** | {run.mean_faithfulness * 100:.1f}% | >= 75.0% | {'✅ Pass' if run.mean_faithfulness >= 0.75 else '⚠️ Review'} |",
            f"| **Hallucination Rate** | {run.hallucination_rate * 100:.1f}% ({run.hallucination_count}/{run.total_queries}) | < 10.0% | {'✅ Pass' if run.hallucination_rate < 0.10 else '⚠️ High'} |",
            "",
            "## Latency Percentile Profile",
            "",
        ]

        if cold_run:
            speedup = cold_run.latency_profile.p50_ms / run.latency_profile.p50_ms if run.latency_profile.p50_ms > 0 else 1.0
            lines.extend([
                "| Cache State | p50 | p95 | p99 | Mean | Std Dev | Speedup |",
                "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
                f"| **Cold** | {cold_run.latency_profile.p50_ms:.1f} ms | {cold_run.latency_profile.p95_ms:.1f} ms | {cold_run.latency_profile.p99_ms:.1f} ms | {cold_run.latency_profile.mean_ms:.1f} ms | {cold_run.latency_profile.std_ms:.1f} ms | 1.0x |",
                f"| **Warm** | {run.latency_profile.p50_ms:.1f} ms | {run.latency_profile.p95_ms:.1f} ms | {run.latency_profile.p99_ms:.1f} ms | {run.latency_profile.mean_ms:.1f} ms | {run.latency_profile.std_ms:.1f} ms | **{speedup:.1f}x** |",
                "",
            ])
        else:
            lat = run.latency_profile
            lines.extend([
                "| Metric | Latency (ms) | Ceiling SLA | Status |",
                "| :--- | :---: | :---: | :---: |",
                f"| **p50 (Median)** | {lat.p50_ms:.1f} ms | - | - |",
                f"| **p95** | {lat.p95_ms:.1f} ms | <= 2500.0 ms | {'✅ Pass' if lat.p95_ms <= 2500.0 else '⚠️ High'} |",
                f"| **p99** | {lat.p99_ms:.1f} ms | <= 5000.0 ms | {'✅ Pass' if lat.p99_ms <= 5000.0 else '⚠️ High'} |",
                f"| **Mean** | {lat.mean_ms:.1f} ms | - | - |",
                "",
            ])

        # Category breakdown
        lines.extend([
            "## Per-Category Performance Breakdown",
            "",
            "| Category | Queries | Precision | Recall | F1 Score | Faithfulness | Hallucination Rate |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        ])
        for cat, vals in run.category_metrics.items():
            lines.append(
                f"| `{cat}` | {vals['count']} | {vals['precision']*100:.1f}% | {vals['recall']*100:.1f}% | "
                f"{vals['f1']*100:.1f}% | {vals['faithfulness']*100:.1f}% | {vals['hallucination_rate']*100:.1f}% |"
            )
        lines.append("")

        # Flagged Hallucinations
        flagged = [s for s in run.question_summaries if s.is_hallucination]
        lines.extend([
            "## Flagged Hallucinations & Low Faithfulness Anomalies",
            "",
        ])
        if flagged:
            for f in flagged:
                lines.append(f"### ⚠️ Question `{f.id}` (`{f.category}`)")
                lines.append(f"**Query:** {f.question}")
                lines.append("")
                lines.append(f"- **LLM Judge Score:** {f.llm_judge_score}/5.0")
                lines.append(f"- **Embedding Similarity:** {f.embedding_similarity:.2f}")
                if f.judge_reasoning:
                    lines.append(f"- **Judge Reasoning:** *{f.judge_reasoning.strip()}*")
                if f.unsupported_claims:
                    lines.append(f"- **Unsupported Claims:** `{f.unsupported_claims}`")
                lines.append("")
        else:
            lines.append("*No hallucinations detected in this evaluation pass.*")
            lines.append("")

        # Collapsible per-question trace
        lines.extend([
            "## Detailed Question-Level Trace",
            "",
            "<details>",
            "<summary><strong>Click to expand full per-question trace (25 queries)</strong></summary>",
            "",
            "| ID | Category | Precision | Recall | F1 | Faithfulness | Latency | Flagged? |",
            "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        ])
        for q in run.question_summaries:
            flag_icon = "⚠️ YES" if q.is_hallucination else "NO"
            lines.append(
                f"| `{q.id}` | `{q.category}` | {q.precision:.2f} | {q.recall:.2f} | {q.f1:.2f} | "
                f"{q.faithfulness_score:.2f} | {q.latency_ms:.1f} ms | {flag_icon} |"
            )
        lines.extend([
            "",
            "</details>",
            "",
        ])

        return "\n".join(lines)

    @classmethod
    def generate_comparison_report(
        cls,
        comparison: dict[str, Any],
        git_commit_sha: Optional[str] = None,
    ) -> str:
        """
        Generate a GitHub-flavored Markdown report for a side-by-side configuration experiment.

        :param comparison: Comparison dictionary from ComparisonReporter.compare().
        :param git_commit_sha: Optional git commit SHA.
        :return: Formatted Markdown string.
        """
        timestamp_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        commit_str = f"`{git_commit_sha[:8]}`" if git_commit_sha else "`N/A`"

        if comparison.get("status") == "unreachable":
            return "\n".join([
                "# Configuration Comparison Report: Aborted",
                "",
                f"- **Timestamp:** {timestamp_str}",
                f"- **Git Commit:** {commit_str}",
                "",
                "> ⚠️ **One or more configurations were unreachable.**",
                "",
                f"- Config A (`{comparison['config_a']['name']}`): {comparison['config_a'].get('error') or 'Online'}",
                f"- Config B (`{comparison['config_b']['name']}`): {comparison['config_b'].get('error') or 'Online'}",
            ])

        cfg_a = comparison["config_a_name"]
        cfg_b = comparison["config_b_name"]
        winner = comparison["winner"]

        lines: list[str] = [
            f"# RAGPatrol Configuration Comparison: `{cfg_a}` vs `{cfg_b}`",
            "",
            "## Experiment Summary",
            "",
            f"- **Timestamp:** {timestamp_str}",
            f"- **Git Commit:** {commit_str}",
            f"- **Overall Experiment Winner:** 🏆 **`{winner}`**",
            f"- **Outcome:** {comparison['summary']}",
            "",
            "## Side-by-Side Metric Comparison",
            "",
            f"| Metric | `{cfg_a}` | `{cfg_b}` | Delta | Winner |",
            "| :--- | :---: | :---: | :---: | :---: |",
        ]

        for row in comparison.get("metrics_comparison", []):
            unit = row["unit"]
            row_winner = row["winner"]
            diff = row["val_a"] - row["val_b"]

            if unit == "%":
                str_a = f"{row['val_a'] * 100:.1f}%"
                str_b = f"{row['val_b'] * 100:.1f}%"
                diff_str = f"{diff * 100:+.1f}%"
            else:
                str_a = f"{row['val_a']:.1f} ms"
                str_b = f"{row['val_b']:.1f} ms"
                diff_str = f"{diff:+.1f} ms"

            winner_display = f"🏆 **`{row_winner}`**" if row_winner != "Tie" else "🤝 Tie"
            lines.append(f"| **{row['metric']}** | {str_a} | {str_b} | {diff_str} | {winner_display} |")

        lines.extend([
            "",
            "## Per-Category Performance Comparison",
            "",
            f"| Category | Metric | `{cfg_a}` | `{cfg_b}` |",
            "| :--- | :--- | :---: | :---: |",
        ])

        for cat, data in comparison.get("category_comparison", {}).items():
            a = data["config_a"]
            b = data["config_b"]
            lines.append(f"| `{cat}` | **Precision** | {a.get('precision', 0)*100:.1f}% | {b.get('precision', 0)*100:.1f}% |")
            lines.append(f"| `{cat}` | **Recall** | {a.get('recall', 0)*100:.1f}% | {b.get('recall', 0)*100:.1f}% |")
            lines.append(f"| `{cat}` | **F1 Score** | {a.get('f1', 0)*100:.1f}% | {b.get('f1', 0)*100:.1f}% |")
            lines.append(f"| `{cat}` | **Faithfulness** | {a.get('faithfulness', 0)*100:.1f}% | {b.get('faithfulness', 0)*100:.1f}% |")

        lines.append("")
        return "\n".join(lines)
