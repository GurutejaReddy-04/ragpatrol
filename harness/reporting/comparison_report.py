"""
Side-by-side configuration comparison and evaluation reporting (Phase 6).

Compares two benchmarked configurations across retrieval quality (precision, recall, F1),
faithfulness groundedness, and latency percentiles (p50, p95, p99), highlighting trade-offs
and identifying metric winners.
"""

import logging
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from harness.runner import StageRunResult

logger = logging.getLogger(__name__)



class ComparisonReporter:
    """
    Renders side-by-side evaluation comparison reports and determines configuration winners.
    """

    METRICS = [
        ("Precision", "mean_precision", True, "%"),
        ("Recall", "mean_recall", True, "%"),
        ("F1", "mean_f1", True, "%"),
        ("Faithfulness", "mean_faithfulness", True, "%"),
        ("Latency p50", "p50_ms", False, "ms"),
        ("Latency p95", "p95_ms", False, "ms"),
        ("Latency p99", "p99_ms", False, "ms"),
    ]

    @classmethod
    def compare(
        cls,
        config_a_name: str,
        result_a: Optional["StageRunResult"],
        config_b_name: str,
        result_b: Optional["StageRunResult"],
        error_a: Optional[str] = None,
        error_b: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Produce a structured comparison dictionary between two evaluation runs.

        :param config_a_name: Label for configuration A.
        :param result_a: StageRunResult for configuration A, or None if failed.
        :param config_b_name: Label for configuration B.
        :param result_b: StageRunResult for configuration B, or None if failed.
        :param error_a: Optional error/unreachable message for A.
        :param error_b: Optional error/unreachable message for B.
        :return: Structured comparison dictionary.
        """
        # 1. Unreachable / Execution Failure Detection
        unreachable: list[str] = []
        if error_a or result_a is None or result_a.successful_queries == 0:
            unreachable.append(config_a_name)
        if error_b or result_b is None or result_b.successful_queries == 0:
            unreachable.append(config_b_name)

        if unreachable:
            err_msg_a = error_a or ("No successful queries" if result_a and result_a.successful_queries == 0 else "Server unreachable")
            err_msg_b = error_b or ("No successful queries" if result_b and result_b.successful_queries == 0 else "Server unreachable")

            return {
                "status": "unreachable",
                "unreachable_configs": unreachable,
                "config_a": {"name": config_a_name, "reachable": config_a_name not in unreachable, "error": err_msg_a if config_a_name in unreachable else None},
                "config_b": {"name": config_b_name, "reachable": config_b_name not in unreachable, "error": err_msg_b if config_b_name in unreachable else None},
                "summary": f"Comparison aborted: configuration(s) {', '.join(unreachable)} unreachable.",
                "metrics_comparison": [],
                "category_comparison": {},
            }

        if result_a is None or result_b is None:
            raise ValueError("Comparison requires both result_a and result_b to be non-None.")

        # 2. Metric comparisons
        metrics_rows: list[dict[str, Any]] = []
        wins_a = 0
        wins_b = 0
        ties = 0

        for metric_label, attr, higher_is_better, unit in cls.METRICS:
            if attr in ("p50_ms", "p95_ms", "p99_ms"):
                lat_a = result_a.latency_profile if result_a.latency_profile else None
                lat_b = result_b.latency_profile if result_b.latency_profile else None
                val_a = getattr(lat_a, attr, 0.0) if lat_a else 0.0
                val_b = getattr(lat_b, attr, 0.0) if lat_b else 0.0
            else:
                val_a = getattr(result_a, attr, 0.0)
                val_b = getattr(result_b, attr, 0.0)

            # Determine winner
            diff = val_a - val_b
            if abs(diff) < 1e-4:
                winner = "Tie"
                ties += 1
            elif (diff > 0 and higher_is_better) or (diff < 0 and not higher_is_better):
                winner = config_a_name
                wins_a += 1
            else:
                winner = config_b_name
                wins_b += 1

            metrics_rows.append({
                "metric": metric_label,
                "val_a": val_a,
                "val_b": val_b,
                "higher_is_better": higher_is_better,
                "unit": unit,
                "winner": winner,
            })

        overall_winner = (
            config_a_name if wins_a > wins_b
            else config_b_name if wins_b > wins_a
            else "Tie"
        )
        total_eval = len(metrics_rows)
        summary_text = (
            f"Config '{config_a_name}' wins on {wins_a}/{total_eval} metrics. "
            f"Config '{config_b_name}' wins on {wins_b}/{total_eval} metrics."
        )
        if ties > 0:
            summary_text += f" ({ties} ties)"

        # 3. Category breakdown
        all_categories = sorted(set(result_a.category_metrics.keys()) | set(result_b.category_metrics.keys()))
        category_rows: dict[str, dict[str, Any]] = {}
        for cat in all_categories:
            cat_a = result_a.category_metrics.get(cat, {})
            cat_b = result_b.category_metrics.get(cat, {})
            category_rows[cat] = {
                "config_a": cat_a,
                "config_b": cat_b,
            }

        return {
            "status": "success",
            "config_a_name": config_a_name,
            "config_b_name": config_b_name,
            "winner": overall_winner,
            "wins_a": wins_a,
            "wins_b": wins_b,
            "ties": ties,
            "summary": summary_text,
            "metrics_comparison": metrics_rows,
            "category_comparison": category_rows,
        }

    @classmethod
    def print_comparison_report(cls, report: dict[str, Any]) -> None:
        """Render side-by-side comparison table to terminal."""
        print("\n" + "=" * 80)
        print(" CONFIGURATION EXPERIMENT COMPARISON REPORT")
        print("=" * 80)

        if report.get("status") == "unreachable":
            print(" [!] COMPARISON ABORTED: One or more configurations are unreachable.")
            print("-" * 80)
            for cfg in (report["config_a"], report["config_b"]):
                status = "ONLINE" if cfg["reachable"] else f"UNREACHABLE ({cfg['error']})"
                print(f"  * Config '{cfg['name']}': {status}")
            print("=" * 80)
            return

        cfg_a = report["config_a_name"]
        cfg_b = report["config_b_name"]

        # Main metrics table
        print(f" Comparing: '{cfg_a}' vs '{cfg_b}'")
        print("+" + "-" * 20 + "+" + "-" * 20 + "+" + "-" * 20 + "+" + "-" * 15 + "+")
        print(f"| {'Metric':<18} | {cfg_a:<18} | {cfg_b:<18} | {'Winner':<13} |")
        print("+" + "-" * 20 + "+" + "-" * 20 + "+" + "-" * 20 + "+" + "-" * 15 + "+")

        for row in report["metrics_comparison"]:
            unit = row["unit"]
            winner = row["winner"]

            if unit == "%":
                str_a = f"{row['val_a'] * 100:.1f}%"
                str_b = f"{row['val_b'] * 100:.1f}%"
            else:
                str_a = f"{row['val_a']:.1f} ms"
                str_b = f"{row['val_b']:.1f} ms"

            # Highlight the winner
            disp_a = f"* {str_a}" if winner == cfg_a else f"  {str_a}"
            disp_b = f"* {str_b}" if winner == cfg_b else f"  {str_b}"

            print(f"| {row['metric']:<18} | {disp_a:<18} | {disp_b:<18} | {winner:<13} |")

        print("+" + "-" * 20 + "+" + "-" * 20 + "+" + "-" * 20 + "+" + "-" * 15 + "+")
        print(f" Summary: {report['summary']}")
        print("-" * 80)

        # Per-Category Breakdown
        print(" PER-CATEGORY BREAKDOWN:")
        print("+" + "-" * 12 + "+" + "-" * 32 + "+" + "-" * 32 + "+")
        print(f"| {'Category':<10} | {cfg_a + ' (P/R/F1/Faith)':<30} | {cfg_b + ' (P/R/F1/Faith)':<30} |")
        print("+" + "-" * 12 + "+" + "-" * 32 + "+" + "-" * 32 + "+")

        for cat, data in report.get("category_comparison", {}).items():
            a = data["config_a"]
            b = data["config_b"]

            fmt_a = f"{a.get('precision', 0)*100:.0f}% / {a.get('recall', 0)*100:.0f}% / {a.get('f1', 0)*100:.0f}% / {a.get('faithfulness', 0)*100:.0f}%"
            fmt_b = f"{b.get('precision', 0)*100:.0f}% / {b.get('recall', 0)*100:.0f}% / {b.get('f1', 0)*100:.0f}% / {b.get('faithfulness', 0)*100:.0f}%"

            print(f"| {cat:<10} | {fmt_a:<30} | {fmt_b:<30} |")

        print("+" + "-" * 12 + "+" + "-" * 32 + "+" + "-" * 32 + "+")
        print("=" * 80)
