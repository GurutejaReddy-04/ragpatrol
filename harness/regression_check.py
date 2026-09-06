"""
RAGPatrol — Regression detection and quality gating module.

Compares the latest evaluation run against historical baselines to flag
statistically meaningful degradations in retrieval, faithfulness, or latency percentiles.
Serves as the primary CI/CD automated gate.
"""

import argparse
import logging
import sys
from typing import Any, Optional

from harness.config import get_settings
from harness.storage.db import DatabaseManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("regression_check")

DEFAULT_THRESHOLDS = {
    "retrieval_precision_drop": 0.05,
    "retrieval_recall_drop": 0.05,
    "retrieval_f1_drop": 0.05,
    "faithfulness_avg_drop": 0.10,
    "latency_p95_increase": 0.20,  # 20% relative increase
}


class RegressionChecker:
    """
    Evaluates deltas between the most recent run and its preceding historical run.
    """

    def __init__(
        self,
        db_manager: Optional[DatabaseManager] = None,
        thresholds: Optional[dict[str, float]] = None,
    ) -> None:
        """
        Initialize the regression checker with database access and gating thresholds.
        """
        self.db = db_manager or DatabaseManager()
        self.thresholds = dict(DEFAULT_THRESHOLDS)
        if thresholds:
            self.thresholds.update(thresholds)

    def check(self, config_name: str = "default", stage: str = "all") -> dict[str, Any]:
        """
        Compare current run against the preceding historical run for the given config and stage.

        :param config_name: Targeted configuration tag.
        :param stage: Evaluation pipeline stage.
        :return: Structured regression result dictionary.
        """
        history = self.db.get_run_history(config_name=config_name, stage=stage, limit=2)

        if not history:
            logger.info("No runs found in database for config='%s', stage='%s'.", config_name, stage)
            return {
                "passed": True,
                "status": "no_prior_run",
                "regressions": [],
                "current": {},
                "previous": {},
            }

        current_run = history[0]
        current_metrics = {m.metric_name: round(m.value, 4) for m in current_run.metrics if m.category is None}

        if len(history) < 2:
            logger.info("Only 1 run found for config='%s', stage='%s'. Baseline established.", config_name, stage)
            return {
                "passed": True,
                "status": "no_prior_run",
                "regressions": [],
                "current": current_metrics,
                "previous": {},
            }

        previous_run = history[1]
        previous_metrics = {m.metric_name: round(m.value, 4) for m in previous_run.metrics if m.category is None}

        regressions: list[dict[str, Any]] = []

        # 1. Retrieval Precision Drop
        if "retrieval_precision" in current_metrics and "retrieval_precision" in previous_metrics:
            drop = previous_metrics["retrieval_precision"] - current_metrics["retrieval_precision"]
            max_drop = self.thresholds.get("retrieval_precision_drop", 0.05)
            if drop > max_drop:
                regressions.append({
                    "metric": "retrieval_precision",
                    "type": "drop",
                    "current": current_metrics["retrieval_precision"],
                    "previous": previous_metrics["retrieval_precision"],
                    "delta": -round(drop, 4),
                    "threshold": max_drop,
                })

        # 2. Retrieval Recall Drop
        if "retrieval_recall" in current_metrics and "retrieval_recall" in previous_metrics:
            drop = previous_metrics["retrieval_recall"] - current_metrics["retrieval_recall"]
            max_drop = self.thresholds.get("retrieval_recall_drop", 0.05)
            if drop > max_drop:
                regressions.append({
                    "metric": "retrieval_recall",
                    "type": "drop",
                    "current": current_metrics["retrieval_recall"],
                    "previous": previous_metrics["retrieval_recall"],
                    "delta": -round(drop, 4),
                    "threshold": max_drop,
                })

        # 3. Retrieval F1 Drop
        if "retrieval_f1" in current_metrics and "retrieval_f1" in previous_metrics:
            drop = previous_metrics["retrieval_f1"] - current_metrics["retrieval_f1"]
            max_drop = self.thresholds.get("retrieval_f1_drop", 0.05)
            if drop > max_drop:
                regressions.append({
                    "metric": "retrieval_f1",
                    "type": "drop",
                    "current": current_metrics["retrieval_f1"],
                    "previous": previous_metrics["retrieval_f1"],
                    "delta": -round(drop, 4),
                    "threshold": max_drop,
                })

        # 4. Faithfulness Average Drop
        if "faithfulness_avg" in current_metrics and "faithfulness_avg" in previous_metrics:
            drop = previous_metrics["faithfulness_avg"] - current_metrics["faithfulness_avg"]
            max_drop = self.thresholds.get("faithfulness_avg_drop", 0.10)
            if drop > max_drop:
                regressions.append({
                    "metric": "faithfulness_avg",
                    "type": "drop",
                    "current": current_metrics["faithfulness_avg"],
                    "previous": previous_metrics["faithfulness_avg"],
                    "delta": -round(drop, 4),
                    "threshold": max_drop,
                })

        # 5. Latency p95 Relative Increase
        if "latency_p95" in current_metrics and "latency_p95" in previous_metrics:
            curr_lat = current_metrics["latency_p95"]
            prev_lat = previous_metrics["latency_p95"]
            if prev_lat > 0.0:
                rel_increase = (curr_lat - prev_lat) / prev_lat
                max_increase = self.thresholds.get("latency_p95_increase", 0.20)
                if rel_increase > max_increase:
                    regressions.append({
                        "metric": "latency_p95",
                        "type": "increase",
                        "current": curr_lat,
                        "previous": prev_lat,
                        "delta": round(rel_increase, 4),
                        "threshold": max_increase,
                    })

        passed = len(regressions) == 0
        status = "passed" if passed else "regression_detected"

        return {
            "passed": passed,
            "status": status,
            "regressions": regressions,
            "current": current_metrics,
            "previous": previous_metrics,
            "current_run_id": current_run.id,
            "previous_run_id": previous_run.id,
        }

    def print_report(self, result: dict[str, Any], config_name: str, stage: str) -> None:
        """Output human-readable terminal report."""
        print("\n" + "=" * 80)
        print(f" REGRESSION CHECK REPORT: Config='{config_name}' | Stage='{stage}'")
        print("=" * 80)

        if result.get("status") == "no_prior_run":
            print(" Status: PASS (Baseline Run - No preceding historical run to compare)")
            print("-" * 80)
            print(" Current Metrics Recorded:")
            for k, v in result.get("current", {}).items():
                print(f"  * {k:<25}: {v}")
            print("=" * 80)
            return

        status_label = "PASS" if result["passed"] else "FAIL (REGRESSION DETECTED)"
        print(f" Status: {status_label}")
        print(f" Current Run ID:  {result.get('current_run_id')}")
        print(f" Previous Run ID: {result.get('previous_run_id')}")
        print("-" * 80)

        # Comparison Table
        print("+-------------------------+------------+------------+------------+----------------+")
        print("| Metric                  | Previous   | Current    | Delta      | Threshold      |")
        print("+-------------------------+------------+------------+------------+----------------+")

        curr = result.get("current", {})
        prev = result.get("previous", {})
        all_metrics = sorted(set(curr.keys()) | set(prev.keys()))

        reg_map = {r["metric"]: r for r in result.get("regressions", [])}

        for m in all_metrics:
            c_val = curr.get(m, "N/A")
            p_val = prev.get(m, "N/A")

            delta_str = "N/A"
            thresh_str = "None"
            if isinstance(c_val, (int, float)) and isinstance(p_val, (int, float)):
                diff = c_val - p_val
                delta_str = f"{diff:+.4f}"

            if m in reg_map:
                thresh_str = f"LIMIT {reg_map[m]['threshold']}"
                delta_str = f"{delta_str} [FAIL]"

            print(f"| {m:<23} | {str(p_val):<10} | {str(c_val):<10} | {delta_str:<10} | {thresh_str:<14} |")

        print("+-------------------------+------------+------------+------------+----------------+")

        if result["regressions"]:
            print("\n DETECTED REGRESSIONS:")
            for r in result["regressions"]:
                print(f"  * {r['metric']}: {r['type']} of {abs(r['delta']):.4f} exceeds threshold {r['threshold']}")
            print("-" * 80)


def main() -> None:
    """CLI entrypoint: execute RAGPatrol regression gate check against latest evaluation run."""
    parser = argparse.ArgumentParser(
        description="RAGPatrol — Automated Regression Checker & Quality Gate",
    )
    parser.add_argument("--config", default="default", help="Configuration label.")
    parser.add_argument("--stage", default="all", help="Evaluation pipeline stage.")
    args = parser.parse_args()

    checker = RegressionChecker()
    result = checker.check(config_name=args.config, stage=args.stage)
    checker.print_report(result, config_name=args.config, stage=args.stage)

    if not result["passed"]:
        logger.error("Regression check failed! Quality gate blocked.")
        sys.exit(1)

    logger.info("Regression check passed successfully.")
    sys.exit(0)


if __name__ == "__main__":
    main()
