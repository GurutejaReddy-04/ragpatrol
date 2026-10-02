"""
Longitudinal Quality Regression Walkthrough Example.

Demonstrates RAGPatrol's automated regression gating pipeline:
1. Seeds an isolated historical database with a healthy baseline evaluation run.
2. Simulates an intentional quality regression caused by a problematic prompt/model change.
3. Invokes the RegressionChecker gate to compare matching warm runs.
4. Outputs the diagnostic regression failure report and delta table.
5. Verifies internally that regression detection triggered gate failure (asserting exit code 1 logic)
   and exits with code 0 to certify the demonstration in automated CI environments.
"""

from datetime import datetime, timezone, timedelta
from pathlib import Path
import sys

# Ensure repository root is on sys.path for direct script execution
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sqlalchemy.pool import StaticPool

from harness.regression_check import RegressionChecker
from harness.storage.db import DatabaseManager
from harness.storage.models import EvalRun, RunMetric


def main() -> int:
    print("=" * 70)
    print("RAGPatrol Longitudinal Regression Gating Walkthrough")
    print("=" * 70)

    # 1. Initialize isolated in-memory SQLite database
    print("\n[Step 1] Initializing isolated evaluation storage...")
    db = DatabaseManager(
        database_url="sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    t_baseline = datetime.now(timezone.utc) - timedelta(hours=2)
    t_regressed = datetime.now(timezone.utc)

    # 2. Seed healthy baseline run (Commit SHA: a1b2c3d4)
    print("[Step 2] Recording baseline evaluation run (healthy metrics)...")
    baseline_run = EvalRun(
        id="run_baseline_a1b2c3d4",
        timestamp=t_baseline,
        config_name="default",
        git_commit_sha="a1b2c3d4",
        cache_state="warm",
        stage="all",
        total_queries=25,
        successful_queries=25,
        failed_queries=0,
        evaluation_schema_version="2.0",
        benchmark_id="citebase_25",
        embedding_model="all-MiniLM-L6-v2",
        judge_model="gemini-2.5-flash",
    )
    baseline_metrics = {
        "retrieval_precision": 0.88,
        "retrieval_recall": 0.88,
        "retrieval_f1": 0.88,
        "faithfulness_avg": 0.90,
        "latency_p95": 210.0,
    }
    for name, val in baseline_metrics.items():
        baseline_run.metrics.append(RunMetric(metric_name=name, value=val))
    db.save_run(baseline_run)
    print("  [OK] Baseline run saved: Precision=0.88, Recall=0.88, F1=0.88, Faithfulness=0.90, p95=210ms")

    # 3. Simulate regressed run from a PR / code change (Commit SHA: e5f6g7h8)
    print("\n[Step 3] Recording current evaluation run with intentional regressions...")
    regressed_run = EvalRun(
        id="run_current_e5f6g7h8",
        timestamp=t_regressed,
        config_name="default",
        git_commit_sha="e5f6g7h8",
        cache_state="warm",
        stage="all",
        total_queries=25,
        successful_queries=25,
        failed_queries=0,
        evaluation_schema_version="2.0",
        benchmark_id="citebase_25",
        embedding_model="all-MiniLM-L6-v2",
        judge_model="gemini-2.5-flash",
    )
    regressed_metrics = {
        "retrieval_precision": 0.70,   # -0.18 drop (exceeds 0.05 limit)
        "retrieval_recall": 0.88,
        "retrieval_f1": 0.72,          # -0.16 drop (exceeds 0.05 limit)
        "faithfulness_avg": 0.78,      # -0.12 drop (exceeds 0.05 limit)
        "latency_p95": 320.0,          # +52.4% increase (exceeds 20% limit)
    }
    for name, val in regressed_metrics.items():
        regressed_run.metrics.append(RunMetric(metric_name=name, value=val))
    db.save_run(regressed_run)
    print("  [ALERT] Regressed run saved: Precision=0.70 (-0.18), F1=0.72 (-0.16), Faithfulness=0.78 (-0.12), p95=320ms (+52%)")

    # 4. Execute regression check gating
    print("\n[Step 4] Executing RegressionChecker gate comparison...")
    checker = RegressionChecker(db_manager=db)
    result = checker.check(config_name="default", stage="all", cache_state="warm")

    print("\n" + "-" * 70)
    print("GATE DIAGNOSTIC REPORT")
    print("-" * 70)
    checker.print_report(result)
    print("-" * 70)

    # 5. Assert gate rejection invariants
    print("\n[Step 5] Verifying quality gate enforcement logic...")
    assert result["passed"] is False, "Expected regression check to FAIL, but it passed!"
    assert result["status"] == "regression_detected"
    assert len(result["regressions"]) >= 4, f"Expected at least 4 regressions, got {len(result['regressions'])}"

    regressed_metric_names = {r["metric"] for r in result["regressions"]}
    assert "retrieval_precision" in regressed_metric_names
    assert "retrieval_f1" in regressed_metric_names
    assert "faithfulness_avg" in regressed_metric_names
    assert "latency_p95" in regressed_metric_names

    print("  [OK] Regression gate correctly detected all metric degradations.")
    print("  [OK] Gate evaluation would return EXIT CODE 1 in production CI pipelines.")
    print("  [OK] Walkthrough verified successfully (exiting 0 for automated test harness).")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
