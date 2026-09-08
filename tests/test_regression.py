"""
Unit tests for regression detection and quality gating.

Uses in-memory SQLite with StaticPool for fast, isolated, deterministic testing.
"""

from datetime import datetime, timezone, timedelta
import pytest
from sqlalchemy.pool import StaticPool

from harness.regression_check import RegressionChecker
from harness.storage.db import DatabaseManager
from harness.storage.models import EvalRun, RunMetric


@pytest.fixture
def in_memory_db() -> DatabaseManager:
    """Fixture providing an isolated in-memory SQLite database instance."""
    db = DatabaseManager(
        database_url="sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    return db


def create_mock_run(
    db: DatabaseManager,
    run_id: str,
    timestamp: datetime,
    metrics_dict: dict[str, float],
    config_name: str = "default",
    stage: str = "all",
) -> EvalRun:
    """Helper to persist a test evaluation run with metrics."""
    run = EvalRun(
        id=run_id,
        timestamp=timestamp,
        config_name=config_name,
        stage=stage,
        cache_state="warm",
        total_queries=25,
        successful_queries=25,
        failed_queries=0,
    )
    for m_name, val in metrics_dict.items():
        run.metrics.append(RunMetric(metric_name=m_name, value=val))

    return db.save_run(run)


def test_no_prior_run(in_memory_db: DatabaseManager) -> None:
    """A baseline run without preceding historical runs must pass with 'no_prior_run'."""
    now = datetime.now(timezone.utc)
    create_mock_run(
        in_memory_db,
        run_id="run_baseline",
        timestamp=now,
        metrics_dict={"retrieval_precision": 0.80, "retrieval_recall": 0.80},
    )

    checker = RegressionChecker(db_manager=in_memory_db)
    res = checker.check(config_name="default", stage="all")

    assert res["passed"] is True
    assert res["status"] == "no_prior_run"
    assert len(res["regressions"]) == 0
    assert res["current"]["retrieval_precision"] == 0.80


def test_clear_regression_precision_drop(in_memory_db: DatabaseManager) -> None:
    """A 0.10 precision drop (exceeding 0.05 limit) must fail the regression check."""
    t0 = datetime.now(timezone.utc) - timedelta(hours=1)
    t1 = datetime.now(timezone.utc)

    # Historical run
    create_mock_run(
        in_memory_db,
        run_id="run_prev",
        timestamp=t0,
        metrics_dict={"retrieval_precision": 0.85, "retrieval_f1": 0.85},
    )
    # Current run with drop of 0.10
    create_mock_run(
        in_memory_db,
        run_id="run_curr",
        timestamp=t1,
        metrics_dict={"retrieval_precision": 0.75, "retrieval_f1": 0.85},
    )

    checker = RegressionChecker(db_manager=in_memory_db)
    res = checker.check(config_name="default", stage="all")

    assert res["passed"] is False
    assert res["status"] == "regression_detected"
    assert len(res["regressions"]) == 1
    reg = res["regressions"][0]
    assert reg["metric"] == "retrieval_precision"
    assert reg["delta"] == -0.10


def test_clear_improvement(in_memory_db: DatabaseManager) -> None:
    """Metric improvements must pass with zero regression flags."""
    t0 = datetime.now(timezone.utc) - timedelta(hours=1)
    t1 = datetime.now(timezone.utc)

    create_mock_run(
        in_memory_db,
        run_id="run_prev",
        timestamp=t0,
        metrics_dict={"retrieval_precision": 0.70, "faithfulness_avg": 0.75},
    )
    create_mock_run(
        in_memory_db,
        run_id="run_curr",
        timestamp=t1,
        metrics_dict={"retrieval_precision": 0.85, "faithfulness_avg": 0.90},
    )

    checker = RegressionChecker(db_manager=in_memory_db)
    res = checker.check(config_name="default", stage="all")

    assert res["passed"] is True
    assert len(res["regressions"]) == 0


def test_within_tolerance_noise(in_memory_db: DatabaseManager) -> None:
    """A 0.03 drop within the 0.05 tolerance limit must pass."""
    t0 = datetime.now(timezone.utc) - timedelta(hours=1)
    t1 = datetime.now(timezone.utc)

    create_mock_run(
        in_memory_db,
        run_id="run_prev",
        timestamp=t0,
        metrics_dict={"retrieval_precision": 0.85},
    )
    create_mock_run(
        in_memory_db,
        run_id="run_curr",
        timestamp=t1,
        metrics_dict={"retrieval_precision": 0.82},
    )

    checker = RegressionChecker(db_manager=in_memory_db)
    res = checker.check(config_name="default", stage="all")

    assert res["passed"] is True
    assert len(res["regressions"]) == 0


def test_missing_metrics_handled_gracefully(in_memory_db: DatabaseManager) -> None:
    """Disparate metrics across runs must be handled without throwing errors."""
    t0 = datetime.now(timezone.utc) - timedelta(hours=1)
    t1 = datetime.now(timezone.utc)

    create_mock_run(
        in_memory_db,
        run_id="run_prev",
        timestamp=t0,
        metrics_dict={"retrieval_precision": 0.80, "extra_metric": 0.99},
    )
    create_mock_run(
        in_memory_db,
        run_id="run_curr",
        timestamp=t1,
        metrics_dict={"retrieval_precision": 0.80, "different_metric": 0.50},
    )

    checker = RegressionChecker(db_manager=in_memory_db)
    res = checker.check(config_name="default", stage="all")

    assert res["passed"] is True
    assert len(res["regressions"]) == 0


def test_latency_p95_relative_increase(in_memory_db: DatabaseManager) -> None:
    """Latency p95 increasing by >20% relative must trigger regression."""
    t0 = datetime.now(timezone.utc) - timedelta(hours=1)
    t1 = datetime.now(timezone.utc)

    # 100ms -> 130ms is +30% increase
    create_mock_run(
        in_memory_db,
        run_id="run_prev",
        timestamp=t0,
        metrics_dict={"latency_p95": 100.0},
    )
    create_mock_run(
        in_memory_db,
        run_id="run_curr",
        timestamp=t1,
        metrics_dict={"latency_p95": 130.0},
    )

    checker = RegressionChecker(db_manager=in_memory_db)
    res = checker.check(config_name="default", stage="all")

    assert res["passed"] is False
    assert len(res["regressions"]) == 1
    assert res["regressions"][0]["metric"] == "latency_p95"
    assert res["regressions"][0]["delta"] == 0.30
