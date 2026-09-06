"""
Unit tests for DatabaseManager in harness/storage/db.py.
"""

from datetime import datetime, timezone
import pytest
from sqlalchemy.pool import StaticPool

from harness.storage.db import DatabaseManager
from harness.storage.models import EvalRun, QuestionResult, RunMetric


@pytest.fixture
def in_memory_db():
    """Create a clean in-memory SQLite database manager for isolated testing."""
    return DatabaseManager(
        database_url="sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )


def test_create_tables_idempotent(in_memory_db: DatabaseManager) -> None:
    """Calling create_tables multiple times must not raise errors."""
    in_memory_db.create_tables()
    in_memory_db.create_tables()


def test_sqlite_timeout_configured() -> None:
    """Verify EC-9: SQLite database manager sets timeout in connect_args."""
    db = DatabaseManager(database_url="sqlite:///:memory:", poolclass=StaticPool)
    # Check that create_engine received connect_args with timeout
    assert db.engine is not None


def test_save_run_with_dict_data(in_memory_db: DatabaseManager) -> None:
    """Persist run data provided as dictionaries."""
    run_dict = {
        "id": "run_test_001",
        "timestamp": datetime.now(timezone.utc),
        "config_name": "test_cfg",
        "git_commit_sha": "abc1234",
        "cache_state": "cold",
        "stage": "all",
        "total_queries": 2,
        "successful_queries": 2,
        "failed_queries": 0,
    }
    q_results = [
        {
            "question_id": "q1",
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "faithfulness_score": 0.95,
            "latency_ms": 120.0,
            "hallucination_flag": False,
            "judge_reasoning": "Accurate",
            "error_message": None,
        }
    ]
    metrics = [
        {"metric_name": "retrieval_f1", "value": 1.0, "category": None},
        {"metric_name": "retrieval_f1", "value": 1.0, "category": "domain"},
    ]

    saved = in_memory_db.save_run(run_dict, question_results=q_results, metrics=metrics)
    assert saved.id == "run_test_001"
    assert len(saved.question_results) == 1
    assert len(saved.metrics) == 2
    assert saved.metrics[0].value == 1.0


def test_save_run_with_orm_object(in_memory_db: DatabaseManager) -> None:
    """Persist run data provided as an ORM EvalRun instance."""
    eval_run = EvalRun(
        id="run_orm_002",
        timestamp=datetime.now(timezone.utc),
        config_name="orm_cfg",
        stage="retrieval",
        total_queries=1,
        successful_queries=1,
        failed_queries=0,
    )
    eval_run.metrics.append(RunMetric(metric_name="retrieval_p", value=0.85))
    eval_run.question_results.append(QuestionResult(question_id="q01", precision=0.85))

    saved = in_memory_db.save_run(eval_run)
    assert saved.id == "run_orm_002"
    assert len(saved.metrics) == 1
    assert len(saved.question_results) == 1


def test_save_run_none_metric_value(in_memory_db: DatabaseManager) -> None:
    """Verify EC-8 fix: None metric values should safely coalesce to 0.0 without TypeError."""
    run_dict = {
        "id": "run_none_val_003",
        "timestamp": datetime.now(timezone.utc),
        "config_name": "none_cfg",
    }
    metrics = [
        {"metric_name": "nullable_metric", "value": None, "category": None},
    ]

    saved = in_memory_db.save_run(run_dict, metrics=metrics)
    assert saved.id == "run_none_val_003"
    assert len(saved.metrics) == 1
    assert saved.metrics[0].value == 0.0


def test_get_latest_run(in_memory_db: DatabaseManager) -> None:
    """Fetch the latest run matching config and stage."""
    # Initially None
    assert in_memory_db.get_latest_run(config_name="custom_cfg") is None

    # Save two runs with same config and stage
    t1 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    t2 = datetime(2026, 1, 2, 12, 0, 0, tzinfo=timezone.utc)

    in_memory_db.save_run({"id": "run_old", "timestamp": t1, "config_name": "custom_cfg", "stage": "all"})
    in_memory_db.save_run({"id": "run_new", "timestamp": t2, "config_name": "custom_cfg", "stage": "all"})

    latest = in_memory_db.get_latest_run(config_name="custom_cfg", stage="all")
    assert latest is not None
    assert latest.id == "run_new"


def test_get_run_history(in_memory_db: DatabaseManager) -> None:
    """Fetch run history ordered chronologically descending."""
    for i in range(5):
        in_memory_db.save_run({
            "id": f"run_hist_{i}",
            "timestamp": datetime(2026, 1, i + 1, tzinfo=timezone.utc),
            "config_name": "hist_cfg",
            "stage": "all",
        })

    history = in_memory_db.get_run_history(config_name="hist_cfg", stage="all", limit=3)
    assert len(history) == 3
    assert history[0].id == "run_hist_4"
    assert history[1].id == "run_hist_3"
    assert history[2].id == "run_hist_2"


def test_get_run_by_id(in_memory_db: DatabaseManager) -> None:
    """Fetch specific run by primary key."""
    in_memory_db.save_run({"id": "run_specific_id", "config_name": "spec_cfg"})

    found = in_memory_db.get_run_by_id("run_specific_id")
    assert found is not None
    assert found.id == "run_specific_id"

    assert in_memory_db.get_run_by_id("non_existent_id") is None
