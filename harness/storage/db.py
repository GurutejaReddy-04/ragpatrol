"""
Database persistence manager for evaluation runs and trend tracking (Phase 5).

Supports both SQLite (default) and PostgreSQL (via DATABASE_URL env var).
Provides atomic multi-entity run persistence and query helpers for regression analysis.
"""

import logging
import os
from pathlib import Path
from typing import Any, Optional, Union
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker, selectinload

from harness.storage.models import Base, EvalRun, QuestionResult, RunMetric

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Manages database connection lifecycle, schema initialization, and run queries.
    """

    def __init__(self, database_url: Optional[str] = None, **engine_kwargs: Any) -> None:
        """
        Initialize database engine and session maker.

        :param database_url: Optional connection URL. Defaults to DATABASE_URL env var or sqlite:///eval_runs.db.
        :param engine_kwargs: Additional arguments passed to create_engine (e.g. poolclass, connect_args).
        """
        raw_url = database_url or os.getenv("DATABASE_URL") or "sqlite:///eval_runs.db"
        self.database_url: str = str(raw_url)

        # Auto-create local directories for file-based SQLite databases
        if self.database_url.startswith("sqlite:///") and not self.database_url.startswith("sqlite:///:memory:"):
            db_file_path = self.database_url.replace("sqlite:///", "")
            parent_dir = Path(db_file_path).parent
            if parent_dir and not parent_dir.exists():
                parent_dir.mkdir(parents=True, exist_ok=True)

        # Apply SQLite-specific timeout to handle locked databases gracefully
        if self.database_url.startswith("sqlite:") and "connect_args" not in engine_kwargs:
            engine_kwargs.setdefault("connect_args", {"timeout": 30})

        self.engine = create_engine(self.database_url, **engine_kwargs)
        self.session_factory = sessionmaker(bind=self.engine, expire_on_commit=False)
        self.create_tables()
        logger.debug("DatabaseManager initialized with URL: %s", self.database_url)

    def create_tables(self) -> None:
        """Create all required relational tables if they do not exist."""
        Base.metadata.create_all(self.engine)

    def get_session(self) -> Session:
        """Produce a new scoped database session."""
        return self.session_factory()

    def save_run(
        self,
        run_data: Union[EvalRun, dict[str, Any]],
        question_results: Optional[list[dict[str, Any]]] = None,
        metrics: Optional[list[dict[str, Any]]] = None,
    ) -> EvalRun:
        """
        Persist a complete evaluation run with associated metrics and question records atomically.

        :param run_data: Populated EvalRun ORM instance or dictionary of run fields.
        :param question_results: Optional list of question result dictionaries if run_data is a dict.
        :param metrics: Optional list of metric dictionaries if run_data is a dict.
        :return: Persisted EvalRun object.
        """
        with self.get_session() as session:
            with session.begin():
                if isinstance(run_data, EvalRun):
                    eval_run = run_data
                else:
                    eval_run = EvalRun(
                        id=run_data["id"],
                        timestamp=run_data.get("timestamp"),
                        config_name=run_data.get("config_name", "default"),
                        git_commit_sha=run_data.get("git_commit_sha"),
                        cache_state=run_data.get("cache_state", "cold"),
                        stage=run_data.get("stage", "all"),
                        total_queries=run_data.get("total_queries", 0),
                        successful_queries=run_data.get("successful_queries", 0),
                        failed_queries=run_data.get("failed_queries", 0),
                    )

                    if question_results:
                        for qr in question_results:
                            eval_run.question_results.append(
                                QuestionResult(
                                    question_id=qr["question_id"],
                                    precision=qr.get("precision", 0.0),
                                    recall=qr.get("recall", 0.0),
                                    f1=qr.get("f1", 0.0),
                                    faithfulness_score=qr.get("faithfulness_score", 0.0),
                                    latency_ms=qr.get("latency_ms", 0.0),
                                    hallucination_flag=qr.get("hallucination_flag", False),
                                    judge_reasoning=qr.get("judge_reasoning"),
                                    error_message=qr.get("error_message"),
                                )
                            )

                    if metrics:
                        for m in metrics:
                            eval_run.metrics.append(
                                RunMetric(
                                    metric_name=m["metric_name"],
                                    value=float(m["value"] if m["value"] is not None else 0.0),
                                    category=m.get("category"),
                                )
                            )

                session.add(eval_run)

            session.refresh(eval_run)
            logger.info(
                "Persisted EvalRun %s with %d metrics and %d question records.",
                eval_run.id, len(eval_run.metrics), len(eval_run.question_results),
            )
            return eval_run

    def get_latest_run(self, config_name: str = "default", stage: str = "all") -> Optional[EvalRun]:
        """
        Fetch the most recent evaluation run matching a specified config and stage.

        :param config_name: Targeted configuration tag.
        :param stage: Targeted evaluation pipeline stage.
        :return: Most recent EvalRun or None if no matching runs exist.
        """
        with self.get_session() as session:
            stmt = (
                select(EvalRun)
                .options(selectinload(EvalRun.metrics), selectinload(EvalRun.question_results))
                .where(
                    EvalRun.config_name == config_name,
                    EvalRun.stage == stage,
                )
                .order_by(EvalRun.timestamp.desc())
                .limit(1)
            )
            return session.scalars(stmt).first()

    def get_run_history(
        self,
        config_name: str = "default",
        stage: str = "all",
        limit: int = 20,
    ) -> list[EvalRun]:
        """
        Retrieve historical evaluation runs for trend tracking and regression analysis.

        :param config_name: Targeted configuration tag.
        :param stage: Targeted evaluation pipeline stage.
        :param limit: Maximum historical records to return.
        :return: Chronological list of historical runs (newest first).
        """
        with self.get_session() as session:
            stmt = (
                select(EvalRun)
                .options(selectinload(EvalRun.metrics), selectinload(EvalRun.question_results))
                .where(
                    EvalRun.config_name == config_name,
                    EvalRun.stage == stage,
                )
                .order_by(EvalRun.timestamp.desc())
                .limit(limit)
            )
            return list(session.scalars(stmt).all())

    def get_run_by_id(self, run_id: str) -> Optional[EvalRun]:
        """
        Retrieve a specific run by ID for detailed analysis or report generation.

        :param run_id: Primary key of targeted run.
        :return: Matching EvalRun or None.
        """
        with self.get_session() as session:
            stmt = (
                select(EvalRun)
                .options(selectinload(EvalRun.metrics), selectinload(EvalRun.question_results))
                .where(EvalRun.id == run_id)
            )
            return session.scalars(stmt).first()
