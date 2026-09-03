"""
SQLAlchemy 2.0 relational models for persisting evaluation runs and metrics over time.
"""

from datetime import datetime, timezone
from typing import Optional
from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Boolean
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for relational evaluation models."""
    pass


class EvalRun(Base):
    """
    Represents a single evaluation run snapshot against a named RAG configuration.
    """
    __tablename__ = "eval_runs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
    config_name: Mapped[str] = mapped_column(String(128), default="default")
    git_commit_sha: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    cache_state: Mapped[str] = mapped_column(String(32), default="cold")  # cold | warm
    passed: Mapped[bool] = mapped_column(Boolean, default=True)

    metrics: Mapped[list["RunMetric"]] = relationship(
        "RunMetric", back_populates="run", cascade="all, delete-orphan"
    )
    question_results: Mapped[list["QuestionResult"]] = relationship(
        "QuestionResult", back_populates="run", cascade="all, delete-orphan"
    )


class RunMetric(Base):
    """
    Aggregate metric value recorded for an evaluation run.
    """
    __tablename__ = "run_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(64), ForeignKey("eval_runs.id"), nullable=False)
    metric_name: Mapped[str] = mapped_column(String(128), nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    category: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    run: Mapped["EvalRun"] = relationship("EvalRun", back_populates="metrics")


class QuestionResult(Base):
    """
    Detailed per-question evaluation breakdown for diagnostic traceability.
    """
    __tablename__ = "question_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(64), ForeignKey("eval_runs.id"), nullable=False)
    question_id: Mapped[str] = mapped_column(String(64), nullable=False)
    precision: Mapped[float] = mapped_column(Float, default=0.0)
    recall: Mapped[float] = mapped_column(Float, default=0.0)
    f1: Mapped[float] = mapped_column(Float, default=0.0)
    faithfulness_score: Mapped[float] = mapped_column(Float, default=0.0)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    hallucination_flag: Mapped[bool] = mapped_column(Boolean, default=False)

    run: Mapped["EvalRun"] = relationship("EvalRun", back_populates="question_results")
