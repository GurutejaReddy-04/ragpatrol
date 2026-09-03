"""
Database persistence manager for evaluation runs and trend tracking.
"""

import logging
from typing import Optional
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from harness.storage.models import Base, EvalRun

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Manages SQLAlchemy engine, session lifecycle, and run persistence.
    """

    def __init__(self, database_url: str = "sqlite:///eval_runs.db") -> None:
        """
        Initialize database engine and session maker.

        :param database_url: SQLAlchemy compatible connection URL.
        """
        self.database_url = database_url
        self.engine = create_engine(self.database_url, echo=False)
        self.session_factory = sessionmaker(bind=self.engine, expire_on_commit=False)
        logger.debug("DatabaseManager initialized with URL: %s", self.database_url)

    def create_tables(self) -> None:
        """Create all required relational tables if they do not exist."""
        logger.info("Ensuring database schema exists.")
        Base.metadata.create_all(self.engine)

    def get_session(self) -> Session:
        """Produce a new scoped database session."""
        return self.session_factory()

    def save_run(self, run: EvalRun) -> None:
        """
        Persist a complete evaluation run with associated metrics and question records.

        :param run: Populated EvalRun ORM instance.
        """
        with self.get_session() as session:
            session.add(run)
            session.commit()
            logger.info("Persisted EvalRun %s to database.", run.id)

    def get_latest_run(self, config_name: str = "default") -> Optional[EvalRun]:
        """
        Fetch the most recent evaluation run for a specified configuration name.

        :param config_name: Targeted configuration tag.
        :return: Most recent EvalRun or None if no runs exist.
        """
        with self.get_session() as session:
            stmt = (
                select(EvalRun)
                .where(EvalRun.config_name == config_name)
                .order_by(EvalRun.timestamp.desc())
                .limit(1)
            )
            return session.scalars(stmt).first()

    def get_run_history(self, config_name: str = "default", limit: int = 10) -> list[EvalRun]:
        """
        Retrieve historical evaluation runs for trend tracking and regression analysis.

        :param config_name: Targeted configuration tag.
        :param limit: Maximum historical records to return.
        :return: Chronological list of historical runs.
        """
        with self.get_session() as session:
            stmt = (
                select(EvalRun)
                .where(EvalRun.config_name == config_name)
                .order_by(EvalRun.timestamp.desc())
                .limit(limit)
            )
            return list(session.scalars(stmt).all())
