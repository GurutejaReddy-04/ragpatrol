"""
Central orchestrator for the LLM Evaluation & Observability Harness.

Coordinates test set loading, target client execution, multi-stage scoring
(retrieval, faithfulness, latency), persistence, and report emission.
"""

import argparse
import logging
import sys
from typing import Optional

from harness.clients.rag_client import RAGClient
from harness.config import HarnessSettings, get_settings
from harness.scorers.faithfulness import FaithfulnessScorer
from harness.scorers.latency import LatencyProfiler
from harness.scorers.retrieval import RetrievalScorer
from harness.storage.db import DatabaseManager

logger = logging.getLogger(__name__)


class EvaluationRunner:
    """
    Execution runner driving the evaluation pipeline through configured stages.
    """

    def __init__(self, settings: Optional[HarnessSettings] = None) -> None:
        """
        Initialize runner with settings, client, scorers, and storage.

        :param settings: HarnessSettings instance (defaults to loaded config.yaml).
        """
        self.settings = settings or get_settings()
        self.client = RAGClient(
            base_url=self.settings.target_api.base_url,
            api_key=self.settings.citebase_api_key,
            timeout_seconds=self.settings.target_api.timeout_seconds,
            max_retries=self.settings.target_api.max_retries,
        )
        self.retrieval_scorer = RetrievalScorer()
        self.faithfulness_scorer = FaithfulnessScorer(
            embedding_weight=self.settings.metrics.faithfulness.embedding_weight,
            llm_judge_weight=self.settings.metrics.faithfulness.llm_judge_weight,
            hallucination_threshold=self.settings.metrics.faithfulness.min_score,
        )
        self.latency_profiler = LatencyProfiler()
        self.db_manager = DatabaseManager(database_url=self.settings.storage.database_url)
        logger.info("EvaluationRunner initialized against target %s", self.settings.target_api.base_url)

    def run(self, stage: str = "all", config_name: str = "default") -> bool:
        """
        Execute evaluation pass for designated stage.

        :param stage: Evaluation stage ('retrieval', 'faithfulness', 'latency', 'all').
        :param config_name: Label tag for the current target configuration.
        :return: True if evaluation ran and met thresholds, False otherwise.
        """
        logger.info("Starting evaluation run (stage=%s, config=%s)...", stage, config_name)
        # Phase 0 stub: confirms connectivity and basic wiring
        is_healthy = self.client.check_health()
        if not is_healthy:
            logger.error("Target RAG API health check failed. Aborting evaluation run.")
            return False

        logger.info("Target RAG API is reachable. Pipeline ready for Phase 1+ execution.")
        return True


def main() -> None:
    """CLI entrypoint for executing harness runs."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="LLM Evaluation & Observability Harness Runner")
    parser.add_argument(
        "--stage",
        choices=["retrieval", "faithfulness", "latency", "all"],
        default="all",
        help="Evaluation pipeline stage to execute.",
    )
    parser.add_argument(
        "--config",
        default="default",
        help="Named configuration tag for this run.",
    )
    args = parser.parse_args()

    runner = EvaluationRunner()
    success = runner.run(stage=args.stage, config_name=args.config)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
