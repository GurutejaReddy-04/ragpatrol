"""
RAGPatrol — Central orchestrator for the LLM Evaluation & Observability Harness.

Coordinates test set loading, target client execution, multi-stage scoring
(retrieval, faithfulness, latency percentiles), cold/warm cache comparison,
and formatted terminal reporting.
"""

import argparse
import logging
import os
import shutil
import sys
import time
import subprocess
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
import yaml
from pydantic import BaseModel, Field

from harness.clients.rag_client import RAGClient
from harness.config import HarnessSettings, get_settings
from harness.reporting.comparison_report import ComparisonReporter
from harness.reporting.html_report import HTMLReportGenerator
from harness.reporting.markdown_report import MarkdownReportGenerator
from harness.scorers.faithfulness import FaithfulnessResult, FaithfulnessScorer
from harness.scorers.latency import LatencyProfile, LatencyProfiler
from harness.scorers.retrieval import RetrievalResult, RetrievalScorer
from harness.storage.db import DatabaseManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ragpatrol")


@contextmanager
def apply_env_overrides(overrides: dict[str, str]):
    """Temporarily apply environment variable overrides during an evaluation pass."""
    original: dict[str, Optional[str]] = {}
    for k, v in overrides.items():
        original[k] = os.environ.get(k)
        os.environ[k] = str(v)
    try:
        yield
    finally:
        for k, orig_val in original.items():
            if orig_val is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = orig_val



def get_git_commit_sha() -> Optional[str]:
    """Capture current HEAD git commit SHA if executing within a git repository."""
    git_bin = shutil.which("git")
    if not git_bin:
        return None
    try:
        res = subprocess.run(  # nosec B603
            [git_bin, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return res.stdout.strip()
    except Exception as e:
        logger.warning("Could not resolve git commit SHA: %s", e)
        return None



class QuestionEvalSummary(BaseModel):
    """Evaluation metrics captured for a single question."""
    id: str
    category: str
    question: str
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    ground_truth_chunk_ids: list[str] = Field(default_factory=list)
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    faithfulness_score: float = 0.0
    embedding_similarity: float = 0.0
    llm_judge_score: float = 0.0
    is_hallucination: bool = False
    judge_reasoning: str = ""
    unsupported_claims: list[str] = Field(default_factory=list)
    latency_ms: float = 0.0
    faithfulness_latency_ms: float = 0.0
    status: str = "success"  # success | error
    error_message: Optional[str] = None


class StageRunResult(BaseModel):
    """Aggregate result from an evaluation stage pass."""
    config_name: str
    cache_state: str  # cold | warm
    stage: str
    total_queries: int
    successful_queries: int
    failed_queries: int
    mean_precision: float
    mean_recall: float
    mean_f1: float
    mean_faithfulness: float
    hallucination_count: int
    hallucination_rate: float
    latency_profile: LatencyProfile
    category_metrics: dict[str, dict[str, float]]
    question_summaries: list[QuestionEvalSummary]


class EvaluationRunner:
    """
    Drives evaluation passes across target RAG endpoints.
    """

    def __init__(
        self,
        settings: Optional[HarnessSettings] = None,
        base_url: Optional[str] = None,
        adapter: str = "auto",
    ) -> None:
        self.settings = settings or get_settings()
        target_base_url = base_url or self.settings.target_api.base_url
        self.client = RAGClient(
            base_url=target_base_url,
            api_key=self.settings.citebase_api_key,
            timeout_seconds=self.settings.target_api.timeout_seconds,
            max_retries=self.settings.target_api.max_retries,
            adapter=adapter,
        )
        self.retrieval_scorer = RetrievalScorer()
        self.faithfulness_scorer = FaithfulnessScorer(
            embedding_weight=self.settings.metrics.faithfulness.embedding_weight,
            llm_judge_weight=self.settings.metrics.faithfulness.llm_judge_weight,
            model_name=self.settings.judge.model,
        )
        self.latency_profiler = LatencyProfiler()
        self.db = DatabaseManager(database_url=self.settings.storage.database_url)
        logger.info(
            "RAGPatrol EvaluationRunner initialized against target: %s (adapter=%s)",
            target_base_url,
            self.client.adapter,
        )

    def load_testset(self, testset_path: Optional[str] = None) -> list[dict[str, Any]]:
        """Load and parse test questions from YAML."""
        path = Path(testset_path or self.settings.testset.path)
        if not path.exists():
            raise FileNotFoundError(f"Testset file not found at {path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            from harness.exceptions import ConfigValidationError
            raise ConfigValidationError(
                f"Failed to parse testset YAML at {path}: {e}"
            ) from e

        if not isinstance(data, list):
            raise ValueError(f"Expected a list of questions in {path}")

        logger.info("Loaded %d questions from %s", len(data), path)
        return data

    def execute_pass(
        self,
        questions: list[dict[str, Any]],
        stage: str = "all",
        cache_state: str = "cold",
        config_name: str = "default",
        dry_run: bool = False,
    ) -> StageRunResult:
        """
        Execute an evaluation pass across the question dataset.
        """
        logger.info(
            "=== RAGPatrol: Starting Evaluation Pass (stage=%s, cache=%s, queries=%d, dry_run=%s) ===",
            stage,
            cache_state,
            len(questions),
            dry_run,
        )

        summaries: list[QuestionEvalSummary] = []
        latencies: list[float] = []

        for idx, q_item in enumerate(questions, start=1):
            qid = q_item.get("id", f"q{idx:02d}")
            category = q_item.get("category", "general")
            q_text = q_item.get("question", "")
            gt_chunk_ids = q_item.get("ground_truth_chunk_ids", [])
            meta = q_item.get("metadata", {})
            collections = meta.get("collection_names", [])

            logger.info("[%d/%d] Querying '%s': %s", idx, len(questions), qid, q_text[:50])

            # Prepare client call parameters
            primary_coll = collections[0] if collections else None
            extra_payload = {"collection_names": collections} if collections else None

            t_query_start = time.perf_counter()
            try:
                # 1. Query target RAG system
                rag_resp = self.client.query(
                    question=q_text,
                    collection_name=primary_coll,
                    extra_payload=extra_payload,
                )
                total_latency_ms = rag_resp.latency_ms
                retrieved_chunk_ids = [c.chunk_id for c in rag_resp.retrieved_chunks]
                retrieved_texts = [c.text for c in rag_resp.retrieved_chunks]

                # 2. Retrieval Scoring (Phase 2)
                ret_res: RetrievalResult = self.retrieval_scorer.score(
                    retrieved_chunk_ids=retrieved_chunk_ids,
                    ground_truth_chunk_ids=gt_chunk_ids,
                )

                # 3. Faithfulness Scoring (Phase 3)
                if stage in ("faithfulness", "latency", "all"):
                    faith_res: FaithfulnessResult = self.faithfulness_scorer.score(
                        question=q_text,
                        answer=rag_resp.answer,
                        contexts=retrieved_texts,
                        dry_run=dry_run,
                    )
                else:
                    faith_res = FaithfulnessResult()

                latencies.append(total_latency_ms)

                summary = QuestionEvalSummary(
                    id=qid,
                    category=category,
                    question=q_text,
                    retrieved_chunk_ids=retrieved_chunk_ids,
                    ground_truth_chunk_ids=gt_chunk_ids,
                    precision=ret_res.precision,
                    recall=ret_res.recall,
                    f1=ret_res.f1,
                    faithfulness_score=faith_res.faithfulness_score,
                    embedding_similarity=faith_res.embedding_similarity,
                    llm_judge_score=faith_res.llm_judge_score,
                    is_hallucination=faith_res.is_hallucination,
                    judge_reasoning=faith_res.judge_reasoning,
                    unsupported_claims=faith_res.unsupported_claims,
                    latency_ms=total_latency_ms,
                    faithfulness_latency_ms=faith_res.embedding_latency_ms + faith_res.judge_latency_ms,
                    status="success",
                )
                summaries.append(summary)

            except Exception as e:
                query_lat_ms = (time.perf_counter() - t_query_start) * 1000.0
                logger.error("Query failed for '%s': %s", qid, e)
                # Never crash the whole eval run on single question network blips
                summaries.append(
                    QuestionEvalSummary(
                        id=qid,
                        category=category,
                        question=q_text,
                        ground_truth_chunk_ids=gt_chunk_ids,
                        latency_ms=query_lat_ms,
                        status="error",
                        error_message=str(e),
                    )
                )

        # Compute Aggregates
        successful = [s for s in summaries if s.status == "success"]
        n_success = len(successful)

        mean_precision = sum(s.precision for s in successful) / n_success if n_success else 0.0
        mean_recall = sum(s.recall for s in successful) / n_success if n_success else 0.0
        mean_f1 = sum(s.f1 for s in successful) / n_success if n_success else 0.0
        mean_faithfulness = sum(s.faithfulness_score for s in successful) / n_success if n_success else 0.0
        hallucinations = [s for s in successful if s.is_hallucination]
        hallucination_rate = len(hallucinations) / n_success if n_success else 0.0

        lat_profile = self.latency_profiler.compute_percentiles(latencies)

        # Category Breakdown
        categories: set[str] = {s.category for s in summaries}
        cat_metrics: dict[str, dict[str, float]] = {}
        for cat in sorted(categories):
            cat_items = [s for s in successful if s.category == cat]
            cn = len(cat_items)
            cat_metrics[cat] = {
                "count": cn,
                "precision": round(sum(s.precision for s in cat_items) / cn, 4) if cn else 0.0,
                "recall": round(sum(s.recall for s in cat_items) / cn, 4) if cn else 0.0,
                "f1": round(sum(s.f1 for s in cat_items) / cn, 4) if cn else 0.0,
                "faithfulness": round(sum(s.faithfulness_score for s in cat_items) / cn, 4) if cn else 0.0,
                "hallucination_rate": round(sum(1 for s in cat_items if s.is_hallucination) / cn, 4) if cn else 0.0,
            }

        run_res = StageRunResult(
            config_name=config_name,
            cache_state=cache_state,
            stage=stage,
            total_queries=len(summaries),
            successful_queries=n_success,
            failed_queries=len(summaries) - n_success,
            mean_precision=round(mean_precision, 4),
            mean_recall=round(mean_recall, 4),
            mean_f1=round(mean_f1, 4),
            mean_faithfulness=round(mean_faithfulness, 4),
            hallucination_count=len(hallucinations),
            hallucination_rate=round(hallucination_rate, 4),
            latency_profile=lat_profile,
            category_metrics=cat_metrics,
            question_summaries=summaries,
        )

        # Persist run and associated metrics to database
        run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        git_sha = get_git_commit_sha()

        run_data = {
            "id": run_id,
            "timestamp": datetime.now(timezone.utc),
            "config_name": config_name,
            "git_commit_sha": git_sha,
            "cache_state": cache_state,
            "stage": stage,
            "total_queries": run_res.total_queries,
            "successful_queries": run_res.successful_queries,
            "failed_queries": run_res.failed_queries,
        }

        metrics_data = [
            {"metric_name": "retrieval_precision", "value": run_res.mean_precision, "category": None},
            {"metric_name": "retrieval_recall", "value": run_res.mean_recall, "category": None},
            {"metric_name": "retrieval_f1", "value": run_res.mean_f1, "category": None},
            {"metric_name": "faithfulness_avg", "value": run_res.mean_faithfulness, "category": None},
            {"metric_name": "hallucination_rate", "value": run_res.hallucination_rate, "category": None},
            {"metric_name": "latency_p50", "value": run_res.latency_profile.p50_ms, "category": None},
            {"metric_name": "latency_p95", "value": run_res.latency_profile.p95_ms, "category": None},
            {"metric_name": "latency_p99", "value": run_res.latency_profile.p99_ms, "category": None},
            {"metric_name": "latency_mean", "value": run_res.latency_profile.mean_ms, "category": None},
        ]

        for cat, vals in run_res.category_metrics.items():
            metrics_data.append(
                {"metric_name": "retrieval_precision", "value": vals["precision"], "category": cat}
            )
            metrics_data.append(
                {"metric_name": "retrieval_recall", "value": vals["recall"], "category": cat}
            )
            metrics_data.append(
                {"metric_name": "retrieval_f1", "value": vals["f1"], "category": cat}
            )
            metrics_data.append(
                {"metric_name": "faithfulness_avg", "value": vals["faithfulness"], "category": cat}
            )
            metrics_data.append(
                {"metric_name": "hallucination_rate", "value": vals["hallucination_rate"], "category": cat}
            )

        question_data = [
            {
                "question_id": s.id,
                "precision": s.precision,
                "recall": s.recall,
                "f1": s.f1,
                "faithfulness_score": s.faithfulness_score,
                "latency_ms": s.latency_ms,
                "hallucination_flag": s.is_hallucination,
                "judge_reasoning": s.judge_reasoning,
                "error_message": s.error_message,
            }
            for s in run_res.question_summaries
        ]

        try:
            self.db.save_run(run_data, question_data, metrics_data)
            logger.info("Persisted run '%s' to database.", run_id)
        except Exception as e:
            logger.error("Failed to persist evaluation run '%s' to database: %s", run_id, e)

        return run_res

    def print_stage_report(self, run: StageRunResult) -> None:
        """Render readable ASCII evaluation report to stdout."""
        print("\n" + "=" * 80)
        print(f" EVALUATION SUMMARY: Stage='{run.stage}' | Cache='{run.cache_state}' | Config='{run.config_name}'")
        print("=" * 80)
        print(
            f" Total Queries: {run.total_queries} | Successful: {run.successful_queries} | "
            f"Failed: {run.failed_queries}"
        )
        print(f" Mean Precision:     {run.mean_precision * 100:.1f}%")
        print(f" Mean Recall:        {run.mean_recall * 100:.1f}%")
        print(f" Mean F1:            {run.mean_f1 * 100:.1f}%")
        if run.stage in ("faithfulness", "latency", "all"):
            print(f" Mean Faithfulness:  {run.mean_faithfulness * 100:.1f}%")
            print(
                f" Hallucination Rate: {run.hallucination_rate * 100:.1f}% "
                f"({run.hallucination_count}/{run.successful_queries})"
            )

        p = run.latency_profile
        print(
            f" Latency Profile:    p50={p.p50_ms:.1f}ms | p95={p.p95_ms:.1f}ms | "
            f"p99={p.p99_ms:.1f}ms | mean={p.mean_ms:.1f}ms (std={p.std_ms:.1f}ms)"
        )
        print("-" * 80)

        # Category Breakdown Table
        print(" CATEGORY BREAKDOWN:")
        print(" Category     | Count | Precision | Recall | F1    | Faithfulness | Hallucination %")
        print("-" * 80)
        for cat, vals in run.category_metrics.items():
            print(
                f" {cat:<12} | {vals['count']:<5} | {vals['precision']*100:>8.1f}% | "
                f"{vals['recall']*100:>5.1f}% | {vals['f1']*100:>4.1f}% | "
                f"{vals['faithfulness']*100:>11.1f}% | {vals['hallucination_rate']*100:>14.1f}%"
            )
        print("-" * 80)

        # Flagged Hallucinations
        flagged = [s for s in run.question_summaries if s.is_hallucination]
        if flagged:
            print(f" FLAGGED HALLUCINATIONS ({len(flagged)}):")
            for f in flagged[:3]:  # Print up to 3 samples
                print(f"  * [{f.id}] Q: {f.question[:60]}")
                print(f"    Judge Score: {f.llm_judge_score}/5 | Emb Sim: {f.embedding_similarity:.2f}")
                if f.judge_reasoning:
                    print(f"    Reasoning: {f.judge_reasoning[:120]}...")
                if f.unsupported_claims:
                    print(f"    Unsupported Claims: {f.unsupported_claims}")
            if len(flagged) > 3:
                print(f"    ... and {len(flagged) - 3} more flagged entries.")
            print("-" * 80)

    def print_latency_comparison(self, cold: StageRunResult, warm: StageRunResult) -> None:
        """Render side-by-side cold vs warm latency comparison table."""
        comp = LatencyProfiler.compare_profiles(cold.latency_profile, warm.latency_profile)
        c = cold.latency_profile
        w = warm.latency_profile

        sep = "+" + "-" * 15 + "+" + "-" * 12 + "+" + "-" * 12 + "+" + "-" * 12 + "+" + "-" * 12 + "+" + "-" * 14 + "+"
        print("\n" + sep)
        print(
            f"| {'Cache State':<13} | {'p50 (ms)':<10} | {'p95 (ms)':<10} | "
            f"{'p99 (ms)':<10} | {'Mean (ms)':<10} | {'Improvement':<12} |"
        )
        print(sep)
        print(
            f"| {'Cold':<13} | {c.p50_ms:>10.1f} | {c.p95_ms:>10.1f} | "
            f"{c.p99_ms:>10.1f} | {c.mean_ms:>10.1f} | {'1.0x':>12} |"
        )
        print(
            f"| {'Warm':<13} | {w.p50_ms:>10.1f} | {w.p95_ms:>10.1f} | "
            f"{w.p99_ms:>10.1f} | {w.mean_ms:>10.1f} | {comp['p50_speedup']:>11.1f}x |"
        )
        print(sep)



    def run_comparison(
        self,
        config_a_name: str,
        config_b_name: str,
        stage: str = "all",
        dry_run: bool = False,
        cache_mode: str = "warm",
        testset_path: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Execute full benchmark against two named configurations and output side-by-side comparison report.
        """
        questions = self.load_testset(testset_path)

        cfg_a = self.settings.configs.get(config_a_name)
        cfg_b = self.settings.configs.get(config_b_name)

        base_url_a = cfg_a.base_url if cfg_a else self.settings.target_api.base_url
        base_url_b = cfg_b.base_url if cfg_b else self.settings.target_api.base_url

        overrides_a = cfg_a.env_overrides if cfg_a else {}
        overrides_b = cfg_b.env_overrides if cfg_b else {}

        res_a: Optional[StageRunResult] = None
        error_a: Optional[str] = None
        res_b: Optional[StageRunResult] = None
        error_b: Optional[str] = None

        # Pre-flight health checks to prevent zero-score confusion on offline endpoints
        client_a = RAGClient(
            base_url=base_url_a,
            api_key=self.settings.citebase_api_key,
            timeout_seconds=self.settings.target_api.timeout_seconds,
            max_retries=1,
        )
        try:
            client_a.check_health()
        except Exception as e:
            error_a = f"Cannot connect to target service at {base_url_a}: {e}"
            logger.error("Pre-flight check failed for config '%s': %s", config_a_name, error_a)

        client_b = RAGClient(
            base_url=base_url_b,
            api_key=self.settings.citebase_api_key,
            timeout_seconds=self.settings.target_api.timeout_seconds,
            max_retries=1,
        )
        try:
            client_b.check_health()
        except Exception as e:
            error_b = f"Cannot connect to target service at {base_url_b}: {e}"
            logger.error("Pre-flight check failed for config '%s': %s", config_b_name, error_b)

        # Run Config A if reachable
        if not error_a:
            logger.info(
                "=== Running Configuration A: '%s' (url=%s, overrides=%s) ===",
                config_a_name, base_url_a, overrides_a,
            )
            self.client = client_a
            with apply_env_overrides(overrides_a):
                try:
                    res_a = self.execute_pass(
                        questions=questions,
                        stage=stage,
                        cache_state=cache_mode,
                        config_name=config_a_name,
                        dry_run=dry_run,
                    )
                except Exception as e:
                    error_a = f"Execution failed for '{config_a_name}': {e}"
                    logger.error(error_a)

        # Run Config B if reachable
        if not error_b:
            logger.info(
                "=== Running Configuration B: '%s' (url=%s, overrides=%s) ===",
                config_b_name, base_url_b, overrides_b,
            )
            self.client = client_b
            with apply_env_overrides(overrides_b):
                try:
                    res_b = self.execute_pass(
                        questions=questions,
                        stage=stage,
                        cache_state=cache_mode,
                        config_name=config_b_name,
                        dry_run=dry_run,
                    )
                except Exception as e:
                    error_b = f"Execution failed for '{config_b_name}': {e}"
                    logger.error(error_b)

        report = ComparisonReporter.compare(
            config_a_name=config_a_name,
            result_a=res_a,
            config_b_name=config_b_name,
            result_b=res_b,
            error_a=error_a,
            error_b=error_b,
        )
        ComparisonReporter.print_comparison_report(report)
        return report

    def save_run_report(
        self,
        run_result: StageRunResult,
        format_type: str = "both",
        cold_result: Optional[StageRunResult] = None,
    ) -> list[Path]:
        """Save Markdown and/or HTML reports to reports/ directory."""
        reports_dir = Path("reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        saved_paths: list[Path] = []

        if format_type in ("markdown", "both"):
            md_path = reports_dir / f"run_{run_result.config_name}_{timestamp}.md"
            md_content = MarkdownReportGenerator.generate_run_report(
                run=run_result,
                git_commit_sha=get_git_commit_sha(),
                cold_run=cold_result,
            )
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(md_content)
            logger.info("Saved Markdown report to %s", md_path)
            saved_paths.append(md_path)

        if format_type in ("html", "both"):
            html_path = reports_dir / f"run_{run_result.config_name}_{timestamp}.html"
            html_content = HTMLReportGenerator.generate_run_report(
                run=run_result,
                git_commit_sha=get_git_commit_sha(),
                cold_run=cold_result,
            )
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            logger.info("Saved HTML report to %s", html_path)
            saved_paths.append(html_path)

        return saved_paths

    def save_comparison_report(
        self,
        comparison: dict[str, Any],
        format_type: str = "both",
    ) -> list[Path]:
        """Save comparison Markdown and/or HTML reports to reports/ directory."""
        reports_dir = Path("reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        cfg_a = comparison.get("config_a_name", "configA")
        cfg_b = comparison.get("config_b_name", "configB")
        saved_paths: list[Path] = []

        if format_type in ("markdown", "both"):
            md_path = reports_dir / f"comparison_{cfg_a}_vs_{cfg_b}_{timestamp}.md"
            md_content = MarkdownReportGenerator.generate_comparison_report(
                comparison=comparison,
                git_commit_sha=get_git_commit_sha(),
            )
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(md_content)
            logger.info("Saved comparison Markdown report to %s", md_path)
            saved_paths.append(md_path)

        if format_type in ("html", "both"):
            html_path = reports_dir / f"comparison_{cfg_a}_vs_{cfg_b}_{timestamp}.html"
            html_content = HTMLReportGenerator.generate_comparison_report(
                comparison=comparison,
                git_commit_sha=get_git_commit_sha(),
            )
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            logger.info("Saved comparison HTML report to %s", html_path)
            saved_paths.append(html_path)

        return saved_paths


def main() -> None:
    """CLI entrypoint: parse arguments and execute the RAGPatrol evaluation pipeline."""
    parser = argparse.ArgumentParser(
        description="RAGPatrol — LLM Evaluation & Observability Harness",
    )
    parser.add_argument(
        "--stage",
        choices=["retrieval", "faithfulness", "latency", "all"],
        default="all",
        help="Evaluation pipeline stage to execute.",
    )
    parser.add_argument(
        "--cache-mode",
        choices=["cold", "warm", "both"],
        default="both",
        help="Cache evaluation state (cold, warm, or side-by-side both).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip LLM judge calls to conserve API quota and test locally with embeddings.",
    )
    parser.add_argument(
        "--config",
        default="default",
        help="Configuration label for this evaluation run.",
    )
    parser.add_argument(
        "--testset",
        default="testset/questions.yaml",
        help="Path to questions YAML dataset.",
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("CONFIG_A", "CONFIG_B"),
        help="Compare two named configurations side-by-side.",
    )
    parser.add_argument(
        "--report",
        choices=["markdown", "html", "both", "none"],
        default="both",
        help="Export report format (markdown, html, both, or none). Default: both.",
    )
    parser.add_argument(
        "--adapter",
        choices=["auto", "citebase", "stub"],
        default="auto",
        help="Client adapter for target RAG API (auto, citebase, or stub). Default: auto.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Override target API base URL (e.g., http://localhost:8001).",
    )
    args = parser.parse_args()

    # Automatically route to stub testset if stub adapter is active and default testset was untouched
    if args.adapter == "stub" and args.testset == "testset/questions.yaml":
        stub_testset = Path("testset/stub_questions.yaml")
        if stub_testset.exists():
            args.testset = str(stub_testset)
            logger.info("Automatically selected stub testset: %s", args.testset)

    runner = EvaluationRunner(
        base_url=args.base_url,
        adapter=args.adapter,
    )

    # EH-2: Validate required environment credentials early
    try:
        runner.settings.validate_env(require_judge_key=(not args.dry_run))
    except Exception as e:
        logger.error("Environment validation failed: %s", e)
        print(f"\n[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    if args.compare:
        config_a, config_b = args.compare
        report = runner.run_comparison(
            config_a_name=config_a,
            config_b_name=config_b,
            stage=args.stage,
            dry_run=args.dry_run,
            cache_mode=args.cache_mode if args.cache_mode != "both" else "warm",
            testset_path=args.testset,
        )
        if args.report != "none":
            runner.save_comparison_report(report, format_type=args.report)
        sys.exit(0 if report.get("status") == "success" else 1)

    questions = runner.load_testset(args.testset)

    cold_result: Optional[StageRunResult] = None
    warm_result: Optional[StageRunResult] = None

    if args.cache_mode in ("cold", "both"):
        # Attempt cache flush before cold pass
        runner.client.flush_cache()
        cold_result = runner.execute_pass(
            questions=questions,
            stage=args.stage,
            cache_state="cold",
            config_name=args.config,
            dry_run=args.dry_run,
        )
        runner.print_stage_report(cold_result)

    if args.cache_mode in ("warm", "both"):
        # Warm pass executes immediately after cold pass or uses populated cache
        warm_result = runner.execute_pass(
            questions=questions,
            stage=args.stage,
            cache_state="warm",
            config_name=args.config,
            dry_run=args.dry_run,
        )
        runner.print_stage_report(warm_result)

    if args.cache_mode == "both" and cold_result and warm_result:
        runner.print_latency_comparison(cold_result, warm_result)

    # Save artifact reports
    target_result = warm_result or cold_result
    if target_result and args.report != "none":
        runner.save_run_report(
            run_result=target_result,
            format_type=args.report,
            cold_result=cold_result if cold_result != target_result else None,
        )

    sys.exit(0)




if __name__ == "__main__":
    main()
