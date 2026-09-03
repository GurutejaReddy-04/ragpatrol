"""
Configuration management using pydantic-settings.

Loads baseline parameters from config.yaml while strictly injecting credentials
(such as API keys) via environment variables to guarantee zero secret leakage.
"""

import logging
import os
from pathlib import Path
from typing import Any, Optional
import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from harness.exceptions import ConfigValidationError

logger = logging.getLogger(__name__)


class TargetAPIConfig(BaseModel):
    """Network connection parameters for the target RAG API under test."""
    base_url: str = Field(default="http://127.0.0.1:8000", description="Base URL of the RAG service.")
    timeout_seconds: float = Field(default=30.0, gt=0.0, description="Network request timeout in seconds.")
    max_retries: int = Field(default=3, ge=0, description="Max retry attempts on transient network errors.")
    retry_backoff_factor: float = Field(default=1.5, ge=1.0, description="Exponential backoff factor for retries.")


class TestsetConfig(BaseModel):
    """Location and metadata for the evaluation question dataset."""
    path: str = Field(default="testset/questions.yaml", description="Path to versioned test questions YAML.")


class StorageConfig(BaseModel):
    """Persistence configuration for benchmark runs and metric history."""
    database_url: str = Field(
        default="sqlite:///eval_runs.db",
        description="SQLAlchemy database URL (SQLite, PostgreSQL).",
    )


class RetrievalMetricConfig(BaseModel):
    """Regression thresholds for retrieval quality."""
    min_precision: float = Field(default=0.70, ge=0.0, le=1.0)
    min_recall: float = Field(default=0.70, ge=0.0, le=1.0)
    min_f1: float = Field(default=0.70, ge=0.0, le=1.0)


class FaithfulnessMetricConfig(BaseModel):
    """Regression thresholds and weighting for answer groundedness."""
    min_score: float = Field(default=0.75, ge=0.0, le=1.0)
    embedding_weight: float = Field(default=0.40, ge=0.0, le=1.0)
    llm_judge_weight: float = Field(default=0.60, ge=0.0, le=1.0)


class LatencyMetricConfig(BaseModel):
    """Performance SLA ceilings in milliseconds."""
    max_p95_ms: float = Field(default=2500.0, gt=0.0)
    max_p99_ms: float = Field(default=5000.0, gt=0.0)


class MetricsConfig(BaseModel):
    """Collection of evaluation metric targets and thresholds."""
    retrieval: RetrievalMetricConfig = Field(default_factory=RetrievalMetricConfig)
    faithfulness: FaithfulnessMetricConfig = Field(default_factory=FaithfulnessMetricConfig)
    latency: LatencyMetricConfig = Field(default_factory=LatencyMetricConfig)


class JudgeConfig(BaseModel):
    """LLM-as-a-judge provider and generation settings."""
    provider: str = Field(default="gemini", description="LLM provider: gemini, openai, etc.")
    model: str = Field(default="gemini-2.5-flash", description="Model identifier for judge scoring.")
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)


class HarnessSettings(BaseSettings):
    """
    Central evaluation harness settings.

    Reads baseline parameters from `config.yaml` and overlays sensitive secrets
    directly from system environment variables.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Core sections
    target_api: TargetAPIConfig = Field(default_factory=TargetAPIConfig)
    testset: TestsetConfig = Field(default_factory=TestsetConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    judge: JudgeConfig = Field(default_factory=JudgeConfig)

    # Secrets strictly mapped via environment variables
    citebase_api_key: Optional[str] = Field(
        default=None,
        description="Tenant API key injected via CITEBASE_API_KEY or BOOTSTRAP_API_KEY.",
    )
    judge_api_key: Optional[str] = Field(
        default=None,
        description="LLM Judge API key injected via JUDGE_API_KEY or GEMINI_API_KEY.",
    )

    @classmethod
    def load_from_yaml(cls, config_path: str = "config.yaml") -> "HarnessSettings":
        """
        Load configuration from YAML file, layering in environment variable overrides.

        Prevents secrets from being hardcoded into configuration files.
        """
        data: dict[str, Any] = {}
        path = Path(config_path)

        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    loaded = yaml.safe_load(f)
                    if isinstance(loaded, dict):
                        data = loaded
                logger.info("Loaded baseline configuration from %s", config_path)
            except Exception as e:
                logger.error("Failed to parse config file at %s: %s", config_path, e)
                raise ConfigValidationError(f"Invalid YAML config file: {e}") from e
        else:
            logger.warning("Config file %s not found. Falling back to default settings.", config_path)

        # Environment variable overrides for target base URL
        env_base_url = os.getenv("TARGET_BASE_URL") or os.getenv("CITEBASE_BASE_URL")
        if env_base_url:
            if "target_api" not in data:
                data["target_api"] = {}
            data["target_api"]["base_url"] = env_base_url

        # Inject sensitive secrets directly from OS environment
        citebase_key = os.getenv("CITEBASE_API_KEY") or os.getenv("BOOTSTRAP_API_KEY")
        judge_key = os.getenv("JUDGE_API_KEY") or os.getenv("GEMINI_API_KEY")

        settings = cls(
            **data,
            citebase_api_key=citebase_key,
            judge_api_key=judge_key,
        )
        return settings

    def validate_env(self, require_judge_key: bool = False, require_target_key: bool = False) -> None:
        """
        Verify that necessary environment variables are set before starting an eval run.

        Raises ConfigValidationError early so we fail loudly instead of mid-benchmark.
        """
        missing: list[str] = []

        if require_target_key and not self.citebase_api_key:
            missing.append("CITEBASE_API_KEY (or BOOTSTRAP_API_KEY)")

        if require_judge_key and not self.judge_api_key:
            missing.append("JUDGE_API_KEY (or GEMINI_API_KEY)")

        if missing:
            msg = f"Missing required environment credentials: {', '.join(missing)}"
            logger.error(msg)
            raise ConfigValidationError(msg)

        logger.info("Environment credentials validation passed.")


# Global default configuration instance
_settings: Optional[HarnessSettings] = None


def get_settings(config_path: str = "config.yaml") -> HarnessSettings:
    """Return singleton configuration instance initialized from config.yaml and environment."""
    global _settings
    if _settings is None:
        _settings = HarnessSettings.load_from_yaml(config_path)
    return _settings
