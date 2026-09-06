"""
Tests for configuration management in RAGPatrol (LLM Evaluation & Observability Harness).
"""

import os
from unittest import mock

import pytest
import yaml

from harness.config import (
    HarnessSettings,
    TargetAPIConfig,
    TestsetConfig,
    StorageConfig,
    MetricsConfig,
    JudgeConfig,
    get_settings,
)
from harness.exceptions import ConfigValidationError


def test_default_settings_have_sensible_values() -> None:
    """Check defaults for target_api, testset, storage, metrics, judge."""
    settings = HarnessSettings()
    
    assert isinstance(settings.target_api, TargetAPIConfig)
    assert settings.target_api.base_url == "http://127.0.0.1:8000"
    
    assert isinstance(settings.testset, TestsetConfig)
    assert settings.testset.path == "testset/questions.yaml"
    
    assert isinstance(settings.storage, StorageConfig)
    assert settings.storage.database_url == "sqlite:///eval_runs.db"
    
    assert isinstance(settings.metrics, MetricsConfig)
    assert settings.metrics.retrieval.min_precision == 0.70
    assert settings.metrics.faithfulness.min_score == 0.75
    assert settings.metrics.latency.max_p95_ms == 2500.0
    
    assert isinstance(settings.judge, JudgeConfig)
    assert settings.judge.provider == "gemini"
    assert settings.judge.model == "gemini-2.5-flash"


def test_load_from_yaml_valid_config(tmp_path, monkeypatch) -> None:
    """Load from real config.yaml and verify values."""
    monkeypatch.delenv("TARGET_BASE_URL", raising=False)
    monkeypatch.delenv("CITEBASE_BASE_URL", raising=False)
    config_file = tmp_path / "valid_config.yaml"
    yaml_content = {
        "target_api": {
            "base_url": "http://api.example.com",
            "timeout_seconds": 45.0,
        },
        "judge": {
            "provider": "openai",
            "model": "gpt-4",
        }
    }
    with open(config_file, "w") as f:
        yaml.dump(yaml_content, f)
        
    settings = HarnessSettings.load_from_yaml(str(config_file))
    
    assert settings.target_api.base_url == "http://api.example.com"
    assert settings.target_api.timeout_seconds == 45.0
    assert settings.judge.provider == "openai"
    assert settings.judge.model == "gpt-4"
    # Check default still applies for missing sections
    assert settings.testset.path == "testset/questions.yaml"


def test_load_from_yaml_missing_file_falls_back() -> None:
    """Non-existent file falls back to defaults."""
    settings = HarnessSettings.load_from_yaml("non_existent_config_file_12345.yaml")
    
    # Should fall back to default settings without crashing
    assert settings.target_api.base_url == "http://127.0.0.1:8000"
    assert settings.judge.provider == "gemini"


def test_load_from_yaml_invalid_yaml_raises_config_error(tmp_path) -> None:
    """Corrupted YAML raises ConfigValidationError."""
    config_file = tmp_path / "invalid_config.yaml"
    with open(config_file, "w") as f:
        f.write("unclosed_string: 'this yaml is broken\n")
        
    with pytest.raises(ConfigValidationError) as exc_info:
        HarnessSettings.load_from_yaml(str(config_file))
        
    assert "Invalid YAML config file" in str(exc_info.value)


def test_validate_env_missing_judge_key_raises() -> None:
    """validate_env raises when judge key missing and required."""
    settings = HarnessSettings(judge_api_key=None)
    
    with pytest.raises(ConfigValidationError) as exc_info:
        settings.validate_env(require_judge_key=True)
        
    assert "JUDGE_API_KEY" in str(exc_info.value)


def test_validate_env_passes_when_keys_present() -> None:
    """validate_env passes when keys are set."""
    settings = HarnessSettings(
        citebase_api_key="fake-target-key",
        judge_api_key="fake-judge-key",
    )
    
    # Should not raise any exceptions
    settings.validate_env(require_judge_key=True, require_target_key=True)


@mock.patch.dict(os.environ, {
    "TARGET_BASE_URL": "http://override.example.com",
    "JUDGE_API_KEY": "secret-judge-key",
    "CITEBASE_API_KEY": "secret-citebase-key",
}, clear=True)
def test_env_variable_overrides(tmp_path) -> None:
    """Environment vars override config.yaml values."""
    config_file = tmp_path / "env_config.yaml"
    yaml_content = {
        "target_api": {
            "base_url": "http://yaml.example.com",
        }
    }
    with open(config_file, "w") as f:
        yaml.dump(yaml_content, f)
        
    settings = HarnessSettings.load_from_yaml(str(config_file))
    
    assert settings.target_api.base_url == "http://override.example.com"
    assert settings.judge_api_key == "secret-judge-key"
    assert settings.citebase_api_key == "secret-citebase-key"


def test_get_settings_singleton() -> None:
    """Verify singleton pattern returns same object."""
    import harness.config
    
    # Reset the singleton for the test
    harness.config._settings = None
    
    settings_1 = get_settings("non_existent.yaml")
    settings_2 = get_settings("non_existent.yaml")
    
    assert settings_1 is settings_2
    
    # Clean up
    harness.config._settings = None
