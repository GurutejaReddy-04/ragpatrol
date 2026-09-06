"""
Unit tests for RAGPatrol EvaluationRunner and orchestration in harness/runner.py.
"""

import os
from pathlib import Path
import httpx
import pytest
import yaml
from sqlalchemy.pool import StaticPool

from harness.config import HarnessSettings
from harness.exceptions import ConfigValidationError
from harness.runner import EvaluationRunner, apply_env_overrides, get_git_commit_sha
from harness.storage.db import DatabaseManager


@pytest.fixture
def mock_runner(tmp_path):
    """Create an EvaluationRunner wired to an in-memory database and mock client."""
    settings = HarnessSettings()
    settings.storage.database_url = "sqlite:///:memory:"

    # Custom mock transport that returns a valid stub response
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/query":
            return httpx.Response(
                200,
                json={
                    "answer_text": "Python asyncio provides cooperative multitasking via an event loop.",
                    "sources": [
                        {"doc": "asyncio_guide", "page": 1, "content": "Asyncio uses async def and await."}
                    ],
                    "response_time_ms": 42.5,
                },
            )
        return httpx.Response(404)

    mock_client = httpx.Client(
        base_url="http://mock-target:8000",
        transport=httpx.MockTransport(handler),
    )

    runner = EvaluationRunner(
        settings=settings,
        base_url="http://mock-target:8000",
        adapter="stub",
    )
    # Inject mock client and in-memory DB
    runner.client._client = mock_client
    runner.db = DatabaseManager(database_url="sqlite:///:memory:", poolclass=StaticPool)
    return runner


def test_apply_env_overrides_context_manager():
    """Verify apply_env_overrides sets and safely restores environment variables."""
    var_name = "TEST_HARNESS_VAR_OVERRIDE_123"
    os.environ[var_name] = "original"

    with apply_env_overrides({var_name: "temporary"}):
        assert os.environ[var_name] == "temporary"

    assert os.environ[var_name] == "original"
    del os.environ[var_name]

    # Test setting a previously unset variable
    with apply_env_overrides({"NEW_HARNESS_VAR_456": "created"}):
        assert os.environ["NEW_HARNESS_VAR_456"] == "created"
    assert "NEW_HARNESS_VAR_456" not in os.environ


def test_get_git_commit_sha():
    """get_git_commit_sha should return a string SHA or None without raising an uncaught exception."""
    sha = get_git_commit_sha()
    assert sha is None or (isinstance(sha, str) and len(sha) >= 7)


def test_load_testset_valid_yaml(mock_runner):
    """Load valid testset from stub_questions.yaml."""
    path = Path("testset/stub_questions.yaml")
    if path.exists():
        questions = mock_runner.load_testset(str(path))
        assert isinstance(questions, list)
        assert len(questions) > 0
        assert "id" in questions[0]
        assert "question" in questions[0]


def test_load_testset_missing_file_raises(mock_runner):
    """Loading from a non-existent path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        mock_runner.load_testset("non_existent_path_to_testset_999.yaml")


def test_load_testset_invalid_yaml_raises(mock_runner, tmp_path):
    """Verify EC-12 fix: Corrupted YAML raises ConfigValidationError instead of dumping a raw trace."""
    corrupt_file = tmp_path / "corrupt.yaml"
    with open(corrupt_file, "w", encoding="utf-8") as f:
        f.write("this: is: invalid: [yaml: broken")

    with pytest.raises(ConfigValidationError) as exc_info:
        mock_runner.load_testset(str(corrupt_file))
    assert "Failed to parse testset YAML" in str(exc_info.value)


def test_load_testset_non_list_raises(mock_runner, tmp_path):
    """A YAML file containing a dict instead of a list must raise ValueError."""
    dict_file = tmp_path / "dict_testset.yaml"
    with open(dict_file, "w", encoding="utf-8") as f:
        yaml.dump({"not_a": "list"}, f)

    with pytest.raises(ValueError) as exc_info:
        mock_runner.load_testset(str(dict_file))
    assert "Expected a list of questions" in str(exc_info.value)


def test_execute_pass_with_mock_client(mock_runner):
    """Execute a full pass and verify StageRunResult fields and metrics aggregation."""
    questions = [
        {
            "id": "q01",
            "category": "asyncio",
            "question": "How does asyncio run coroutines?",
            "ground_truth_chunk_ids": ["doc_asyncio_guide_chunk_1"],
        }
    ]

    result = mock_runner.execute_pass(
        questions=questions,
        stage="retrieval",
        cache_state="cold",
        config_name="mock_config",
        dry_run=True,
    )

    assert result.config_name == "mock_config"
    assert result.total_queries == 1
    assert result.successful_queries == 1
    assert result.failed_queries == 0
    assert result.mean_precision == 1.0
    assert result.mean_recall == 1.0
    assert result.mean_f1 == 1.0
    assert len(result.question_summaries) == 1
    assert result.question_summaries[0].status == "success"


def test_execute_pass_handles_query_failure_gracefully(mock_runner):
    """Simulate a network exception on one query; the pass must continue and record error status."""
    # Transport that returns 500 error
    error_client = httpx.Client(
        base_url="http://mock-target:8000",
        transport=httpx.MockTransport(lambda req: httpx.Response(500, text="Internal Server Error")),
    )
    mock_runner.client._client = error_client

    questions = [
        {
            "id": "q_failing",
            "category": "faulty",
            "question": "Will this crash the run?",
            "ground_truth_chunk_ids": ["doc_1"],
        }
    ]

    result = mock_runner.execute_pass(
        questions=questions,
        stage="all",
        cache_state="cold",
        config_name="error_cfg",
        dry_run=True,
    )

    assert result.total_queries == 1
    assert result.successful_queries == 0
    assert result.failed_queries == 1
    assert len(result.question_summaries) == 1
    assert result.question_summaries[0].status == "error"
    assert "HTTP 500" in (result.question_summaries[0].error_message or "")
