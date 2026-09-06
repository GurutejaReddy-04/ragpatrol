"""
Smoke test verifying importability of all RAGPatrol modules and stubs.

Ensures no circular dependencies, syntax errors, or missing pinned requirements
break the RAGPatrol package from day zero.
"""

import importlib
import logging
import pkgutil
from typing import Set

import pytest

import harness
import harness.clients
import harness.clients.contracts
import harness.clients.rag_client
import harness.config
import harness.exceptions
import harness.reporting
import harness.reporting.html_report
import harness.reporting.markdown_report
import harness.runner
import harness.scorers
import harness.scorers.faithfulness
import harness.scorers.latency
import harness.scorers.retrieval
import harness.storage
import harness.storage.db
import harness.storage.models
import stub_app.fake_rag_api

logger = logging.getLogger(__name__)


def test_explicit_module_imports() -> None:
    """Verify that core classes and functions can be resolved from their modules."""
    # Contracts
    from harness.clients.contracts import (
        CiteBaseQueryResponse,
        CiteBaseSourceReference,
        RAGQueryRequest,
        RAGResponse,
        RetrievedChunk,
        normalize_citebase_response,
        normalize_target_response,
    )
    assert RetrievedChunk is not None
    assert RAGResponse is not None
    assert RAGQueryRequest is not None
    assert CiteBaseSourceReference is not None
    assert CiteBaseQueryResponse is not None
    assert callable(normalize_citebase_response)
    assert callable(normalize_target_response)

    # Client
    from harness.clients.rag_client import RAGClient
    assert RAGClient is not None

    # Config
    from harness.config import HarnessSettings, get_settings
    settings = get_settings()
    assert isinstance(settings, HarnessSettings)

    # Exceptions
    from harness.exceptions import (
        ConfigValidationError,
        HarnessError,
        RAGClientError,
        RAGConnectionError,
        RAGResponseError,
        RAGTimeoutError,
    )
    assert issubclass(ConfigValidationError, HarnessError)
    assert issubclass(RAGConnectionError, RAGClientError)
    assert issubclass(RAGTimeoutError, RAGClientError)
    assert issubclass(RAGResponseError, RAGClientError)

    # Scorers
    from harness.scorers.retrieval import RetrievalResult, RetrievalScorer
    from harness.scorers.faithfulness import FaithfulnessResult, FaithfulnessScorer
    from harness.scorers.latency import LatencyResult, LatencyProfiler
    assert RetrievalScorer is not None
    assert RetrievalResult is not None
    assert FaithfulnessScorer is not None
    assert FaithfulnessResult is not None
    assert LatencyProfiler is not None
    assert LatencyResult is not None

    # Storage
    from harness.storage.models import EvalRun, QuestionResult, RunMetric
    from harness.storage.db import DatabaseManager
    assert EvalRun is not None
    assert RunMetric is not None
    assert QuestionResult is not None
    assert DatabaseManager is not None

    # Reporting
    from harness.reporting.markdown_report import MarkdownReportGenerator
    from harness.reporting.html_report import HTMLReportGenerator
    assert MarkdownReportGenerator is not None
    assert HTMLReportGenerator is not None

    # Runner
    from harness.runner import EvaluationRunner
    assert EvaluationRunner is not None

    # Stub API
    from stub_app.fake_rag_api import app as stub_app_instance
    assert stub_app_instance is not None

    logger.info("All explicit module imports and symbol assertions passed.")


def test_walk_packages_imports() -> None:
    """Recursively discover and import all modules under the harness package namespace."""
    discovered: Set[str] = set()

    for _, module_name, _ in pkgutil.walk_packages(
        path=harness.__path__,
        prefix=f"{harness.__name__}.",
    ):
        mod = importlib.import_module(module_name)
        assert mod is not None
        discovered.add(module_name)

    logger.info("Successfully discovered and imported %d submodules via pkgutil.", len(discovered))
    assert "harness.clients.contracts" in discovered
    assert "harness.clients.rag_client" in discovered
    assert "harness.scorers.retrieval" in discovered
    assert "harness.storage.db" in discovered
