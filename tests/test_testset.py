"""
Pytest integration for testset validation.

Ensures that CI automatically executes validate_testset.main() and blocks
on any dataset corruption, missing fields, or empty ground truths.
"""

from pathlib import Path
import pytest
from testset.validate_testset import main as validate_main, validate_testset


def test_validate_testset_execution() -> None:
    """Execute testset validation directly and assert a clean exit code 0."""
    result = validate_main()
    assert result == 0, "testset/validate_testset.py main() returned non-zero exit code"


def test_validate_testset_detailed() -> None:
    """Verify detailed validation output and ensure zero structural errors."""
    is_valid, errors = validate_testset(Path("testset/questions.yaml"))
    assert is_valid is True, f"Validation errors found in testset/questions.yaml: {errors}"
    assert len(errors) == 0
