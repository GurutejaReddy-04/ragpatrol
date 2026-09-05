"""
Validation script enforcing integrity and schema constraints on testset/questions.yaml.

Checks:
1. Unique question IDs (no duplicates).
2. Every ground_truth_chunk_ids entry is non-empty.
3. Required fields exist with valid types.
4. Categories conform to expected taxonomy (easy, ambiguous, edge, general).
5. (Non-fatal Warning) Chunk ID pattern conforms to doc_<doc>_chunk_<page>.
"""

import logging
import re
import sys
from pathlib import Path
from typing import Any, Tuple
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("validate_testset")

DEFAULT_TESTSET_PATH = Path("testset/questions.yaml")
EXPECTED_CHUNK_ID_PATTERN = re.compile(r"^doc_[a-zA-Z0-9_-]+_chunk_[a-zA-Z0-9_-]+$")
ALLOWED_CATEGORIES = {"easy", "ambiguous", "edge", "general"}
REQUIRED_FIELDS = {
    "id": str,
    "question": str,
    "ground_truth_chunk_ids": list,
    "category": str,
}


def validate_testset(path: Path = DEFAULT_TESTSET_PATH) -> Tuple[bool, list[str]]:
    """
    Validate the specified questions.yaml dataset against schema invariants.

    :param path: Path to questions.yaml.
    :return: (is_valid, list_of_error_messages)
    """
    if not path.exists():
        return False, [f"Testset file not found: {path}"]

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception as e:
        return False, [f"Failed to parse YAML file at {path}: {e}"]

    if not isinstance(data, list):
        return False, [f"Root of {path} must be a list of questions, got {type(data)}"]

    errors: list[str] = []
    seen_ids: set[str] = set()

    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            errors.append(f"Item #{idx} is not a dictionary.")
            continue

        # 1. Required field checks and types
        for field_name, expected_type in REQUIRED_FIELDS.items():
            if field_name not in item:
                errors.append(f"Item #{idx} is missing required field '{field_name}'.")
            elif not isinstance(item[field_name], expected_type):
                errors.append(
                    f"Item #{idx} field '{field_name}' must be of type {expected_type.__name__}, "
                    f"got {type(item[field_name]).__name__}."
                )

        qid = item.get("id")
        if qid:
            # 2. Duplicate ID check
            if qid in seen_ids:
                errors.append(f"Duplicate question id detected: '{qid}'.")
            seen_ids.add(qid)

            # Check question length
            q_text = str(item.get("question", "")).strip()
            if not q_text:
                errors.append(f"Question '{qid}' has an empty or whitespace-only question body.")

        # 3. Non-empty ground_truth_chunk_ids check
        chunk_ids = item.get("ground_truth_chunk_ids")
        if isinstance(chunk_ids, list):
            if len(chunk_ids) == 0:
                errors.append(
                    f"Question '{qid or idx}' has an empty 'ground_truth_chunk_ids' list. "
                    f"Every evaluation query must define at least one target or sentinel chunk."
                )
            else:
                # 4. Pattern check (warning only — non-blocking)
                for chunk_id in chunk_ids:
                    if not isinstance(chunk_id, str):
                        errors.append(f"Question '{qid}' contains non-string chunk ID: {chunk_id}")
                    elif not EXPECTED_CHUNK_ID_PATTERN.match(chunk_id):
                        logger.warning(
                            "Chunk ID '%s' in question '%s' doesn't match expected pattern "
                            "'doc_<doc_id>_chunk_<page>'. Suggestion: follow doc_[name]_chunk_[id].",
                            chunk_id,
                            qid,
                        )
        elif "ground_truth_chunk_ids" in item:
            errors.append(f"Question '{qid}' ground_truth_chunk_ids must be a list.")

        # 5. Category taxonomy check
        category = item.get("category")
        if category and str(category).lower() not in ALLOWED_CATEGORIES:
            errors.append(
                f"Question '{qid}' has invalid category '{category}'. "
                f"Allowed categories: {', '.join(sorted(ALLOWED_CATEGORIES))}."
            )

    is_valid = len(errors) == 0
    return is_valid, errors


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for running test set validation."""
    if argv is not None:
        target_path = Path(argv[0]) if argv else DEFAULT_TESTSET_PATH
    elif len(sys.argv) > 1 and Path(sys.argv[0]).name.endswith("validate_testset.py"):
        target_path = Path(sys.argv[1])
    else:
        target_path = DEFAULT_TESTSET_PATH

    logger.info("Validating evaluation testset at: %s", target_path)

    is_valid, errors = validate_testset(target_path)

    if not is_valid:
        logger.error("Testset validation FAILED with %d error(s):", len(errors))
        for err in errors:
            logger.error("  - %s", err)
        return 1

    logger.info("Testset validation PASSED. All schema invariants and constraints verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
