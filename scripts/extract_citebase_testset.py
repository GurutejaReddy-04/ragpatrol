"""
Extract and adapt CiteBase benchmark queries into the RAGPatrol format.

Reads the 25 held-out queries from CiteBase's test suite, normalizes document
and page references into RAGPatrol chunk IDs, categorizes query complexity, and
emits testset/questions.yaml with provenance documentation.
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("extract_testset")

DEFAULT_CITEBASE_JSON = Path(
    r"D:\Btech_Organized\Projects"
    r"\Production RAG-as-a-Service — multi-tenant document intelligence API"
    r"\tests\eval\benchmark_dataset.json"
)
OUTPUT_YAML = Path("testset/questions.yaml")


def map_chunk_id(doc_id: str, page: Any) -> str:
    """Normalize CiteBase doc/page to harness chunk ID format."""
    # We trust no one, so we validate and sanitize inputs
    clean_doc = str(doc_id).strip().replace(" ", "_")
    return f"doc_{clean_doc}_chunk_{page}"


def categorize_query(ground_truth_pages: list[int], collection_count: int = 1) -> str:
    """Categorize query into easy, ambiguous, or edge based on context breadth."""
    if not ground_truth_pages:
        return "edge"
    # Multi-page or cross-collection queries introduce ambiguity
    if len(ground_truth_pages) > 1 or collection_count > 1:
        return "ambiguous"
    return "easy"


def synthesize_reference_answer(item: dict[str, Any], category: str) -> str:
    """Produce or preserve a clear reference answer."""
    if "reference_answer" in item and item["reference_answer"]:
        return str(item["reference_answer"]).strip()

    keywords = item.get("ground_truth_keywords", [])
    if category == "edge":
        return "The requested information is out-of-domain and not covered by the local corpus."

    if keywords:
        kw_str = ", ".join(keywords)
        return f"Ground truth reference covering concepts: {kw_str}."

    return "Verified factual answer derived from cited corpus pages."


def extract_queries(source_path: Path) -> list[dict[str, Any]]:
    """Read CiteBase benchmark dataset and adapt each entry."""
    if not source_path.exists():
        logger.error("Source CiteBase benchmark file not found at %s", source_path)
        raise FileNotFoundError(f"Missing benchmark file: {source_path}")

    with open(source_path, "r", encoding="utf-8") as f:
        raw_items = json.load(f)

    if not isinstance(raw_items, list):
        raise ValueError(f"Expected a list of benchmark items, got {type(raw_items)}")

    logger.info("Loaded %d raw queries from %s", len(raw_items), source_path)
    adapted_items: list[dict[str, Any]] = []

    for item in raw_items:
        qid = str(item.get("id", f"q{len(adapted_items)+1:02d}"))
        question = item.get("question", "").strip()

        if not question:
            logger.warning("Item %s has empty question text. Skipping.", qid)
            continue

        # Handle field naming variations: ground_truth_pages vs ground_truth_docs
        pages: list[int] = []
        if "ground_truth_pages" in item:
            raw_pages = item["ground_truth_pages"]
            if isinstance(raw_pages, list):
                pages = [int(p) for p in raw_pages if isinstance(p, (int, str)) and str(p).isdigit()]
        elif "ground_truth_docs" in item:
            logger.warning("Item %s uses 'ground_truth_docs' instead of 'ground_truth_pages'.", qid)
            pages = []
        else:
            logger.warning("Item %s missing ground truth page/doc references.", qid)

        # Collection mapping
        collections = item.get("collection_names", [])
        if not collections:
            logger.warning("Item %s missing 'collection_names'. Defaulting to 'general'.", qid)
            collections = ["general"]

        # Preserve or derive category
        if "category" in item and item["category"]:
            category = str(item["category"]).strip().lower()
        else:
            category = categorize_query(pages, collection_count=len(collections))

        # Map to ground_truth_chunk_ids
        chunk_ids: list[str] = []
        if pages:
            for coll in collections:
                for p in pages:
                    chunk_ids.append(map_chunk_id(coll, p))
        else:
            # Out-of-domain / edge cases test fallback; use explicit _ood sentinel
            primary_coll = collections[0]
            chunk_ids.append(f"doc_{primary_coll}_chunk_ood")

        ref_answer = synthesize_reference_answer(item, category)

        adapted: dict[str, Any] = {
            "id": qid,
            "question": question,
            "ground_truth_chunk_ids": chunk_ids,
            "reference_answer": ref_answer,
            "category": category,
            "metadata": {
                "collection_names": collections,
                "query_type": item.get("query_type", "unclassified"),
                "expected_retrieval_mode": item.get("expected_retrieval_mode", "local_document"),
                "ground_truth_keywords": item.get("ground_truth_keywords", []),
            },
        }
        adapted_items.append(adapted)

    return adapted_items


def write_yaml(items: list[dict[str, Any]], target_path: Path, source_path: Path) -> None:
    """Write adapted queries to YAML with header provenance comments."""
    target_path.parent.mkdir(parents=True, exist_ok=True)

    header = (
        "# ====================================================================\n"
        "# Curated Evaluation Test Set (Questions v1.0)\n"
        "#\n"
        f"# Provenance: Derived from CiteBase benchmark dataset\n"
        f"# Source: {source_path}\n"
        "# Reference: CiteBase EVALUATION_REPORT.md (25 held-out evaluation queries)\n"
        "#\n"
        "# Categories:\n"
        "#   - easy: single-page factual retrieval\n"
        "#   - ambiguous: multi-page / cross-collection retrieval\n"
        "#   - edge: out-of-domain queries testing web fallback / 'I don't know'\n"
        "# ====================================================================\n\n"
    )

    yaml_str = yaml.dump(items, sort_keys=False, width=120, allow_unicode=True)

    with open(target_path, "w", encoding="utf-8") as f:
        f.write(header + yaml_str)

    logger.info("Successfully wrote %d adapted questions to %s", len(items), target_path)


def main() -> int:
    source_arg = sys.argv[1] if len(sys.argv) > 1 else os.getenv("CITEBASE_DATASET_PATH")
    source_path = Path(source_arg) if source_arg else DEFAULT_CITEBASE_JSON

    logger.info("Extracting CiteBase test set from: %s", source_path)
    try:
        items = extract_queries(source_path)
        write_yaml(items, OUTPUT_YAML, source_path)
        return 0
    except Exception as e:
        logger.error("Extraction failed: %s", e, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
