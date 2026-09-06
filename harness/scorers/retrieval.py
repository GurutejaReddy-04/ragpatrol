"""
Retrieval quality evaluation module (Phase 2).

Calculates pure set-based Precision, Recall, and F1 comparing retrieved chunk IDs
against curated ground-truth chunk IDs.

Design Note: Why Set Math over scikit-learn?
-------------------------------------------
Classical evaluation metrics in scikit-learn (e.g., precision_score, recall_score)
assume fixed-length classification vectors over a static universe of binary labels.
In production RAG systems, each query retrieves a variable number of document chunks
(e.g., 1 to 5 chunks from collections spanning thousands of pages). Set-based intersection
(True Positives = |Retrieved ∩ GroundTruth|) treats retrieval as an unordered set retrieval
problem, avoiding artificial zero-padding, index alignments, or label-space explosion.
"""

import logging
from typing import Sequence
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class RetrievalResult(BaseModel):
    """Evaluation result for a single question's retrieval pass."""
    precision: float = Field(default=0.0, ge=0.0, le=1.0)
    recall: float = Field(default=0.0, ge=0.0, le=1.0)
    f1: float = Field(default=0.0, ge=0.0, le=1.0)
    retrieved_count: int = Field(default=0, ge=0)
    ground_truth_count: int = Field(default=0, ge=0)
    matched_count: int = Field(default=0, ge=0)


def compute_retrieval_metrics(
    retrieved_ids: Sequence[str],
    ground_truth_ids: Sequence[str],
) -> dict[str, float]:
    """
    Pure Python set math for Precision, Recall, and harmonic F1.

    :param retrieved_ids: List of chunk IDs returned by target system.
    :param ground_truth_ids: List of relevant ground truth chunk IDs.
    :return: Dictionary with precision, recall, and f1 scores.
    """
    retrieved_set = set(str(x).strip() for x in retrieved_ids)
    ground_truth_set = set(str(x).strip() for x in ground_truth_ids)

    tp = len(retrieved_set & ground_truth_set)
    precision = tp / len(retrieved_set) if retrieved_set else 0.0
    recall = tp / len(ground_truth_set) if ground_truth_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


class RetrievalScorer:
    """
    Evaluates whether the RAG retrieval pipeline fetched the expected chunks.

    Uses pure set operations to handle variable chunk counts cleanly without
    fixed-length classification assumptions.
    """

    def __init__(self) -> None:
        logger.debug("Initialized RetrievalScorer.")

    def compute(
        self,
        retrieved_chunk_ids: Sequence[str],
        ground_truth_chunk_ids: Sequence[str],
    ) -> RetrievalResult:
        """Alias for score() conforming to runner computation conventions."""
        return self.score(retrieved_chunk_ids, ground_truth_chunk_ids)

    def score(
        self,
        retrieved_chunk_ids: Sequence[str],
        ground_truth_chunk_ids: Sequence[str],
    ) -> RetrievalResult:
        """
        Compute precision, recall, and F1 across chunk ID sequences.

        :param retrieved_chunk_ids: Chunk IDs returned by the target API.
        :param ground_truth_chunk_ids: Curated relevant chunk IDs from testset.
        :return: Populated RetrievalResult DTO.
        """
        # We trust no one, so we validate inputs into clean sets
        retrieved_set = set(str(cid).strip() for cid in retrieved_chunk_ids if cid)
        gt_set = set(str(gid).strip() for gid in ground_truth_chunk_ids if gid)

        if not gt_set:
            logger.warning("Empty ground truth chunk list provided to RetrievalScorer.")
            return RetrievalResult(
                precision=0.0,
                recall=0.0,
                f1=0.0,
                retrieved_count=len(retrieved_set),
                ground_truth_count=0,
                matched_count=0,
            )

        matched = retrieved_set & gt_set
        tp = len(matched)
        precision = tp / len(retrieved_set) if retrieved_set else 0.0
        recall = tp / len(gt_set) if gt_set else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        return RetrievalResult(
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1=round(f1, 4),
            retrieved_count=len(retrieved_set),
            ground_truth_count=len(gt_set),
            matched_count=tp,
        )
