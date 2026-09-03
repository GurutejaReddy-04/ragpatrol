"""
Retrieval quality evaluation module.

Calculates set-based Precision, Recall, and F1 comparing retrieved chunk IDs
against curated ground-truth chunk IDs.
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


class RetrievalScorer:
    """
    Evaluates whether the RAG retrieval pipeline fetched the expected chunks.

    Uses set-based overlap rather than fixed-length classification metrics because
    different queries may retrieve varying numbers of chunks.
    """

    def __init__(self) -> None:
        logger.debug("Initialized RetrievalScorer.")

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
        retrieved_set = set(retrieved_chunk_ids)
        gt_set = set(ground_truth_chunk_ids)

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
        matched_count = len(matched)
        precision = matched_count / len(retrieved_set) if retrieved_set else 0.0
        recall = matched_count / len(gt_set) if gt_set else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        return RetrievalResult(
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1=round(f1, 4),
            retrieved_count=len(retrieved_set),
            ground_truth_count=len(gt_set),
            matched_count=matched_count,
        )
