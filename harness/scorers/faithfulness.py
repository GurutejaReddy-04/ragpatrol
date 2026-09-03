"""
Faithfulness evaluation module (Phase 3 stub).

Combines embedding-based cosine similarity (fast groundedness proxy) with
an LLM-as-a-judge signal to detect unsupported claims and hallucinations.
"""

import logging
from typing import Optional, Sequence
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class FaithfulnessResult(BaseModel):
    """Result of faithfulness and groundedness evaluation for an answer."""
    faithfulness_score: float = Field(default=0.0, ge=0.0, le=1.0)
    embedding_similarity: Optional[float] = Field(default=None, ge=-1.0, le=1.0)
    llm_judge_score: Optional[float] = Field(default=None, ge=1.0, le=5.0)
    is_hallucination: bool = Field(default=False)
    judge_reasoning: Optional[str] = None
    unsupported_claims: list[str] = Field(default_factory=list)


class FaithfulnessScorer:
    """
    Evaluates whether generated answers are strictly grounded in retrieved context chunks.

    Explicitly measures groundedness, NOT independent real-world fact verification.
    """

    def __init__(
        self,
        embedding_weight: float = 0.4,
        llm_judge_weight: float = 0.6,
        hallucination_threshold: float = 0.5,
    ) -> None:
        self.embedding_weight = embedding_weight
        self.llm_judge_weight = llm_judge_weight
        self.hallucination_threshold = hallucination_threshold
        logger.debug("Initialized FaithfulnessScorer stub.")

    def score(
        self,
        question: str,
        answer: str,
        contexts: Sequence[str],
    ) -> FaithfulnessResult:
        """
        Evaluate faithfulness of the answer against retrieved context.

        Phase 3 implementation placeholder.
        """
        logger.info("Faithfulness evaluation stub invoked for question: %s", question[:50])
        return FaithfulnessResult(
            faithfulness_score=1.0,
            embedding_similarity=1.0,
            llm_judge_score=5.0,
            is_hallucination=False,
            judge_reasoning="Phase 0 stub - full scoring enabled in Phase 3.",
            unsupported_claims=[],
        )
