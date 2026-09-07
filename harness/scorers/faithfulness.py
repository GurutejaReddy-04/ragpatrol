"""
Dual-signal faithfulness evaluation module (Phase 3).

Combines fast semantic embedding cosine similarity (all-MiniLM-L6-v2) for gross
topical drift detection with a Gemini LLM-as-a-judge for nuanced factual grounding.
"""

import json
import logging
import os
import re
import threading
import time
from typing import Any, Optional, Sequence
import numpy as np
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer, util

from harness.exceptions import EmbeddingModelError

logger = logging.getLogger(__name__)

# Global singleton for sentence-transformers model
_EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
_EMBEDDING_MODEL: Optional[SentenceTransformer] = None
_EMBEDDING_LOCK = threading.Lock()


def get_embedding_model() -> SentenceTransformer:
    """Singleton getter for SentenceTransformer to prevent redundant memory allocation."""
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        with _EMBEDDING_LOCK:
            if _EMBEDDING_MODEL is None:
                logger.info("Loading sentence-transformers model: %s", _EMBEDDING_MODEL_NAME)
                try:
                    _EMBEDDING_MODEL = SentenceTransformer(_EMBEDDING_MODEL_NAME)
                except Exception as e:
                    logger.error(
                        "Failed to load embedding model '%s': %s",
                        _EMBEDDING_MODEL_NAME, e,
                    )
                    raise EmbeddingModelError(
                        f"Cannot load embedding model '{_EMBEDDING_MODEL_NAME}': {e}"
                    ) from e
    return _EMBEDDING_MODEL


class FaithfulnessResult(BaseModel):
    """Result of dual-signal faithfulness and groundedness evaluation."""
    faithfulness_score: float = Field(default=0.0, ge=0.0, le=1.0)
    embedding_similarity: float = Field(default=0.0, ge=0.0, le=1.0)
    llm_judge_score: float = Field(default=1.0, ge=1.0, le=5.0)
    is_hallucination: bool = Field(default=False)
    judge_reasoning: str = Field(default="")
    unsupported_claims: list[str] = Field(default_factory=list)
    embedding_latency_ms: float = Field(default=0.0, ge=0.0)
    judge_latency_ms: float = Field(default=0.0, ge=0.0)
    raw_judge_response: Optional[str] = Field(default=None)


class FaithfulnessScorer:
    """
    Evaluates whether generated answers are strictly grounded in retrieved context chunks.

    Combines:
    1. Fast embedding cosine similarity proxy (captures gross topic drift)
    2. Nuanced LLM-as-a-judge evaluation (captures unsupported claims)
    """

    def __init__(
        self,
        embedding_weight: float = 0.4,
        llm_judge_weight: float = 0.6,
        embedding_threshold: float = 0.5,
        judge_score_threshold: float = 2.0,
        gemini_client: Optional[Any] = None,
        model_name: str = "gemini-2.5-flash",
    ) -> None:
        """
        Initialize the dual-signal scorer.

        :param embedding_weight: Weight assigned to embedding similarity (hyperparameter).
        :param llm_judge_weight: Weight assigned to normalized LLM judge score (hyperparameter).
        :param embedding_threshold: Floor below which answer is flagged as hallucination.
        :param judge_score_threshold: Ceiling at or below which judge triggers hallucination flag.
        :param gemini_client: Optional injected client (e.g. mock client for tests).
        :param model_name: Gemini model identifier.
        """
        self.embedding_weight = embedding_weight
        self.llm_judge_weight = llm_judge_weight
        self.embedding_threshold = embedding_threshold
        self.judge_score_threshold = judge_score_threshold
        self._gemini_client = gemini_client
        self.model_name = model_name
        logger.debug(
            "FaithfulnessScorer initialized (weights: emb=%.2f, judge=%.2f)",
            embedding_weight,
            llm_judge_weight,
        )

    def compute_embedding_similarity(self, answer: str, context_text: str) -> tuple[float, float]:
        """
        Compute normalized cosine similarity between generated answer and retrieved context.

        :return: (normalized_similarity_in_0_to_1, latency_ms)
        """
        t0 = time.perf_counter()
        if not answer or not context_text:
            if not answer:
                logger.warning("Empty or None answer provided to embedding similarity.")
            return 0.0, (time.perf_counter() - t0) * 1000.0
        if not answer.strip() or not context_text.strip():
            return 0.0, (time.perf_counter() - t0) * 1000.0

        model = get_embedding_model()
        # Encode answer and context passages
        emb_answer = model.encode(answer.strip(), convert_to_tensor=True)
        emb_context = model.encode(context_text.strip(), convert_to_tensor=True)

        raw_sim = util.cos_sim(emb_answer, emb_context).item()
        # Clip negative cosine values to 0.0
        normalized_sim = max(0.0, min(1.0, float(raw_sim)))
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return round(normalized_sim, 4), round(latency_ms, 2)

    def call_llm_judge(
        self,
        question: str,
        context_text: str,
        answer: str,
    ) -> tuple[float, str, list[str], float, Optional[str]]:
        """
        Invoke Gemini LLM-as-a-judge with structured JSON prompt and defensive retries.

        :return: (judge_score_1_to_5, reasoning, unsupported_claims, latency_ms, raw_response)
        """
        t0 = time.perf_counter()
        prompt = (
            f"Question: {question}\n"
            f"Retrieved Context:\n"
            f"{context_text}\n"
            f"Generated Answer:\n"
            f"{answer}\n\n"
            f"Is the answer fully supported by the retrieved context?\n"
            f"Respond ONLY in JSON:\n"
            f'{{"faithfulness_score": 1-5, "reasoning": "...", "unsupported_claims": [...]}}'
        )

        client = self._get_gemini_client()
        if client is None:
            logger.warning("Gemini client unavailable. Skipping LLM judge call.")
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return 3.0, "Gemini client unconfigured or unavailable", [], round(latency_ms, 2), None

        raw_text = None
        for attempt in range(1, 3):
            try:
                logger.debug("Dispatching LLM judge call (attempt %d)...", attempt)
                response = client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                )
                raw_text = response.text if hasattr(response, "text") else str(response)
                score, reasoning, claims = self._parse_judge_json(raw_text)
                latency_ms = (time.perf_counter() - t0) * 1000.0
                return score, reasoning, claims, round(latency_ms, 2), raw_text
            except Exception as e:
                logger.warning("LLM judge attempt %d failed: %s", attempt, e)
                if attempt == 2:
                    latency_ms = (time.perf_counter() - t0) * 1000.0
                    return (
                        3.0,
                        f"Judge parsing failed after 2 attempts: {e}",
                        [],
                        round(latency_ms, 2),
                        raw_text,
                    )

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return 3.0, "Judge invocation error", [], round(latency_ms, 2), raw_text

    def _get_gemini_client(self) -> Any:
        """Resolve or initialize the Gemini client."""
        if self._gemini_client is not None:
            return self._gemini_client

        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("JUDGE_API_KEY")
        if not api_key:
            return None

        try:
            from google import genai
            self._gemini_client = genai.Client(api_key=api_key)
            return self._gemini_client
        except Exception as e:
            logger.error("Failed to instantiate google.genai.Client: %s", e)
            return None

    def _parse_judge_json(self, raw_text: str) -> tuple[float, str, list[str]]:
        """
        Defensively parse JSON from model output, stripping markdown code fences.
        """
        if not raw_text:
            raise ValueError("Empty response from LLM judge.")

        # Clean markdown code blocks
        clean_text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text.strip(), flags=re.MULTILINE)
        match = re.search(r"\{.*\}", clean_text, re.DOTALL)
        if not match:
            raise ValueError(f"No JSON object detected in response: {raw_text[:100]}")

        payload = json.loads(match.group(0))
        raw_score = payload.get("faithfulness_score", 3)
        score = float(raw_score)
        # Clamp to 1.0 - 5.0
        score = max(1.0, min(5.0, score))
        reasoning = str(payload.get("reasoning", ""))
        claims = list(payload.get("unsupported_claims", []))
        return score, reasoning, claims

    def score(
        self,
        question: str,
        answer: str,
        contexts: Sequence[str],
        dry_run: bool = False,
    ) -> FaithfulnessResult:
        """
        Calculate dual-signal faithfulness score.

        :param question: Evaluation test question.
        :param answer: Generated answer from target system.
        :param contexts: List of retrieved context chunk text strings.
        :param dry_run: If True, skips external LLM judge calls and uses embedding proxy.
        :return: Populated FaithfulnessResult DTO.
        """
        context_text = "\n---\n".join(c.strip() for c in contexts if c and c.strip())

        # 1. Embedding Signal
        emb_sim, emb_lat = self.compute_embedding_similarity(answer, context_text)

        # 2. LLM Judge Signal
        if dry_run:
            logger.info("Dry-run flag enabled: skipping Gemini judge call.")
            # Synthesize judge proxy from embedding similarity
            judge_score = max(1.0, min(5.0, round(emb_sim * 5.0, 1)))
            reasoning = "Dry-run mode: evaluated via embedding similarity proxy."
            claims: list[str] = []
            judge_lat = 0.0
            raw_judge = None
        else:
            judge_score, reasoning, claims, judge_lat, raw_judge = self.call_llm_judge(
                question=question,
                context_text=context_text,
                answer=answer,
            )

        # 3. Combined Score with dynamic weight fallback.
        # When the LLM judge is unavailable or failed, fall back to 100% embedding
        # instead of contributing a misleading constant from the default score.
        _JUDGE_FAILURE_MARKERS = (
            "unconfigured", "unavailable", "failed after", "invocation error",
        )
        judge_failed = any(marker in reasoning.lower() for marker in _JUDGE_FAILURE_MARKERS)

        if judge_failed and not dry_run:
            logger.warning(
                "LLM judge unavailable — falling back to 100%% embedding weight."
            )
            effective_emb_weight = 1.0
            effective_judge_weight = 0.0
        else:
            effective_emb_weight = self.embedding_weight
            effective_judge_weight = self.llm_judge_weight

        normalized_judge = judge_score / 5.0
        combined_score = round(
            (effective_emb_weight * emb_sim) + (effective_judge_weight * normalized_judge),
            4,
        )

        # 4. Hallucination Flagging
        # Also flag as hallucination when the judge is unavailable, because the
        # absence of verification is itself a reliability concern.
        is_hallucination = (
            (judge_score <= self.judge_score_threshold)
            or (emb_sim < self.embedding_threshold)
            or judge_failed
        )

        return FaithfulnessResult(
            faithfulness_score=combined_score,
            embedding_similarity=emb_sim,
            llm_judge_score=judge_score,
            is_hallucination=is_hallucination,
            judge_reasoning=reasoning,
            unsupported_claims=claims,
            embedding_latency_ms=emb_lat,
            judge_latency_ms=judge_lat,
            raw_judge_response=raw_judge,
        )
