"""
Latency profiling and percentile evaluation module (Phase 4).

Calculates p50 (median), p95, and p99 percentiles alongside min, max, mean,
and standard deviation to guard against outlier skew. Quantifies cold vs. warm
cache performance improvements.
"""

import logging
from typing import Sequence
import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class LatencyProfile(BaseModel):
    """Statistical summary of query response latencies."""
    sample_count: int = Field(default=0, ge=0)
    p50_ms: float = Field(default=0.0, ge=0.0)
    p95_ms: float = Field(default=0.0, ge=0.0)
    p99_ms: float = Field(default=0.0, ge=0.0)
    mean_ms: float = Field(default=0.0, ge=0.0)
    std_ms: float = Field(default=0.0, ge=0.0)
    min_ms: float = Field(default=0.0, ge=0.0)
    max_ms: float = Field(default=0.0, ge=0.0)


# Backward compatibility alias
LatencyResult = LatencyProfile


class LatencyProfiler:
    """
    Computes percentile latencies across evaluation runs and evaluates cache acceleration.
    """

    def __init__(self) -> None:
        logger.debug("Initialized LatencyProfiler.")

    def compute_percentiles(self, latencies_ms: Sequence[float]) -> LatencyProfile:
        """
        Compute latency percentiles and distribution statistics from millisecond timings.

        :param latencies_ms: Sequence of wall-clock latency measurements.
        :return: Populated LatencyProfile DTO.
        """
        if not latencies_ms:
            logger.warning("Empty latency list provided to LatencyProfiler.")
            return LatencyProfile()

        arr = np.array([float(x) for x in latencies_ms if x >= 0.0], dtype=np.float64)
        if len(arr) == 0:
            return LatencyProfile()

        return LatencyProfile(
            sample_count=len(arr),
            p50_ms=round(float(np.percentile(arr, 50)), 2),
            p95_ms=round(float(np.percentile(arr, 95)), 2),
            p99_ms=round(float(np.percentile(arr, 99)), 2),
            mean_ms=round(float(np.mean(arr)), 2),
            std_ms=round(float(np.std(arr)), 2),
            min_ms=round(float(np.min(arr)), 2),
            max_ms=round(float(np.max(arr)), 2),
        )

    @staticmethod
    def compare_profiles(cold: LatencyProfile, warm: LatencyProfile) -> dict[str, float]:
        """
        Compute empirical cache speedup factors between cold and warm runs.

        :param cold: Baseline LatencyProfile from cold cache run.
        :param warm: Benchmark LatencyProfile from warm cache run.
        :return: Dictionary containing speedup multipliers across percentiles.
        """
        def safe_ratio(c: float, w: float) -> float:
            if w <= 0.0 or c <= 0.0:
                return 1.0
            return round(c / w, 2)

        return {
            "p50_speedup": safe_ratio(cold.p50_ms, warm.p50_ms),
            "p95_speedup": safe_ratio(cold.p95_ms, warm.p95_ms),
            "p99_speedup": safe_ratio(cold.p99_ms, warm.p99_ms),
            "mean_speedup": safe_ratio(cold.mean_ms, warm.mean_ms),
        }
