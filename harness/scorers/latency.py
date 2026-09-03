"""
Latency profiling and percentile evaluation module.

Calculates p50, p95, and p99 percentiles alongside min/max/mean to capture
system latency distribution across cold and warm cache states.
"""

import logging
from typing import Sequence
import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class LatencyResult(BaseModel):
    """Statistical summary of query response latencies."""
    sample_count: int = Field(default=0, ge=0)
    p50_ms: float = Field(default=0.0, ge=0.0)
    p95_ms: float = Field(default=0.0, ge=0.0)
    p99_ms: float = Field(default=0.0, ge=0.0)
    mean_ms: float = Field(default=0.0, ge=0.0)
    min_ms: float = Field(default=0.0, ge=0.0)
    max_ms: float = Field(default=0.0, ge=0.0)


class LatencyProfiler:
    """
    Computes percentile latencies across evaluation runs.
    """

    def __init__(self) -> None:
        logger.debug("Initialized LatencyProfiler.")

    def compute_percentiles(self, latencies_ms: Sequence[float]) -> LatencyResult:
        """
        Compute latency percentiles from an array of millisecond timings.

        :param latencies_ms: Sequence of wall-clock latency measurements.
        :return: Populated LatencyResult DTO.
        """
        if not latencies_ms:
            logger.warning("Empty latency list provided to LatencyProfiler.")
            return LatencyResult()

        arr = np.array(latencies_ms, dtype=np.float64)
        return LatencyResult(
            sample_count=len(arr),
            p50_ms=round(float(np.percentile(arr, 50)), 2),
            p95_ms=round(float(np.percentile(arr, 95)), 2),
            p99_ms=round(float(np.percentile(arr, 99)), 2),
            mean_ms=round(float(np.mean(arr)), 2),
            min_ms=round(float(np.min(arr)), 2),
            max_ms=round(float(np.max(arr)), 2),
        )
