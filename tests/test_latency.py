"""
Unit tests for percentile latency profiling and cache speedup evaluation.

Tests percentile math, standard deviation, and cold/warm cache comparison without network I/O.
"""

import pytest
from harness.scorers.latency import LatencyProfile, LatencyProfiler


@pytest.fixture
def profiler() -> LatencyProfiler:
    return LatencyProfiler()


def test_percentile_computation_known_array(profiler: LatencyProfiler) -> None:
    """Test percentile calculations against mathematically defined array [1, 2, 3, 4, 5]."""
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    profile = profiler.compute_percentiles(data)

    assert profile.sample_count == 5
    assert profile.p50_ms == 3.0
    assert profile.p95_ms == 4.8
    assert profile.p99_ms == 4.96  # linear interpolation across 5 points
    assert profile.mean_ms == 3.0
    assert profile.min_ms == 1.0
    assert profile.max_ms == 5.0
    assert profile.std_ms > 1.4  # std dev for 1..5 is sqrt(2) ~ 1.414


def test_empty_latency_array_handling(profiler: LatencyProfiler) -> None:
    """Empty list returns zeroed profile without raising ZeroDivisionError."""
    profile = profiler.compute_percentiles([])
    assert profile.sample_count == 0
    assert profile.p50_ms == 0.0
    assert profile.mean_ms == 0.0


def test_single_element_array(profiler: LatencyProfiler) -> None:
    """Single sample yields identical percentiles."""
    profile = profiler.compute_percentiles([125.5])
    assert profile.sample_count == 1
    assert profile.p50_ms == 125.5
    assert profile.p95_ms == 125.5
    assert profile.mean_ms == 125.5
    assert profile.std_ms == 0.0


def test_cold_warm_cache_comparison() -> None:
    """Verify empirical improvement ratio calculation."""
    cold = LatencyProfile(
        sample_count=25,
        p50_ms=400.0,
        p95_ms=800.0,
        p99_ms=1200.0,
        mean_ms=500.0,
        std_ms=150.0,
        min_ms=200.0,
        max_ms=1500.0,
    )
    warm = LatencyProfile(
        sample_count=25,
        p50_ms=20.0,
        p95_ms=40.0,
        p99_ms=60.0,
        mean_ms=25.0,
        std_ms=10.0,
        min_ms=15.0,
        max_ms=80.0,
    )

    comparison = LatencyProfiler.compare_profiles(cold, warm)
    # p50 speedup: 400 / 20 = 20.0x
    assert comparison["p50_speedup"] == 20.0
    # p95 speedup: 800 / 40 = 20.0x
    assert comparison["p95_speedup"] == 20.0
    # mean speedup: 500 / 25 = 20.0x
    assert comparison["mean_speedup"] == 20.0
