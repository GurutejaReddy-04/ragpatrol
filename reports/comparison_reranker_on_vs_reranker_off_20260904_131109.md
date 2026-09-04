# Configuration Comparison: `reranker_on` vs `reranker_off`

## Experiment Summary

- **Timestamp:** 2026-09-04 13:11:09 UTC
- **Git Commit:** `5c6f0814`
- **Overall Experiment Winner:** 🏆 **`reranker_off`**
- **Outcome:** Config 'reranker_on' wins on 2/7 metrics. Config 'reranker_off' wins on 5/7 metrics.

## Side-by-Side Metric Comparison

| Metric | `reranker_on` | `reranker_off` | Delta | Winner |
| :--- | :---: | :---: | :---: | :---: |
| **Precision** | 16.0% | 10.0% | +6.0% | 🏆 **`reranker_on`** |
| **Recall** | 16.0% | 20.0% | -4.0% | 🏆 **`reranker_off`** |
| **F1** | 16.0% | 13.3% | +2.7% | 🏆 **`reranker_on`** |
| **Faithfulness** | 39.5% | 39.9% | -0.4% | 🏆 **`reranker_off`** |
| **Latency p50** | 48.3 ms | 46.8 ms | +1.5 ms | 🏆 **`reranker_off`** |
| **Latency p95** | 58.2 ms | 54.9 ms | +3.3 ms | 🏆 **`reranker_off`** |
| **Latency p99** | 62.8 ms | 60.0 ms | +2.8 ms | 🏆 **`reranker_off`** |

## Per-Category Performance Comparison

| Category | Metric | `reranker_on` | `reranker_off` |
| :--- | :--- | :---: | :---: |
| `ambiguous` | **Precision** | 0.0% | 5.6% |
| `ambiguous` | **Recall** | 0.0% | 11.1% |
| `ambiguous` | **F1 Score** | 0.0% | 7.4% |
| `ambiguous` | **Faithfulness** | 33.3% | 36.8% |
| `easy` | **Precision** | 36.4% | 18.2% |
| `easy` | **Recall** | 36.4% | 36.4% |
| `easy` | **F1 Score** | 36.4% | 24.2% |
| `easy` | **Faithfulness** | 41.1% | 38.3% |
| `edge` | **Precision** | 0.0% | 0.0% |
| `edge` | **Recall** | 0.0% | 0.0% |
| `edge` | **F1 Score** | 0.0% | 0.0% |
| `edge` | **Faithfulness** | 46.9% | 49.0% |
