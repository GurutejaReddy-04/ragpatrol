"""
RAGPatrol — Streamlit interactive observability dashboard.

Provides metric trend tracking over time, run inspection, and side-by-side configuration analysis.
Run with: streamlit run harness/reporting/dashboard.py
"""

import sys
from pathlib import Path
import pandas as pd
from sqlalchemy import select

# Ensure harness is importable when running via `streamlit run harness/reporting/dashboard.py`
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    import streamlit as st
except ImportError:
    st = None  # type: ignore[assignment]

from harness.storage.db import DatabaseManager
from harness.storage.models import EvalRun, RunMetric


def load_data() -> pd.DataFrame:
    """Load all evaluation runs from the database into a Pandas DataFrame for dashboard rendering."""
    db = DatabaseManager()
    with db.get_session() as session:
        runs = session.scalars(select(EvalRun).order_by(EvalRun.timestamp.asc())).all()
        run_data = []
        for r in runs:
            row = {
                "run_id": r.id,
                "timestamp": r.timestamp,
                "config_name": r.config_name,
                "stage": r.stage,
                "cache_state": r.cache_state,
                "git_commit": r.git_commit_sha[:7] if r.git_commit_sha else "N/A",
                "total_queries": r.total_queries,
            }
            for m in r.metrics:
                if m.category is None:
                    row[m.metric_name] = m.value
            run_data.append(row)
    return pd.DataFrame(run_data)


def main() -> None:
    """Launch the Streamlit interactive observability dashboard."""
    if st is None:
        print("Streamlit is not installed. Run: pip install streamlit")
        sys.exit(1)

    st.set_page_config(page_title="RAGPatrol Dashboard", page_icon="📈", layout="wide")
    st.title("📈 RAGPatrol — LLM Evaluation & Observability Dashboard")
    st.markdown("Automated quality gating, latency profiling, and regression tracking over time.")

    df = load_data()

    if df.empty:
        st.warning("No evaluation runs found in database. Execute `python -m harness.runner` to seed runs.")
        return

    # Sidebar filtering
    configs = sorted(df["config_name"].unique())
    selected_config = st.sidebar.selectbox("Select Configuration", options=["All"] + configs)

    filtered_df = df if selected_config == "All" else df[df["config_name"] == selected_config]

    # Overview KPI Cards
    latest_run = filtered_df.iloc[-1]
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Latest F1", f"{latest_run.get('retrieval_f1', 0)*100:.1f}%")
    with col2:
        st.metric("Latest Precision", f"{latest_run.get('retrieval_precision', 0)*100:.1f}%")
    with col3:
        st.metric("Faithfulness", f"{latest_run.get('faithfulness_avg', 0)*100:.1f}%")
    with col4:
        st.metric("p50 Latency", f"{latest_run.get('latency_p50', 0):.1f} ms")
    with col5:
        st.metric("p95 Latency", f"{latest_run.get('latency_p95', 0):.1f} ms")

    # Trend Charts
    st.subheader("📊 Metric Trends Over Time")
    trend_cols = ["timestamp", "retrieval_precision", "retrieval_recall", "retrieval_f1", "faithfulness_avg"]
    avail_cols = [c for c in trend_cols if c in filtered_df.columns]

    if len(avail_cols) > 1:
        chart_df = filtered_df[avail_cols].copy()
        chart_df["timestamp"] = pd.to_datetime(chart_df["timestamp"]).dt.strftime("%m-%d %H:%M")
        st.line_chart(chart_df.set_index("timestamp"))

    st.subheader("⚡ Latency Percentiles (p50 / p95 / p99)")
    lat_cols = ["timestamp", "latency_p50", "latency_p95", "latency_p99"]
    avail_lat = [c for c in lat_cols if c in filtered_df.columns]
    if len(avail_lat) > 1:
        lat_df = filtered_df[avail_lat].copy()
        lat_df["timestamp"] = pd.to_datetime(lat_df["timestamp"]).dt.strftime("%m-%d %H:%M")
        st.line_chart(lat_df.set_index("timestamp"))

    # Recent Runs Table
    st.subheader("📜 Historical Evaluation Runs")
    st.dataframe(filtered_df.sort_values(by="timestamp", ascending=False), use_container_width=True)


if __name__ == "__main__":
    main()
