from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data_loader import load_csv  # noqa: E402
from predict import detect_anomalies  # noqa: E402
from utils import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_SCALER_PATH,
    DEFAULT_SMAP_METRICS_PATH,
    DEFAULT_SMAP_SUMMARY_PATH,
    load_json,
)

SAMPLE_DATA_PATH = ROOT / "data" / "raw" / "train.csv"

st.set_page_config(
    page_title="NetSentrix",
    page_icon=".",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root {
        --card-bg: rgba(255, 255, 255, 0.04);
        --card-border: rgba(148, 163, 184, 0.28);
        --card-text: inherit;
    }
    .main .block-container {padding-top: 1.4rem; max-width: 1380px;}
    [data-testid="stMetricValue"] {
        font-size: 1.55rem;
        color: var(--card-text);
    }
    [data-testid="stMetricLabel"], [data-testid="stMetricDelta"] {
        color: var(--card-text);
    }
    div[data-testid="stMetric"] {
        border: 1px solid var(--card-border);
        border-radius: 8px;
        padding: 0.85rem 1rem;
        background: var(--card-bg);
        color: var(--card-text);
    }
    .status-ok {color: #047857; font-weight: 700;}
    .status-warn {color: #b45309; font-weight: 700;}
    .status-critical {color: #b91c1c; font-weight: 700;}
    @media (prefers-color-scheme: light) {
        :root {
            --card-bg: #ffffff;
            --card-border: #e5e7eb;
            --card-text: #111827;
        }
    }
    @media (prefers-color-scheme: dark) {
        :root {
            --card-bg: rgba(15, 23, 42, 0.55);
            --card-border: rgba(148, 163, 184, 0.35);
            --card-text: #f8fafc;
        }
        .status-ok {color: #34d399;}
        .status-warn {color: #fbbf24;}
        .status-critical {color: #f87171;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def model_assets_ready() -> bool:
    return DEFAULT_MODEL_PATH.exists() and DEFAULT_SCALER_PATH.exists() and DEFAULT_CONFIG_PATH.exists()


def load_benchmark_artifacts() -> tuple[dict | None, pd.DataFrame | None]:
    summary = load_json(DEFAULT_SMAP_SUMMARY_PATH) if DEFAULT_SMAP_SUMMARY_PATH.exists() else None
    metrics = pd.read_csv(DEFAULT_SMAP_METRICS_PATH) if DEFAULT_SMAP_METRICS_PATH.exists() else None
    return summary, metrics


def x_axis(df: pd.DataFrame, timestamp_col: str | None):
    return df[timestamp_col] if timestamp_col and timestamp_col in df.columns else df.index


def severity_for_scores(scores: np.ndarray, threshold: float) -> np.ndarray:
    return np.select(
        [scores >= threshold * 1.6, scores > threshold],
        ["Critical", "Warning"],
        default="Normal",
    )


def health_score(scores: np.ndarray, threshold: float) -> int:
    if len(scores) == 0 or threshold <= 0:
        return 100
    anomaly_ratio = float(np.mean(scores > threshold))
    peak_pressure = float(np.clip((np.nanmax(scores) / threshold - 1) / 2, 0, 1))
    score = 100 - (anomaly_ratio * 65 + peak_pressure * 35)
    return int(np.clip(round(score), 0, 100))


def health_label(score: int) -> str:
    if score >= 85:
        return "Healthy"
    if score >= 60:
        return "Warning"
    return "Critical"


def top_root_causes(result: dict, indices: np.ndarray, limit: int = 5) -> pd.DataFrame:
    feature_scores = result["feature_scores"]
    feature_columns = result["feature_columns"]
    if len(indices) == 0:
        return pd.DataFrame(columns=["sensor", "impact"])

    impact = feature_scores[indices].mean(axis=0)
    order = np.argsort(impact)[::-1][:limit]
    return pd.DataFrame(
        {
            "sensor": [feature_columns[i] for i in order],
            "impact": impact[order],
        }
    )


def threshold_table(scores: np.ndarray, trained_threshold: float, manual_threshold: float) -> pd.DataFrame:
    methods = [
        ("Trained threshold", trained_threshold),
        ("Percentile 95", float(np.percentile(scores, 95))),
        ("Percentile 99", float(np.percentile(scores, 99))),
        ("Mean + 3*std", float(scores.mean() + 3 * scores.std())),
        ("Manual threshold", manual_threshold),
    ]
    return pd.DataFrame(
        [
            {
                "method": name,
                "threshold": value,
                "anomalies": int(np.sum(scores > value)),
                "health_score": health_score(scores, value),
            }
            for name, value in methods
        ]
    )


def make_score_figure(
    df: pd.DataFrame,
    scores,
    threshold: float,
    anomaly_indices,
    timestamp_col: str | None,
    end: int | None = None,
):
    view_df = df.iloc[:end] if end else df
    view_scores = scores[: len(view_df)]
    view_anomalies = anomaly_indices[anomaly_indices < len(view_df)]
    x = x_axis(view_df, timestamp_col)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=view_scores,
            mode="lines",
            name="Anomaly score",
            line=dict(color="#2563eb", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=[threshold] * len(view_scores),
            mode="lines",
            name="Threshold",
            line=dict(color="#ef4444", width=2, dash="dash"),
        )
    )
    if len(view_anomalies):
        fig.add_trace(
            go.Scatter(
                x=x.iloc[view_anomalies] if hasattr(x, "iloc") else view_anomalies,
                y=view_scores[view_anomalies],
                mode="markers",
                name="Anomalies",
                marker=dict(color="#dc2626", size=9, symbol="x"),
            )
        )

    fig.update_layout(
        title="Reconstruction Error Over Time",
        xaxis_title=timestamp_col or "Row index",
        yaxis_title="Smoothed reconstruction error",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=55, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def make_sensor_figure(df: pd.DataFrame, feature_columns: list[str], anomaly_indices, timestamp_col: str | None):
    x = x_axis(df, timestamp_col)
    fig = go.Figure()
    for column in feature_columns[:6]:
        fig.add_trace(go.Scatter(x=x, y=df[column], mode="lines", name=column))
    if len(anomaly_indices):
        marker_y = [df[feature_columns[0]].max()] * len(anomaly_indices)
        fig.add_trace(
            go.Scatter(
                x=x.iloc[anomaly_indices] if hasattr(x, "iloc") else anomaly_indices,
                y=marker_y,
                mode="markers",
                name="Anomaly markers",
                marker=dict(color="#dc2626", size=8, symbol="triangle-down"),
            )
        )
    fig.update_layout(
        title="Sensor Stream Overlay",
        xaxis_title=timestamp_col or "Row index",
        yaxis_title="Sensor value",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=55, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


st.title("NetSentrix")
st.caption("LSTM autoencoder anomaly monitoring with root-cause insight and live replay")

if not model_assets_ready():
    st.warning(
        "No trained model was found. Train first with `python src/train.py --input data/raw/train.csv`."
    )
    st.stop()

config = load_json(DEFAULT_CONFIG_PATH)
benchmark_summary, benchmark_metrics = load_benchmark_artifacts()

with st.sidebar:
    st.header("Data")
    data_source = st.radio("Monitoring source", ["Demo data", "Upload CSV"], horizontal=True)
    uploaded_file = None
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    else:
        st.caption("Demo data powers the live monitoring tabs. NASA SMAP results live in the Benchmark tab.")
    timestamp_col = st.text_input("Timestamp column", value=config.get("timestamp_col") or "")
    label_col = st.text_input("Label column", value=config.get("label_col") or "label")

    st.header("Detection")
    smoothing_window = st.slider(
        "Smoothing window",
        min_value=1,
        max_value=50,
        value=int(config.get("smoothing_window", 5)),
    )
    use_saved_threshold = st.toggle("Use trained threshold", value=True)
    manual_threshold_value = st.number_input(
        "Manual threshold",
        min_value=0.0,
        value=float(config.get("threshold", 0.01)),
        format="%.8f",
    )
    manual_threshold = None if use_saved_threshold else manual_threshold_value

if data_source == "Upload CSV" and uploaded_file is None:
    st.info("Upload a CSV file or switch to Demo data in the sidebar.")
    st.stop()

try:
    df = load_csv(SAMPLE_DATA_PATH if data_source == "Demo data" else uploaded_file)
    result = detect_anomalies(
        df,
        label_col=label_col,
        timestamp_col=timestamp_col or None,
        threshold=manual_threshold,
        smoothing_window=smoothing_window,
    )
except Exception as exc:
    st.error(str(exc))
    st.stop()

anomaly_indices = result["anomaly_indices"]
scores = result["scores"]
threshold = result["threshold"]
report = result["report"]
feature_columns = result["feature_columns"]
severities = severity_for_scores(scores, threshold)
health = health_score(scores, threshold)
status = health_label(health)
first_anomaly = int(anomaly_indices[0]) if len(anomaly_indices) else None
top_causes = top_root_causes(result, anomaly_indices)
primary_cause = top_causes.iloc[0]["sensor"] if not top_causes.empty else "None"

metric_cols = st.columns(6)
metric_cols[0].metric("Rows", f"{len(df):,}")
metric_cols[1].metric("Sensors", f"{len(feature_columns):,}")
metric_cols[2].metric("Anomalies", f"{len(anomaly_indices):,}")
metric_cols[3].metric("Health", f"{health}/100", status)
metric_cols[4].metric("First anomaly", "None" if first_anomaly is None else first_anomaly)
metric_cols[5].metric("Top cause", primary_cause)

if status == "Healthy":
    st.markdown('<span class="status-ok">System status: Healthy</span>', unsafe_allow_html=True)
elif status == "Warning":
    st.markdown('<span class="status-warn">System status: Warning</span>', unsafe_allow_html=True)
else:
    st.markdown('<span class="status-critical">System status: Critical</span>', unsafe_allow_html=True)

monitor_tab, live_tab, explain_tab, threshold_tab, benchmark_tab, report_tab = st.tabs(
    ["Monitor", "Live Replay", "Root Cause", "Threshold Lab", "Benchmark", "Report"]
)

with monitor_tab:
    st.plotly_chart(
        make_score_figure(df, scores, threshold, anomaly_indices, timestamp_col or None),
        use_container_width=True,
    )
    st.plotly_chart(
        make_sensor_figure(df, feature_columns, anomaly_indices, timestamp_col or None),
        use_container_width=True,
    )

with live_tab:
    st.subheader("Real-Time Replay")
    replay_cols = st.columns([0.7, 0.3])
    replay_rows = replay_cols[0].slider("Rows to replay", 80, len(df), min(220, len(df)))
    replay_speed = replay_cols[1].slider("Delay per frame", 0.01, 0.25, 0.04)
    frame = st.empty()
    if st.button("Run replay", type="primary"):
        step = max(5, replay_rows // 60)
        for end in range(60, replay_rows + 1, step):
            frame.plotly_chart(
                make_score_figure(df, scores, threshold, anomaly_indices, timestamp_col or None, end=end),
                use_container_width=True,
                key=f"replay-{end}",
            )
            time.sleep(replay_speed)
    else:
        frame.plotly_chart(
            make_score_figure(df, scores, threshold, anomaly_indices, timestamp_col or None, end=replay_rows),
            use_container_width=True,
        )

with explain_tab:
    left, right = st.columns([0.48, 0.52])
    with left:
        st.subheader("Root-Cause Sensors")
        st.dataframe(top_causes, use_container_width=True, hide_index=True)
        if len(anomaly_indices):
            selected_index = st.selectbox("Inspect anomaly index", anomaly_indices)
            selected_errors = pd.DataFrame(
                {
                    "sensor": feature_columns,
                    "reconstruction_error": result["feature_scores"][int(selected_index)],
                }
            ).sort_values("reconstruction_error", ascending=False)
            st.dataframe(selected_errors, use_container_width=True, hide_index=True)
    with right:
        cause_fig = go.Figure()
        if not top_causes.empty:
            cause_fig.add_trace(
                go.Bar(
                    x=top_causes["impact"],
                    y=top_causes["sensor"],
                    orientation="h",
                    marker_color="#2563eb",
                )
            )
        cause_fig.update_layout(
            title="Average Contribution During Anomalies",
            xaxis_title="Mean reconstruction error",
            yaxis_title="Sensor",
            margin=dict(l=20, r=20, t=55, b=20),
        )
        st.plotly_chart(cause_fig, use_container_width=True)

with threshold_tab:
    st.subheader("Threshold Comparison")
    comparison = threshold_table(scores, float(config.get("threshold", threshold)), manual_threshold_value)
    st.dataframe(
        comparison.style.format({"threshold": "{:.6f}"}),
        use_container_width=True,
        hide_index=True,
    )
    threshold_fig = go.Figure()
    threshold_fig.add_trace(go.Histogram(x=scores, nbinsx=40, name="Scores", marker_color="#94a3b8"))
    for _, row in comparison.iterrows():
        threshold_fig.add_vline(
            x=row["threshold"],
            line_width=2,
            line_dash="dash",
            annotation_text=row["method"],
        )
    threshold_fig.update_layout(
        title="Anomaly Score Distribution",
        xaxis_title="Smoothed reconstruction error",
        yaxis_title="Rows",
        margin=dict(l=20, r=20, t=55, b=20),
    )
    st.plotly_chart(threshold_fig, use_container_width=True)

with benchmark_tab:
    st.subheader("NASA SMAP Benchmark")
    st.caption("These results come from saved SMAP benchmark artifacts and are independent of the demo/uploaded monitoring data.")
    if benchmark_summary is None or benchmark_metrics is None:
        st.info(
            "Run `python src/download_nasa_data.py` and "
            "`python src/benchmark_smap.py --spacecraft SMAP` to populate benchmark results."
        )
    else:
        summary_cols = st.columns(5)
        ae_summary = benchmark_summary["lstm_autoencoder"]
        if_summary = benchmark_summary["isolation_forest"]
        summary_cols[0].metric("Dataset", benchmark_summary["dataset"])
        summary_cols[1].metric("Channels", f"{benchmark_summary['channel_count']}")
        summary_cols[2].metric("AE F1", f"{ae_summary['f1']:.3f}")
        summary_cols[3].metric("IF F1", f"{if_summary['f1']:.3f}")
        summary_cols[4].metric(
            "Adj F1 lift",
            f"{benchmark_summary.get('adjusted_f1_improvement_vs_isolation_forest_pct', 0):+.2f} pts",
        )

        comparison = pd.DataFrame(
            [
                {
                    "model": "LSTM Autoencoder",
                    "precision": ae_summary["precision"],
                    "recall": ae_summary["recall"],
                    "f1": ae_summary["f1"],
                    "adjusted_f1": ae_summary.get("adjusted_f1", np.nan),
                    "roc_auc": ae_summary["roc_auc"],
                },
                {
                    "model": "Isolation Forest",
                    "precision": if_summary["precision"],
                    "recall": if_summary["recall"],
                    "f1": if_summary["f1"],
                    "adjusted_f1": if_summary.get("adjusted_f1", np.nan),
                    "roc_auc": if_summary["roc_auc"],
                },
            ]
        )
        st.dataframe(
            comparison.style.format(
                {
                    "precision": "{:.4f}",
                    "recall": "{:.4f}",
                    "f1": "{:.4f}",
                    "adjusted_f1": "{:.4f}",
                    "roc_auc": "{:.4f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.dataframe(
            benchmark_metrics.sort_values("ae_f1", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

with report_tab:
    left, right = st.columns([0.65, 0.35])
    with left:
        st.subheader("Anomaly Report")
        if len(anomaly_indices):
            st.dataframe(report, use_container_width=True, hide_index=True)
        else:
            st.success("No anomalies detected at the current threshold.")
    with right:
        st.subheader("Sensors Monitored")
        st.dataframe(pd.DataFrame({"column": feature_columns}), use_container_width=True, hide_index=True)
        severity_summary = pd.DataFrame(severities, columns=["severity"]).value_counts().reset_index(name="rows")
        st.dataframe(severity_summary, use_container_width=True, hide_index=True)

    csv = report.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download anomaly report",
        data=csv,
        file_name="anomaly_report.csv",
        mime="text/csv",
        disabled=report.empty,
        use_container_width=True,
    )
