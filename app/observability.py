import pandas as pd
import streamlit as st
from pathlib import Path


st.set_page_config(
    page_title="Observability Dashboard",
    page_icon="📈",
    layout="wide"
)


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

LOG_PATH = (
    Path(__file__).parent.parent
    / "evals"
    / "observability_log.csv"
)


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.title("📈 LLM Observability Dashboard")

st.write(
    "Monitor evaluation quality, cost, latency, and failure "
    "categories over time."
)


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

if not LOG_PATH.exists():
    st.error("Observability log not found.")
    st.stop()


df = pd.read_csv(LOG_PATH)


if df.empty:
    st.warning("No observability data found.")
    st.stop()


# ---------------------------------------------------------
# Prepare data
# ---------------------------------------------------------

# Handles both:
# 2026-09-14T10:00:00
# 2026-09-14T18:23:28.353932
df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    format="mixed"
)

df["passed"] = pd.to_numeric(
    df["passed"],
    errors="coerce"
).fillna(0).astype(int)

df["latency_ms"] = pd.to_numeric(
    df["latency_ms"],
    errors="coerce"
)

df["cost_usd"] = pd.to_numeric(
    df["cost_usd"],
    errors="coerce"
)

df = df.dropna(
    subset=[
        "timestamp",
        "latency_ms",
        "cost_usd"
    ]
)

df = df.sort_values("timestamp")


# ---------------------------------------------------------
# Sidebar Filters
# ---------------------------------------------------------

st.sidebar.header("Filters")


prompt_versions = sorted(
    df["prompt_version"].dropna().unique()
)

selected_versions = st.sidebar.multiselect(
    "Prompt Version",
    prompt_versions,
    default=prompt_versions
)


query_types = sorted(
    df["query_type"].dropna().unique()
)

selected_query_types = st.sidebar.multiselect(
    "Query Type",
    query_types,
    default=query_types
)


filtered_df = df[
    df["prompt_version"].isin(selected_versions)
    & df["query_type"].isin(selected_query_types)
].copy()


if filtered_df.empty:
    st.warning(
        "No data matches the selected filters."
    )
    st.stop()


# ---------------------------------------------------------
# Calculate Current Metrics
# ---------------------------------------------------------

total_requests = len(filtered_df)

total_passed = filtered_df["passed"].sum()

pass_rate = (
    total_passed / total_requests
    if total_requests
    else 0
)

total_cost = filtered_df["cost_usd"].sum()

p95_latency = filtered_df[
    "latency_ms"
].quantile(0.95)

failure_count = (
    filtered_df["passed"] == 0
).sum()


# ---------------------------------------------------------
# Top Metrics
# ---------------------------------------------------------

st.header("Current Metrics")


col1, col2, col3, col4 = st.columns(4)


with col1:
    st.metric(
        "Pass Rate",
        f"{pass_rate:.1%}"
    )


with col2:
    st.metric(
        "Total Cost",
        f"${total_cost:.4f}"
    )


with col3:
    st.metric(
        "p95 Latency",
        f"{p95_latency:.2f} ms"
    )


with col4:
    st.metric(
        "Failures",
        failure_count
    )


# ---------------------------------------------------------
# Data Note
# ---------------------------------------------------------

st.info(
    "Note: this project currently uses a deterministic local "
    "classifier. Therefore the evaluator records $0.0000 "
    "LLM cost and measures local execution latency. These "
    "metrics are for dashboard/evaluation demonstration, "
    "not production LLM billing or production model latency."
)


# ---------------------------------------------------------
# Pass Rate Over Time
# ---------------------------------------------------------

st.header("Pass Rate Over Time")


filtered_df["pass_rate"] = (
    filtered_df["passed"]
    .rolling(
        window=5,
        min_periods=1
    )
    .mean()
    * 100
)


pass_rate_chart = filtered_df[
    [
        "timestamp",
        "pass_rate"
    ]
].set_index("timestamp")


st.line_chart(
    pass_rate_chart
)


# ---------------------------------------------------------
# Cost Over Time
# ---------------------------------------------------------

st.header("Cost Over Time")


filtered_df["cumulative_cost"] = (
    filtered_df["cost_usd"].cumsum()
)


cost_chart = filtered_df[
    [
        "timestamp",
        "cumulative_cost"
    ]
].set_index("timestamp")


st.line_chart(
    cost_chart
)


# ---------------------------------------------------------
# Latency Over Time
# ---------------------------------------------------------

st.header("Latency Over Time")


filtered_df["rolling_p95_latency"] = (
    filtered_df["latency_ms"]
    .rolling(
        window=10,
        min_periods=1
    )
    .quantile(0.95)
)


latency_chart = filtered_df[
    [
        "timestamp",
        "rolling_p95_latency"
    ]
].set_index("timestamp")


st.line_chart(
    latency_chart
)


# ---------------------------------------------------------
# Failure Categories Over Time
# ---------------------------------------------------------

st.header("Failure Categories Over Time")


failures = filtered_df[
    filtered_df["passed"] == 0
].copy()


if failures.empty:

    st.success(
        "No failures recorded for the selected filters."
    )

else:

    failure_over_time = (
        failures
        .groupby(
            [
                pd.Grouper(
                    key="timestamp",
                    freq="5min"
                ),
                "failure_category"
            ]
        )
        .size()
        .unstack(
            fill_value=0
        )
    )

    st.line_chart(
        failure_over_time
    )


# ---------------------------------------------------------
# Failure Summary
# ---------------------------------------------------------

st.header("Failure Summary")


if failures.empty:

    st.success(
        "No failures recorded."
    )

else:

    failure_counts = (
        failures[
            "failure_category"
        ]
        .value_counts()
        .rename_axis(
            "Failure Category"
        )
        .reset_index(
            name="Count"
        )
    )

    st.bar_chart(
        failure_counts.set_index(
            "Failure Category"
        )
    )


# ---------------------------------------------------------
# Query Type Performance
# ---------------------------------------------------------

st.header("Performance by Query Type")


query_performance = (
    filtered_df
    .groupby("query_type")
    .agg(
        Total=("passed", "count"),
        Passed=("passed", "sum"),
        Average_Latency_ms=(
            "latency_ms",
            "mean"
        ),
        Total_Cost=(
            "cost_usd",
            "sum"
        )
    )
    .reset_index()
)


query_performance["Pass_Rate"] = (
    query_performance["Passed"]
    / query_performance["Total"]
    * 100
)


query_performance = query_performance[
    [
        "query_type",
        "Total",
        "Passed",
        "Pass_Rate",
        "Average_Latency_ms",
        "Total_Cost"
    ]
]


query_performance = query_performance.rename(
    columns={
        "query_type": "Query Type",
        "Pass_Rate": "Pass Rate (%)",
        "Average_Latency_ms": "Avg Latency (ms)",
        "Total_Cost": "Cost (USD)"
    }
)


st.dataframe(
    query_performance,
    use_container_width=True,
    hide_index=True
)


# ---------------------------------------------------------
# Failure Details
# ---------------------------------------------------------

st.header("Failure Details")


if failures.empty:

    st.write(
        "No failure records."
    )

else:

    failure_display = failures[
        [
            "timestamp",
            "prompt_version",
            "query_type",
            "failure_category",
            "latency_ms",
            "cost_usd"
        ]
    ].copy()


    failure_display = failure_display.rename(
        columns={
            "timestamp": "Timestamp",
            "prompt_version": "Prompt Version",
            "query_type": "Query Type",
            "failure_category": "Failure Category",
            "latency_ms": "Latency (ms)",
            "cost_usd": "Cost (USD)"
        }
    )


    st.dataframe(
        failure_display,
        use_container_width=True,
        hide_index=True
    )


# ---------------------------------------------------------
# Raw Observability Data
# ---------------------------------------------------------

st.header("Observability Data")


st.dataframe(
    filtered_df,
    use_container_width=True,
    hide_index=True
)

