
import pandas as pd
import streamlit as st
from pathlib import Path


st.set_page_config(
    page_title="Prompt & Eval Log",
    page_icon="📊",
    layout="wide"
)


LOG_PATH = Path(__file__).parent.parent / "evals" / "eval_log.csv"


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.title("📊 Prompt & Eval Log")

st.write(
    "Evaluation history for prompt versions, pass rates, "
    "query-type performance, and regression results."
)


# ---------------------------------------------------------
# Load evaluation log
# ---------------------------------------------------------

if not LOG_PATH.exists():
    st.error("Evaluation log not found.")
    st.stop()


df = pd.read_csv(LOG_PATH)


if df.empty:
    st.warning("No evaluation results found.")
    st.stop()


# ---------------------------------------------------------
# Latest Evaluation
# ---------------------------------------------------------

latest = df.iloc[-1]


st.header("Latest Evaluation")


col1, col2, col3, col4 = st.columns(4)


with col1:
    st.metric(
        "Prompt Version",
        latest["prompt_version"]
    )


with col2:
    st.metric(
        "Pass Rate",
        latest["overall_pass_rate"]
    )


with col3:
    st.metric(
        "Passed",
        int(latest["passed"])
    )


with col4:
    st.metric(
        "Failed",
        int(latest["failed"])
    )


# ---------------------------------------------------------
# Latest Results by Query Type
# ---------------------------------------------------------

st.header("Latest Results by Query Type")


category_df = pd.DataFrame({
    "Query Type": [
        "Billing",
        "Technical",
        "Delivery",
        "Account"
    ],
    "Pass Rate": [
        latest["billing_rate"],
        latest["technical_rate"],
        latest["delivery_rate"],
        latest["account_rate"]
    ]
})


st.dataframe(
    category_df,
    use_container_width=True,
    hide_index=True
)


# ---------------------------------------------------------
# Evaluation History
# ---------------------------------------------------------

st.header("Evaluation History")


display_df = df.rename(
    columns={
        "prompt_version": "Version",
        "date": "Date",
        "total_cases": "Total",
        "passed": "Passed",
        "failed": "Failed",
        "overall_pass_rate": "Overall",
        "billing_rate": "Billing",
        "technical_rate": "Technical",
        "delivery_rate": "Delivery",
        "account_rate": "Account",
        "result": "Result"
    }
)


st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)


# ---------------------------------------------------------
# Pass Rate Trend
# ---------------------------------------------------------

st.header("Pass Rate Trend")


trend_df = pd.DataFrame({
    "Evaluation": range(1, len(df) + 1),
    "Pass Rate": [
        float(
            rate.replace("%", "")
        )
        for rate in df["overall_pass_rate"]
    ]
})


trend_df = trend_df.set_index("Evaluation")


st.line_chart(trend_df)


# ---------------------------------------------------------
# Evaluation Results
# ---------------------------------------------------------

st.header("Evaluation Results")


for _, row in df.iterrows():

    if row["result"] == "PASS":
        icon = "✅"
    else:
        icon = "❌"

    st.write(
        f"{icon} "
        f"**{row['prompt_version']}** — "
        f"{row['overall_pass_rate']} — "
        f"{row['result']}"
    )

