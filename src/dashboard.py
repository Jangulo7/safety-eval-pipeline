"""
Streamlit dashboard for LLM evaluation results visualization.

Features:
- Pagination for large datasets (prevents loading 1000+ results)
- Shared S3 client for connection pooling
- Date range filtering
- Caching for performance
- Export functionality
"""
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import altair as alt
import pandas as pd
import streamlit as st
from botocore.exceptions import ClientError

# Use shared client
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from aws_clients import get_s3_client

st.set_page_config(layout="wide", page_title="AI Eval Dashboard")


@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_data_paginated(
    max_results: int = 100,
    days_back: int = 30,
    model_filter: Optional[str] = None
) -> pd.DataFrame:
    """
    Load evaluation results with pagination and filtering.

    Args:
        max_results: Maximum number of results to load (default: 100)
        days_back: Only load results from last N days (default: 30)
        model_filter: Filter by model name substring (optional)

    Returns:
        DataFrame with evaluation results
    """
    s3 = get_s3_client()  # Use shared client
    bucket = os.getenv("S3_MODEL_BUCKET")
    prefix = "benchmarks/artificial_analysis/"

    if not bucket:
        st.error("S3_MODEL_BUCKET environment variable not set")
        return pd.DataFrame()

    data = []
    cutoff_date = datetime.now() - timedelta(days=days_back)

    try:
        paginator = s3.get_paginator("list_objects_v2")

        # Paginate but stop at max_results
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            if "Contents" not in page:
                continue

            for obj in page["Contents"]:
                # Stop if we've reached the limit
                if len(data) >= max_results:
                    break

                # Filter by date
                if obj["LastModified"] < cutoff_date:
                    continue

                # Only process JSON result files
                if not obj["Key"].endswith(".json"):
                    continue

                try:
                    obj_body = s3.get_object(Bucket=bucket, Key=obj["Key"])
                    content = json.loads(obj_body["Body"].read())

                    # Extract model info
                    model_name = content.get("config", {}).get("model_name", "Unknown")

                    # Apply model filter if specified
                    if model_filter and model_filter.lower() not in model_name.lower():
                        continue

                    # Build result row
                    res = {
                        "model": model_name,
                        "timestamp": obj["LastModified"],
                        "s3_key": obj["Key"],
                    }

                    # Extract benchmark results
                    if "results" in content and isinstance(content["results"], dict):
                        for k, v in content["results"].items():
                            # Clean metric name: "lighteval|mmlu|5|0" -> "MMLU"
                            short_name = k.split("|")[1].upper() if "|" in k else k

                            if isinstance(v, dict) and v:
                                # Get the first metric value (usually accuracy)
                                metric_val = next(iter(v.values()), 0)
                                res[short_name] = round(metric_val * 100, 2)
                            else:
                                res[short_name] = 0.0

                    data.append(res)

                except Exception as e:
                    # Log error but continue processing
                    st.warning(f"Error parsing {obj['Key']}: {e}")
                    continue

            # Break outer loop if we've hit max_results
            if len(data) >= max_results:
                break

    except ClientError as e:
        st.error(f"S3 Connection Error: {e}")
        return pd.DataFrame()

    df = pd.DataFrame(data)

    # Sort by timestamp descending (newest first)
    if not df.empty and "timestamp" in df.columns:
        df = df.sort_values("timestamp", ascending=False)

    return df


def display_summary_stats(df: pd.DataFrame):
    """Display summary statistics in columns."""
    if df.empty:
        return

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Evaluations", len(df))

    with col2:
        unique_models = df["model"].nunique() if "model" in df.columns else 0
        st.metric("Unique Models", unique_models)

    with col3:
        if "timestamp" in df.columns:
            latest = df["timestamp"].max()
            days_ago = (datetime.now() - latest).days
            st.metric("Latest Eval", f"{days_ago} days ago")

    with col4:
        # Average score across all benchmarks
        numeric_cols = df.select_dtypes(include=["float", "int"]).columns
        numeric_cols = [c for c in numeric_cols if c != "timestamp"]
        if len(numeric_cols) > 0:
            avg_score = df[numeric_cols].mean().mean()
            st.metric("Avg Score", f"{avg_score:.1f}%")


def display_leaderboard(df: pd.DataFrame):
    """Display ranked leaderboard by average score."""
    if df.empty:
        return

    st.subheader("🏆 Leaderboard")

    # Calculate average score across all benchmarks
    numeric_cols = df.select_dtypes(include=["float", "int"]).columns
    numeric_cols = [c for c in numeric_cols if c not in ["timestamp"]]

    if len(numeric_cols) > 0:
        df["avg_score"] = df[numeric_cols].mean(axis=1)
        leaderboard = df[["model", "avg_score"] + list(numeric_cols)].copy()
        leaderboard = leaderboard.sort_values("avg_score", ascending=False)

        # Format for display
        st.dataframe(
            leaderboard.style.background_gradient(
                subset=["avg_score"], cmap="RdYlGn"
            ),
            use_container_width=True,
            height=400,
        )
    else:
        st.info("No numeric benchmark scores found")


def main():
    """Main dashboard application."""
    st.title("LLM Evaluation Dashboard")

    # Sidebar filters
    st.sidebar.header("Filters")

    max_results = st.sidebar.slider(
        "Max Results to Load",
        min_value=10,
        max_value=500,
        value=100,
        step=10,
        help="Limit results to prevent slow loading",
    )

    days_back = st.sidebar.slider(
        "Days of History",
        min_value=1,
        max_value=365,
        value=30,
        help="Only show results from last N days",
    )

    model_filter = st.sidebar.text_input(
        "Model Name Filter",
        value="",
        help="Filter by model name (case-insensitive)",
    )

    # Load data button
    if st.sidebar.button("Refresh Data", type="primary"):
        st.cache_data.clear()

    # Load data with filters
    with st.spinner("Loading evaluation data..."):
        df = load_data_paginated(
            max_results=max_results,
            days_back=days_back,
            model_filter=model_filter if model_filter else None,
        )

    if df.empty:
        st.info("No evaluation results found. Check your filters or S3 configuration.")
        return

    # Display summary statistics
    display_summary_stats(df)

    st.markdown("---")

    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Results Table", "Leaderboard", "Visualizations"])

    with tab1:
        st.subheader("Evaluation Results")
        st.dataframe(df, use_container_width=True, height=500)

        # Export button
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download as CSV",
            data=csv,
            file_name=f"eval_results_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )

    with tab2:
        display_leaderboard(df)

    with tab3:
        st.subheader("Benchmark Comparisons")

        numeric_cols = df.select_dtypes(include=["float", "int"]).columns
        numeric_cols = [c for c in numeric_cols if c not in ["timestamp"]]

        if len(numeric_cols) >= 2:
            col1, col2 = st.columns(2)

            with col1:
                x_axis = st.selectbox("X Axis", numeric_cols, index=0)

            with col2:
                y_axis = st.selectbox(
                    "Y Axis",
                    numeric_cols,
                    index=min(1, len(numeric_cols) - 1),
                )

            # Scatter plot
            chart = (
                alt.Chart(df)
                .mark_circle(size=100)
                .encode(
                    x=alt.X(x_axis, scale=alt.Scale(domain=[0, 100])),
                    y=alt.Y(y_axis, scale=alt.Scale(domain=[0, 100])),
                    tooltip=["model", x_axis, y_axis],
                    color=alt.Color("model:N", legend=None),
                )
                .interactive()
                .properties(height=500)
            )

            st.altair_chart(chart, use_container_width=True)

            # Bar chart comparison
            st.subheader("All Benchmarks Comparison")

            # Get top N models
            top_n = st.slider("Number of models to compare", 3, 20, 10)

            if "avg_score" not in df.columns:
                numeric_cols_temp = df.select_dtypes(include=["float", "int"]).columns
                numeric_cols_temp = [c for c in numeric_cols_temp if c != "timestamp"]
                df["avg_score"] = df[numeric_cols_temp].mean(axis=1)

            top_models = df.nlargest(top_n, "avg_score")

            # Reshape data for grouped bar chart
            chart_data = []
            for _, row in top_models.iterrows():
                for col in numeric_cols:
                    chart_data.append({
                        "Model": row["model"],
                        "Benchmark": col,
                        "Score": row[col],
                    })

            chart_df = pd.DataFrame(chart_data)

            bar_chart = (
                alt.Chart(chart_df)
                .mark_bar()
                .encode(
                    x=alt.X("Model:N", axis=alt.Axis(labelAngle=-45)),
                    y=alt.Y("Score:Q", scale=alt.Scale(domain=[0, 100])),
                    color="Benchmark:N",
                    column="Benchmark:N",
                )
                .properties(width=150, height=400)
            )

            st.altair_chart(bar_chart, use_container_width=True)

        else:
            st.info("Need at least 2 numeric columns for visualizations")

    # Footer
    st.markdown("---")
    st.caption(
        f"Dashboard updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
        f"Showing {len(df)} results"
    )


if __name__ == "__main__":
    main()
