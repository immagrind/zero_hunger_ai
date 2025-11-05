from __future__ import annotations

import io
import os
from datetime import datetime

import pandas as pd
import streamlit as st

# Support running from parent dir or repo root
try:
    from app.utils import (
        validate_columns, aggregate, compute_ppc, classify,
        classify_ml_enhanced, predict_future_ppc, detect_anomalies
    )
    from app.visuals import plot_ppc_bar, plot_risk_pie, make_pdf_summary
except ModuleNotFoundError:
    import os
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
    from app.utils import (
        validate_columns, aggregate, compute_ppc, classify,
        classify_ml_enhanced, predict_future_ppc, detect_anomalies
    )
    from app.visuals import plot_ppc_bar, plot_risk_pie, make_pdf_summary


st.set_page_config(page_title="Zero Hunger AI Agent", layout="wide")

st.title("Zero Hunger AI Agent")
st.info(
    "This tool is an explainable decision-support aid. It does not replace professional judgement. "
    "Decisions should be verified with additional local data."
)

with st.sidebar:
    st.header("Data & Settings")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])  # Required columns enforced later
    threshold = st.slider("PPC Threshold (tonnes/person)", min_value=0.0, max_value=1.0, value=0.25, step=0.01)
    aggregate_crops = st.checkbox("Aggregate across crops", value=True)
    impute_missing = st.checkbox("Impute missing production", value=True)
    
    st.header("ML Features")
    use_ml_classification = st.checkbox("Use ML-enhanced classification", value=False, help="Uses Random Forest to refine risk predictions based on historical patterns")
    show_forecast = st.checkbox("Show future PPC forecasts", value=False, help="Predict next 3 years using ML regression")
    show_anomalies = st.checkbox("Detect anomalies", value=False, help="Flag unusual patterns using Isolation Forest")


def process(df: pd.DataFrame) -> dict:
    clean_df, invalid_df, issues = validate_columns(df)

    if not issues:
        st.success("Validation completed: no issues detected.")
    else:
        st.warning("Validation completed with issues: " + "; ".join(issues))

    if impute_missing:
        aggregated = aggregate(clean_df, aggregate_across_crops=aggregate_crops)
    else:
        # Drop rows with missing Production_Tonnes prior to aggregation
        no_missing = clean_df.dropna(subset=["Production_Tonnes"]).copy()
        aggregated = aggregate(no_missing, aggregate_across_crops=aggregate_crops)

    with_ppc = compute_ppc(aggregated)
    
    # Use ML-enhanced classification if enabled
    if use_ml_classification:
        classified = classify_ml_enhanced(with_ppc, threshold, use_ml=True)
    else:
        classified = classify(with_ppc, threshold)
    
    # Anomaly detection if enabled
    if show_anomalies:
        classified = detect_anomalies(classified)

    return {
        "invalid": invalid_df,
        "aggregated": aggregated,
        "classified": classified,
    }


def kpis(df: pd.DataFrame):
    avg_ppc = df["Production_per_Capita"].mean()
    counts = df["Risk_Category"].value_counts().to_dict()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg PPC", f"{avg_ppc:.4f}")
    c2.metric("#Critical", f"{counts.get('Critical Risk', 0)}")
    c3.metric("#Moderate", f"{counts.get('Moderate Risk', 0)}")
    c4.metric("#Safe", f"{counts.get('Safe Zone', 0)}")


def main():
    if uploaded is None:
        st.info("Upload a CSV to begin. Required columns: Region, Crop, Year, Production_Tonnes, Population.")
        return


    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        return

    try:
        outputs = process(df)
    except Exception as e:
        st.error(f"Processing error: {e}")
        return

    invalid = outputs["invalid"]
    classified = outputs["classified"]

    if not invalid.empty:
        st.warning(f"Invalid rows: {len(invalid)} (Population invalid or non-imputable missing production)")
        buf_invalid = io.BytesIO()
        invalid.to_csv(buf_invalid, index=False)
        st.download_button("Download invalid rows CSV", data=buf_invalid.getvalue(), file_name="invalid_rows.csv", mime="text/csv")

    # KPI row
    kpis(classified)

    # ML Forecasts
    if show_forecast:
        st.subheader("ML-Predicted Future PPC Trends")
        with st.spinner("Training ML model and generating forecasts..."):
            forecasts = predict_future_ppc(classified, forecast_years=3)
            if not forecasts.empty:
                st.dataframe(forecasts.sort_values(["Region", "Year"]))
                st.info("Predictions based on Random Forest regression trained on historical patterns")
            else:
                st.warning("Insufficient data for forecasting (need at least 10 rows with multiple years per region)")
    
    # Anomaly alerts
    if show_anomalies and "Is_Anomaly" in classified.columns:
        anomalies = classified[classified["Is_Anomaly"] == True]
        if not anomalies.empty:
            st.subheader("⚠️ Anomaly Detection Results")
            st.warning(f"Found {len(anomalies)} anomalous regions that may require investigation")
            st.dataframe(
                anomalies[["Region", "Year", "Production_per_Capita", "Anomaly_Score"]].sort_values("Anomaly_Score", ascending=False)
            )
    
    # Visuals
    st.plotly_chart(plot_ppc_bar(classified), use_container_width=True)
    st.plotly_chart(plot_risk_pie(classified), use_container_width=True)

    # Data table
    display_df = classified.copy()
    # Build column list dynamically
    base_cols = [
        "Region",
        "Year",
        "Total_Production_Tonnes",
        "Population",
        "Production_per_Capita",
        "Risk_Category",
        "Suggested_Intervention",
    ]
    # Add ML columns if present
    ml_cols = []
    if "ML_Confidence" in display_df.columns:
        ml_cols.extend(["ML_Risk_Category", "ML_Confidence"])
    if "Is_Anomaly" in display_df.columns:
        ml_cols.extend(["Is_Anomaly", "Anomaly_Score"])
    
    display_cols = base_cols + ml_cols
    available_cols = [c for c in display_cols if c in display_df.columns]
    
    st.dataframe(
        display_df[available_cols].sort_values(["Year", "Region"]).reset_index(drop=True)
    )

    # Exports
    flagged = classified[classified["Risk_Category"].isin(["Critical Risk", "Moderate Risk"])].copy()
    csv_buf = io.BytesIO()
    flagged_cols = [
        "Region",
        "Year",
        "Total_Production_Tonnes",
        "Population",
        "Production_per_Capita",
        "Risk_Category",
        "Suggested_Intervention",
    ]
    flagged.to_csv(csv_buf, index=False, columns=flagged_cols)
    st.download_button(
        "Download flagged (Critical + Moderate) CSV",
        data=csv_buf.getvalue(),
        file_name="flagged_rows.csv",
        mime="text/csv",
    )

    # PDF export via Matplotlib only
    exports_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "exports")
    os.makedirs(exports_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_path = os.path.join(exports_dir, f"report_summary_{ts}.pdf")
    try:
        make_pdf_summary(classified, threshold, pdf_path)
        with open(pdf_path, "rb") as f:
            st.download_button(
                "Download PDF Summary",
                data=f.read(),
                file_name=os.path.basename(pdf_path),
                mime="application/pdf",
            )
        st.success(f"PDF saved to {pdf_path}")
    except Exception as e:
        st.error(f"Failed to generate PDF: {e}")


if __name__ == "__main__":
    main()



