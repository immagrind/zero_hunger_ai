from __future__ import annotations

from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import LabelEncoder


REQUIRED_COLUMNS: Tuple[str, ...] = (
    "Region",
    "Crop",
    "Year",
    "Production_Tonnes",
    "Population",
)


def validate_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Implements report §6 and §5: Validate columns, filter invalids, imputation eligibility.

    - Ensures required columns exist.
    - Invalid rows: Population <= 0 or NaN → listed in invalids, excluded.
    - Missing Production_Tonnes → imputed later if (Region, Year) mean exists; otherwise flagged as invalid and excluded.

    Returns (clean_df_for_processing, invalid_df, issues_log).
    """
    issues: List[str] = []

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    work = df.copy()

    # Coerce dtypes conservatively
    work["Region"] = work["Region"].astype(str)
    work["Crop"] = work["Crop"].astype(str)
    work["Year"] = pd.to_numeric(work["Year"], errors="coerce").astype('Int64')
    work["Production_Tonnes"] = pd.to_numeric(work["Production_Tonnes"], errors="coerce")
    work["Population"] = pd.to_numeric(work["Population"], errors="coerce")

    invalid_mask_pop = work["Population"].isna() | (work["Population"] <= 0)
    invalid_pop = work[invalid_mask_pop]
    if not invalid_pop.empty:
        issues.append(f"Invalid population rows: {len(invalid_pop)}")

    # Rows with missing production might be imputable; keep temporarily
    missing_prod_mask = work["Production_Tonnes"].isna()

    # Build (Region, Year) mean for production
    prod_mean_by_region_year = (
        work.loc[~missing_prod_mask & ~invalid_mask_pop]
        .groupby(["Region", "Year"])['Production_Tonnes']
        .mean()
    )

    # Determine which missing production rows are imputable
    imputable_mask = (
        missing_prod_mask
        & ~invalid_mask_pop
        & work.set_index(["Region", "Year"]).index.isin(prod_mean_by_region_year.index)
    )

    non_imputable_mask = missing_prod_mask & ~imputable_mask
    invalid_missing_prod = work[non_imputable_mask]
    if not invalid_missing_prod.empty:
        issues.append(f"Non-imputable missing production rows: {len(invalid_missing_prod)}")

    # Construct invalid df and exclude from clean
    invalid_df = pd.concat([invalid_pop, invalid_missing_prod], axis=0).drop_duplicates().reset_index(drop=True)

    clean_df = work.loc[~invalid_mask_pop & ~non_imputable_mask].copy()

    return clean_df.reset_index(drop=True), invalid_df, issues


def aggregate(df: pd.DataFrame, aggregate_across_crops: bool) -> pd.DataFrame:
    """Implements report §7.4 Aggregation.

    Sums Production_Tonnes by (Region, Year) if aggregate_across_crops=True, otherwise by (Region, Year, Crop).
    Population is merged as the first/unique per (Region, Year).
    """
    work = df.copy()

    # Impute missing Production_Tonnes where possible using (Region, Year) mean
    missing_mask = work["Production_Tonnes"].isna()
    if missing_mask.any():
        means = work.groupby(["Region", "Year"])['Production_Tonnes'].transform('mean')
        work.loc[missing_mask, "Production_Tonnes"] = means[missing_mask]

    if aggregate_across_crops:
        grouped = work.groupby(["Region", "Year"], as_index=False).agg(
            Total_Production_Tonnes=("Production_Tonnes", "sum"),
            Population=("Population", "first"),
        )
    else:
        grouped = work.groupby(["Region", "Year", "Crop"], as_index=False).agg(
            Total_Production_Tonnes=("Production_Tonnes", "sum"),
            Population=("Population", "first"),
        )

    return grouped


def compute_ppc(df: pd.DataFrame) -> pd.DataFrame:
    """Implements report §7.5 PPC formula.

    Adds Production_per_Capita = Total_Production_Tonnes / Population.
    """
    work = df.copy()
    work["Production_per_Capita"] = (work["Total_Production_Tonnes"] / work["Population"]).astype(float)
    return work


def classify(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Implements report §7.6 Decision Rules.

    Deterministic classification:
    - PPC < 0.5 * threshold → Critical Risk; Immediate Aid Deployment
    - 0.5 * threshold ≤ PPC < threshold → Moderate Risk; Monitoring & Contingency Planning
    - PPC ≥ threshold → Safe Zone; No Immediate Action Required
    """
    work = df.copy()

    low = 0.5 * threshold

    conditions = [
        work["Production_per_Capita"] < low,
        (work["Production_per_Capita"] >= low) & (work["Production_per_Capita"] < threshold),
    ]
    choices = ["Critical Risk", "Moderate Risk"]
    work["Risk_Category"] = np.select(conditions, choices, default="Safe Zone")

    suggestion_map = {
        "Critical Risk": "Immediate Aid Deployment",
        "Moderate Risk": "Monitoring & Contingency Planning",
        "Safe Zone": "No Immediate Action Required",
    }
    work["Suggested_Intervention"] = work["Risk_Category"].map(suggestion_map)

    return work


def predict_future_ppc(df: pd.DataFrame, forecast_years: int = 3) -> pd.DataFrame:
    """ML-based forecasting: Predict future PPC trends using Random Forest regression.
    
    Trains on historical (Region, Year, PPC) patterns and forecasts next N years.
    Returns dataframe with Region, Year, Predicted_PPC columns.
    """
    if len(df) < 10:  # Need minimum data for ML
        return pd.DataFrame(columns=["Region", "Year", "Predicted_PPC"])
    
    work = df.copy()
    
    # Prepare features: Region (encoded), Year, and historical PPC trends
    le = LabelEncoder()
    work["Region_encoded"] = le.fit_transform(work["Region"])
    
    # Create features for each region: Year, rolling averages, trends
    features_list = []
    targets = []
    
    for region in work["Region"].unique():
        region_data = work[work["Region"] == region].sort_values("Year")
        if len(region_data) < 2:
            continue
            
        for idx, row in region_data.iterrows():
            year = row["Year"]
            ppc = row["Production_per_Capita"]
            
            # Features: Year, Region encoding, rolling stats
            region_subset = region_data[region_data["Year"] <= year]
            if len(region_subset) > 1:
                rolling_mean = region_subset["Production_per_Capita"].iloc[:-1].mean()
                rolling_std = region_subset["Production_per_Capita"].iloc[:-1].std()
                if pd.isna(rolling_std):
                    rolling_std = 0.0
                trend = (region_subset["Production_per_Capita"].iloc[-1] - region_subset["Production_per_Capita"].iloc[0]) / len(region_subset) if len(region_subset) > 1 else 0.0
            else:
                rolling_mean = ppc
                rolling_std = 0.0
                trend = 0.0
            
            features_list.append([year, row["Region_encoded"], rolling_mean, rolling_std, trend])
            targets.append(ppc)
    
    if len(features_list) < 5:
        return pd.DataFrame(columns=["Region", "Year", "Predicted_PPC"])
    
    X = np.array(features_list)
    y = np.array(targets)
    
    # Train model
    model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)
    model.fit(X, y)
    
    # Forecast future years
    max_year = work["Year"].max()
    predictions = []
    
    for region in work["Region"].unique():
        region_data = work[work["Region"] == region].sort_values("Year")
        if len(region_data) < 2:
            continue
        
        region_encoded = le.transform([region])[0]
        last_row = region_data.iloc[-1]
        last_ppc = last_row["Production_per_Capita"]
        last_rolling_mean = region_data["Production_per_Capita"].mean()
        last_rolling_std = region_data["Production_per_Capita"].std()
        if pd.isna(last_rolling_std):
            last_rolling_std = 0.0
        trend = (region_data["Production_per_Capita"].iloc[-1] - region_data["Production_per_Capita"].iloc[0]) / len(region_data) if len(region_data) > 1 else 0.0
        
        for year_offset in range(1, forecast_years + 1):
            future_year = max_year + year_offset
            features = np.array([[future_year, region_encoded, last_rolling_mean, last_rolling_std, trend]])
            pred_ppc = model.predict(features)[0]
            predictions.append({"Region": region, "Year": future_year, "Predicted_PPC": max(0, pred_ppc)})
    
    return pd.DataFrame(predictions)


def classify_ml_enhanced(df: pd.DataFrame, threshold: float, use_ml: bool = False) -> pd.DataFrame:
    """ML-enhanced risk classification: Combines deterministic rules with ML predictions.
    
    If use_ml=True and sufficient data, trains a classifier on historical patterns
    to refine risk predictions. Falls back to deterministic rules if ML unavailable.
    """
    work = df.copy()
    
    # First apply deterministic classification
    work = classify(work, threshold)
    
    if not use_ml or len(work) < 20:
        return work
    
    # ML enhancement: learn from historical patterns
    try:
        # Prepare features
        le = LabelEncoder()
        work["Region_encoded"] = le.fit_transform(work["Region"])
        
        # Features: PPC, Year, Region, Population, Production
        X = work[["Production_per_Capita", "Year", "Region_encoded", "Population", "Total_Production_Tonnes"]].values
        y = work["Risk_Category"].values
        
        # Train classifier
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)
        clf.fit(X, y)
        
        # Get ML predictions
        ml_predictions = clf.predict(X)
        ml_proba = clf.predict_proba(X)
        
        # Combine: use ML if confidence > 0.7, else keep deterministic
        confidence = np.max(ml_proba, axis=1)
        ml_override = confidence > 0.7
        
        work["ML_Risk_Category"] = work["Risk_Category"]
        work.loc[ml_override, "Risk_Category"] = ml_predictions[ml_override]
        work["ML_Confidence"] = confidence
        
        # Update suggestions based on ML predictions
        suggestion_map = {
            "Critical Risk": "Immediate Aid Deployment",
            "Moderate Risk": "Monitoring & Contingency Planning",
            "Safe Zone": "No Immediate Action Required",
        }
        work["Suggested_Intervention"] = work["Risk_Category"].map(suggestion_map)
        
    except Exception:
        # Fallback to deterministic if ML fails
        pass
    
    return work


def detect_anomalies(df: pd.DataFrame, contamination: float = 0.1) -> pd.DataFrame:
    """ML-based anomaly detection: Identify unusual regions using Isolation Forest.
    
    Flags regions with anomalous PPC patterns that may indicate data quality issues
    or genuinely unusual circumstances requiring investigation.
    """
    if len(df) < 10:
        df["Is_Anomaly"] = False
        df["Anomaly_Score"] = 0.0
        return df
    
    work = df.copy()
    
    # Features for anomaly detection
    features = work[["Production_per_Capita", "Total_Production_Tonnes", "Population", "Year"]].values
    
    # Train Isolation Forest
    iso_forest = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
    anomaly_labels = iso_forest.fit_predict(features)
    anomaly_scores = iso_forest.score_samples(features)
    
    work["Is_Anomaly"] = anomaly_labels == -1
    work["Anomaly_Score"] = -anomaly_scores  # Higher score = more anomalous
    
    return work


