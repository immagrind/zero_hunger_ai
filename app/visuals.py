from __future__ import annotations

from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px


def plot_ppc_bar(df: pd.DataFrame):
    """Implements report §8 Visualization: Bar chart of PPC per region (color by risk)."""
    color_map = {
        "Critical Risk": "#d62728",
        "Moderate Risk": "#ff7f0e",
        "Safe Zone": "#2ca02c",
    }
    fig = px.bar(
        df,
        x="Region",
        y="Production_per_Capita",
        color="Risk_Category",
        color_discrete_map=color_map,
        hover_data={"Year": True, "Population": True, "Production_per_Capita": ":.4f"},
    )
    fig.update_layout(yaxis_title="PPC (tonnes/person)", xaxis_title="Region")
    return fig


def plot_risk_pie(df: pd.DataFrame):
    """Implements report §8 Visualization: Pie chart of risk distribution."""
    counts = df["Risk_Category"].value_counts().reset_index()
    counts.columns = ["Risk_Category", "Count"]
    color_map = {
        "Critical Risk": "#d62728",
        "Moderate Risk": "#ff7f0e",
        "Safe Zone": "#2ca02c",
    }
    fig = px.pie(counts, names="Risk_Category", values="Count", color="Risk_Category", color_discrete_map=color_map)
    return fig


def make_pdf_summary(df: pd.DataFrame, threshold: float, path: str) -> None:
    """Implements report §9 Reporting: Single-page PDF via Matplotlib only.

    Contains:
    - Static bar chart (top 10 PPC)
    - Table (Critical + Moderate rows)
    - Text block: average PPC, counts of Critical/Moderate/Safe, top 3 vulnerable regions
    """
    # Prepare data
    df_sorted = df.sort_values("Production_per_Capita", ascending=False)
    top10 = df_sorted.head(10)

    crit_mod = df[df["Risk_Category"].isin(["Critical Risk", "Moderate Risk"])].copy()
    avg_ppc = df["Production_per_Capita"].mean()
    counts = df["Risk_Category"].value_counts().to_dict()
    vulnerable = df.sort_values("Production_per_Capita", ascending=True).head(3)["Region"].tolist()

    # Compose figure
    fig = plt.figure(figsize=(11.69, 8.27))  # A4 landscape in inches roughly

    # Bar chart axis
    ax_bar = fig.add_axes([0.06, 0.55, 0.6, 0.38])
    ax_bar.bar(top10["Region"], top10["Production_per_Capita"], color="#1f77b4")
    ax_bar.set_title("Top 10 Production per Capita (PPC)")
    ax_bar.set_ylabel("PPC (tonnes/person)")
    ax_bar.set_xticklabels(top10["Region"], rotation=45, ha="right")

    # Table axis
    ax_table = fig.add_axes([0.06, 0.08, 0.6, 0.38])
    table_cols = [
        "Region",
        "Year",
        "Total_Production_Tonnes",
        "Population",
        "Production_per_Capita",
        "Risk_Category",
    ]
    table_df = crit_mod[table_cols].copy()
    table_df["Production_per_Capita"] = table_df["Production_per_Capita"].map(lambda v: f"{v:.4f}")
    ax_table.axis("off")
    table = ax_table.table(cellText=table_df.values, colLabels=table_df.columns, loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.2)

    # Text axis
    ax_text = fig.add_axes([0.7, 0.08, 0.28, 0.85])
    ax_text.axis("off")
    lines: List[str] = []
    lines.append("Zero Hunger AI Agent - Summary")
    lines.append("")
    lines.append(f"Threshold: {threshold:.2f} tonnes/person")
    lines.append(f"Average PPC: {avg_ppc:.4f}")
    lines.append("")
    lines.append("Counts by Risk:")
    lines.append(f"  Critical: {counts.get('Critical Risk', 0)}")
    lines.append(f"  Moderate: {counts.get('Moderate Risk', 0)}")
    lines.append(f"  Safe: {counts.get('Safe Zone', 0)}")
    lines.append("")
    lines.append("Top 3 Vulnerable Regions:")
    for r in vulnerable:
        lines.append(f"  - {r}")

    ax_text.text(0, 1, "\n".join(lines), va="top", fontsize=10)

    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)



