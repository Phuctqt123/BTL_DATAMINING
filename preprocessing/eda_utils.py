"""Descriptive statistics and reproducible EDA plots for TV1."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .preprocess import FEATURE_COLUMNS, clean_transactions, log_transform_features

sns.set_theme(style="whitegrid", palette="deep")


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_eda_outputs(raw: pd.DataFrame, features: pd.DataFrame, root: str | Path) -> dict:
    """Save EDA tables and charts. Can run repeatedly without fitting models."""
    root = Path(root)
    fig_dir = root / "preprocessing" / "figures"
    out_dir = root / "data" / "processed"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    missing = pd.DataFrame({
        "MissingCount": raw.isna().sum().reindex(
            ["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "Customer ID", "Country"]
        ),
        "MissingPct": raw.isna().mean().reindex(
            ["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "Customer ID", "Country"]
        ) * 100,
    })
    missing.index.name = "Column"
    missing.to_csv(out_dir / "eda_missing_values.csv")
    fig, ax = plt.subplots(figsize=(9.0, 4.5))
    sns.barplot(x=missing.index, y=missing["MissingPct"], ax=ax, color="steelblue")
    ax.set(xlabel="Column", ylabel="Missing (%)", title="Missing values in the combined raw workbook")
    ax.tick_params(axis="x", rotation=32)
    for i, pct in enumerate(missing["MissingPct"]):
        ax.text(i, pct + max(0.25, missing["MissingPct"].max() * 0.01), f"{pct:.2f}%",
                ha="center", fontsize=9)
    ax.set_ylim(0, max(1, missing["MissingPct"].max() * 1.18))
    _save(fig, fig_dir / "01_missing_values.png")

    raw_countries = raw["Country"].fillna("<MISSING>").value_counts()
    country_summary = raw_countries.rename_axis("Country").reset_index(name="RawRows")
    country_summary.to_csv(out_dir / "eda_countries.csv", index=False)
    fig, ax = plt.subplots(figsize=(11, 6))
    subset = country_summary.head(15).sort_values("RawRows", ascending=True)
    sns.barplot(data=subset, x="RawRows", y="Country", ax=ax, color="steelblue")
    ax.set(title="Top 15 countries by transaction line count (raw)", xlabel="Line items", ylabel="")
    _save(fig, fig_dir / "02_countries.png")

    dates = raw.loc[raw["InvoiceDate"].notna(), ["InvoiceDate", "Invoice"]].copy()
    dates["Month"] = dates["InvoiceDate"].dt.to_period("M").astype(str)
    monthly = dates.groupby("Month").agg(
        RawRows=("Invoice", "size"), DistinctInvoices=("Invoice", "nunique")
    ).reset_index()
    monthly.to_csv(out_dir / "eda_monthly_raw.csv", index=False)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(monthly["Month"], monthly["RawRows"], marker="o", linewidth=2, markersize=3)
    ax.set(title="Transaction lines by month (raw)", xlabel="Month", ylabel="Line items")
    ax.tick_params(axis="x", rotation=55)
    ax.grid(alpha=0.25)
    _save(fig, fig_dir / "03_monthly_raw.png")

    cleaned = clean_transactions(raw)
    valid_sales = cleaned.purchases.copy()
    valid_sales["Month"] = valid_sales["InvoiceDate"].dt.to_period("M").astype(str)
    monthly_sales = valid_sales.groupby("Month").agg(
        SalesRows=("Invoice", "size"), SalesInvoices=("Invoice", "nunique"),
        GrossSalesGBP=("LineTotal", "sum"),
    ).reset_index()
    monthly_sales.to_csv(out_dir / "eda_monthly_clean_sales.csv", index=False)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(monthly_sales["Month"], monthly_sales["GrossSalesGBP"], marker="o", markersize=3)
    ax.set(title="Gross sales per month, valid purchases only (GBP)", xlabel="Month", ylabel="GBP")
    ax.tick_params(axis="x", rotation=55)
    ax.grid(alpha=0.25)
    _save(fig, fig_dir / "04_monthly_clean_sales.png")

    # Seasonality comparison by calendar month, only complete years. 2009 and
    # 2011 cover partial years; don't compare December 2011 as a full month.
    monthly_sales["Year"] = monthly_sales["Month"].str[:4].astype(int)
    monthly_sales["MonthNumber"] = monthly_sales["Month"].str[5:7].astype(int)
    fig, ax = plt.subplots(figsize=(9, 5))
    pivot = monthly_sales.pivot(index="MonthNumber", columns="Year", values="GrossSalesGBP")
    sns.heatmap(pivot, annot=False, fmt=".0f", cmap="YlGnBu", ax=ax)
    ax.set(title="Monthly gross sales by year (check partial coverage)", xlabel="Year", ylabel="Calendar month")
    _save(fig, fig_dir / "05_month_year_heatmap.png")

    log_data = log_transform_features(features)
    skews_before = features.skew(numeric_only=True)
    skews_after = log_data.skew(numeric_only=True)
    skew = pd.DataFrame({"SkewBefore": skews_before, "SkewAfterLog": skews_after}).reindex(FEATURE_COLUMNS)
    skew.index.name = "Feature"
    skew.to_csv(out_dir / "eda_skewness_comparison.csv")

    # One pair of plots per feature, avoiding cross-feature scales that would
    # render near-zero variables invisible.
    for i, column in enumerate(FEATURE_COLUMNS, start=1):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        sns.histplot(features[column], bins=35, ax=axes[0], kde=False, color="steelblue")
        sns.histplot(log_data[column], bins=35, ax=axes[1], kde=False, color="steelblue")
        axes[0].set(title=f"{column}: raw (skew={skews_before[column]:.2f})", xlabel=column)
        axes[1].set(title=f"{column}: after log (skew={skews_after[column]:.2f})",
                    xlabel=f"log1p({column})" if column != "CancelRate" else "unchanged CancelRate")
        fig.suptitle("Feature distribution before and after transform")
        _save(fig, fig_dir / f"feature_{i:02d}_{column.lower()}_hist.png")

        fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
        sns.boxplot(x=features[column], ax=axes[0], showfliers=True, color="steelblue")
        sns.boxplot(x=log_data[column], ax=axes[1], showfliers=True, color="steelblue")
        axes[0].set(title=f"{column}: raw", xlabel=column)
        axes[1].set(title=f"{column}: transformed", xlabel="log1p" if column != "CancelRate" else "original")
        _save(fig, fig_dir / f"feature_{i:02d}_{column.lower()}_box.png")

    fig, ax = plt.subplots(figsize=(11, 4.5))
    temp = skew.reset_index().melt(id_vars="Feature", var_name="Stage", value_name="Skewness")
    sns.barplot(data=temp, x="Feature", y="Skewness", hue="Stage", ax=ax, palette=["steelblue", "darkorange"])
    ax.tick_params(axis="x", rotation=25)
    ax.set(title="Skewness before vs. after log transform", xlabel="Feature")
    _save(fig, fig_dir / "06_skewness_comparison.png")

    # Pearson detects linear association; Spearman is less sensitive to heavy
    # tails and better matches the ordinal interpretation of RFM features.
    for method in ("pearson", "spearman"):
        corr = log_data.corr(method=method).reindex(index=FEATURE_COLUMNS, columns=FEATURE_COLUMNS)
        corr.to_csv(out_dir / f"eda_correlation_{method}.csv", index_label="Feature")
        fig, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                    center=0, vmin=-1, vmax=1, ax=ax, square=True)
        ax.set(title=f"{method.title()} correlation of transformed features")
        _save(fig, fig_dir / f"07_correlation_{method}.png")

    # Save a short, measured interpretation for the notebook/report generator.
    summary = {
        "countries_top5": {str(k): int(v) for k, v in raw_countries.head(5).items()},
        "months_in_raw": int(len(monthly)),
        "monthly_peak_raw": (monthly.loc[monthly["RawRows"].idxmax(), "Month"] if len(monthly) else None),
        "monthly_peak_raw_lines": int(monthly["RawRows"].max()) if len(monthly) else 0,
        "monthly_peak_sales": (monthly_sales.loc[monthly_sales["GrossSalesGBP"].idxmax(), "Month"] if len(monthly_sales) else None),
        "monthly_peak_sales_gbp": float(monthly_sales["GrossSalesGBP"].max()) if len(monthly_sales) else 0,
        "gross_sales_total_gbp": float(valid_sales["LineTotal"].sum()),
        "features_min_max_median": {
            col: {
                "min": float(features[col].min()), "max": float(features[col].max()),
                "median": float(features[col].median()),
            } for col in FEATURE_COLUMNS
        },
    }
    (out_dir / "eda_observations.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary
