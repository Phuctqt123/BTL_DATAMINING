"""Online Retail II: deterministic, reusable customer-level RFM pipeline.

Usage (from repository root):
    python -m preprocessing.preprocess --input data/raw/online_retail_II.xlsx

The eight output features are defined by the TV1 assignment. The scaler is
fit only once on the development data. Reuse the same scaler for new customers.
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

SHEETS = ("Year 2009-2010", "Year 2010-2011")
RAW_COLUMNS = (
    "Invoice", "StockCode", "Description", "Quantity", "InvoiceDate",
    "Price", "Customer ID", "Country",
)
FEATURE_COLUMNS = (
    "Recency", "Frequency", "Monetary", "AvgOrderValue",
    "UniqueProducts", "AvgBasketSize", "Tenure", "CancelRate",
)
# Tenure is left-skewed on UCI and log1p would increase |skewness|.
# CancelRate is bounded in [0,1]. Log only the six right-skewed metrics.
LOG_FEATURES = tuple(col for col in FEATURE_COLUMNS if col not in ("Tenure", "CancelRate"))
RANDOM_STATE = 42

# Only known charges/non-merchandise codes are excluded. Do NOT use a
# digits-only regex: genuine StockCode values such as 85123A are alphanumeric.
NON_PRODUCT_CODES = frozenset({
    "POST", "POSTAGE", "DOT", "M", "D", "BANK CHARGES", "AMAZONFEE",
    "C2", "CRUK", "S", "B", "PADS", "TEST001", "TEST002",
    "ADJUST", "ADJUST2", "MANUAL", "DISCOUNT",
})
NON_PRODUCT_PREFIXES = ("GIFT_",)  # vouchers, not distinct merchandise SKUs


@dataclass
class CleaningResult:
    purchases: pd.DataFrame
    cancellations: pd.DataFrame
    audit: pd.DataFrame
    retained_rows_before_split: int


def _canonical_id(value: Any) -> str | pd._libs.missing.NAType:
    """Preserve identifier identity but strip Excel's integer .0 formatting."""
    if value is None or pd.isna(value):
        return pd.NA
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return pd.NA
        return str(int(value)) if float(value).is_integer() else str(value)
    result = str(value).strip()
    if not result or result.casefold() in {"nan", "none", "null", "<na>"}:
        return pd.NA
    return re.sub(r"\.0+$", "", result)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names between the two UCI workbook versions."""
    aliases = {
        "invoice": "Invoice", "invoiceno": "Invoice", "stockcode": "StockCode",
        "description": "Description", "quantity": "Quantity",
        "invoicedate": "InvoiceDate", "price": "Price", "unitprice": "Price",
        "customerid": "Customer ID", "country": "Country",
    }
    rename = {}
    for col in df.columns:
        key = re.sub(r"[^a-z0-9]", "", str(col).strip().lower())
        if key in aliases:
            rename[col] = aliases[key]
    normalized = df.rename(columns=rename).copy()
    if normalized.columns.duplicated().any():
        raise ValueError("Duplicate canonical columns detected in workbook.")
    missing = [col for col in RAW_COLUMNS if col not in normalized.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Available: {list(df.columns)}")
    for col in ("Invoice", "StockCode", "Customer ID"):
        normalized[col] = pd.Series(
            (_canonical_id(v) for v in normalized[col].to_numpy()),
            index=normalized.index, dtype="string",
        )
    for col in ("Country", "Description"):
        normalized[col] = normalized[col].astype("string").str.strip().replace("", pd.NA)
    for col in ("Quantity", "Price"):
        normalized[col] = pd.to_numeric(normalized[col], errors="coerce")
    normalized["InvoiceDate"] = pd.to_datetime(
        normalized["InvoiceDate"], errors="coerce"
    )
    return normalized


def load_retail_data(workbook: str | Path) -> pd.DataFrame:
    """Read all rows from BOTH mandatory sheets, retaining SourceSheet for EDA."""
    workbook = Path(workbook)
    if not workbook.is_file():
        raise FileNotFoundError(
            f"Source workbook not found: {workbook}. "
            "Download UCI Online Retail II into data/raw/ first."
        )
    # The official 45 MB workbook has >8 million XML cells. Fast streaming
    # avoids multi-minute openpyxl object decoding while preserving both sheets.
    if workbook.stat().st_size > 10_000_000:
        try:
            from .fast_xlsx import load_retail_xlsx_fast
        except ImportError:
            print("[preprocess] lxml unavailable; falling back to openpyxl", flush=True)
        else:
            return load_retail_xlsx_fast(workbook)
    with pd.ExcelFile(workbook, engine="openpyxl") as xl:
        missing = [s for s in SHEETS if s not in xl.sheet_names]
        if missing:
            raise ValueError(f"Missing sheets {missing}; found {xl.sheet_names}")
        frames = []
        for sheet in SHEETS:
            frame = normalize_columns(pd.read_excel(xl, sheet_name=sheet))
            frame["SourceSheet"] = sheet
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _non_product_mask(df: pd.DataFrame) -> pd.Series:
    codes = df["StockCode"].fillna("").str.upper().str.strip()
    return codes.eq("") | codes.isin(NON_PRODUCT_CODES) | codes.str.startswith(NON_PRODUCT_PREFIXES)


def _cancellation_mask(df: pd.DataFrame) -> pd.Series:
    return df["Invoice"].fillna("").str.upper().str.startswith("C")


def describe_raw(df: pd.DataFrame) -> dict[str, Any]:
    """Return raw-data EDA summaries (without substituting cleaned figures)."""
    summary: dict[str, Any] = {
        "raw_rows": int(len(df)),
        "raw_columns": list(RAW_COLUMNS),
        "sheet_rows": {str(k): int(v) for k, v in df["SourceSheet"].value_counts().items()}
        if "SourceSheet" in df else {},
        "unique_invoices": int(df["Invoice"].nunique()),
        "unique_customers": int(df["Customer ID"].nunique()),
        "unique_products": int(df["StockCode"].nunique()),
        "unique_countries": int(df["Country"].nunique()),
        "date_min": str(df["InvoiceDate"].min()),
        "date_max": str(df["InvoiceDate"].max()),
    }
    summary["missing"] = {
        col: {"count": int(df[col].isna().sum()),
              "pct": round(float(df[col].isna().mean() * 100), 4)}
        for col in RAW_COLUMNS
    }
    mask_nonprod = _non_product_mask(df)
    mask_cancel = _cancellation_mask(df)
    summary["anomalies"] = {
        "negative_quantity": int(df["Quantity"].lt(0).sum()),
        "zero_quantity": int(df["Quantity"].eq(0).sum()),
        "zero_price": int(df["Price"].eq(0).sum()),
        "negative_price": int(df["Price"].lt(0).sum()),
        "cancelled_rows": int(mask_cancel.sum()),
        "cancelled_unique_invoices": int(df.loc[mask_cancel, "Invoice"].nunique()),
        "exact_duplicate_rows": int(df.duplicated(subset=list(RAW_COLUMNS)).sum()),
        "non_product_rows": int(mask_nonprod.sum()),
    }
    summary["non_product_codes_top"] = {
        str(k): int(v) for k, v in df.loc[mask_nonprod, "StockCode"]
        .fillna("<MISSING>").value_counts().head(25).items()
    }
    return summary


def clean_transactions(df: pd.DataFrame) -> CleaningResult:
    """Keep valid purchases and returns separately; never mix return spend into RFM."""
    working = df.copy()
    audit_rows: list[dict[str, Any]] = []

    def record(step: str, previous: int, remaining: int, notes: str) -> None:
        audit_rows.append({
            "Step": step, "RowsBefore": int(previous),
            "RowsRemoved": int(previous - remaining),
            "RowsRemaining": int(remaining), "Notes": notes,
        })

    record("Raw combined", len(working), len(working), "Both workbook sheets")
    before = len(working)
    working = working.loc[working["Customer ID"].notna()].copy()
    record("Remove missing Customer ID", before, len(working), "Customers cannot be identified")

    before = len(working)
    working = working.drop_duplicates(subset=list(RAW_COLUMNS)).copy()
    record("Remove exact duplicates", before, len(working), "Exclude SourceSheet from matching")

    before = len(working)
    working = working.loc[~_non_product_mask(working)].copy()
    record("Remove non-product StockCode", before, len(working), "Keep genuine alphanumeric SKUs")

    before = len(working)
    working = working.loc[
        working["Invoice"].notna() & working["InvoiceDate"].notna()
    ].copy()
    record("Remove invalid invoice/date", before, len(working), "Identifiers and dates required for RFM")

    before = len(working)
    mask_cancel = _cancellation_mask(working)
    cancellations = working.loc[mask_cancel].copy()
    purchases = working.loc[~mask_cancel].copy()
    record("Separate cancellation invoices", before, len(purchases),
           f"{len(cancellations)} rows retained separately for CancelRate")

    before = len(purchases)
    finite = np.isfinite(purchases["Quantity"]) & np.isfinite(purchases["Price"])
    purchases = purchases.loc[
        finite & purchases["Quantity"].gt(0) & purchases["Price"].gt(0)
    ].copy()
    record("Keep positive quantity and price", before, len(purchases), "Purchase rows only")

    purchases["LineTotal"] = purchases["Quantity"] * purchases["Price"]
    if (purchases["LineTotal"] <= 0).any() or not np.isfinite(purchases["LineTotal"]).all():
        raise ValueError("Invalid line totals after cleaning")
    if purchases.empty:
        raise ValueError("No valid purchases remain after cleaning")
    return CleaningResult(
        purchases=purchases.reset_index(drop=True),
        cancellations=cancellations.reset_index(drop=True),
        audit=pd.DataFrame(audit_rows),
        retained_rows_before_split=int(len(working)),
    )


def get_reference_date(df: pd.DataFrame) -> pd.Timestamp:
    """Calendar midnight after the final transaction date in the raw data."""
    last = pd.to_datetime(df["InvoiceDate"], errors="coerce").max()
    if pd.isna(last):
        raise ValueError("No valid InvoiceDate for reference date")
    return last.normalize() + pd.Timedelta(days=1)


def build_customer_features(
    purchases: pd.DataFrame,
    cancellations: pd.DataFrame,
    reference_date: str | pd.Timestamp,
) -> pd.DataFrame:
    """One customer per row, all eight mandatory RFM + behavioural features."""
    ref = pd.Timestamp(reference_date).normalize()
    if purchases.empty:
        raise ValueError("Cannot aggregate an empty purchase set")
    if purchases["InvoiceDate"].max().normalize() >= ref:
        raise ValueError("Reference date must be after the most recent purchase date")
    sales = purchases.groupby("Customer ID", sort=True).agg(
        first_purchase=("InvoiceDate", "min"),
        last_purchase=("InvoiceDate", "max"),
        Frequency=("Invoice", "nunique"),
        Monetary=("LineTotal", "sum"),
        UniqueProducts=("StockCode", "nunique"),
        purchased_items=("Quantity", "sum"),
    )
    sales["Recency"] = (ref - sales["last_purchase"].dt.normalize()).dt.days
    sales["Tenure"] = (ref - sales["first_purchase"].dt.normalize()).dt.days
    sales["AvgOrderValue"] = sales["Monetary"] / sales["Frequency"]
    sales["AvgBasketSize"] = sales["purchased_items"] / sales["Frequency"]

    cancel_counts = cancellations.groupby("Customer ID")["Invoice"].nunique()
    sales["CancelRate"] = (
        cancel_counts.reindex(sales.index, fill_value=0)
        / (sales["Frequency"] + cancel_counts.reindex(sales.index, fill_value=0))
    )

    features = sales.loc[:, FEATURE_COLUMNS].copy()
    for col in ("Recency", "Frequency", "UniqueProducts", "Tenure"):
        features[col] = features[col].astype("int64")
    if features.isna().any().any() or not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError("Feature matrix has NaN or infinite values")
    if not features["CancelRate"].between(0, 1).all():
        raise ValueError("CancelRate is outside [0,1]")
    if not (features["Recency"] <= features["Tenure"]).all():
        raise ValueError("Recency exceeds Tenure")
    features.index.name = "Customer ID"
    return features


def log_transform_features(features: pd.DataFrame) -> pd.DataFrame:
    """Fixed transform spec, so training and Streamlit always agree."""
    frame = features.loc[:, FEATURE_COLUMNS].astype(float).copy()
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError("Non-finite values in the input features")
    if (frame.loc[:, LOG_FEATURES] < 0).any().any():
        raise ValueError("Log-transformed features must be >= 0")
    frame.loc[:, LOG_FEATURES] = np.log1p(frame.loc[:, LOG_FEATURES])
    return frame


def fit_transform_features(features: pd.DataFrame) -> tuple[pd.DataFrame, RobustScaler]:
    transformed = log_transform_features(features)
    scaler = RobustScaler(quantile_range=(25.0, 75.0))
    values = scaler.fit_transform(transformed)
    scaled = pd.DataFrame(values, index=features.index, columns=FEATURE_COLUMNS)
    scaled.index.name = "Customer ID"
    return scaled, scaler


def transform_features(features: pd.DataFrame, scaler: RobustScaler) -> pd.DataFrame:
    """For the app: log first, then apply the SAVED (never refitted) scaler."""
    transformed = log_transform_features(features)
    scaled = pd.DataFrame(
        scaler.transform(transformed), index=features.index, columns=FEATURE_COLUMNS
    )
    scaled.index.name = "Customer ID"
    return scaled


def feature_statistics(features: pd.DataFrame) -> pd.DataFrame:
    log_features = log_transform_features(features)
    result = pd.DataFrame({
        "SkewBefore": features.skew(numeric_only=True),
        "SkewAfterLog": log_features.skew(numeric_only=True),
        "Median": features.median(numeric_only=True),
        "P25": features.quantile(0.25),
        "P75": features.quantile(0.75),
        "Min": features.min(numeric_only=True),
        "Max": features.max(numeric_only=True),
    }).reindex(FEATURE_COLUMNS)
    result.index.name = "Feature"
    return result


def find_high_correlations(features: pd.DataFrame, threshold: float = 0.90) -> list[dict[str, Any]]:
    """Flag, but do not silently drop, rubric-mandated features."""
    corr = log_transform_features(features).corr(method="spearman")
    pairs = []
    for i, a in enumerate(FEATURE_COLUMNS):
        for b in FEATURE_COLUMNS[i + 1:]:
            value = corr.loc[a, b]
            if pd.notna(value) and abs(value) >= threshold:
                pairs.append({"FeatureA": a, "FeatureB": b, "Spearman": round(float(value), 5)})
    return sorted(pairs, key=lambda p: abs(p["Spearman"]), reverse=True)


def run_pipeline_from_frame(raw: pd.DataFrame, project_root: str | Path) -> dict[str, Any]:
    """Write all TV1 delivery files from the already-loaded two-sheet dataframe."""
    root = Path(project_root).resolve()
    processed = root / "data" / "processed"
    models = root / "models"
    processed.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)

    raw_summary = describe_raw(raw)
    reference_date = get_reference_date(raw)
    cleaned = clean_transactions(raw)
    features = build_customer_features(cleaned.purchases, cleaned.cancellations, reference_date)
    scaled, scaler = fit_transform_features(features)

    features.to_csv(processed / "customers_features.csv", index_label="Customer ID", float_format="%.10g")
    scaled.to_csv(processed / "X_scaled.csv", index_label="Customer ID", float_format="%.12g")
    cleaned.audit.to_csv(processed / "cleaning_audit.csv", index=False)
    cleaned.cancellations.to_csv(processed / "cancelled_transactions.csv", index=False)
    feature_statistics(features).to_csv(processed / "feature_skewness.csv")
    log_transform_features(features).corr(method="spearman").to_csv(
        processed / "feature_correlation_spearman.csv", index_label="Feature"
    )
    log_transform_features(features).corr(method="pearson").to_csv(
        processed / "feature_correlation_pearson.csv", index_label="Feature"
    )
    joblib.dump(scaler, models / "scaler.joblib")

    metadata = {
        **raw_summary,
        "reference_date": reference_date.date().isoformat(),
        "clean_purchases_rows": int(len(cleaned.purchases)),
        "cancellation_rows_retained": int(len(cleaned.cancellations)),
        "cancellation_unique_invoices_retained": int(cleaned.cancellations["Invoice"].nunique()),
        "customers_with_purchases": int(len(features)),
        "customers_cancel_only": int(
            len(set(cleaned.cancellations["Customer ID"].unique()) - set(features.index))
        ),
        "feature_columns": list(FEATURE_COLUMNS),
        "log_features": list(LOG_FEATURES),
        "non_product_codes": sorted(NON_PRODUCT_CODES),
        "non_product_prefixes": list(NON_PRODUCT_PREFIXES),
        "high_correlations": find_high_correlations(features),
        "scaler_type": "sklearn.preprocessing.RobustScaler",
        "scaler_center": {k: float(v) for k, v in zip(FEATURE_COLUMNS, scaler.center_)},
        "scaler_scale": {k: float(v) for k, v in zip(FEATURE_COLUMNS, scaler.scale_)},
        "random_state": RANDOM_STATE,
        "notes": [
            "Cancellation invoices are excluded from purchase RFM but counted in CancelRate.",
            "Customers with cancellations only have no valid RFM profile and are not clustered.",
            "All eight required features are retained; correlated pairs are flagged for TV2/TV3.",
            "CancelRate is not log-transformed because it is bounded in [0, 1].",
            "Reference date is midnight after the last date in the combined RAW workbook.",
            "SourceSheet is metadata and is not part of duplicate detection.",
        ],
    }
    (processed / "pipeline_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # Round-trip checks on disk and model re-use, not just in-memory checks.
    loaded_scaler = joblib.load(models / "scaler.joblib")
    np.testing.assert_allclose(
        transform_features(features, loaded_scaler).to_numpy(), scaled.to_numpy(),
        atol=1e-11, rtol=1e-11,
    )
    if len(pd.read_csv(processed / "customers_features.csv")) != len(scaled):
        raise AssertionError("Saved feature CSV has inconsistent row count")
    print(json.dumps({
        "raw_rows": metadata["raw_rows"],
        "clean_purchases_rows": metadata["clean_purchases_rows"],
        "customers_with_purchases": metadata["customers_with_purchases"],
        "reference_date": metadata["reference_date"],
        "generated_at": str(processed),
    }, ensure_ascii=False, indent=2))
    return metadata


def run_pipeline(raw_path: str | Path, project_root: str | Path) -> dict[str, Any]:
    """Convenient command line entrypoint that loads the workbook first."""
    return run_pipeline_from_frame(load_retail_data(raw_path), project_root)


def main() -> None:
    parser = argparse.ArgumentParser(description="TV1 RFM preprocessing pipeline")
    parser.add_argument("--input", default="data/raw/online_retail_II.xlsx", type=Path)
    parser.add_argument("--root", default=Path(__file__).resolve().parents[1], type=Path)
    args = parser.parse_args()
    run_pipeline(args.input, args.root)


if __name__ == "__main__":
    main()
