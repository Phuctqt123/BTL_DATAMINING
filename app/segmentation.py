"""Phân khúc dữ liệu mới bằng đúng pipeline của nhóm:
TV1 (làm sạch + đặc trưng + scaler đã lưu) -> TV2 (K-Means đã lưu) -> TV3 (persona)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

from evaluation.personas import assign_personas
from preprocessing.preprocess import (
    FEATURE_COLUMNS, build_customer_features, clean_transactions,
    get_reference_date, normalize_columns, transform_features,
)

ROOT = Path(__file__).resolve().parents[1]
SCALER_PATH = ROOT / "models" / "scaler.joblib"
KMEANS_PATH = ROOT / "models" / "kmeans.joblib"
SEGMENTS_PATH = ROOT / "data" / "processed" / "customers_segments.csv"
RAW_COLUMNS = ["Invoice", "StockCode", "Description", "Quantity",
               "InvoiceDate", "Price", "Customer ID", "Country"]


@dataclass
class Artifacts:
    scaler: object
    kmeans: object
    segments: pd.DataFrame


def load_artifacts() -> Artifacts:
    missing = [p for p in (SCALER_PATH, KMEANS_PATH, SEGMENTS_PATH) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Thiếu " + ", ".join(str(p.relative_to(ROOT)) for p in missing)
            + ". Chạy `python -m preprocessing.run_all` rồi `python -m app.build_artifacts`."
        )
    segments = pd.read_csv(SEGMENTS_PATH, dtype={"Customer ID": "string"}).set_index("Customer ID")
    return Artifacts(joblib.load(SCALER_PATH), joblib.load(KMEANS_PATH), segments)


def predict_features(features: pd.DataFrame, art: Artifacts) -> pd.DataFrame:
    """features: 8 cột FEATURE_COLUMNS, index là Customer ID."""
    scaled = transform_features(features.loc[:, FEATURE_COLUMNS], art.scaler)
    out = features.loc[:, FEATURE_COLUMNS].copy()
    out["Cluster"] = art.kmeans.predict(scaled)
    out["Persona"] = assign_personas(out["Cluster"].tolist())
    return out


def detect_input_type(df: pd.DataFrame) -> str:
    cols = set(df.columns)
    if set(FEATURE_COLUMNS) <= cols:
        return "features"
    try:
        normalize_columns(df.head(5))
        return "transactions"
    except ValueError:
        return "unknown"


def segment_transactions(raw: pd.DataFrame, art: Artifacts,
                         reference_date: pd.Timestamp | None = None
                         ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Giao dịch thô (cùng 8 cột với Online Retail II) -> phân khúc từng khách.

    Trả về (kết quả theo khách, nhật ký làm sạch, ngày mốc đã dùng)."""
    normalized = normalize_columns(raw)
    cleaned = clean_transactions(normalized)
    ref = pd.Timestamp(reference_date) if reference_date is not None else get_reference_date(normalized)
    features = build_customer_features(cleaned.purchases, cleaned.cancellations, ref)
    return predict_features(features, art), cleaned.audit, ref


def segment_feature_table(df: pd.DataFrame, art: Artifacts) -> pd.DataFrame:
    frame = df.copy()
    if "Customer ID" in frame.columns:
        frame["Customer ID"] = frame["Customer ID"].astype("string").str.replace(r"\.0+$", "", regex=True)
        frame = frame.set_index("Customer ID")
    return predict_features(frame, art)


def read_upload(name: str, data) -> pd.DataFrame:
    if name.lower().endswith((".xlsx", ".xls")):
        sheets = pd.read_excel(data, sheet_name=None)
        return pd.concat(sheets.values(), ignore_index=True)
    return pd.read_csv(data, encoding_errors="replace")
