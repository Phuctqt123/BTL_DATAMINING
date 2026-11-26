"""Chuẩn bị các file app cần đọc, dùng lại code của TV1, TV2, TV3.

Chạy từ thư mục gốc repo, sau khi đã chạy `python -m preprocessing.run_all`:
    python -m app.build_artifacts

Tạo ra:
    models/kmeans.joblib                  K-Means k=4 (cùng tham số TV2 đã chọn)
    data/processed/customers_segments.csv đặc trưng + nhãn cụm + persona + cờ ngoại biên DBSCAN
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from clustering.models import (
    CUSTOMER_ID, DATA_DIR, RAW_FEATURES_PATH, RESULTS_DIR,
    fit_kmeans, load_customer_features, load_scaled_data, save_model,
)
from evaluation.personas import assign_personas

N_CLUSTERS = 4
SEGMENTS_PATH = DATA_DIR / "customers_segments.csv"


def main() -> None:
    customer_ids, X = load_scaled_data()
    model, labels = fit_kmeans(X, n_clusters=N_CLUSTERS)

    # Nhãn phải trùng với labels.csv TV2 đã bàn giao, vì personas.json của TV3
    # ánh xạ theo đúng số thứ tự cụm này.
    delivered = pd.read_csv(RESULTS_DIR / "labels.csv", dtype={CUSTOMER_ID: "string"})
    delivered[CUSTOMER_ID] = delivered[CUSTOMER_ID].str.strip()
    if not delivered[CUSTOMER_ID].equals(customer_ids.astype("string")):
        raise AssertionError("Thứ tự khách trong labels.csv khác X_scaled.csv")
    if not np.array_equal(delivered["kmeans"].to_numpy(), labels):
        raise AssertionError("Nhãn K-Means khác labels.csv, cần kiểm tra lại dữ liệu đầu vào")

    save_model(model, "kmeans.joblib")

    features = load_customer_features(RAW_FEATURES_PATH)
    segments = features.copy()
    segments["Cluster"] = labels
    segments["Persona"] = assign_personas(labels)
    segments["DBSCAN_Outlier"] = delivered["dbscan"].to_numpy() == -1
    segments.to_csv(SEGMENTS_PATH, index_label=CUSTOMER_ID)

    print(f"Đã lưu {SEGMENTS_PATH} ({len(segments):,} khách)")
    print(segments["Persona"].value_counts().to_string())


if __name__ == "__main__":
    main()
