"""
Module dùng chung cho phần gom cụm (TV2 - Clustering).

Chức năng chính:
1. Đọc dữ liệu đã chuẩn hóa từ TV1 (X_scaled.csv và customers_features.csv).
2. Cung cấp hàm fit 3 thuật toán: K-Means, Hierarchical (Agglomerative), DBSCAN.
3. Chạy lưới tham số (ParameterGrid) và chấm các chỉ số nội bộ (Silhouette, Davies-Bouldin, Calinski-Harabasz).
4. Lưu bảng kết quả tinh chỉnh, file nhãn labels.csv cho TV3 và model kmeans.joblib cho TV4.
"""

from __future__ import annotations

import time
from os import getenv
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.model_selection import KFold, ParameterGrid

from preprocessing.preprocess import (
    FEATURE_COLUMNS,
    fit_transform_features,
    transform_features,
)

RANDOM_STATE = 42

CLUSTERING_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CLUSTERING_DIR.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed"
DATA_DIR = PROJECT_ROOT / Path(DEFAULT_DATA_DIR).expanduser()
DEFAULT_DATA_PATH = DATA_DIR / "X_scaled.csv"
RAW_FEATURES_PATH = DEFAULT_DATA_PATH.with_name("customers_features.csv")
RESULTS_DIR = CLUSTERING_DIR / "results"
MODELS_DIR = PROJECT_ROOT / "models"
CUSTOMER_ID = "Customer ID"


def _display_path(path: Path | str) -> str:
    resolved = Path(path).resolve()
    if resolved.is_relative_to(PROJECT_ROOT):
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    return resolved.name


def load_scaled_data(
    path: Path | str = DEFAULT_DATA_PATH,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Đọc bảng đặc trưng đã chuẩn hóa X_scaled.csv do TV1 bàn giao.

    Trả về:
        customer_ids: Series chứa mã khách hàng (Customer ID).
        X_df: DataFrame chỉ gồm các cột đặc trưng dạng số để đưa vào mô hình.
    """
    data_path = Path(path)
    features = load_customer_features(data_path)
    print(f"Đã đọc {len(features):,} khách hàng từ {_display_path(data_path)}")
    customer_ids = features.index.to_series().reset_index(drop=True)
    return customer_ids, features.reset_index(drop=True)


def load_customer_features(path: Path | str) -> pd.DataFrame:
    """Đọc bảng đặc trưng TV1 và đưa các biến về đúng thứ tự FEATURE_COLUMNS."""
    if not Path(path).is_file():
        raise FileNotFoundError(f"Không tìm thấy '{_display_path(path)}'. Kiểm tra tệp TV1.")
    frame = pd.read_csv(path, dtype={CUSTOMER_ID: "string"})
    customer_ids = frame[CUSTOMER_ID].str.strip()
    features = frame.loc[:, FEATURE_COLUMNS]
    return features.set_axis(pd.Index(customer_ids, name=CUSTOMER_ID))


def cross_validate_kmeans(features: pd.DataFrame, k_values: list[int]) -> pd.DataFrame:
    """Đo lỗi gán tâm ngoài mẫu, fit bộ chuẩn hóa riêng trong mỗi fold train."""
    folds = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    records = []
    for fold, (train_index, test_index) in enumerate(folds.split(features), start=1):
        X_train, scaler = fit_transform_features(features.iloc[train_index])
        X_test = transform_features(features.iloc[test_index], scaler)
        for k in k_values:
            model, _ = fit_kmeans(X_train, n_clusters=k)
            records.append({
                "fold": fold,
                "n_clusters": k,
                "test_wcss_per_customer": -model.score(X_test) / len(X_test),
            })
    return pd.DataFrame(records)


def evaluate_clustering(
    X: pd.DataFrame | np.ndarray,
    labels: np.ndarray,
    ignore_noise: bool = False,
) -> dict[str, float | int]:
    """
    Tính bộ 3 chỉ số đánh giá nội bộ cho bài toán gom cụm không nhãn:
    - Silhouette Score: càng gần 1 càng tốt (tách biệt rõ, gọn cụm).
    - Davies-Bouldin Index: càng nhỏ càng tốt (gần 0 là tốt).
    - Calinski-Harabasz Index: càng cao càng tốt.
    """
    X_arr = np.asarray(X)
    labels_arr = np.asarray(labels)

    noise_mask = labels_arr == -1
    noise_ratio = float(np.mean(noise_mask))

    if ignore_noise:
        valid_mask = ~noise_mask
        X_eval = X_arr[valid_mask]
        labels_eval = labels_arr[valid_mask]
    else:
        X_eval = X_arr
        labels_eval = labels_arr

    _, cluster_sizes = np.unique(labels_arr[~noise_mask], return_counts=True)
    n_clusters = len(cluster_sizes)
    stats = {
        "n_clusters": n_clusters,
        "n_noise": int(noise_mask.sum()),
        "noise_ratio": noise_ratio,
        "min_cluster_size": int(cluster_sizes.min()) if n_clusters else 0,
        "max_cluster_size": int(cluster_sizes.max()) if n_clusters else 0,
    }

    # Các chỉ số nội bộ chỉ tính được khi có ít nhất 2 cụm và số điểm > số cụm
    if n_clusters < 2 or len(labels_eval) <= len(np.unique(labels_eval)):
        return {
            **stats,
            "silhouette": np.nan,
            "davies_bouldin": np.nan,
            "calinski_harabasz": np.nan,
        }

    return {
        **stats,
        "silhouette": float(silhouette_score(X_eval, labels_eval)),
        "davies_bouldin": float(davies_bouldin_score(X_eval, labels_eval)),
        "calinski_harabasz": float(calinski_harabasz_score(X_eval, labels_eval)),
    }


def run_grid_search(
    X: pd.DataFrame | np.ndarray,
    model_builder: Callable[..., Any],
    param_grid: dict[str, list[Any]],
    ignore_noise: bool = False,
) -> pd.DataFrame:
    """
    Quét tham số và tính các chỉ số nội bộ trên tập dữ liệu đem gom cụm.
    Đánh giá K-fold cho K-means được thực hiện riêng trên các fold giữ lại.
    """
    records: list[dict[str, Any]] = []

    for params in ParameterGrid(param_grid):
        t0 = time.perf_counter()
        model = model_builder(**params)
        labels = model.fit_predict(X)
        elapsed = time.perf_counter() - t0

        metrics = evaluate_clustering(X, labels, ignore_noise=ignore_noise)
        row: dict[str, Any] = {**params, **metrics, "fit_time_sec": round(elapsed, 3)}

        # Nếu là KMeans thì lưu thêm inertia để vẽ biểu đồ Elbow
        if hasattr(model, "inertia_"):
            row["inertia"] = float(model.inertia_)

        records.append(row)

    return pd.DataFrame(records)


def fit_kmeans(
    X: pd.DataFrame | np.ndarray,
    n_clusters: int = 4,
    init: str = "k-means++",
    n_init: int = 10,
    random_state: int = RANDOM_STATE,
) -> tuple[KMeans, np.ndarray]:
    """Huấn luyện mô hình K-Means với bộ tham số đã chọn."""
    model = KMeans(
        n_clusters=n_clusters,
        init=init,
        n_init=n_init,
        random_state=random_state,
    )
    labels = model.fit_predict(X)
    return model, labels


def fit_hierarchical(
    X: pd.DataFrame | np.ndarray,
    n_clusters: int = 4,
    linkage: str = "ward",
    metric: str = "euclidean",
) -> tuple[AgglomerativeClustering, np.ndarray]:
    """Huấn luyện mô hình Agglomerative Hierarchical Clustering."""
    # Theo ràng buộc của scikit-learn, linkage='ward' bắt buộc đi cùng metric='euclidean'
    used_metric = "euclidean" if linkage == "ward" else metric
    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage=linkage,
        metric=used_metric,
    )
    labels = model.fit_predict(X)
    return model, labels


def fit_dbscan(
    X: pd.DataFrame | np.ndarray,
    eps: float = 0.5,
    min_samples: int = 5,
    metric: str = "euclidean",
) -> tuple[DBSCAN, np.ndarray]:
    """Huấn luyện mô hình DBSCAN (nhãn -1 là điểm nhiễu / ngoại biên)."""
    model = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
    labels = model.fit_predict(X)
    return model, labels


def save_tuning_results(
    df: pd.DataFrame,
    filename: str,
    results_dir: Path | str = RESULTS_DIR,
) -> Path:
    """Lưu bảng kết quả tinh chỉnh siêu tham số vào clustering/results/."""
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    df.to_csv(out_path, index=False)
    print(f"[SAVED] Đã lưu bảng tinh chỉnh -> {_display_path(out_path)}")
    return out_path


def save_cluster_labels(
    customer_ids: pd.Series,
    algo_name: str,
    labels: np.ndarray,
    output_path: Path | str = RESULTS_DIR / "labels.csv",
) -> Path:
    """Cập nhật nhãn theo mã khách, giữ nguyên các cột của thuật toán khác."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    new_ids = pd.Series(customer_ids, dtype="string").str.strip()
    if len(new_ids) != len(labels):
        raise ValueError("Số nhãn phải bằng số Customer ID.")
    new_labels = pd.Series(labels, index=pd.Index(new_ids, name=CUSTOMER_ID), name=algo_name)

    if out_path.exists():
        out_df = pd.read_csv(out_path, dtype={CUSTOMER_ID: "string"})
        out_df = out_df.set_index(out_df.pop(CUSTOMER_ID).str.strip())
        if set(out_df.index) != set(new_labels.index):
            raise ValueError(
                f"Tập Customer ID đã đổi. Hãy chuyển {_display_path(out_path)} "
                "sang một thư mục lưu kết quả cũ."
            )
        out_df[algo_name] = new_labels
    else:
        out_df = new_labels.to_frame()

    with TemporaryDirectory(dir=out_path.parent) as temp_dir:
        temporary_path = Path(temp_dir) / out_path.name
        out_df.to_csv(temporary_path, index_label=CUSTOMER_ID)
        temporary_path.replace(out_path)
    print(f"[SAVED] Đã cập nhật cột '{algo_name}' -> {_display_path(out_path)}")
    return out_path


def save_model(
    model: Any,
    filename: str = "kmeans.joblib",
    models_dir: Path | str = MODELS_DIR,
) -> Path:
    """Lưu mô hình đã huấn luyện bằng joblib (phục vụ app Streamlit của TV4)."""
    out_dir = Path(models_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    joblib.dump(model, out_path)
    print(f"[SAVED] Đã lưu model -> {_display_path(out_path)}")
    return out_path
