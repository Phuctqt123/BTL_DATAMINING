"""
Module dùng chung cho phần gom cụm (TV2 - Clustering).

Chức năng chính:
1. Đọc dữ liệu đã chuẩn hóa từ TV1 (hoặc tạo dữ liệu giả lập để test khung khi chưa có data thật).
2. Cung cấp hàm fit 3 thuật toán: K-Means, Hierarchical (Agglomerative), DBSCAN.
3. Chạy lưới tham số (ParameterGrid) và chấm các chỉ số nội bộ (Silhouette, Davies-Bouldin, Calinski-Harabasz).
4. Lưu bảng kết quả tinh chỉnh, file nhãn labels.csv cho TV3 và model kmeans.joblib cho TV4.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import RobustScaler

# Cố định seed theo quy ước của nhóm để kết quả tái lập được trên mọi máy
RANDOM_STATE = 42

# Định nghĩa các đường dẫn chuẩn theo cấu trúc thư mục repo
CLUSTERING_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CLUSTERING_DIR.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "X_scaled.csv"
RESULTS_DIR = CLUSTERING_DIR / "results"
MODELS_DIR = PROJECT_ROOT / "models"

# Danh sách đặc trưng dự kiến theo preprocessing/README.md (chỉ dùng cho dữ liệu giả lập)
DEFAULT_FEATURE_NAMES = [
    "Recency",
    "Frequency",
    "Monetary",
    "AvgOrderValue",
    "UniqueProducts",
    "AvgBasketSize",
    "Tenure",
    "CancelRate",
]


def make_mock_data(
    n_samples: int = 1200,
    n_features: int = 8,
    n_outliers: int = 40,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Tạo dữ liệu giả lập đã chuẩn hóa (gồm 4 cụm + một ít điểm ngoại biên)
    để chạy thử toàn bộ notebook trong thời gian chờ TV1 bàn giao X_scaled.csv.
    """
    rng = np.random.default_rng(random_state)

    # Tạo 4 cụm chính mô phỏng 4 phân khúc khách hàng
    X_blobs, _ = make_blobs(
        n_samples=n_samples - n_outliers,
        n_features=n_features,
        centers=4,
        cluster_std=1.05,
        center_box=(-6.0, 6.0),
        random_state=random_state,
    )

    # Thêm một ít điểm ngoại biên rải xa tâm cụm để test khả năng lọc nhiễu của DBSCAN
    X_outliers = rng.uniform(low=-10.0, high=10.0, size=(n_outliers, n_features))
    X_raw = np.vstack([X_blobs, X_outliers])

    # Xáo trộn thứ tự các dòng và chuẩn hóa bằng RobustScaler cho giống đầu ra của TV1
    perm = rng.permutation(len(X_raw))
    X_scaled = RobustScaler().fit_transform(X_raw[perm])

    cols = (
        DEFAULT_FEATURE_NAMES
        if n_features == len(DEFAULT_FEATURE_NAMES)
        else [f"feature_{i + 1}" for i in range(n_features)]
    )
    customer_ids = pd.Series(
        np.arange(10001, 10001 + len(X_scaled)), name="Customer ID"
    )
    X_df = pd.DataFrame(X_scaled, columns=cols)
    return customer_ids, X_df


def load_scaled_data(
    path: Path | str = DEFAULT_DATA_PATH,
    allow_mock: bool = True,
) -> tuple[pd.Series, pd.DataFrame, bool]:
    """
    Đọc bảng đặc trưng đã chuẩn hóa X_scaled.csv do TV1 bàn giao.

    Trả về:
        customer_ids: Series chứa mã khách hàng (Customer ID).
        X_df: DataFrame chỉ gồm các cột đặc trưng dạng số để đưa vào mô hình.
        is_mock: True nếu đang chạy dữ liệu giả lập, False nếu đọc từ file thật.
    """
    data_path = Path(path)

    if not data_path.exists():
        if not allow_mock:
            raise FileNotFoundError(
                f"Không tìm thấy dữ liệu tại '{data_path}'. "
                "Hãy đặt file X_scaled.csv của TV1 vào đúng thư mục."
            )
        print(
            f"[CẢNH BÁO] Chưa có file '{data_path}'.\n"
            "-> Đang dùng DỮ LIỆU GIẢ LẬP để chạy thử khung. "
            "Khi TV1 bàn giao X_scaled.csv, chỉ cần chạy lại (Restart & Run All)."
        )
        customer_ids, X_df = make_mock_data()
        return customer_ids, X_df, True

    df = pd.read_csv(data_path)

    # Linh hoạt nhận diện cột Customer ID (kể cả khi TV1 lưu dưới dạng index Unnamed: 0)
    id_candidates = ["Customer ID", "CustomerID", "customer_id", "Unnamed: 0"]
    id_col = next((col for col in id_candidates if col in df.columns), None)

    if id_col is not None:
        customer_ids = df[id_col].rename("Customer ID")
        feature_df = df.drop(columns=[id_col])
    else:
        customer_ids = pd.Series(df.index, name="Customer ID")
        feature_df = df

    # Chỉ giữ các cột số để tránh lỗi nếu file có cột phụ dạng chữ
    X_df = feature_df.select_dtypes(include=[np.number]).copy()
    if X_df.empty:
        raise ValueError(f"File '{data_path}' không có cột đặc trưng dạng số nào.")

    if X_df.isna().any().any() or np.isinf(X_df.to_numpy()).any():
        raise ValueError(
            f"Dữ liệu '{data_path}' còn giá trị NaN hoặc Inf. Cần kiểm tra lại bước tiền xử lý."
        )

    print(
        f"[OK] Đã đọc dữ liệu thật: {X_df.shape[0]} khách hàng, "
        f"{X_df.shape[1]} đặc trưng ({list(X_df.columns)})."
    )
    return customer_ids, X_df, False


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

    Với DBSCAN, đặt ignore_noise=True để bỏ điểm nhiễu (nhãn -1) khi tính 3 chỉ số,
    nhưng vẫn giữ lại tỷ lệ nhiễu (noise_ratio) trên toàn tập.
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

    unique_clusters = set(labels_eval) - {-1}
    n_clusters = len(unique_clusters)

    # Các chỉ số nội bộ chỉ tính được khi có ít nhất 2 cụm và số điểm > số cụm
    if n_clusters < 2 or len(labels_eval) <= n_clusters:
        return {
            "n_clusters": n_clusters,
            "noise_ratio": round(noise_ratio, 4),
            "silhouette": np.nan,
            "davies_bouldin": np.nan,
            "calinski_harabasz": np.nan,
        }

    return {
        "n_clusters": n_clusters,
        "noise_ratio": round(noise_ratio, 4),
        "silhouette": round(float(silhouette_score(X_eval, labels_eval)), 4),
        "davies_bouldin": round(float(davies_bouldin_score(X_eval, labels_eval)), 4),
        "calinski_harabasz": round(
            float(calinski_harabasz_score(X_eval, labels_eval)), 2
        ),
    }


def run_grid_search(
    X: pd.DataFrame | np.ndarray,
    model_builder: Callable[..., Any],
    param_grid: dict[str, list[Any]],
    ignore_noise: bool = False,
) -> pd.DataFrame:
    """
    Thay thế GridSearchCV (vốn dành cho học có giám sát) bằng vòng lặp ParameterGrid
    kết hợp bộ chỉ số nội bộ để tinh chỉnh siêu tham số cho thuật toán gom cụm.
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
            row["inertia"] = round(float(model.inertia_), 2)

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
    print(f"[SAVED] Đã lưu bảng tinh chỉnh -> {out_path}")
    return out_path


def save_cluster_labels(
    customer_ids: pd.Series,
    algo_name: str,
    labels: np.ndarray,
    output_path: Path | str = RESULTS_DIR / "labels.csv",
) -> Path:
    """
    Cập nhật cột nhãn của một thuật toán vào file clustering/results/labels.csv.
    Giúp 3 notebook cập nhật độc lập cột 'kmeans', 'hierarchical', 'dbscan'
    mà không ghi đè mất kết quả của nhau.
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    new_ids = pd.Series(customer_ids).reset_index(drop=True)
    new_labels = pd.Series(labels, name=algo_name)

    if out_path.exists():
        existing_df = pd.read_csv(out_path)
        same_length = len(existing_df) == len(new_ids)
        same_ids = (
            same_length
            and "Customer ID" in existing_df.columns
            and (
                existing_df["Customer ID"].astype(str).values
                == new_ids.astype(str).values
            ).all()
        )
        if same_ids:
            existing_df[algo_name] = new_labels
            out_df = existing_df
        else:
            print(
                "[THÔNG BÁO] Danh sách Customer ID thay đổi (ví dụ chuyển từ mock sang data thật) "
                "-> tạo mới labels.csv."
            )
            out_df = pd.DataFrame({"Customer ID": new_ids, algo_name: new_labels})
    else:
        out_df = pd.DataFrame({"Customer ID": new_ids, algo_name: new_labels})

    out_df.to_csv(out_path, index=False)
    print(f"[SAVED] Đã cập nhật cột '{algo_name}' -> {out_path}")
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
    print(f"[SAVED] Đã lưu model -> {out_path}")
    return out_path
