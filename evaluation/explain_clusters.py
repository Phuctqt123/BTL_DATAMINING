"""Giải thích mô hình K-Means (k=4) bằng cây đại diện và độ quan trọng đặc trưng.

Chạy từ thư mục gốc repo:
    python -m evaluation.explain_clusters

Các bước:
1. Huấn luyện cây quyết định nông (cây đại diện, surrogate) trên 8 đặc trưng ở
   thang gốc để dự đoán nhãn K-Means; đo độ trung thành (fidelity) bằng
   5-fold stratified CV và trên toàn bộ dữ liệu.
2. Độ quan trọng hoán vị (permutation importance) của từng đặc trưng với cây
   đại diện, và độc lập với cây, độ nhạy của chính K-Means: xáo trộn một cột
   trong X_scaled rồi đo tỷ lệ khách bị đổi cụm (trung bình 10 lần lặp).
3. SHAP cho cây đại diện nếu gói `shap` có sẵn (không tự cài đặt).
4. Xuất luật của cây ra văn bản và lưu hai hình cho báo cáo.

Chọn max_depth=3: cây có 8 lá, mỗi luật tối đa ba điều kiện nên đọc được trực
tiếp; tăng lên độ sâu 4 chỉ thêm khoảng 2 điểm phần trăm độ trung thành nhưng
gấp đôi số lá (16), nên không còn dễ diễn giải.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_text

from evaluation.personas import get_persona
from preprocessing.preprocess import FEATURE_COLUMNS as _FEATURES

FEATURE_COLUMNS = list(_FEATURES)
RANDOM_STATE = 42
SURROGATE_DEPTH = 3
N_REPEATS = 10
CV_FOLDS = 5

ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = ROOT / "data" / "processed" / "customers_features.csv"
SCALED_PATH = ROOT / "data" / "processed" / "X_scaled.csv"
LABELS_PATH = ROOT / "clustering" / "results" / "labels.csv"
KMEANS_PATH = ROOT / "models" / "kmeans.joblib"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
FIGURES_DIR = Path(__file__).resolve().parent / "figures"

IMPORTANCE_PATH = RESULTS_DIR / "feature_importance.csv"
FIDELITY_PATH = RESULTS_DIR / "surrogate_fidelity.json"
RULES_PATH = RESULTS_DIR / "surrogate_rules.txt"

FEATURE_UNITS = {
    "Recency": "ngày", "Frequency": "hóa đơn", "Monetary": "GBP", "AvgOrderValue": "GBP",
    "UniqueProducts": "sản phẩm", "AvgBasketSize": "đơn vị/hóa đơn", "Tenure": "ngày",
    "CancelRate": "",
}


# ----------------------------------------------------------------- dữ liệu
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Trả về (đặc trưng gốc, đặc trưng đã chuẩn hóa, nhãn K-Means), cùng thứ tự khách."""
    def read(path: Path) -> pd.DataFrame:
        return pd.read_csv(path, dtype={"Customer ID": "string"}).set_index("Customer ID")

    features = read(FEATURES_PATH)[FEATURE_COLUMNS]
    scaled = read(SCALED_PATH)[FEATURE_COLUMNS]
    labels = read(LABELS_PATH)["kmeans"].astype(int)
    if not (features.index.equals(scaled.index) and features.index.equals(labels.index)):
        raise ValueError("Thứ tự Customer ID giữa các file đặc trưng và nhãn không khớp.")
    return features, scaled, labels


def persona_names(classes) -> list[str]:
    return [get_persona(int(c))["name"] for c in classes]


# ------------------------------------------------------------ cây đại diện
def fit_surrogate(features: pd.DataFrame, labels: pd.Series,
                  max_depth: int = SURROGATE_DEPTH) -> DecisionTreeClassifier:
    tree = DecisionTreeClassifier(max_depth=max_depth, random_state=RANDOM_STATE)
    return tree.fit(features[FEATURE_COLUMNS], labels)


def surrogate_fidelity(features: pd.DataFrame, labels: pd.Series,
                       depths=(2, 3, 4, 5)) -> dict:
    """Độ trung thành của cây đại diện so với nhãn K-Means ở nhiều độ sâu."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for depth in depths:
        tree = DecisionTreeClassifier(max_depth=depth, random_state=RANDOM_STATE)
        scores = cross_val_score(tree, features[FEATURE_COLUMNS], labels, cv=cv, scoring="accuracy")
        tree.fit(features[FEATURE_COLUMNS], labels)
        rows.append({
            "max_depth": depth, "n_leaves": int(tree.get_n_leaves()),
            "cv_mean": float(scores.mean()), "cv_std": float(scores.std()),
            "full_data": float(tree.score(features[FEATURE_COLUMNS], labels)),
        })
    chosen = next(r for r in rows if r["max_depth"] == SURROGATE_DEPTH)
    tree = fit_surrogate(features, labels)
    pred = tree.predict(features[FEATURE_COLUMNS])
    per_cluster = {
        get_persona(int(c))["name"]: float((pred[labels.to_numpy() == c] == c).mean())
        for c in sorted(labels.unique())
    }
    return {"chosen_depth": SURROGATE_DEPTH, "cv_folds": CV_FOLDS, "random_state": RANDOM_STATE,
            "chosen": chosen, "per_cluster_recall": per_cluster, "by_depth": rows}


def surrogate_rules(tree: DecisionTreeClassifier) -> str:
    text = export_text(tree, feature_names=FEATURE_COLUMNS, decimals=2, show_weights=False)
    names = dict(zip((str(c) for c in tree.classes_), persona_names(tree.classes_)))
    lines = []
    for line in text.splitlines():
        if "class:" in line:
            head, label = line.rsplit("class:", 1)
            line = f"{head}cụm {label.strip()} - {names[label.strip()]}"
        lines.append(line)
    return "\n".join(lines) + "\n"


def _fmt(value: float) -> str:
    """Định dạng số kiểu Việt Nam: dấu chấm phân nhóm nghìn, dấu phẩy thập phân."""
    text = f"{value:,.2f}".rstrip("0").rstrip(".")
    return text.replace(",", "_").replace(".", ",").replace("_", ".")


def decision_path(tree: DecisionTreeClassifier, row: pd.Series) -> list[str]:
    """Đường đi của một khách qua cây đại diện, mỗi phần tử là một điều kiện.

    Phần tử cuối là persona tại lá. Ví dụ: ["Recency ≤ 75 ngày", "Frequency > 5,5 hóa đơn", "VIP"].
    """
    t = tree.tree_
    x = row[FEATURE_COLUMNS].to_numpy(dtype=float)
    node, steps = 0, []
    while t.children_left[node] != t.children_right[node]:
        feat = FEATURE_COLUMNS[t.feature[node]]
        thr = t.threshold[node]
        unit = FEATURE_UNITS.get(feat, "")
        go_left = x[t.feature[node]] <= thr
        sign = "≤" if go_left else ">"
        steps.append(f"{feat} {sign} {_fmt(thr)}{(' ' + unit) if unit else ''}")
        node = t.children_left[node] if go_left else t.children_right[node]
    label = tree.classes_[int(np.argmax(t.value[node][0]))]
    steps.append(get_persona(int(label))["name"])
    return steps


# -------------------------------------------------------- độ quan trọng
def kmeans_shuffle_importance(kmeans, scaled: pd.DataFrame, n_repeats: int = N_REPEATS,
                              random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Tỷ lệ khách bị K-Means đổi cụm khi xáo trộn riêng từng đặc trưng (đã chuẩn hóa)."""
    rng = np.random.default_rng(random_state)
    base_frame = scaled[FEATURE_COLUMNS]
    base = kmeans.predict(base_frame)
    rows = []
    for j, feat in enumerate(FEATURE_COLUMNS):
        shares = []
        for _ in range(n_repeats):
            shuffled = base_frame.copy()
            shuffled[feat] = rng.permutation(shuffled[feat].to_numpy())
            shares.append(float((kmeans.predict(shuffled) != base).mean()))
        rows.append({"feature": feat, "kmeans_change_mean": np.mean(shares),
                     "kmeans_change_std": np.std(shares)})
    return pd.DataFrame(rows).set_index("feature")


def surrogate_permutation_importance(tree, features: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    result = permutation_importance(tree, features[FEATURE_COLUMNS], labels, scoring="accuracy",
                                    n_repeats=N_REPEATS, random_state=RANDOM_STATE)
    return pd.DataFrame({"surrogate_perm_mean": result.importances_mean,
                         "surrogate_perm_std": result.importances_std},
                        index=pd.Index(FEATURE_COLUMNS, name="feature"))


def surrogate_shap_importance(tree, features: pd.DataFrame) -> pd.Series | None:
    """Trung bình |SHAP| của cây đại diện (gộp mọi lớp). Trả về None nếu thiếu gói shap."""
    try:
        import shap  # noqa: PLC0415
    except ImportError:
        return None
    values = shap.TreeExplainer(tree).shap_values(features[FEATURE_COLUMNS])
    arr = np.abs(np.stack(values, axis=-1) if isinstance(values, list) else np.asarray(values))
    # arr: (n_mẫu, n_đặc_trưng, n_lớp)
    return pd.Series(arr.mean(axis=(0, 2)), index=pd.Index(FEATURE_COLUMNS, name="feature"),
                     name="surrogate_shap_mean_abs")


# --------------------------------------------------------------- hình vẽ
def plot_importance(importance: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = importance.sort_values("kmeans_change_mean").index
    y = np.arange(len(order))
    h = 0.38
    fig, ax = plt.subplots(figsize=(6.2, 2.5), facecolor="white")
    ax.barh(y + h / 2, importance.loc[order, "kmeans_change_mean"], h,
            xerr=importance.loc[order, "kmeans_change_std"], color="#1f6feb",
            label="K-Means: tỷ lệ khách đổi cụm khi xáo trộn", error_kw={"lw": 0.8})
    ax.barh(y - h / 2, importance.loc[order, "surrogate_perm_mean"], h,
            xerr=importance.loc[order, "surrogate_perm_std"], color="#d29922",
            label="Cây đại diện: mức giảm độ chính xác (hoán vị)", error_kw={"lw": 0.8})

    def label(v: float) -> str:
        return "0" if abs(v) < 5e-4 else f"{v:.3f}".replace(".", ",")

    for yi, feat in zip(y, order):
        for col, dy in (("kmeans_change_mean", h / 2), ("surrogate_perm_mean", -h / 2)):
            ax.text(importance.loc[feat, col] + 0.008, yi + dy, label(importance.loc[feat, col]),
                    va="center", fontsize=8)
    ax.set_yticks(y, order)
    ax.set_xlabel("Độ quan trọng (tỷ lệ, trung bình 10 lần xáo trộn)", fontsize=9)
    ax.tick_params(labelsize=9)
    ax.set_xlim(0, max(importance["kmeans_change_mean"].max(),
                       importance["surrogate_perm_mean"].max()) * 1.18)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.2f}".replace(".", ","))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


PERSONA_COLORS = {"VIP": "#1f6feb", "Tiềm năng": "#2da44e",
                  "Cần níu kéo": "#d29922", "Ngủ đông": "#8c959f"}


def plot_surrogate_tree(tree: DecisionTreeClassifier, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb
    from sklearn.tree import plot_tree

    names = persona_names(tree.classes_)
    fig, ax = plt.subplots(figsize=(11, 4.6), facecolor="white")
    annotations = plot_tree(tree, feature_names=FEATURE_COLUMNS, class_names=names,
                            filled=True, impurity=False, proportion=True, rounded=True,
                            precision=2, fontsize=9, ax=ax)
    # Rút gọn nhãn nút: bỏ mảng value, thay bằng độ thuần của lớp đa số; tô màu theo persona.
    for ann in annotations:
        text = ann.get_text()
        if text.strip() in ("True", "False"):  # nhãn nhánh mặc định của plot_tree
            ann.set_text(text.replace("True", "Đúng").replace("False", "Sai"))
            continue
        if "value" not in text:
            continue
        lines = text.split("\n")
        values = [float(v) for v in re.findall(r"[\d.]+", next(l for l in lines if l.startswith("value")))]
        persona = next(l for l in lines if l.startswith("class")).split("=", 1)[1].strip()
        out = []
        for line in lines:
            if line.startswith("value"):
                continue
            if line.startswith("samples"):
                line = "khách: " + line.split("=", 1)[1].strip()
            elif line.startswith("class"):
                line = f"{persona} ({max(values):.0%})"
            out.append(re.sub(r"(\d)\.(\d)", r"\1,\2", line.replace("<=", "≤")))
        ann.set_text("\n".join(out))
        purity = max(values)
        rgb = np.array(to_rgb(PERSONA_COLORS.get(persona, "#cccccc")))
        ann.get_bbox_patch().set_facecolor(tuple(1 - (1 - rgb) * (0.25 + 0.5 * purity)))
    ax.set_title(f"Cây đại diện (độ sâu {tree.get_depth()}) mô phỏng nhãn K-Means; "
                 "trong ngoặc: tỷ lệ khách của lớp đa số tại nút", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


# ------------------------------------------------------------------ chạy
def run() -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    features, scaled, labels = load_data()
    kmeans = joblib.load(KMEANS_PATH)
    if (kmeans.predict(scaled[FEATURE_COLUMNS]) != labels.to_numpy()).any():
        raise ValueError("models/kmeans.joblib không tái tạo được nhãn trong labels.csv.")

    fidelity = surrogate_fidelity(features, labels)
    tree = fit_surrogate(features, labels)

    importance = surrogate_permutation_importance(tree, features, labels).join(
        kmeans_shuffle_importance(kmeans, scaled))
    shap_imp = surrogate_shap_importance(tree, features)
    if shap_imp is not None:
        importance = importance.join(shap_imp)
    importance["surrogate_gini_importance"] = tree.feature_importances_
    importance["rank_kmeans"] = importance["kmeans_change_mean"].rank(ascending=False, method="min").astype(int)
    importance["rank_surrogate"] = importance["surrogate_perm_mean"].rank(ascending=False, method="min").astype(int)
    importance = importance.sort_values("kmeans_change_mean", ascending=False)
    importance.round(6).to_csv(IMPORTANCE_PATH)

    fidelity["shap_available"] = shap_imp is not None
    FIDELITY_PATH.write_text(json.dumps(fidelity, ensure_ascii=False, indent=2), encoding="utf-8")
    RULES_PATH.write_text(surrogate_rules(tree), encoding="utf-8")
    plot_importance(importance, FIGURES_DIR / "feature_importance.png")
    plot_surrogate_tree(tree, FIGURES_DIR / "surrogate_tree.png")
    return {"fidelity": fidelity, "importance": importance}


def main() -> None:
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # console Windows mặc định cp1252
    out = run()
    fid = out["fidelity"]
    c = fid["chosen"]
    print(f"Cây đại diện max_depth={fid['chosen_depth']} ({c['n_leaves']} lá): "
          f"độ trung thành CV = {c['cv_mean']:.4f} ± {c['cv_std']:.4f}, toàn bộ dữ liệu = {c['full_data']:.4f}")
    for r in fid["by_depth"]:
        print(f"  depth {r['max_depth']}: CV {r['cv_mean']:.4f}, full {r['full_data']:.4f}, lá {r['n_leaves']}")
    print("Recall theo cụm:", {k: round(v, 4) for k, v in fid["per_cluster_recall"].items()})
    print("SHAP:", "có" if fid["shap_available"] else "bỏ qua (chưa cài gói shap)")
    print(out["importance"].round(4).to_string())
    print(RULES_PATH.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
