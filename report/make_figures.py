"""Vẽ lại các hình minh họa của báo cáo với nhãn tiếng Việt.

Chạy từ thư mục gốc repo: python report/make_figures.py
Ảnh được ghi đè vào report/graphics/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "report" / "graphics"

BLUE, LIGHT, ACCENT, GREY = "#1F4E79", "#9DC3E6", "#F9B800", "#8C8C8C"
PERSONA = {0: "VIP", 1: "Tiềm năng", 2: "Cần níu kéo", 3: "Ngủ đông"}
PCOLOR = {0: "#E8A202", 1: "#2E86AB", 2: "#C0392B", 3: "#7F8C8D"}
FEAT_VI = {
    "Recency": "Recency", "Frequency": "Frequency", "Monetary": "Monetary",
    "AvgOrderValue": "AvgOrderValue", "UniqueProducts": "UniqueProducts",
    "AvgBasketSize": "AvgBasketSize", "Tenure": "Tenure", "CancelRate": "CancelRate",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False,
    "axes.spines.right": False, "axes.titleweight": "bold", "axes.titlesize": 12,
})


def vn(x: float, d: int = 0) -> str:
    """Định dạng số kiểu Việt Nam: dấu chấm phân nghìn, dấu phẩy thập phân."""
    s = f"{x:,.{d}f}"
    return s.replace(",", " ").replace(".", ",").replace(" ", ".")


def save(fig, name):
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", name)


raw = pd.read_csv(ROOT / "data/raw/online_retail_II.csv",
                  dtype={"Invoice": str, "StockCode": str, "Customer ID": str})

# 2.1 Tỷ lệ giá trị thiếu theo trường
miss = raw.isna().mean().mul(100)
fig, ax = plt.subplots(figsize=(7.5, 3.6))
bars = ax.bar(miss.index, miss.values, color=[ACCENT if v > 1 else BLUE for v in miss.values])
for b, v, n in zip(bars, miss.values, raw.isna().sum().values):
    if n:
        ax.text(b.get_x() + b.get_width() / 2, v + 0.6, f"{vn(v, 2)}%\n({vn(n)} dòng)",
                ha="center", fontsize=9)
ax.set_ylabel("Tỷ lệ thiếu (%)"); ax.set_ylim(0, miss.max() * 1.3)
ax.tick_params(axis="x", rotation=20)
save(fig, "c2_missing.png")

# 2.2 Năm quốc gia có nhiều dòng giao dịch nhất (trục log)
top = raw["Country"].value_counts().head(5)[::-1]
names = {"United Kingdom": "Vương quốc Anh", "EIRE": "Ireland (EIRE)", "Germany": "Đức",
         "France": "Pháp", "Netherlands": "Hà Lan"}
fig, ax = plt.subplots(figsize=(7.5, 3.2))
ax.barh([names.get(c, c) for c in top.index], top.values,
        color=[ACCENT if c == "United Kingdom" else BLUE for c in top.index])
ax.set_xscale("log"); ax.set_xlim(1e3, 4e6)
for i, v in enumerate(top.values):
    ax.text(v * 1.08, i, f"{vn(v)} ({vn(v / len(raw) * 100, 1)}%)", va="center", fontsize=10)
ax.set_xlabel("Số dòng giao dịch (thang logarit)")
save(fig, "c2_countries.png")

# 2.3 Tổng giá trị mua hợp lệ theo tháng
from preprocessing.preprocess import clean_transactions, normalize_columns
purchases = clean_transactions(normalize_columns(raw)).purchases
monthly = purchases.groupby(purchases["InvoiceDate"].dt.to_period("M"))["LineTotal"].sum() / 1e6
fig, ax = plt.subplots(figsize=(9, 3.6))
xs = np.arange(len(monthly))
ax.fill_between(xs, monthly.values, color=LIGHT, alpha=0.5)
ax.plot(xs, monthly.values, color=BLUE, lw=2, marker="o", ms=4)
ticks = [i for i, p in enumerate(monthly.index) if p.month in (1, 4, 7, 10)]
ax.set_xticks(ticks, [f"{monthly.index[i].month:02d}/{monthly.index[i].year}" for i in ticks])
ax.set_ylabel("Doanh thu (triệu GBP)"); ax.grid(axis="y", alpha=0.3)
pk = int(np.argmax(monthly.values))
ax.annotate(f"Cao nhất {monthly.index[pk].month:02d}/{monthly.index[pk].year}: "
            f"{vn(monthly.values[pk], 2)} triệu GBP",
            xy=(pk, monthly.values[pk]), xytext=(pk - 9.5, monthly.values[pk] * 0.98),
            fontsize=10, arrowprops=dict(arrowstyle="->", color=BLUE))
save(fig, "c2_monthly_sales.png")

# 3.1 Độ lệch trước và sau log
sk = pd.read_csv(ROOT / "data/processed/feature_skewness.csv")
fig, ax = plt.subplots(figsize=(8.5, 3.8))
x = np.arange(len(sk)); w = 0.38
ax.bar(x - w / 2, sk["SkewBefore"], w, color=GREY, label="Trước biến đổi")
ax.bar(x + w / 2, sk["SkewAfterLog"], w, color=BLUE, label="Sau log1p")
for i, (a, b) in enumerate(zip(sk["SkewBefore"], sk["SkewAfterLog"])):
    ax.text(i - w / 2, max(a, 0) + 1, vn(a, 1), ha="center", fontsize=8.5)
    ax.text(i + w / 2, max(b, 0) + 1, vn(b, 1), ha="center", fontsize=8.5, color=BLUE)
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(x, sk["Feature"], rotation=20); ax.set_ylabel("Hệ số độ lệch (skewness)")
ax.legend(frameon=False)
save(fig, "c3_skewness.png")

# 3.2 Tương quan Spearman
corr = pd.read_csv(ROOT / "data/processed/feature_correlation_spearman.csv", index_col=0)
cmap = LinearSegmentedColormap.from_list("bw", ["#2E86AB", "#FFFFFF", "#C0392B"])
fig, ax = plt.subplots(figsize=(6.6, 5.6))
im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1)
ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
ax.set_yticks(range(len(corr)), corr.index)
for i in range(len(corr)):
    for j in range(len(corr)):
        v = corr.values[i, j]
        ax.text(j, i, vn(v, 2), ha="center", va="center", fontsize=8.5,
                color="white" if abs(v) > 0.7 else "black")
ax.spines[:].set_visible(False)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Hệ số Spearman")
save(fig, "c3_corr_spearman.png")

# 5.1 Hình chiếu PCA 2D
X = pd.read_csv(ROOT / "data/processed/X_scaled.csv", dtype={"Customer ID": str})
lab = pd.read_csv(ROOT / "clustering/results/labels.csv", dtype={"Customer ID": str})
df = X.merge(lab, on="Customer ID")
feats = [c for c in X.columns if c != "Customer ID"]
pca = PCA(n_components=2, random_state=42).fit(df[feats])
Z = pca.transform(df[feats]); ev = pca.explained_variance_ratio_ * 100
fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), sharex=True, sharey=True)
hier_colors = ["#E8A202", "#2E86AB", "#C0392B", "#7F8C8D", "#27AE60"]
for ax, col, title in zip(axes, ["kmeans", "hierarchical", "dbscan"],
                          ["K-Means (k = 4)", "Hierarchical Ward (k = 5)", "DBSCAN (nhiễu = −1)"]):
    for c in sorted(df[col].unique()):
        m = df[col] == c
        if col == "kmeans":
            color, name = PCOLOR[c], f"{c} – {PERSONA[c]}"
        elif col == "dbscan":
            color, name = ("#C0392B", "Nhiễu (−1)") if c == -1 else (LIGHT, f"Cụm {c}")
        else:
            color, name = hier_colors[c % 5], f"Cụm {c}"
        ax.scatter(Z[m, 0], Z[m, 1], s=4 if c != -1 else 9, alpha=0.6, color=color, label=name,
                   zorder=3 if c == -1 else 2)
    ax.set_title(title); ax.set_xlabel(f"PC1 ({vn(ev[0], 1)}% phương sai)")
    ax.legend(markerscale=3, fontsize=8.5, frameon=False, loc="upper left")
axes[0].set_ylabel(f"PC2 ({vn(ev[1], 1)}% phương sai)")
save(fig, "c5_pca.png")

# 5.2 Radar chart hồ sơ cụm (min-max trên trung bình)
radar = pd.read_csv(ROOT / "evaluation/results/cluster_profile_radar_minmax.csv")
key = radar.columns[0]
radar = radar.set_index(key)
cols = [c for c in radar.columns if c in FEAT_VI]
ang = np.linspace(0, 2 * np.pi, len(cols), endpoint=False).tolist(); ang += ang[:1]
fig = plt.figure(figsize=(6.6, 6.2)); ax = plt.subplot(polar=True)
for idx, row in radar.iterrows():
    c = int(str(idx).split()[-1]) if not isinstance(idx, (int, np.integer)) else int(idx)
    v = row[cols].tolist(); v += v[:1]
    ax.plot(ang, v, color=PCOLOR[c], lw=2, label=f"Cụm {c} – {PERSONA[c]}")
    ax.fill(ang, v, color=PCOLOR[c], alpha=0.12)
ax.set_xticks(ang[:-1], [FEAT_VI[c] for c in cols], fontsize=10)
ax.set_yticks([0.25, 0.5, 0.75, 1.0], ["0,25", "0,5", "0,75", "1"], fontsize=8, color=GREY)
ax.set_ylim(0, 1.05)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=2, frameon=False)
save(fig, "c5_radar.png")

print("PCA explained variance:", np.round(ev, 2))
