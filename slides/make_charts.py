"""Vẽ lại các biểu đồ chính cho slide theo tông màu deck (vàng #F9B800, đen, xám).

Chạy từ thư mục gốc repo: python slides/make_charts.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "slides" / "charts"
OUT.mkdir(parents=True, exist_ok=True)

YELLOW, BLACK, GREY, LIGHT = "#F9B800", "#1E1A1A", "#9A9A9A", "#E6E6E6"
PERSONA_ORDER = ["VIP", "Tiềm năng", "Cần níu kéo", "Ngủ đông"]
PERSONA_COLORS = {"VIP": YELLOW, "Tiềm năng": "#8FB3A9", "Cần níu kéo": "#D9A3A3", "Ngủ đông": GREY}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 13, "axes.edgecolor": BLACK,
    "axes.labelcolor": BLACK, "xtick.color": BLACK, "ytick.color": BLACK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 1.4,
})


def save(fig, name):
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print("saved", name)


seg = pd.read_csv(ROOT / "data/processed/customers_segments.csv", dtype={"Customer ID": str})

# 1) Tỷ trọng khách vs doanh thu theo persona
g = seg.groupby("Persona").agg(n=("Recency", "size"), rev=("Monetary", "sum")).reindex(PERSONA_ORDER)
share_c, share_r = g["n"] / g["n"].sum() * 100, g["rev"] / g["rev"].sum() * 100
fig, ax = plt.subplots(figsize=(8, 4.6))
x = np.arange(len(PERSONA_ORDER)); w = 0.38
b1 = ax.bar(x - w / 2, share_c, w, color=LIGHT, edgecolor=BLACK, linewidth=1.4, label="% số khách")
b2 = ax.bar(x + w / 2, share_r, w, color=YELLOW, edgecolor=BLACK, linewidth=1.4, label="% doanh thu")
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, f"{b.get_height():.1f}%",
                ha="center", fontsize=12, fontweight="bold", color=BLACK)
ax.set_xticks(x, PERSONA_ORDER, fontweight="bold"); ax.set_ylim(0, 95); ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.legend(frameon=False, loc="upper right")
save(fig, "s_share_persona.png")

# 2) Elbow + Silhouette
t = pd.read_csv(ROOT / "clustering/results/tuning_kmeans.csv")
fig, ax1 = plt.subplots(figsize=(8, 4.4))
ax1.plot(t["n_clusters"], t["inertia"], "-o", color=BLACK, lw=2.5, ms=7, label="WCSS (Elbow)")
ax1.set_xlabel("Số cụm k"); ax1.set_ylabel("WCSS")
ax2 = ax1.twinx(); ax2.spines["right"].set_visible(True)
ax2.plot(t["n_clusters"], t["silhouette"], "-s", color=YELLOW, lw=2.5, ms=7, mec=BLACK, label="Silhouette")
ax2.set_ylabel("Silhouette")
ax1.axvline(4, color=YELLOW, lw=8, alpha=0.35, zorder=0)
ax1.annotate("k = 4", xy=(4, t.loc[t.n_clusters == 4, "inertia"].item()), xytext=(5.2, 16500),
             fontsize=14, fontweight="bold", arrowprops=dict(arrowstyle="->", color=BLACK, lw=1.5))
lines = ax1.get_legend_handles_labels(); lines2 = ax2.get_legend_handles_labels()
ax1.legend(lines[0] + lines2[0], lines[1] + lines2[1], frameon=False, loc="upper center")
save(fig, "s_elbow.png")

# 3) So sánh mô hình
m = pd.read_csv(ROOT / "evaluation/results/model_comparison.csv")
names = ["K-Means\nk=4", "Hierarchical\nward k=5", "DBSCAN\n(eps=0.735)"]
fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
for ax, col, title, better in zip(axes, ["Silhouette", "Davies-Bouldin", "Calinski-Harabasz"],
                                  ["Silhouette ↑", "Davies–Bouldin ↓", "Calinski–Harabasz ↑"], [1, -1, 1]):
    vals = m[col].to_numpy()
    colors = [YELLOW, LIGHT, LIGHT]
    ax.bar(range(3), vals, color=colors, edgecolor=BLACK, linewidth=1.4)
    for i, v in enumerate(vals):
        ax.text(i, v * 1.03, f"{v:,.3f}" if v < 100 else f"{v:,.0f}", ha="center", fontsize=11, fontweight="bold")
    ax.set_title(title, fontweight="bold"); ax.set_xticks(range(3), names, fontsize=10)
    ax.set_yticks([]); ax.spines["left"].set_visible(False); ax.set_ylim(0, vals.max() * 1.2)
fig.tight_layout()
save(fig, "s_model_compare.png")

# 4) Radar trung vị chuẩn hóa theo persona (Recency đảo chiều)
feats = ["Recency", "Frequency", "Monetary", "AvgOrderValue", "UniqueProducts", "AvgBasketSize", "Tenure", "CancelRate"]
labels = ["Mua gần đây", "Tần suất", "Chi tiêu", "Giá trị/đơn", "Số SP", "SL/đơn", "Gắn bó", "Tỷ lệ hủy"]
mean = seg.groupby("Persona")[feats].mean().reindex(PERSONA_ORDER)
mean = np.log1p(mean.clip(lower=0))
norm = (mean - mean.min()) / (mean.max() - mean.min()).replace(0, 1)
norm["Recency"] = 1 - norm["Recency"]
ang = np.linspace(0, 2 * np.pi, len(feats), endpoint=False).tolist(); ang += ang[:1]
fig = plt.figure(figsize=(6.4, 6.4)); ax = plt.subplot(polar=True)
for p in PERSONA_ORDER:
    v = norm.loc[p].tolist(); v += v[:1]
    ax.plot(ang, v, color=BLACK if p == "VIP" else PERSONA_COLORS[p], lw=2.2)
    ax.fill(ang, v, color=PERSONA_COLORS[p], alpha=0.35, label=p)
ax.set_xticks(ang[:-1], labels, fontsize=12, fontweight="bold"); ax.set_yticks([])
ax.spines["polar"].set_color(BLACK)
ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.1), frameon=False)
save(fig, "s_radar.png")

# 5) Phễu làm sạch dữ liệu
steps = [("Dữ liệu thô", 1067371), ("Có Customer ID", 824364), ("Bỏ trùng", 797885),
         ("Bỏ mã phi sản phẩm", 794225), ("Giao dịch mua hợp lệ", 776579)]
fig, ax = plt.subplots(figsize=(8, 4))
mx = steps[0][1]
for i, (lab, v) in enumerate(steps):
    wdt = v / mx
    ax.barh(len(steps) - i, wdt, left=(1 - wdt) / 2, height=0.72,
            color=YELLOW if i == len(steps) - 1 else LIGHT, edgecolor=BLACK, linewidth=1.4)
    ax.text(0.5, len(steps) - i, f"{lab}: {v:,}".replace(",", "."), ha="center", va="center",
            fontsize=12, fontweight="bold")
ax.axis("off")
save(fig, "s_funnel.png")

# 6) Doanh thu theo tháng (giao dịch mua hợp lệ)
from preprocessing.preprocess import clean_transactions, normalize_columns
raw = pd.read_csv(ROOT / "data/raw/online_retail_II.csv", dtype={"Invoice": str, "StockCode": str, "Customer ID": str})
purchases = clean_transactions(normalize_columns(raw)).purchases
monthly = purchases.groupby(purchases["InvoiceDate"].dt.to_period("M"))["LineTotal"].sum() / 1e6
fig, ax = plt.subplots(figsize=(9, 3.8))
xs = np.arange(len(monthly))
ax.fill_between(xs, monthly.values, color=YELLOW, alpha=0.35)
ax.plot(xs, monthly.values, color=BLACK, lw=2.4, marker="o", ms=5, mfc=YELLOW)
ticks = [i for i, p in enumerate(monthly.index) if p.month in (1, 4, 7, 10)]
ax.set_xticks(ticks, [f"{monthly.index[i].month:02d}/{monthly.index[i].year}" for i in ticks], fontsize=11)
ax.set_ylabel("Doanh thu (triệu GBP)")
pk = int(np.argmax(monthly.values))
ax.annotate("Đỉnh mùa lễ", xy=(pk, monthly.values[pk]), xytext=(pk - 7, monthly.values[pk] * 0.95),
            fontsize=12, fontweight="bold", arrowprops=dict(arrowstyle="->", color=BLACK))
save(fig, "s_monthly.png")
