"""Giao diện báo cáo phân khúc khách hàng & đề xuất chiến lược marketing.

Chạy từ thư mục gốc repo:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.segmentation import (  # noqa: E402
    detect_input_type, load_artifacts, read_upload,
    segment_feature_table, segment_transactions,
)
from app.strategies import render_email, strategy_for  # noqa: E402
from evaluation import explain_clusters as xc  # noqa: E402
from evaluation.personas import get_persona  # noqa: E402
from preprocessing.preprocess import FEATURE_COLUMNS as _FEATURES  # noqa: E402

FEATURE_COLUMNS = list(_FEATURES)

PERSONA_ORDER = ["VIP", "Tiềm năng", "Cần níu kéo", "Ngủ đông"]
PERSONA_COLORS = {"VIP": "#1f6feb", "Tiềm năng": "#2da44e",
                  "Cần níu kéo": "#d29922", "Ngủ đông": "#8c959f"}
FEATURE_LABELS = {
    "Recency": "Recency (ngày)", "Frequency": "Frequency (đơn)", "Monetary": "Monetary (£)",
    "AvgOrderValue": "Giá trị TB/đơn (£)", "UniqueProducts": "Số SP khác nhau",
    "AvgBasketSize": "SL TB/đơn", "Tenure": "Tenure (ngày)", "CancelRate": "Tỷ lệ hủy",
}

st.set_page_config(page_title="Phân khúc khách hàng RFM", layout="wide")


@st.cache_resource
def get_artifacts():
    return load_artifacts()


@st.cache_resource
def get_surrogate():
    """Cây đại diện (cùng cấu hình với evaluation/explain_clusters.py) và bảng đặc trưng gốc."""
    features, _, labels = xc.load_data()
    return xc.fit_surrogate(features, labels), features, labels


try:
    art = get_artifacts()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

seg = art.segments
persona_ids = {get_persona(c)["name"]: c for c in sorted(seg["Cluster"].unique())}


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("Persona")
    out = pd.DataFrame({
        "Số khách": g.size(),
        "Recency (trung vị)": g["Recency"].median(),
        "Frequency (trung vị)": g["Frequency"].median(),
        "Monetary (trung vị)": g["Monetary"].median(),
        "Doanh thu": g["Monetary"].sum(),
    })
    out["% khách"] = out["Số khách"] / out["Số khách"].sum()
    out["% doanh thu"] = out["Doanh thu"] / out["Doanh thu"].sum()
    return out.reindex([p for p in PERSONA_ORDER if p in out.index])


def fmt_summary(df: pd.DataFrame):
    return df.style.format({
        "Số khách": "{:,.0f}", "Recency (trung vị)": "{:,.0f}", "Frequency (trung vị)": "{:,.0f}",
        "Monetary (trung vị)": "£{:,.0f}", "Doanh thu": "£{:,.0f}",
        "% khách": "{:.1%}", "% doanh thu": "{:.1%}",
    })


st.title("Phân khúc khách hàng theo RFM & hành vi")
st.caption("Online Retail II (UCI) · 01/12/2009 – 09/12/2011 · K-Means k=4 trên 8 đặc trưng")

tabs = st.tabs(["Tổng quan", "Hồ sơ phân khúc", "Chiến lược marketing",
                "Tra cứu khách hàng", "Phân khúc dữ liệu mới", "Giải thích mô hình"])

# ------------------------------------------------------------------ Tổng quan
with tabs[0]:
    summ = summary_table(seg)
    c = st.columns(4)
    c[0].metric("Khách hàng", f"{len(seg):,}")
    c[1].metric("Phân khúc", len(summ))
    c[2].metric("Tổng doanh thu", f"£{seg['Monetary'].sum():,.0f}")
    c[3].metric("Ngoại biên (DBSCAN)", f"{int(seg['DBSCAN_Outlier'].sum()):,}",
                f"{seg['DBSCAN_Outlier'].mean():.1%} số khách", delta_color="off")

    left, right = st.columns(2)
    share = summ[["% khách", "% doanh thu"]].reset_index().melt(
        id_vars="Persona", var_name="Chỉ số", value_name="Tỷ trọng")
    fig = px.bar(share, x="Persona", y="Tỷ trọng", color="Chỉ số", barmode="group",
                 text_auto=".1%", category_orders={"Persona": PERSONA_ORDER})
    fig.update_layout(yaxis_tickformat=".0%", height=380, legend_title=None,
                      title="Tỷ trọng số khách và doanh thu")
    fig.update_traces(textposition="outside", textangle=0, cliponaxis=False)
    left.plotly_chart(fig, width="stretch")

    fig2 = px.scatter(seg.reset_index(), x="Recency", y="Monetary", color="Persona",
                      log_y=True, opacity=0.55, color_discrete_map=PERSONA_COLORS,
                      category_orders={"Persona": PERSONA_ORDER},
                      hover_data=["Customer ID", "Frequency"])
    fig2.update_layout(height=380, title="Recency và Monetary theo phân khúc")
    right.plotly_chart(fig2, width="stretch")

    st.dataframe(fmt_summary(summ), width="stretch")

# ------------------------------------------------------------- Hồ sơ phân khúc
with tabs[1]:
    means = seg.groupby("Persona")[FEATURE_COLUMNS].mean().reindex(
        [p for p in PERSONA_ORDER if p in seg["Persona"].unique()])
    # Recency thấp là tốt: đảo chiều để trục radar "càng xa tâm càng tốt"
    radar = (means - means.min()) / (means.max() - means.min()).replace(0, 1)
    radar["Recency"] = 1 - radar["Recency"]
    fig = go.Figure()
    axes = [("Mua gần đây" if f == "Recency" else FEATURE_LABELS[f]) for f in FEATURE_COLUMNS]
    for persona, row in radar.iterrows():
        vals = row[FEATURE_COLUMNS].tolist()
        fig.add_trace(go.Scatterpolar(r=vals + vals[:1], theta=axes + axes[:1], name=persona,
                                      fill="toself", opacity=0.45,
                                      line_color=PERSONA_COLORS.get(persona)))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0, 1], showticklabels=False)), height=520)
    st.plotly_chart(fig, width="stretch")
    st.caption("Trung bình từng đặc trưng theo phân khúc, chuẩn hóa min-max giữa các phân khúc. "
               "Trục Recency đã đảo chiều: càng xa tâm nghĩa là mua càng gần đây.")

    st.dataframe(means.T.rename(index=FEATURE_LABELS).style.format("{:,.2f}"), width="stretch")

    for persona in means.index:
        p = get_persona(persona_ids[persona])
        with st.expander(f"{persona} · cụm {persona_ids[persona]}"):
            st.write(p["description"])
            st.markdown(f"- **Hành vi:** {p['behavior']}\n- **Chi tiêu:** {p['spending']}\n"
                        f"- **Gắn bó:** {p['loyalty']}\n- **Rủi ro rời bỏ:** {p['churn_risk']}")

# -------------------------------------------------------- Chiến lược marketing
with tabs[2]:
    persona = st.selectbox("Phân khúc", PERSONA_ORDER + ["Ngoại biên DBSCAN"])
    s = strategy_for(persona)
    members = (seg[seg["DBSCAN_Outlier"]] if persona == "Ngoại biên DBSCAN"
               else seg[seg["Persona"] == persona])
    a, b, c = st.columns(3)
    a.metric("Số khách", f"{len(members):,}")
    b.metric("Doanh thu", f"£{members['Monetary'].sum():,.0f}")
    c.metric("Recency trung vị", f"{members['Recency'].median():.0f} ngày")

    st.markdown(f"**Mục tiêu:** {s['muc_tieu']}")
    st.markdown("**Khuyến mãi đề xuất:**\n" + "\n".join(f"- {x}" for x in s["khuyen_mai"]))
    st.markdown(f"**Kênh:** {s['kenh']}  \n**Tần suất:** {s['tan_suat']}  \n**KPI theo dõi:** {s['kpi']}")

    if s["email_tieu_de"]:
        st.subheader("Mẫu email")
        sample = members.sort_values("Monetary", ascending=False).head(1)
        sample_id = sample.index[0] if len(sample) else "12345"
        subj, body = render_email(persona, f"Khách {sample_id}",
                                  int(sample["Frequency"].iloc[0]) if len(sample) else 1,
                                  int(sample["Recency"].iloc[0]) if len(sample) else 0)
        st.text_input("Tiêu đề", subj)
        st.text_area("Nội dung", body, height=170)

    export = members[["Recency", "Frequency", "Monetary", "Persona"]].copy()
    export["Khuyen_mai"] = s["khuyen_mai"][0]
    export["Email_tieu_de"] = [render_email(persona, f"Khách {i}", int(r.Frequency), int(r.Recency))[0]
                               for i, r in members.iterrows()]
    st.download_button(f"Tải danh sách {len(export):,} khách (CSV)",
                       export.to_csv(index_label="Customer ID").encode("utf-8-sig"),
                       f"chien_dich_{persona}.csv", "text/csv")

# --------------------------------------------------------- Tra cứu khách hàng
with tabs[3]:
    cid = st.text_input("Nhập Customer ID", value=str(seg.index[0])).strip().removesuffix(".0")
    if cid in seg.index:
        row = seg.loc[cid]
        st.subheader(f"Khách {cid}: {row['Persona']}")
        if row["DBSCAN_Outlier"]:
            st.warning("DBSCAN đánh dấu khách này là ngoại biên, nên kiểm tra trước khi đưa vào chiến dịch.")
        cols = st.columns(4)
        for i, f in enumerate(FEATURE_COLUMNS):
            med = seg[f].median()
            cols[i % 4].metric(FEATURE_LABELS[f], f"{row[f]:,.2f}" if f in ("CancelRate",) else f"{row[f]:,.0f}",
                               f"trung vị toàn tập {med:,.2f}" if f == "CancelRate" else f"trung vị toàn tập {med:,.0f}",
                               delta_color="off")
        s = strategy_for(row["Persona"])
        st.info(f"**Đề xuất:** {s['khuyen_mai'][0]}  \n**Kênh:** {s['kenh']}")
    else:
        st.warning("Không tìm thấy khách hàng này trong tập đã phân khúc.")

# ----------------------------------------------------- Phân khúc dữ liệu mới
with tabs[4]:
    st.markdown(
        "Tải lên **giao dịch thô** cùng định dạng Online Retail II "
        "(`Invoice, StockCode, Description, Quantity, InvoiceDate, Price, Customer ID, Country`) "
        "hoặc **bảng đặc trưng** có `Customer ID` và 8 cột đặc trưng. "
        "Dữ liệu đi qua đúng các bước của nhóm: làm sạch, tính đặc trưng, log + scaler đã lưu, "
        "dự đoán bằng K-Means đã lưu, gán persona.")
    up = st.file_uploader("File CSV hoặc Excel", type=["csv", "xlsx", "xls"])
    use_custom_ref = st.checkbox("Tự chọn ngày mốc tính Recency (mặc định: ngày giao dịch cuối + 1)")
    ref = st.date_input("Ngày mốc", value=pd.Timestamp("2011-12-10")) if use_custom_ref else None

    if up is not None:
        try:
            raw = read_upload(up.name, up)
            kind = detect_input_type(raw)
            if kind == "transactions":
                result, audit, used_ref = segment_transactions(raw, art, pd.Timestamp(ref) if ref else None)
                st.success(f"Đã xử lý {len(raw):,} dòng giao dịch → {len(result):,} khách. "
                           f"Ngày mốc: {used_ref.date()}")
                with st.expander("Nhật ký làm sạch"):
                    st.dataframe(audit, width="stretch")
            elif kind == "features":
                result = segment_feature_table(raw, art)
                st.success(f"Đã phân khúc {len(result):,} khách từ bảng đặc trưng.")
            else:
                # Không dùng st.stop() để tab "Giải thích mô hình" phía sau vẫn được vẽ.
                raise ValueError("không nhận ra định dạng file. Cần 8 cột giao dịch hoặc 8 cột đặc trưng.")

            result["Khuyen_mai"] = result["Persona"].map(lambda p: strategy_for(p)["khuyen_mai"][0])
            st.dataframe(fmt_summary(summary_table(result)), width="stretch")
            st.dataframe(result, width="stretch")
            st.download_button("Tải kết quả (CSV)", result.to_csv(index_label="Customer ID").encode("utf-8-sig"),
                               "ket_qua_phan_khuc.csv", "text/csv")
        except ValueError as exc:
            st.error(f"Dữ liệu không hợp lệ: {exc}")

# ------------------------------------------------------- Giải thích mô hình
with tabs[5]:
    st.markdown(
        "K-Means không cho luật phân cụm tường minh. Phần này giải thích mô hình theo hai cách: "
        "**cây quyết định đại diện** (độ sâu 3) học lại nhãn K-Means từ 8 đặc trưng gốc, và "
        "**độ quan trọng đặc trưng** đo bằng cách xáo trộn từng đặc trưng. "
        "Kết quả do `python -m evaluation.explain_clusters` tạo ra.")
    missing = [p for p in (xc.IMPORTANCE_PATH, xc.FIDELITY_PATH, xc.RULES_PATH) if not p.exists()]
    if missing:
        st.warning("Chưa có kết quả giải thích mô hình. Chạy `python -m evaluation.explain_clusters` trước.")
    else:
        fidelity = json.loads(xc.FIDELITY_PATH.read_text(encoding="utf-8"))
        chosen = fidelity["chosen"]
        c = st.columns(3)
        c[0].metric("Độ trung thành (CV 5-fold)", f"{chosen['cv_mean']:.1%}",
                    f"± {chosen['cv_std']:.1%}", delta_color="off")
        c[1].metric("Độ trung thành (toàn bộ dữ liệu)", f"{chosen['full_data']:.1%}")
        c[2].metric("Số luật (lá của cây)", chosen["n_leaves"])
        st.caption("Độ trung thành là tỷ lệ khách mà cây đại diện gán cùng phân khúc với K-Means.")

        imp = pd.read_csv(xc.IMPORTANCE_PATH).set_index("feature")
        measures = {"kmeans_change_mean": "K-Means: tỷ lệ khách đổi cụm khi xáo trộn",
                    "surrogate_perm_mean": "Cây đại diện: mức giảm độ chính xác (hoán vị)"}
        if "surrogate_shap_mean_abs" in imp.columns:
            measures["surrogate_shap_mean_abs"] = "Cây đại diện: trung bình |SHAP|"
        long = (imp[list(measures)].rename(columns=measures).reset_index()
                .melt(id_vars="feature", var_name="Thước đo", value_name="Độ quan trọng"))
        order = imp.sort_values("kmeans_change_mean").index.tolist()
        fig = px.bar(long, x="Độ quan trọng", y="feature", color="Thước đo", orientation="h",
                     barmode="group", text_auto=".3f", category_orders={"feature": order[::-1]},
                     color_discrete_sequence=["#1f6feb", "#d29922", "#2da44e"])
        fig.update_layout(height=460, yaxis_title=None, legend_title=None,
                          legend=dict(orientation="h", y=-0.15),
                          title="Độ quan trọng đặc trưng (trung bình 10 lần xáo trộn)")
        st.plotly_chart(fig, width="stretch")

        left, right = st.columns(2)
        with left:
            st.subheader("Luật của cây đại diện")
            st.code(xc.RULES_PATH.read_text(encoding="utf-8"), language="text")
        with right:
            st.subheader("Đường đi của một khách qua cây")
            try:
                surrogate, feat_table, km_labels = get_surrogate()
            except FileNotFoundError as exc:
                st.warning(f"Thiếu dữ liệu đặc trưng để dựng cây đại diện: {exc}")
                st.stop()  # tab cuối cùng nên dừng ở đây không ảnh hưởng tab khác
            xid = st.text_input("Customer ID cần giải thích", value=str(feat_table.index[0]),
                                key="explain_cid").strip().removesuffix(".0")
            if xid in feat_table.index:
                row = feat_table.loc[xid]
                steps = xc.decision_path(surrogate, row)
                actual = get_persona(int(km_labels.loc[xid]))["name"]
                st.markdown("**" + " → ".join(steps) + "**")
                if steps[-1] == actual:
                    st.success(f"Cây đại diện và K-Means cùng xếp khách {xid} vào nhóm {actual}.")
                else:
                    st.warning(f"K-Means xếp khách {xid} vào nhóm {actual}, còn cây đại diện xếp vào "
                               f"{steps[-1]}. Luật rút gọn của cây không mô tả đúng trường hợp này; "
                               "phân khúc chính thức vẫn theo K-Means.")
                st.dataframe(row.rename(index=FEATURE_LABELS).to_frame("Giá trị").style.format("{:,.2f}"),
                             width="stretch")
            else:
                st.warning("Không tìm thấy khách hàng này trong tập đã phân khúc.")
