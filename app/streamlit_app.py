"""Giao diện báo cáo phân khúc khách hàng & đề xuất chiến lược marketing.

Chạy từ thư mục gốc repo:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

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
                "Tra cứu khách hàng", "Phân khúc dữ liệu mới"])

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
                st.error("Không nhận ra định dạng file. Cần 8 cột giao dịch hoặc 8 cột đặc trưng.")
                st.stop()

            result["Khuyen_mai"] = result["Persona"].map(lambda p: strategy_for(p)["khuyen_mai"][0])
            st.dataframe(fmt_summary(summary_table(result)), width="stretch")
            st.dataframe(result, width="stretch")
            st.download_button("Tải kết quả (CSV)", result.to_csv(index_label="Customer ID").encode("utf-8-sig"),
                               "ket_qua_phan_khuc.csv", "text/csv")
        except ValueError as exc:
            st.error(f"Dữ liệu không hợp lệ: {exc}")
