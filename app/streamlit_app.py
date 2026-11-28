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

from app import ui  # noqa: E402
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
PERSONA_COLORS = ui.PERSONA_COLORS
FEATURE_LABELS = {
    "Recency": "Recency (ngày)", "Frequency": "Frequency (đơn)", "Monetary": "Monetary (£)",
    "AvgOrderValue": "Giá trị TB/đơn (£)", "UniqueProducts": "Số SP khác nhau",
    "AvgBasketSize": "SL TB/đơn", "Tenure": "Tenure (ngày)", "CancelRate": "Tỷ lệ hủy",
}

st.set_page_config(page_title="Phân khúc khách hàng RFM", page_icon="🛒", layout="wide")
ui.setup()


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


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})"


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


ui.hero("Bài tập lớn · CO3029 Khai phá dữ liệu · Lớp L01 · GVHD: Bùi Tiến Đức",
        "Phân khúc <mark>khách hàng</mark><br>theo RFM &amp; hành vi",
        "Online Retail II (UCI) · 01/12/2009 – 09/12/2011 · K-Means k = 4 trên 8 đặc trưng")

tabs = st.tabs(["01  Tổng quan", "02  Hồ sơ phân khúc", "03  Marketing",
                "04  Tra cứu khách", "05  Dữ liệu mới", "06  Giải thích mô hình"])

# ------------------------------------------------------------------ Tổng quan
with tabs[0]:
    summ = summary_table(seg)
    c = st.columns(4)
    c[0].metric("Khách hàng", f"{len(seg):,}")
    c[1].metric("Phân khúc", len(summ))
    c[2].metric("Tổng doanh thu", f"£{seg['Monetary'].sum():,.0f}")
    c[3].metric("Ngoại biên (DBSCAN)", f"{int(seg['DBSCAN_Outlier'].sum()):,}",
                f"{seg['DBSCAN_Outlier'].mean():.1%} số khách", delta_color="off")

    ui.section(1, "Bốn nhóm khách hàng")
    cards = st.columns(4)
    for col, (persona, r) in zip(cards, summ.iterrows()):
        col.markdown(ui.persona_card(persona, [
            f"<b>{r['Số khách']:,.0f}</b> khách · {r['% khách']:.1%}",
            f"<b>{r['% doanh thu']:.1%}</b> doanh thu",
            f"Mua lần cuối ~{r['Recency (trung vị)']:.0f} ngày trước",
        ]), unsafe_allow_html=True)

    ui.section(2, "Ai đang tạo ra doanh thu?")
    left, right = st.columns(2)
    share = summ[["% khách", "% doanh thu"]].reset_index().melt(
        id_vars="Persona", var_name="Chỉ số", value_name="Tỷ trọng")
    fig = px.bar(share, x="Persona", y="Tỷ trọng", color="Chỉ số", barmode="group",
                 text_auto=".1%", category_orders={"Persona": PERSONA_ORDER},
                 color_discrete_map={"% khách": ui.LIGHT, "% doanh thu": ui.YELLOW})
    fig.update_layout(yaxis_tickformat=".0%", height=400, legend_title=None, xaxis_title=None,
                      title="Tỷ trọng số khách và doanh thu", legend=dict(orientation="h", y=1.08, x=0.55))
    fig.update_traces(textposition="outside", textangle=0, cliponaxis=False,
                      textfont=dict(size=13, family="Montserrat", color=ui.BLACK))
    left.plotly_chart(fig, width="stretch")

    fig2 = px.scatter(seg.reset_index(), x="Recency", y="Monetary", color="Persona",
                      log_y=True, opacity=0.6, color_discrete_map=PERSONA_COLORS,
                      category_orders={"Persona": PERSONA_ORDER[::-1]},
                      hover_data=["Customer ID", "Frequency"])
    fig2.update_traces(marker=dict(size=6, line=dict(width=0.4, color=ui.BLACK)))
    fig2.update_layout(height=400, title="Recency và Monetary theo phân khúc", legend_title=None,
                       xaxis_title="Số ngày từ lần mua cuối", yaxis_title="Tổng chi tiêu (£, log)")
    right.plotly_chart(fig2, width="stretch")

    ui.section(3, "Bảng tổng hợp")
    st.dataframe(fmt_summary(summ), width="stretch")

# ------------------------------------------------------------- Hồ sơ phân khúc
with tabs[1]:
    means = seg.groupby("Persona")[FEATURE_COLUMNS].mean().reindex(
        [p for p in PERSONA_ORDER if p in seg["Persona"].unique()])
    # Recency thấp là tốt: đảo chiều để trục radar "càng xa tâm càng tốt"
    radar = (means - means.min()) / (means.max() - means.min()).replace(0, 1)
    radar["Recency"] = 1 - radar["Recency"]

    ui.section(1, "So sánh 8 đặc trưng")
    left, right = st.columns([3, 1])
    with left:
        fig = go.Figure()
        axes = [("Mua gần đây" if f == "Recency" else FEATURE_LABELS[f]) for f in FEATURE_COLUMNS]
        for persona, row in radar.iterrows():
            vals = row[FEATURE_COLUMNS].tolist()
            fig.add_trace(go.Scatterpolar(
                r=vals + vals[:1], theta=axes + axes[:1], name=persona, fill="toself", opacity=0.9,
                line=dict(color=ui.BLACK if persona == "VIP" else PERSONA_COLORS.get(persona), width=2.5),
                fillcolor=_rgba(PERSONA_COLORS.get(persona), 0.28)))
        fig.update_layout(polar=dict(radialaxis=dict(range=[0, 1], showticklabels=False)), height=520,
                          legend=dict(orientation="h", y=-0.08))
        st.plotly_chart(fig, width="stretch")
    with right:
        ui.html(ui.panel("Cách đọc biểu đồ", [
            "Trung bình từng đặc trưng theo nhóm, chuẩn hóa min-max giữa các nhóm.",
            "Trục <b>Mua gần đây</b> đã đảo chiều: càng xa tâm là mua càng gần đây.",
            "Nhóm phủ rộng nhất là nhóm giá trị nhất.",
        ], "cream"))

    ui.section(2, "Chân dung từng nhóm")
    grid = st.columns(2)
    for i, persona in enumerate(means.index):
        p = get_persona(persona_ids[persona])
        with grid[i % 2]:
            st.markdown(ui.panel(f'<span class="chip" style="background:{PERSONA_COLORS[persona]}">'
                                  f'Cụm {persona_ids[persona]}</span> {persona}', [
                p["description"], f"<b>Hành vi:</b> {p['behavior']}", f"<b>Chi tiêu:</b> {p['spending']}",
                f"<b>Gắn bó:</b> {p['loyalty']}", f"<b>Rủi ro rời bỏ:</b> {p['churn_risk']}",
            ]), unsafe_allow_html=True)
            st.write("")

    ui.section(3, "Giá trị trung bình")
    st.dataframe(means.T.rename(index=FEATURE_LABELS).style.format("{:,.2f}"), width="stretch")

# -------------------------------------------------------- Chiến lược marketing
with tabs[2]:
    if True:
        persona = st.selectbox("Chọn phân khúc", PERSONA_ORDER + ["Ngoại biên DBSCAN"])
        s = strategy_for(persona)
        members = (seg[seg["DBSCAN_Outlier"]] if persona == "Ngoại biên DBSCAN"
                   else seg[seg["Persona"] == persona])
        a, b, c = st.columns(3)
        a.metric("Số khách", f"{len(members):,}")
        b.metric("Doanh thu", f"£{members['Monetary'].sum():,.0f}")
        c.metric("Recency trung vị", f"{members['Recency'].median():.0f} ngày")
    ui.section(1, "Kế hoạch chiến dịch")
    l, r = st.columns(2)
    l.markdown(ui.panel("Mục tiêu &amp; khuyến mãi", [f"<b>{s['muc_tieu']}</b>"] + s["khuyen_mai"], "yellow"),
               unsafe_allow_html=True)
    r.markdown(ui.panel("Kênh, tần suất &amp; KPI", [
        f"<b>Kênh:</b> {s['kenh']}", f"<b>Tần suất:</b> {s['tan_suat']}", f"<b>KPI theo dõi:</b> {s['kpi']}",
    ], "orange"), unsafe_allow_html=True)

    if s["email_tieu_de"]:
        ui.section(2, "Mẫu email tự động")
        if True:
            sample = members.sort_values("Monetary", ascending=False).head(1)
            sample_id = sample.index[0] if len(sample) else "12345"
            subj, body = render_email(persona, f"Khách {sample_id}",
                                      int(sample["Frequency"].iloc[0]) if len(sample) else 1,
                                      int(sample["Recency"].iloc[0]) if len(sample) else 0)
            st.text_input("Tiêu đề", subj)
            st.text_area("Nội dung", body, height=170)

    ui.section(3 if s["email_tieu_de"] else 2, "Xuất danh sách khách")
    export = members[["Recency", "Frequency", "Monetary", "Persona"]].copy()
    export["Khuyen_mai"] = s["khuyen_mai"][0]
    export["Email_tieu_de"] = [render_email(persona, f"Khách {i}", int(r.Frequency), int(r.Recency))[0]
                               for i, r in members.iterrows()]
    st.download_button(f"Tải danh sách {len(export):,} khách (CSV)",
                       export.to_csv(index_label="Customer ID").encode("utf-8-sig"),
                       f"chien_dich_{persona}.csv", "text/csv")

# --------------------------------------------------------- Tra cứu khách hàng
with tabs[3]:
    in_l, in_r = st.columns([2, 1])
    cid = in_l.text_input("Nhập Customer ID", value=str(seg.index[0])).strip().removesuffix(".0")
    if cid in seg.index:
        row = seg.loc[cid]
        s = strategy_for(row["Persona"])
        card, info = st.columns([1, 3])
        card.markdown(ui.persona_card(row["Persona"], [f"Khách <b>{cid}</b>", f"Cụm {int(row['Cluster'])}"]),
                      unsafe_allow_html=True)
        with info:
            if row["DBSCAN_Outlier"]:
                st.warning("DBSCAN đánh dấu khách này là ngoại biên, nên kiểm tra trước khi đưa vào chiến dịch.")
            cols = st.columns(4)
            for i, f in enumerate(FEATURE_COLUMNS):
                med = seg[f].median()
                cols[i % 4].metric(FEATURE_LABELS[f], f"{row[f]:,.2f}" if f == "CancelRate" else f"{row[f]:,.0f}",
                                   f"trung vị {med:,.2f}" if f == "CancelRate" else f"trung vị {med:,.0f}",
                                   delta_color="off")
        ui.section(1, "Đề xuất cho khách này")
        ui.html(ui.panel("Hành động gợi ý", [f"<b>Khuyến mãi:</b> {s['khuyen_mai'][0]}",
                                             f"<b>Kênh:</b> {s['kenh']}"], "yellow"))
    else:
        st.warning("Không tìm thấy khách hàng này trong tập đã phân khúc.")

# ----------------------------------------------------- Phân khúc dữ liệu mới
with tabs[4]:
    st.markdown(ui.panel("Đưa dữ liệu mới vào mô hình", [
        "Tải lên <b>giao dịch thô</b> cùng định dạng Online Retail II: <code>Invoice, StockCode, Description, "
        "Quantity, InvoiceDate, Price, Customer ID, Country</code>.",
        "Hoặc <b>bảng đặc trưng</b> có <code>Customer ID</code> và 8 cột đặc trưng.",
        "Dữ liệu đi qua đúng các bước của nhóm: làm sạch → tính đặc trưng → log + scaler đã lưu → "
        "K-Means đã lưu → gán persona.",
    ], "cream"), unsafe_allow_html=True)

    st.write("")
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
            new_summ = summary_table(result)
            ui.section(1, "Kết quả phân khúc")
            cards = st.columns(4)
            for col, (persona, r_) in zip(cards, new_summ.iterrows()):
                col.markdown(ui.persona_card(persona, [f"<b>{r_['Số khách']:,.0f}</b> khách · {r_['% khách']:.1%}",
                                                       f"<b>{r_['% doanh thu']:.1%}</b> doanh thu"]),
                             unsafe_allow_html=True)
            st.write("")
            st.dataframe(fmt_summary(new_summ), width="stretch")
            ui.section(2, "Chi tiết từng khách")
            st.dataframe(result, width="stretch")
            st.download_button("Tải kết quả (CSV)", result.to_csv(index_label="Customer ID").encode("utf-8-sig"),
                               "ket_qua_phan_khuc.csv", "text/csv")
        except ValueError as exc:
            st.error(f"Dữ liệu không hợp lệ: {exc}")

# ------------------------------------------------------- Giải thích mô hình
with tabs[5]:
    ui.html(ui.panel("Vì sao khách thuộc nhóm này?", [
        "K-Means không cho luật phân cụm tường minh, nên nhóm giải thích theo hai cách.",
        "<b>Cây quyết định đại diện</b> (độ sâu 3) học lại nhãn K-Means từ 8 đặc trưng gốc.",
        "<b>Độ quan trọng đặc trưng</b>: xáo trộn từng đặc trưng rồi đo mức thay đổi.",
        "Kết quả do <code>python -m evaluation.explain_clusters</code> tạo ra.",
    ], "cream"))
    missing = [p for p in (xc.IMPORTANCE_PATH, xc.FIDELITY_PATH, xc.RULES_PATH) if not p.exists()]
    if missing:
        st.warning("Chưa có kết quả giải thích mô hình. Chạy `python -m evaluation.explain_clusters` trước.")
    else:
        fidelity = json.loads(xc.FIDELITY_PATH.read_text(encoding="utf-8"))
        chosen = fidelity["chosen"]
        st.write("")
        c = st.columns(3)
        c[0].metric("Độ trung thành (CV 5-fold)", f"{chosen['cv_mean']:.1%}",
                    f"± {chosen['cv_std']:.1%}", delta_color="off")
        c[1].metric("Độ trung thành (toàn bộ dữ liệu)", f"{chosen['full_data']:.1%}")
        c[2].metric("Số luật (lá của cây)", chosen["n_leaves"])
        st.caption("Độ trung thành là tỷ lệ khách mà cây đại diện gán cùng phân khúc với K-Means.")

        ui.section(1, "Độ quan trọng đặc trưng")
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
                     color_discrete_sequence=[ui.YELLOW, ui.LIGHT, ui.GREY])
        fig.update_layout(height=480, yaxis_title=None, legend_title=None,
                          legend=dict(orientation="h", y=-0.15),
                          title="Trung bình 10 lần xáo trộn")
        st.plotly_chart(fig, width="stretch")

        ui.section(2, "Luật của cây và đường đi của một khách")
        left, right = st.columns(2)
        with left:
            st.code(xc.RULES_PATH.read_text(encoding="utf-8"), language="text")
        with right:
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
                ui.html('<div style="margin:.4rem 0 .8rem">' +
                        "".join(f'<span class="chip">{x}</span>' for x in steps[:-1]) +
                        f'<span class="chip" style="background:{PERSONA_COLORS.get(steps[-1], ui.YELLOW)}">'
                        f'→ {steps[-1]}</span></div>')
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

ui.html('<div class="footer">Nhóm thực hiện: Bùi Đình Phúc · Hoàng Hữu Nhân · Nguyễn Cảnh Nguyên · '
        'Trương Hoàng Danh — Trường Đại học Bách khoa, ĐHQG-HCM</div>')
