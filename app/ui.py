"""Giao diện dùng chung cho app: màu, CSS và các khối HTML theo phong cách slide thuyết trình.

Bảng màu lấy từ deck: vàng #F9B800, đen #1E1A1A, xám chevron #D9D9D9, cam #F7944D.
"""
from __future__ import annotations

import base64
from functools import lru_cache
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

ASSETS = Path(__file__).resolve().parent / "assets"

YELLOW, BLACK, GREY, LIGHT, ORANGE, CREAM = "#F9B800", "#1E1A1A", "#9A9A9A", "#E6E6E6", "#F7944D", "#FFF6E0"
PERSONA_COLORS = {"VIP": YELLOW, "Tiềm năng": "#8FB3A9", "Cần níu kéo": "#D9A3A3",
                  "Ngủ đông": GREY, "Ngoại biên DBSCAN": ORANGE}

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700;800;900&display=swap');

html, body, .stApp, [data-testid="stMarkdownContainer"], button, input, textarea, select,
[data-baseweb="tab"], [data-testid="stMetric"] * {{
    font-family: 'Montserrat', sans-serif !important;
}}
.stApp {{ background: #FAF9F5; }}
.block-container {{ padding-top: 2.2rem; max-width: 1320px; }}
header[data-testid="stHeader"] {{ background: transparent; }}

h1, h2, h3 {{ font-weight: 800 !important; letter-spacing: .01em; color: {BLACK}; }}

/* ---------- Hero ---------- */
.hero {{ display:flex; align-items:center; gap:28px; margin-bottom:.6rem; }}
.hero-bars {{ display:flex; flex-direction:column; gap:10px; align-self:stretch; }}
.hero-bars span {{ width:14px; border-radius:99px; flex:1; }}
.hero-bars span:first-child {{ background:{YELLOW}; }}
.hero-bars span:last-child {{ background:{BLACK}; }}
.hero-text {{ flex:1; }}
.kicker {{ font-size:.8rem; font-weight:600; letter-spacing:.06em; text-transform:uppercase; color:#555; }}
.hero-title {{ font-size:2.55rem; line-height:1.12; font-weight:900; text-transform:uppercase;
               color:{BLACK}; margin:.35rem 0 .55rem; }}
.hero-title mark {{ background:linear-gradient(transparent 58%, {YELLOW} 58%); color:inherit; padding:0 .1em; }}
.hero-sub {{ font-size:.95rem; color:#333; font-weight:500; }}

/* ---------- Tabs dạng chevron như thanh tiến độ của slide ---------- */
.stTabs [role="tablist"] {{ gap:0 !important; flex-wrap:wrap; border-bottom:none !important;
                            box-shadow:none !important; padding:6px 0 4px; }}
.stTabs [role="tablist"]::after, .stTabs .react-aria-SelectionIndicator {{ display:none !important; }}
.stTabs [role="tab"] {{
    --chev: polygon(0 0, calc(100% - 20px) 0, 100% 50%, calc(100% - 20px) 100%, 0 100%, 20px 50%);
    background:{BLACK} !important; padding:2.5px !important; margin:0 -10px 0 0 !important;
    height:auto !important; border:none !important; clip-path: var(--chev); cursor:pointer;
}}
.stTabs [role="tab"]:first-child {{
    --chev: polygon(0 0, calc(100% - 20px) 0, 100% 50%, calc(100% - 20px) 100%, 0 100%);
}}
.stTabs [role="tab"] > [data-testid="stMarkdownContainer"] {{
    background:#D9D9D9; padding:10px 26px 10px 30px; clip-path: var(--chev); transition: background .15s ease;
}}
.stTabs [role="tab"]:first-child > [data-testid="stMarkdownContainer"] {{ padding-left:22px; }}
.stTabs [role="tab"] p {{ font-weight:700 !important; font-size:.84rem !important; color:{BLACK} !important; white-space:nowrap; }}
.stTabs [role="tab"]:hover > [data-testid="stMarkdownContainer"] {{ background:#EDEDED; }}
.stTabs [role="tab"][aria-selected="true"] > [data-testid="stMarkdownContainer"] {{ background:{YELLOW} !important; }}
.stTabs [role="tabpanel"] {{ padding-top:1.4rem; }}

/* ---------- Thẻ số liệu ---------- */
[data-testid="stMetric"] {{
    background:#fff; border:2px solid {BLACK}; border-radius:18px; padding:16px 20px;
    box-shadow:5px 5px 0 {BLACK};
}}
[data-testid="stMetricLabel"] p {{ font-weight:700; text-transform:uppercase; font-size:.78rem; letter-spacing:.04em; }}
[data-testid="stMetricValue"] {{ font-weight:900; font-size:2rem; }}

/* ---------- Nút ---------- */
.stButton > button, .stDownloadButton > button, [data-testid="stFileUploaderDropzone"] button {{
    background:{YELLOW}; color:{BLACK}; border:2px solid {BLACK}; border-radius:999px;
    font-weight:800; box-shadow:3px 3px 0 {BLACK}; padding:.45rem 1.3rem;
}}
.stButton > button:hover, .stDownloadButton > button:hover {{ background:#FFCB33; color:{BLACK}; border-color:{BLACK}; }}
.stButton > button:active, .stDownloadButton > button:active {{ transform:translate(2px,2px); box-shadow:1px 1px 0 {BLACK}; }}

/* ---------- Ô nhập, bảng, expander ---------- */
[data-baseweb="input"], [data-baseweb="select"] > div, [data-baseweb="textarea"] {{
    border:2px solid {BLACK} !important; border-radius:12px !important; background:#fff !important;
}}
[data-testid="stDataFrame"], [data-testid="stExpander"] details, [data-testid="stFileUploaderDropzone"] {{
    border:2px solid {BLACK} !important; border-radius:14px !important; overflow:hidden;
}}
[data-testid="stFileUploaderDropzone"] {{ background:{CREAM}; }}
[data-testid="stExpander"] summary p {{ font-weight:700; }}
[data-testid="stPlotlyChart"] {{ background:#fff; border:2px solid {BLACK}; border-radius:18px; padding:8px 6px 10px; overflow:visible; }}
[data-testid="stAlert"] {{ border-radius:14px; border:2px solid {BLACK}; }}

/* ---------- Khối tự dựng ---------- */
.section {{ display:flex; align-items:center; gap:12px; margin:1.6rem 0 .8rem; }}
.section .num {{ min-width:40px; height:40px; border-radius:50%; display:flex; align-items:center; justify-content:center;
                 background:radial-gradient(circle at 30% 30%, #6b5520, {BLACK} 70%); color:#fff; font-weight:800; font-size:.8rem; }}
.section h3 {{ margin:0 !important; padding:0 !important; font-size:1.25rem; text-transform:uppercase; }}
.section .line {{ flex:1; height:2px; background:repeating-linear-gradient(90deg,{BLACK} 0 6px,transparent 6px 10px); opacity:.35; }}

.pcard {{ background:#fff; border:2px solid {BLACK}; border-radius:20px; padding:16px 14px 18px; text-align:center;
          box-shadow:5px 5px 0 {BLACK}; height:100%; border-top:12px solid var(--pc, {YELLOW}); }}
.pcard .pstat {{ margin-top:4px; }}
.pcard .pname {{ font-weight:900; text-transform:uppercase; font-size:1.05rem; display:inline-block; padding:4px 16px;
                 border-radius:99px; border:2px solid {BLACK}; margin:6px 0 10px; }}
.pcard .pstat {{ font-size:.86rem; font-weight:600; line-height:1.5; }}
.pcard .pstat b {{ font-size:1.05rem; }}

.panel {{ border:2px solid {BLACK}; border-radius:20px; padding:20px 24px; height:100%; box-shadow:5px 5px 0 {BLACK}; }}
.panel h4 {{ font-weight:900; text-transform:uppercase; margin:0 0 .6rem; font-size:1.05rem; }}
.panel ul {{ margin:0; padding-left:1.1rem; }}
.panel li {{ margin:.28rem 0; font-weight:500; }}
.panel.yellow {{ background:{YELLOW}; }}
.panel.orange {{ background:{ORANGE}; color:#1E1A1A; }}
.panel.white {{ background:#fff; }}
.panel.cream {{ background:{CREAM}; }}

.chip {{ display:inline-block; padding:3px 12px; border-radius:99px; border:2px solid {BLACK};
         font-weight:700; font-size:.8rem; margin:2px 4px 2px 0; background:#fff; }}
.bk-logo {{ position:fixed; right:22px; bottom:18px; width:58px; opacity:.28; z-index:0; pointer-events:none; }}
.footer {{ margin-top:2.5rem; padding-top:.8rem; border-top:2px dashed #cfcfcf; font-size:.8rem; color:#666; }}
</style>
"""


@lru_cache(maxsize=None)
def img_uri(name: str) -> str:
    data = (ASSETS / name).read_bytes()
    return "data:image/png;base64," + base64.b64encode(data).decode()


def setup() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(f'<img class="bk-logo" src="{img_uri("hcmut.png")}">', unsafe_allow_html=True)
    _plotly_template()


def hero(kicker: str, title_html: str, subtitle: str) -> None:
    st.markdown(f"""
<div class="hero">
  <div class="hero-bars"><span></span><span></span></div>
  <div class="hero-text">
    <div class="kicker">{kicker}</div>
    <div class="hero-title">{title_html}</div>
    <div class="hero-sub">{subtitle}</div>
  </div>
</div>""", unsafe_allow_html=True)


def section(num: int, title: str) -> None:
    st.markdown(f'<div class="section"><div class="num">{num:02d}</div><h3>{title}</h3>'
                f'<div class="line"></div></div>', unsafe_allow_html=True)


def persona_card(name: str, lines: list[str]) -> str:
    color = PERSONA_COLORS.get(name, "#fff")
    body = "<br>".join(lines)
    return (f'<div class="pcard" style="--pc:{color}">'
            f'<div class="pname" style="background:{color}">{name}</div>'
            f'<div class="pstat">{body}</div></div>')


def panel(title: str, items: list[str] | str, tone: str = "white") -> str:
    inner = items if isinstance(items, str) else "<ul>" + "".join(f"<li>{x}</li>" for x in items) + "</ul>"
    return f'<div class="panel {tone}"><h4>{title}</h4>{inner}</div>'


def html(s: str) -> None:
    st.markdown(s, unsafe_allow_html=True)


def _plotly_template() -> None:
    t = go.layout.Template()
    t.layout = go.Layout(
        font=dict(family="Montserrat, sans-serif", color=BLACK, size=13),
        title=dict(font=dict(size=16, family="Montserrat, sans-serif"), x=0.02),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        colorway=[YELLOW, "#8FB3A9", "#D9A3A3", GREY, ORANGE, BLACK],
        xaxis=dict(gridcolor=LIGHT, linecolor=BLACK, linewidth=2, zeroline=False, ticks="outside"),
        yaxis=dict(gridcolor=LIGHT, linecolor=BLACK, linewidth=2, zeroline=False),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=12)),
        margin=dict(l=20, r=20, t=60, b=60),
        hoverlabel=dict(font=dict(family="Montserrat, sans-serif")),
        polar=dict(bgcolor="rgba(0,0,0,0)", radialaxis=dict(gridcolor=LIGHT, linecolor=GREY),
                   angularaxis=dict(gridcolor=LIGHT, linecolor=BLACK, linewidth=2)),
    )
    t.data.bar = [go.Bar(marker=dict(line=dict(color=BLACK, width=1.6)))]
    pio.templates["slide"] = t
    pio.templates.default = "plotly_white+slide"
