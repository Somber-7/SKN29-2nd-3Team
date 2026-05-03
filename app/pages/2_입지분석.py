# 경로: app/pages/2_입지분석.py

import streamlit as st
import sys
import os
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from utils.ui import load_css, render_sidebar, page_header, section_badge, info_card, chart_card_open, chart_card_close
from utils.db import load_apart_deals

st.set_page_config(page_title="입지 분석", layout="wide")
load_css()
render_sidebar()

page_header("입지 분석")


@st.cache_data(show_spinner="분석 데이터 로딩 중...", ttl=3600)
def load_analysis() -> pd.DataFrame:
    df = load_apart_deals()
    cols = ["기준금리", "인근역수", "인근학교수", "세대수", "건축년도", "거래금액", "브랜드여부"]
    df = df[cols].copy()
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna()


df = load_analysis()

# ── 상관계수
corr = df.corr()["거래금액"]
factors = [
    ("기준금리",   "기준금리",   "금리 상승 시 거래금액 영향"),
    ("인근 역 수", "인근역수",   "반경 내 지하철역 수"),
    ("인근 학교 수","인근학교수", "반경 내 학교 수"),
    ("세대수",     "세대수",     "단지 규모 (세대수)"),
    ("건축연식",   "건축년도",   "건축년도 (최신일수록 높음)"),
    ("브랜드 여부","브랜드여부", "브랜드 아파트 여부"),
]

# ── 섹션 1: 상관계수 카드
section_badge("📐", "주요 입지 요인 상관계수")
st.caption("거래금액(만원)과 각 요인 간 피어슨 상관계수 — 전체 실거래 데이터 기준")

row1 = st.columns(3)
row2 = st.columns(3)
for i, (label, col, sub) in enumerate(factors):
    val  = corr[col]
    sign = "+" if val > 0 else ""
    target = row1[i] if i < 3 else row2[i - 3]
    target.markdown(info_card(label, f"{sign}{val:.3f}", sub), unsafe_allow_html=True)

st.divider()

# ── 섹션 2: 역세권
section_badge("🚇", "역세권 여부별 평균 거래금액", color="#10B981")
col_a, col_b = st.columns(2)

with col_a:
    chart_card_open()
    df["역세권"] = df["인근역수"].apply(lambda x: "역세권 (1개 이상)" if x >= 1 else "비역세권")
    grp = df.groupby("역세권")["거래금액"].mean().reset_index()
    fig1 = px.bar(
        grp, x="역세권", y="거래금액",
        color="역세권",
        color_discrete_map={"역세권 (1개 이상)": "#3B82F6", "비역세권": "#94A3B8"},
        text=grp["거래금액"].apply(lambda v: f"{int(v):,}만원"),
    )
    fig1.update_traces(textposition="outside")
    fig1.update_layout(
        height=300, showlegend=False,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="", yaxis=dict(title="평균 거래금액 (만원)", showgrid=True, gridcolor="#E2E8F0", tickformat=","),
    )
    st.plotly_chart(fig1, use_container_width=True)
    chart_card_close()

with col_b:
    chart_card_open()
    df["역수_구간"] = df["인근역수"].clip(upper=5).astype(int).astype(str)
    df.loc[df["인근역수"] >= 5, "역수_구간"] = "5+"
    order = ["0", "1", "2", "3", "4", "5+"]
    grp2 = df.groupby("역수_구간")["거래금액"].mean().reindex(order).reset_index()
    grp2.columns = ["인근 역 수", "거래금액"]
    fig2 = px.bar(
        grp2, x="인근 역 수", y="거래금액",
        color="거래금액", color_continuous_scale=["#BFDBFE", "#1D4ED8"],
        text=grp2["거래금액"].apply(lambda v: f"{int(v):,}" if pd.notna(v) else "-"),
    )
    fig2.update_traces(textposition="outside")
    fig2.update_layout(
        height=300, showlegend=False, coloraxis_showscale=False,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="인근 역 수", yaxis=dict(title="평균 거래금액 (만원)", showgrid=True, gridcolor="#E2E8F0", tickformat=","),
    )
    st.plotly_chart(fig2, use_container_width=True)
    chart_card_close()

st.divider()

# ── 섹션 3: 학교 수 + 브랜드
section_badge("🏫", "인근 학교 수 / 브랜드 여부별 평균 거래금액", color="#F97316")
col_c, col_d = st.columns(2)

with col_c:
    chart_card_open()
    df["학교수_구간"] = df["인근학교수"].clip(upper=6).astype(int).astype(str)
    df.loc[df["인근학교수"] >= 6, "학교수_구간"] = "6+"
    order_s = ["0", "1", "2", "3", "4", "5", "6+"]
    grp3 = df.groupby("학교수_구간")["거래금액"].mean().reindex(order_s).reset_index()
    grp3.columns = ["인근 학교 수", "거래금액"]
    fig3 = px.bar(
        grp3, x="인근 학교 수", y="거래금액",
        color="거래금액", color_continuous_scale=["#FED7AA", "#EA580C"],
        text=grp3["거래금액"].apply(lambda v: f"{int(v):,}" if pd.notna(v) else "-"),
    )
    fig3.update_traces(textposition="outside")
    fig3.update_layout(
        height=300, showlegend=False, coloraxis_showscale=False,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="인근 학교 수", yaxis=dict(title="평균 거래금액 (만원)", showgrid=True, gridcolor="#E2E8F0", tickformat=","),
    )
    st.plotly_chart(fig3, use_container_width=True)
    chart_card_close()

with col_d:
    chart_card_open()
    grp4 = df.groupby("브랜드여부")["거래금액"].mean().reset_index()
    grp4["구분"] = grp4["브랜드여부"].map({1: "브랜드", 0: "비브랜드"})
    fig4 = px.bar(
        grp4, x="구분", y="거래금액",
        color="구분",
        color_discrete_map={"브랜드": "#F97316", "비브랜드": "#94A3B8"},
        text=grp4["거래금액"].apply(lambda v: f"{int(v):,}만원"),
    )
    fig4.update_traces(textposition="outside")
    fig4.update_layout(
        height=300, showlegend=False,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="", yaxis=dict(title="평균 거래금액 (만원)", showgrid=True, gridcolor="#E2E8F0", tickformat=","),
    )
    st.plotly_chart(fig4, use_container_width=True)
    chart_card_close()

st.divider()

# ── 섹션 4: 건축연식
section_badge("🏗️", "건축연식 구간별 평균 거래금액", color="#345BCB")

bins   = [1970, 1980, 1990, 2000, 2005, 2010, 2015, 2020, 2025]
labels = ["~1980", "80~90", "90~00", "00~05", "05~10", "10~15", "15~20", "20~"]
df["연식_구간"] = pd.cut(df["건축년도"], bins=bins, labels=labels, right=True)
grp5 = df.groupby("연식_구간", observed=True)["거래금액"].mean().reset_index()
grp5.columns = ["건축연식", "거래금액"]

chart_card_open()
fig5 = px.bar(
    grp5, x="건축연식", y="거래금액",
    color="거래금액", color_continuous_scale=["#BFDBFE", "#1D4ED8"],
    text=grp5["거래금액"].apply(lambda v: f"{int(v):,}" if pd.notna(v) else "-"),
)
fig5.update_traces(textposition="outside")
fig5.update_layout(
    height=320, margin=dict(l=0, r=0, t=10, b=0),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    showlegend=False, coloraxis_showscale=False,
    xaxis_title="건축 연식", yaxis=dict(title="평균 거래금액 (만원)", showgrid=True, gridcolor="#E2E8F0", tickformat=","),
)
st.plotly_chart(fig5, use_container_width=True)
chart_card_close()
