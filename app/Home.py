# 경로: app/Home.py

import streamlit as st
import sys
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.ui import load_css, render_sidebar, page_header, section_badge, stat_card, chart_card_open, chart_card_close
from utils.db import load_apart_deals, fetch_all

st.set_page_config(
    page_title="부동산 분석 플랫폼",
    layout="wide",
    initial_sidebar_state="expanded",
)
load_css()
render_sidebar()

page_header("개요")


@st.cache_data(show_spinner="데이터 로딩 중...", ttl=3600)
def load_home_data():
    df = load_apart_deals()  # parquet 캐시 사용
    df["거래일"] = pd.to_datetime(df["거래일"])
    df["거래금액"] = pd.to_numeric(df["거래금액"], errors="coerce")

    # 시군구 → 시도 매핑
    sido_map_rows = fetch_all("SELECT sigungu, sido FROM tbl_sigungu_stats")
    sg_to_sido = {r["sigungu"]: r["sido"] for r in sido_map_rows}
    df["시도"] = df["시군구"].map(sg_to_sido)

    # 요약
    cnt       = len(df)
    avg_amt   = int(df["거래금액"].mean().round(0))
    min_date  = df["거래일"].min().strftime("%Y.%m")
    max_date  = df["거래일"].max().strftime("%Y.%m")
    top_sg    = df["시군구"].value_counts().idxmax()
    top_sido  = sg_to_sido.get(top_sg, "")
    top_cnt   = int(df["시군구"].value_counts().max())

    # 월별
    df["연월"] = df["거래일"].dt.to_period("M").dt.to_timestamp()
    monthly = df.groupby("연월").agg(거래량=("거래금액", "count"), 평균가=("거래금액", "mean")).reset_index()
    monthly["평균가"] = monthly["평균가"].round(0).astype(int)

    # 연도별
    df["연도"] = df["거래일"].dt.year
    yearly = df.groupby("연도").agg(거래량=("거래금액", "count"), 평균가=("거래금액", "mean")).reset_index()
    yearly["평균가"] = yearly["평균가"].round(0).astype(int)

    return cnt, avg_amt, min_date, max_date, top_sg, top_sido, top_cnt, monthly, yearly


cnt, avg_amt, min_date, max_date, top_sg, top_sido, top_cnt, monthly, yearly = load_home_data()

# ── 상단 통계 카드
section_badge("📊", "데이터 요약")
c1, c2, c3, c4 = st.columns(4)
top_short = top_sg.split()[-1] if top_sg else "-"

c1.markdown(stat_card(f"{cnt:,}건",       "총 거래 건수",    f"{min_date} ~ {max_date}"),       unsafe_allow_html=True)
c2.markdown(stat_card(f"{avg_amt:,}만원", "평균 거래금액",   "전국 전체 기간 평균"),              unsafe_allow_html=True)
c3.markdown(stat_card(top_short,          "최다 거래 구/군", f"{top_sido} · {top_cnt:,}건"),     unsafe_allow_html=True)
c4.markdown(stat_card("2015 ~ 2023",      "데이터 기간",     "실거래가 신고 기준"),               unsafe_allow_html=True)

st.divider()

# ── 월별 거래량 + 평균가 듀얼축 차트
section_badge("📈", "전국 월별 거래 현황", color="#F97316")
chart_card_open()

fig = go.Figure()
fig.add_trace(go.Bar(
    x=monthly["연월"], y=monthly["거래량"],
    name="거래량",
    marker_color="#93C5FD",
    yaxis="y1",
    hovertemplate="%{x|%Y년 %m월}<br>거래량: %{y:,}건<extra></extra>",
))
fig.add_trace(go.Scatter(
    x=monthly["연월"], y=monthly["평균가"],
    name="평균 거래금액",
    line=dict(color="#EF4444", width=2),
    yaxis="y2",
    hovertemplate="%{x|%Y년 %m월}<br>평균가: %{y:,.0f}만원<extra></extra>",
))
fig.update_layout(
    height=380,
    margin=dict(l=0, r=0, t=10, b=0),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    xaxis=dict(showgrid=False, tickformat="%Y"),
    yaxis=dict(title="거래량 (건)", showgrid=True, gridcolor="#E2E8F0", tickformat=","),
    yaxis2=dict(title="평균가 (만원)", overlaying="y", side="right", showgrid=False, tickformat=","),
    hovermode="x unified",
)
st.plotly_chart(fig, use_container_width=True)
chart_card_close()

st.divider()

# ── 연도별 거래량 + 평균가
section_badge("📋", "연도별 요약", color="#10B981")
col_a, col_b = st.columns(2)

with col_a:
    chart_card_open()
    st.markdown("**연도별 거래량**")
    fig_bar = px.bar(
        yearly, x="연도", y="거래량",
        color="거래량",
        color_continuous_scale=["#BFDBFE", "#1D4ED8"],
        text=yearly["거래량"].apply(lambda v: f"{v:,}"),
    )
    fig_bar.update_traces(textposition="outside")
    fig_bar.update_layout(
        height=300, margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False, coloraxis_showscale=False,
        xaxis=dict(dtick=1, tickformat="d", showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="#E2E8F0", tickformat=","),
    )
    st.plotly_chart(fig_bar, use_container_width=True)
    chart_card_close()

with col_b:
    chart_card_open()
    st.markdown("**연도별 평균 거래금액 추이**")
    fig_line = px.line(
        yearly, x="연도", y="평균가",
        labels={"평균가": "평균 거래금액 (만원)"},
        markers=True,
    )
    fig_line.update_traces(line_color="#F97316", marker=dict(size=8, color="#F97316"))
    fig_line.update_layout(
        height=300, margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(dtick=1, tickformat="d", showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="#E2E8F0", tickformat=",", title="평균 거래금액 (만원)"),
    )
    st.plotly_chart(fig_line, use_container_width=True)
    chart_card_close()
