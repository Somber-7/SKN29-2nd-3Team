# 경로: app/Home.py

import streamlit as st
import sys
import os
import joblib
import plotly.express as px
import plotly.graph_objects as go

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.ui import load_css, render_sidebar, page_header, section_badge, stat_card

st.set_page_config(
    page_title="부동산 분석 플랫폼",
    layout="wide",
    initial_sidebar_state="expanded",
)
load_css()
render_sidebar()

page_header("개요")

PKL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "cache", "home_data.pkl")


@st.cache_resource(show_spinner="데이터 로딩 중...")
def load_home_data():
    if not os.path.exists(PKL_PATH):
        st.error(
            "사전 계산 파일이 없습니다. 아래 명령어를 먼저 실행해주세요.\n\n"
            "```\npython scripts/save_page_data.py\n```"
        )
        st.stop()
    return joblib.load(PKL_PATH)


data = load_home_data()
cnt      = data["cnt"]
avg_amt  = data["avg_amt"]
min_date = data["min_date"]
max_date = data["max_date"]
top_sg   = data["top_sg"]
top_sido = data["top_sido"]
top_cnt  = data["top_cnt"]
monthly  = data["monthly"]
yearly   = data["yearly"]

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

st.divider()

# ── 연도별 거래량 + 평균가
section_badge("📋", "연도별 요약", color="#10B981")
col_a, col_b = st.columns(2)

with col_a:
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

with col_b:
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
