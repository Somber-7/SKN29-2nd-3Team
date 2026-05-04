# 경로: app/pages/2_입지분석.py

import streamlit as st
import sys
import os
import joblib
import plotly.graph_objects as go
import plotly.express as px

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from utils.ui import page_header, section_badge, info_card

page_header("입지 분석")

PKL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "cache", "location_data.pkl")


@st.cache_resource(show_spinner="분석 데이터 로딩 중...")
def load_analysis():
    if not os.path.exists(PKL_PATH):
        st.error(
            "사전 계산 파일이 없습니다. 아래 명령어를 먼저 실행해주세요.\n\n"
            "```\npython scripts/save_page_data.py\n```"
        )
        st.stop()
    return joblib.load(PKL_PATH)


data           = load_analysis()
corr           = data["corr"]
subway_df      = data["subway_df"]
station_grp_df = data["station_grp_df"]
school_grp_df  = data["school_grp_df"]
brand_df       = data["brand_df"]
era_df         = data["era_df"]

factors = [
    ("기준금리",    "기준금리",   "금리 상승 시 거래금액 영향"),
    ("인근 역 수",  "인근역수",   "반경 내 지하철역 수"),
    ("인근 학교 수","인근학교수", "반경 내 학교 수"),
    ("세대수",      "세대수",     "단지 규모 (세대수)"),
    ("건축연식",    "건축년도",   "건축년도 (최신일수록 높음)"),
    ("브랜드 여부", "브랜드여부", "브랜드 아파트 여부"),
]

# ── 섹션 1: 상관계수 카드
section_badge("📐", "주요 입지 요인 상관계수")
st.caption("거래금액(만원)과 각 요인 간 피어슨 상관계수 — 전체 실거래 데이터 기준")

row1 = st.columns(3)
row2 = st.columns(3)
for i, (label, col, sub) in enumerate(factors):
    val  = float(corr.get(col, 0) or 0)
    sign = "+" if val > 0 else ""
    target = row1[i] if i < 3 else row2[i - 3]
    target.markdown(info_card(label, f"{sign}{val:.3f}", sub), unsafe_allow_html=True)

st.divider()

# ── 섹션 2: 역세권
section_badge("🚇", "역세권 여부별 평균 거래금액", color="#10B981")
col_a, col_b = st.columns(2)

with col_a:
    fig1 = px.bar(
        subway_df, x="역세권", y="거래금액",
        color="역세권",
        color_discrete_map={"역세권 (1개 이상)": "#3B82F6", "비역세권": "#94A3B8"},
        text=subway_df["거래금액"].apply(lambda v: f"{int(v):,}만원"),
    )
    fig1.update_traces(textposition="outside")
    fig1.update_layout(
        height=300, showlegend=False,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="", yaxis=dict(title="평균 거래금액 (만원)", showgrid=True, gridcolor="#E2E8F0", tickformat=","),
    )
    st.plotly_chart(fig1, use_container_width=True)

with col_b:
    fig2 = px.bar(
        station_grp_df, x="인근 역 수", y="거래금액",
        color="거래금액", color_continuous_scale=["#BFDBFE", "#1D4ED8"],
        text=station_grp_df["거래금액"].apply(lambda v: f"{int(v):,}" if v == v else "-"),
    )
    fig2.update_traces(textposition="outside")
    fig2.update_layout(
        height=300, showlegend=False, coloraxis_showscale=False,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="인근 역 수", yaxis=dict(title="평균 거래금액 (만원)", showgrid=True, gridcolor="#E2E8F0", tickformat=","),
    )
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ── 섹션 3: 학교 수 + 브랜드
section_badge("🏫", "인근 학교 수 / 브랜드 여부별 평균 거래금액", color="#F97316")
col_c, col_d = st.columns(2)

with col_c:
    fig3 = px.bar(
        school_grp_df, x="인근 학교 수", y="거래금액",
        color="거래금액", color_continuous_scale=["#FED7AA", "#EA580C"],
        text=school_grp_df["거래금액"].apply(lambda v: f"{int(v):,}" if v == v else "-"),
    )
    fig3.update_traces(textposition="outside")
    fig3.update_layout(
        height=300, showlegend=False, coloraxis_showscale=False,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="인근 학교 수", yaxis=dict(title="평균 거래금액 (만원)", showgrid=True, gridcolor="#E2E8F0", tickformat=","),
    )
    st.plotly_chart(fig3, use_container_width=True)

with col_d:
    fig4 = px.bar(
        brand_df, x="구분", y="거래금액",
        color="구분",
        color_discrete_map={"브랜드": "#F97316", "비브랜드": "#94A3B8"},
        text=brand_df["거래금액"].apply(lambda v: f"{int(v):,}만원"),
    )
    fig4.update_traces(textposition="outside")
    fig4.update_layout(
        height=300, showlegend=False,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="", yaxis=dict(title="평균 거래금액 (만원)", showgrid=True, gridcolor="#E2E8F0", tickformat=","),
    )
    st.plotly_chart(fig4, use_container_width=True)

st.divider()

# ── 섹션 4: 건축연식
section_badge("🏗️", "건축연식 구간별 평균 거래금액", color="#345BCB")

fig5 = px.bar(
    era_df, x="건축연식", y="거래금액",
    color="거래금액", color_continuous_scale=["#BFDBFE", "#1D4ED8"],
    text=era_df["거래금액"].apply(lambda v: f"{int(v):,}" if v == v else "-"),
)
fig5.update_traces(textposition="outside")
fig5.update_layout(
    height=320, margin=dict(l=0, r=0, t=10, b=0),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    showlegend=False, coloraxis_showscale=False,
    xaxis_title="건축 연식", yaxis=dict(title="평균 거래금액 (만원)", showgrid=True, gridcolor="#E2E8F0", tickformat=","),
)
st.plotly_chart(fig5, use_container_width=True)
