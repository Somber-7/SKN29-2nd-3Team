# 경로: app/pages/1_가격추이분석.py

import streamlit as st
import sys
import os
import joblib
import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from utils.ui import load_css, render_sidebar, page_header, section_badge

st.set_page_config(page_title="가격 추이 분석", layout="wide")
load_css()
render_sidebar()

page_header("가격 추이 분석")

PKL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "cache", "price_trend_data.pkl")


@st.cache_resource(show_spinner="데이터 로딩 중...")
def load_precomputed():
    if not os.path.exists(PKL_PATH):
        st.error(
            "사전 계산 파일이 없습니다. 아래 명령어를 먼저 실행해주세요.\n\n"
            "```\npython scripts/save_page_data.py\n```"
        )
        st.stop()
    return joblib.load(PKL_PATH)


data            = load_precomputed()
national        = data["national"]
sido_monthly    = data["sido_monthly"]
sigungu_monthly = data["sigungu_monthly"]
sido_list       = data["sido_list"]
sigungu_map     = data["sigungu_map"]
sg_full_map     = data["sg_full_map"]

# ── 지역 선택
section_badge("📍", "지역 선택")
col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    sido = st.selectbox("시/도", ["전국"] + sido_list, format_func=lambda x: f"🗺  {x}")

with col2:
    sg_options = sigungu_map.get(sido, []) if sido != "전국" else []
    sigungu_short = st.selectbox(
        "구/군", ["전체"] + sg_options,
        disabled=(sido == "전국"),
        format_func=lambda x: f"🏘  {x}",
    )

with col3:
    freq = st.selectbox("집계 단위", ["월별", "분기별", "연별"])

st.divider()


def resample_col(df: pd.DataFrame, col: str, freq: str) -> pd.DataFrame:
    s = df.set_index("date")[col]
    if freq == "분기별":
        s = s.resample("QE").mean()
    elif freq == "연별":
        s = s.resample("YE").mean()
    return s.round(0).reset_index()


# ── 데이터 준비
if sido == "전국":
    plot_data    = resample_col(national, "전국", freq)
    region_label = "전국"
    main_col     = "전국"
elif sigungu_short == "전체":
    sub = sido_monthly[sido_monthly["시도"] == sido][["date", "avg"]].rename(columns={"avg": sido})
    merged = pd.merge(sub, national, on="date", how="left")
    plot_data    = resample_col(merged, sido, freq).merge(
                      resample_col(merged[["date", "전국"]], "전국", freq),
                      on="date", how="left")
    region_label = sido
    main_col     = sido
else:
    full = sg_full_map.get(sido, {}).get(sigungu_short, sigungu_short)
    sub  = sigungu_monthly[sigungu_monthly["시군구"] == full][["date", "avg"]].rename(columns={"avg": sigungu_short})
    merged = pd.merge(sub, national, on="date", how="left")
    plot_data    = resample_col(merged, sigungu_short, freq).merge(
                      resample_col(merged[["date", "전국"]], "전국", freq),
                      on="date", how="left")
    region_label = sigungu_short
    main_col     = sigungu_short

# ── 추이 라인차트
section_badge("📈", f"{region_label} 평균 거래금액 추이", color="#F97316")

fig = go.Figure()
if main_col == "전국":
    fig.add_trace(go.Scatter(
        x=plot_data["date"], y=plot_data["전국"],
        name="전국", line=dict(color="#3B82F6", width=2.5),
        mode="lines+markers" if freq != "월별" else "lines",
        marker=dict(size=6),
        hovertemplate="전국: %{y:,.0f}만원<extra></extra>",
    ))
else:
    fig.add_trace(go.Scatter(
        x=plot_data["date"], y=plot_data[main_col],
        name=main_col, line=dict(color="#3B82F6", width=2.5),
        mode="lines+markers" if freq != "월별" else "lines",
        marker=dict(size=6),
        hovertemplate=f"{main_col}: %{{y:,.0f}}만원<extra></extra>",
    ))
    if "전국" in plot_data.columns:
        fig.add_trace(go.Scatter(
            x=plot_data["date"], y=plot_data["전국"],
            name="전국 평균", line=dict(color="#94A3B8", width=1.5, dash="dot"),
            mode="lines",
            hovertemplate="전국: %{y:,.0f}만원<extra></extra>",
        ))

fig.update_layout(
    height=400, margin=dict(l=0, r=0, t=10, b=0),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=True, gridcolor="#E2E8F0", tickformat=",", title="평균 거래금액 (만원)"),
    hovermode="x unified",
)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── 연도별 bar
section_badge("📊", f"{region_label} 연도별 평균 거래금액", color="#345BCB")

yearly = resample_col(pd.DataFrame({"date": plot_data["date"], main_col: plot_data[main_col]}), main_col, "연별")
yearly["year"] = yearly["date"].dt.year
if "전국" in plot_data.columns and main_col != "전국":
    yearly_nat = resample_col(plot_data[["date", "전국"]], "전국", "연별")
    yearly_nat["year"] = yearly_nat["date"].dt.year
    yearly = pd.merge(yearly, yearly_nat[["year", "전국"]], on="year", how="left")

fig2 = go.Figure()
fig2.add_trace(go.Bar(
    x=yearly["year"], y=yearly[main_col],
    name=main_col, marker_color="#3B82F6",
    text=yearly[main_col].apply(lambda v: f"{int(v):,}" if pd.notna(v) else ""),
    textposition="outside",
    hovertemplate="연도: %{x}<br>평균가: %{y:,.0f}만원<extra></extra>",
))
if "전국" in yearly.columns and main_col != "전국":
    fig2.add_trace(go.Scatter(
        x=yearly["year"], y=yearly["전국"],
        name="전국 평균", line=dict(color="#94A3B8", width=2, dash="dot"),
        mode="lines+markers", marker=dict(size=7),
        hovertemplate="전국: %{y:,.0f}만원<extra></extra>",
    ))
fig2.update_layout(
    height=320, margin=dict(l=0, r=0, t=30, b=0),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    xaxis=dict(dtick=1, tickformat="d", showgrid=False),
    yaxis=dict(showgrid=True, gridcolor="#E2E8F0", tickformat=",", title="평균 거래금액 (만원)"),
    bargap=0.35,
)
st.plotly_chart(fig2, use_container_width=True)
