# 경로: app/pages/9_프리미엄분석.py

import streamlit as st
import sys
import os
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from utils.ui import page_header, section_badge

st.markdown("""
<style>
.metric-card {
    background: #FFFFFF;
    border: 1px solid #DDE5F0;
    border-radius: 16px;
    padding: 22px 18px;
    text-align: center;
    box-shadow: 0 4px 12px rgba(15, 23, 42, 0.05);
}
.metric-label {
    font-size: 11px;
    color: #6B7280;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.metric-value {
    font-size: 30px;
    font-weight: 800;
    color: #172B4D;
    margin: 8px 0 4px;
}
.metric-sub {
    font-size: 12px;
    color: #9CA3AF;
}
.intro-box {
    background: linear-gradient(135deg, #EFF6FF 0%, #F0FDF4 100%);
    border: 1px solid #BFDBFE;
    border-radius: 14px;
    padding: 18px 22px;
    margin-bottom: 20px;
    font-size: 14px;
    color: #1E3A5F;
    line-height: 1.7;
}
</style>
""", unsafe_allow_html=True)

page_header("저·고평가 매물 분석 — 입지·브랜드 대비 가격 진단")

st.markdown("""
<div class="intro-box">
    <b>분석 방법</b>: XGBoost 모델이 전용면적·층·연식·위치 등 객관적 조건으로 <b>적정 시세</b>를 학습합니다.<br>
    <b>프리미엄률</b> = (실제 거래가 − 모델 적정가) ÷ 모델 적정가<br>
    양수면 고평가(적정가보다 비싸게 팔림), 음수면 저평가(적정가보다 싸게 팔림)를 의미합니다.
</div>
""", unsafe_allow_html=True)

PROJECT_ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS_PATH  = os.path.join(PROJECT_ROOT, "data", "models", "premium_analysis_results.pkl")

GRADE_ORDER  = ["큰 할인", "할인", "보통", "프리미엄", "고프리미엄"]
GRADE_COLORS = {
    "큰 할인":    "#2563EB",
    "할인":       "#60A5FA",
    "보통":       "#9CA3AF",
    "프리미엄":   "#F97316",
    "고프리미엄": "#DC2626",
}

GROUP_COLS = [
    ("역세권여부",  "역세권 여부",  ("역세권",  "비역세권")),
    ("학세권여부",  "학세권 여부",  ("학세권",  "비학세권")),
    ("브랜드구분",  "브랜드 여부",  ("브랜드",  "비브랜드")),
]


# =============================================================================
# 분석 결과 로드 (사전 계산된 pkl)
# =============================================================================

@st.cache_resource
def load_results():
    if not os.path.exists(RESULTS_PATH):
        st.error(
            "분석 결과 파일이 없습니다. 아래 명령어를 먼저 실행해주세요.\n\n"
            "```\npython scripts/save_models.py\n```"
        )
        st.stop()
    return joblib.load(RESULTS_PATH)


results       = load_results()
metrics       = results["metrics"]
sigungu_df    = results["sigungu_df"]
group_summaries = results["group_summaries"]
grade_counts  = results["grade_counts"]
scatter_sample = results["scatter_sample"]


# =============================================================================
# Section 1: KPI 카드
# =============================================================================

section_badge("📊", "모델 성능 — 적정가 예측 신뢰도")

def _metric_card(col, label, value, sub=""):
    col.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-sub">{sub}</div>
    </div>
    """, unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
_metric_card(c1, "분석 건수",   f"{metrics['rows']:,}건",          "검증 데이터")
_metric_card(c2, "R² 스코어",  f"{metrics['R2']:.3f}",            "1.0에 가까울수록 정확")
_metric_card(c3, "평균 오차율", f"{metrics['MAPE']:.1f}%",         "MAPE")
_metric_card(c4, "평균 오차",   f"{metrics['MAE']:,.0f}만원",       "MAE")

st.markdown("<br>", unsafe_allow_html=True)


# =============================================================================
# Section 2: 프리미엄 등급 분포
# =============================================================================

section_badge("🏷️", "프리미엄 등급 분포")

grade_data = pd.DataFrame({
    "등급": GRADE_ORDER,
    "건수": [grade_counts.get(g, 0) for g in GRADE_ORDER],
    "색상": [GRADE_COLORS[g] for g in GRADE_ORDER],
})
total_rows = grade_data["건수"].sum()
grade_data["비율"] = grade_data["건수"] / total_rows * 100

col_pie, col_bar = st.columns([1, 1.6])

with col_pie:
    fig_pie = go.Figure(go.Pie(
        labels=grade_data["등급"],
        values=grade_data["건수"],
        marker_colors=grade_data["색상"],
        hole=0.48,
        textinfo="percent+label",
        textfont_size=12,
        sort=False,
    ))
    fig_pie.update_layout(
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        margin=dict(t=10, b=10, l=10, r=10),
    )
    st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})

with col_bar:
    fig_bar = go.Figure(go.Bar(
        x=grade_data["등급"],
        y=grade_data["건수"],
        marker_color=grade_data["색상"],
        text=grade_data.apply(
            lambda r: f"{r['건수']:,}건<br>({r['비율']:.1f}%)", axis=1
        ),
        textposition="outside",
    ))
    fig_bar.update_layout(
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=30, b=20, l=10, r=10),
        yaxis=dict(showgrid=True, gridcolor="#F0F4F8", title="거래 건수"),
        xaxis=dict(categoryorder="array", categoryarray=GRADE_ORDER),
    )
    st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

st.markdown("<br>", unsafe_allow_html=True)


# =============================================================================
# Section 3: 지역별 고평가 / 저평가 TOP 15
# =============================================================================

section_badge("🗺️", "지역별 고평가 / 저평가 TOP 15", color="#7C3AED")

if sigungu_df.empty:
    st.warning("시군구 데이터가 충분하지 않습니다.")
else:
    rate_col = "중앙값프리미엄률" if "중앙값프리미엄률" in sigungu_df.columns else "평균프리미엄률"
    top_over  = sigungu_df.nlargest(15,  rate_col).reset_index(drop=True)
    top_under = sigungu_df.nsmallest(15, rate_col).sort_values(rate_col).reset_index(drop=True)

    col_over, col_under = st.columns(2)

    with col_over:
        st.markdown("#### 🔴 고평가 지역 TOP 15")
        st.caption("모델 적정가 대비 실거래가가 높은 지역 — 수요 대비 공급 부족 또는 시장 선호도 반영")
        fig_over = go.Figure(go.Bar(
            x=top_over[rate_col] * 100,
            y=top_over["시군구"],
            orientation="h",
            marker_color="#EF4444",
            text=top_over[rate_col].apply(lambda x: f"+{x*100:.1f}%"),
            textposition="outside",
        ))
        fig_over.update_layout(
            height=520,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=70, t=10, b=20),
            xaxis=dict(title="프리미엄률 (%)", showgrid=True, gridcolor="#F0F4F8", zeroline=True, zerolinecolor="#E5E7EB"),
            yaxis=dict(autorange="reversed"),
            font=dict(size=11),
        )
        st.plotly_chart(fig_over, use_container_width=True, config={"displayModeBar": False})

    with col_under:
        st.markdown("#### 🔵 저평가 지역 TOP 15")
        st.caption("모델 적정가 대비 실거래가가 낮은 지역 — 잠재 투자 관심 지역 후보")
        fig_under = go.Figure(go.Bar(
            x=top_under[rate_col] * 100,
            y=top_under["시군구"],
            orientation="h",
            marker_color="#3B82F6",
            text=top_under[rate_col].apply(lambda x: f"{x*100:.1f}%"),
            textposition="outside",
        ))
        fig_under.update_layout(
            height=520,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=70, t=10, b=20),
            xaxis=dict(title="프리미엄률 (%)", showgrid=True, gridcolor="#F0F4F8", zeroline=True, zerolinecolor="#E5E7EB"),
            yaxis=dict(autorange="reversed"),
            font=dict(size=11),
        )
        st.plotly_chart(fig_under, use_container_width=True, config={"displayModeBar": False})

st.markdown("<br>", unsafe_allow_html=True)


# =============================================================================
# Section 4: 입지·브랜드 요소별 프리미엄 비교
# =============================================================================

section_badge("📍", "입지·브랜드 요소별 프리미엄 비교", color="#059669")

if not group_summaries:
    st.warning("그룹 분석에 필요한 컬럼이 없습니다.")
else:
    rate_col = "중앙값프리미엄률" if "중앙값프리미엄률" in next(iter(group_summaries.values())).columns else "평균프리미엄률"
    amt_col  = "중앙값프리미엄금액" if "중앙값프리미엄금액" in next(iter(group_summaries.values())).columns else "평균프리미엄금액"

    visible_groups = [(col, label, pair) for col, label, pair in GROUP_COLS if col in group_summaries]
    cols = st.columns(len(visible_groups))

    for ax, (col, label, _) in zip(cols, visible_groups):
        summary = group_summaries[col]
        with ax:
            st.markdown(f"**{label}**")

            bars = []
            for _, row in summary.iterrows():
                group_name = str(row[col])
                rate = row[rate_col] * 100
                color = "#EF4444" if rate > 0 else "#3B82F6"
                bars.append((group_name, rate, color))

            fig = go.Figure()
            for name, rate, color in bars:
                fig.add_trace(go.Bar(
                    name=name,
                    x=[name],
                    y=[rate],
                    marker_color=color,
                    text=f"{rate:+.2f}%",
                    textposition="outside",
                    showlegend=False,
                ))

            fig.add_hline(y=0, line_dash="dash", line_color="#D1D5DB", line_width=1)
            fig.update_layout(
                height=300,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=30, b=10, l=10, r=10),
                yaxis=dict(
                    title="중앙값 프리미엄률 (%)",
                    showgrid=True,
                    gridcolor="#F0F4F8",
                ),
                xaxis=dict(showgrid=False),
                bargap=0.4,
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            # 그룹별 거래건수·프리미엄금액 요약
            for _, row in summary.iterrows():
                rate_pct = row[rate_col] * 100
                amt = row[amt_col]
                arrow = "▲" if rate_pct > 0 else "▼"
                color = "#DC2626" if rate_pct > 0 else "#2563EB"
                st.markdown(
                    f"<span style='color:{color};font-weight:700;'>{arrow} {row[col]}</span>"
                    f"&nbsp;&nbsp;{rate_pct:+.2f}% &nbsp;|&nbsp; {amt:+,.0f}만원"
                    f"&nbsp;&nbsp;<span style='color:#9CA3AF;font-size:12px;'>({row['거래건수']:,}건)</span>",
                    unsafe_allow_html=True,
                )

st.markdown("<br>", unsafe_allow_html=True)


# =============================================================================
# Section 5: 실거래가 vs 예측가 산점도
# =============================================================================

section_badge("🎯", "실거래가 vs 모델 적정가 산점도", color="#F97316")

max_val = float(max(scatter_sample["예측거래금액"].max(), scatter_sample["거래금액"].max()))

fig_scatter = go.Figure()

fig_scatter.add_trace(go.Scatter(
    x=scatter_sample["예측거래금액"],
    y=scatter_sample["거래금액"],
    mode="markers",
    marker=dict(
        color=scatter_sample["프리미엄률"],
        colorscale=[
            [0.0, "#2563EB"],
            [0.4, "#93C5FD"],
            [0.5, "#D1D5DB"],
            [0.6, "#FCA5A5"],
            [1.0, "#DC2626"],
        ],
        cmin=-0.3,
        cmax=0.3,
        size=4,
        opacity=0.45,
        colorbar=dict(
            title="프리미엄률",
            tickformat=".0%",
            thickness=14,
        ),
    ),
    text=scatter_sample.get("시군구", pd.Series([""] * len(scatter_sample))),
    hovertemplate=(
        "지역: %{text}<br>"
        "예측: %{x:,.0f}만원<br>"
        "실거래: %{y:,.0f}만원<extra></extra>"
    ),
    name="개별 거래",
))

fig_scatter.add_trace(go.Scatter(
    x=[0, max_val],
    y=[0, max_val],
    mode="lines",
    line=dict(color="#6B7280", dash="dash", width=1.5),
    name="예측 = 실거래 (기준선)",
))

fig_scatter.update_layout(
    height=460,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(title="모델 예측가 (만원)", showgrid=True, gridcolor="#F0F4F8"),
    yaxis=dict(title="실제 거래가 (만원)", showgrid=True, gridcolor="#F0F4F8"),
    legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
    margin=dict(t=30, b=50, l=60, r=40),
)

st.plotly_chart(fig_scatter, use_container_width=True, config={"displayModeBar": False})
st.caption(
    "기준선 위 (붉은 점): 고평가 — 적정가보다 비싸게 거래  |  "
    "기준선 아래 (파란 점): 저평가 — 적정가보다 싸게 거래"
)

st.markdown("<br>", unsafe_allow_html=True)


# =============================================================================
# Section 6: 지역별 상세 테이블
# =============================================================================

section_badge("📋", "지역별 프리미엄 상세 테이블", color="#6B7280")

if not sigungu_df.empty:
    display_df = sigungu_df.copy()

    pct_cols = [c for c in ["평균프리미엄률", "중앙값프리미엄률"] if c in display_df.columns]
    amt_cols = [c for c in ["평균프리미엄금액", "중앙값프리미엄금액"] if c in display_df.columns]

    for c in pct_cols:
        display_df[c] = display_df[c].apply(lambda x: f"{x*100:+.2f}%")
    for c in amt_cols:
        display_df[c] = display_df[c].apply(lambda x: f"{x:+,.0f}만원")
    display_df["거래건수"] = display_df["거래건수"].apply(lambda x: f"{x:,}건")

    st.dataframe(display_df, use_container_width=True, height=420)
