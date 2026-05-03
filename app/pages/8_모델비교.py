# 경로: app/pages/8_모델비교.py

import streamlit as st
import sys
import os
import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from utils.ui import (
    load_css, render_sidebar, page_header,
    section_badge, chart_card_open, chart_card_close,
)

st.set_page_config(page_title="모델 비교", layout="wide")
load_css()
render_sidebar()

page_header("모델 비교 — 회귀 / 분류 성능 비교")

# TODO: 실제 학습 완료 후 DB 또는 파일에서 로드
# ── 백엔드 연결 포인트 ──────────────────────────────
# from utils.db import fetch_all
# rows = fetch_all("SELECT model, mae, rmse, r2 FROM regression_results ORDER BY r2 DESC", ())
# df_reg = pd.DataFrame(rows)
# ────────────────────────────────────────────────────

# 회귀 성능 (하드코딩 되어있지만 실제 결과임)
REG_DATA = pd.DataFrame({
    "모델":      ["Linear", "RandomForest", "LightGBM", "XGBoost"],
    "MAE":       [8692,      4106,           4020,       3976],
    "RMSE":      [12169,     5543,           5387,       5298],
    "R²":        [0.705,      0.906,           0.926,       0.929],
})

# 더미 분류 성능
# TODO: classification_metrics() 결과 연결
CLS_DATA = pd.DataFrame({
    "모델":        ["SVM", "DecisionTree", "Ensemble", "DNN"],
    "Accuracy":    [0.78,   0.81,           0.88,       0.90],
    "Precision":   [0.76,   0.80,           0.87,       0.89],
    "Recall":      [0.75,   0.79,           0.86,       0.88],
    "F1-Score":    [0.75,   0.79,           0.86,       0.89],
    "AUC":         [0.87,   0.89,           0.94,       0.95],
    "학습시간(s)": [18.2,    2.1,            22.4,       45.8],
})

tab_reg, tab_cls = st.tabs(["📊 회귀 모델 비교", "🏷️ 분류 모델 비교"])

# ══════════════════════════════
# 회귀 탭
# ══════════════════════════════
with tab_reg:
    section_badge("📋", "성능 지표 비교표")

    # 최고 성능 행 강조용 스타일 함수
    best_r2 = REG_DATA["R²"].max()
    def highlight_best_reg(row):
        if row["R²"] == best_r2:
            return ["background-color:#EFF6FF; font-weight:bold"] * len(row)
        return [""] * len(row)

    chart_card_open()
    st.dataframe(
        REG_DATA.style.apply(highlight_best_reg, axis=1)
            .format({"MAE": "{:,}", "RMSE": "{:,}", "R²": "{:.3f}", "학습시간(s)": "{:.1f}"}),
        use_container_width=True, hide_index=True,
    )
    chart_card_close()

    st.markdown("<br>", unsafe_allow_html=True)
    section_badge("📈", "시각화 비교", color="#F97316")
    ch1, ch2 = st.columns(2)

    with ch1:
        chart_card_open()
        # TODO: 실제 학습된 모델들의 R² 값으로 교체
        fig = go.Figure(go.Bar(
            x=REG_DATA["R²"],
            y=REG_DATA["모델"],
            orientation="h",
            marker_color=["#93C5FD","#60A5FA","#3B82F6","#2563EB","#1D4ED8"],
            text=REG_DATA["R²"].apply(lambda v: f"{v:.3f}"),
            textposition="outside",
        ))
        fig.update_layout(
            title="모델별 R² 비교", xaxis_range=[0.6, 1.0],
            height=280, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=40, b=0),
            xaxis=dict(showgrid=True, gridcolor="#F0F4F8"),
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        chart_card_close()

    with ch2:
        chart_card_open()
        fig = go.Figure()
        fig.add_trace(go.Bar(name="MAE",  x=REG_DATA["모델"], y=REG_DATA["MAE"],
                             marker_color="#345BCB"))
        fig.add_trace(go.Bar(name="RMSE", x=REG_DATA["모델"], y=REG_DATA["RMSE"],
                             marker_color="#F97316"))
        fig.update_layout(
            title="MAE / RMSE 비교", barmode="group",
            height=280, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=40, b=0),
            yaxis=dict(showgrid=True, gridcolor="#F0F4F8"),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        chart_card_close()

# ══════════════════════════════
# 분류 탭
# ══════════════════════════════
with tab_cls:
    section_badge("📋", "성능 지표 비교표")

    best_f1 = CLS_DATA["F1-Score"].max()
    def highlight_best_cls(row):
        if row["F1-Score"] == best_f1:
            return ["background-color:#EFF6FF; font-weight:bold"] * len(row)
        return [""] * len(row)

    chart_card_open()
    st.dataframe(
        CLS_DATA.style.apply(highlight_best_cls, axis=1)
            .format({c: "{:.2f}" for c in ["Accuracy","Precision","Recall","F1-Score","AUC"]}
                    | {"학습시간(s)": "{:.1f}"}),
        use_container_width=True, hide_index=True,
    )
    chart_card_close()

    st.markdown("<br>", unsafe_allow_html=True)
    section_badge("📈", "시각화 비교", color="#F97316")
    ch1, ch2 = st.columns(2)

    with ch1:
        chart_card_open()
        # TODO: 실제 분류 모델 F1 값으로 교체
        fig = go.Figure(go.Bar(
            x=CLS_DATA["F1-Score"],
            y=CLS_DATA["모델"],
            orientation="h",
            marker_color=["#93C5FD","#60A5FA","#2563EB","#1D4ED8"],
            text=CLS_DATA["F1-Score"].apply(lambda v: f"{v:.2f}"),
            textposition="outside",
        ))
        fig.update_layout(
            title="모델별 F1-Score 비교", xaxis_range=[0.6, 1.0],
            height=260, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=40, b=0),
            xaxis=dict(showgrid=True, gridcolor="#F0F4F8"),
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        chart_card_close()

    with ch2:
        chart_card_open()
        metrics_cols = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC"]
        fig = go.Figure()
        colors = ["#93C5FD","#60A5FA","#2563EB","#1D4ED8"]
        for i, row in CLS_DATA.iterrows():
            vals = [row[c] for c in metrics_cols] + [row[metrics_cols[0]]]
            cats = metrics_cols + [metrics_cols[0]]
            fig.add_trace(go.Scatterpolar(
                r=vals, theta=cats, fill="toself",
                name=row["모델"], line=dict(color=colors[i]),
                opacity=0.6,
            ))
        fig.update_layout(
            title="레이더 차트 비교",
            polar=dict(radialaxis=dict(range=[0.6, 1.0], showticklabels=False)),
            height=280, paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        chart_card_close()
