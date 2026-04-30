# 경로: app/pages/4_회귀모델.py

import streamlit as st
import sys
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.ui import (
    load_css, render_sidebar, page_header,
    section_badge, stat_card, chart_card_open, chart_card_close,
)

st.set_page_config(page_title="회귀 모델", layout="wide")
load_css()
render_sidebar()

page_header("회귀 모델 — 거래금액 예측")

# ── 더미 성능 지표 (모델별)
# TODO: 실제 평가 후 utils/metrics.py의 regression_metrics() 결과로 교체
DUMMY_METRICS = {
    "linear":       {"MAE": 2100, "RMSE": 3200, "R2": 0.72},
    "randomforest": {"MAE": 1400, "RMSE": 2100, "R2": 0.84},
    "lightgbm":     {"MAE": 1200, "RMSE": 1850, "R2": 0.89},
    "xgboost":      {"MAE": 1150, "RMSE": 1780, "R2": 0.90},
}

DUMMY_IMPORTANCE = {
    "linear":       [("전용면적", 0.38), ("지역", 0.25), ("건축연도", 0.12), ("층", 0.15), ("브랜드등급", 0.10)],
    "randomforest": [("지역",    0.28), ("전용면적", 0.32), ("브랜드등급", 0.18), ("건축연도", 0.14), ("층", 0.08)],
    "lightgbm":     [("지역",    0.30), ("전용면적", 0.28), ("브랜드등급", 0.22), ("건축연도", 0.12), ("층", 0.08)],
    "xgboost":      [("지역",    0.31), ("전용면적", 0.27), ("브랜드등급", 0.21), ("건축연도", 0.13), ("층", 0.08)],
}

MODEL_LABELS = ["📏 Linear", "🌲 RandomForest", "⚡ LightGBM", "🚀 XGBoost"]
MODEL_KEYS   = ["linear", "randomforest", "lightgbm", "xgboost"]

tabs = st.tabs(MODEL_LABELS)

for tab, model_key in zip(tabs, MODEL_KEYS):
    with tab:

        col_left, col_right = st.columns([1, 1.5])

        # ── 좌: 입력 폼
        with col_left:
            section_badge("📋", "예측 조건 입력")

            sido  = st.selectbox("시/도", ["서울특별시", "경기도", "부산광역시", "인천광역시"],
                                 key=f"sido_{model_key}")
            area  = st.slider("전용면적 (㎡)", 20, 200, 84,  key=f"area_{model_key}")
            floor = st.slider("층수",          1,  50,  15,  key=f"floor_{model_key}")
            year  = st.slider("건축연도",   1980, 2023, 2010, key=f"year_{model_key}")
            brand = st.selectbox("브랜드 등급", ["프리미엄", "일반브랜드", "공공(LH)", "기타"],
                                 key=f"brand_{model_key}")

            st.markdown("<br>", unsafe_allow_html=True)
            predict_btn = st.button("예측하기", type="primary",
                                    use_container_width=True, key=f"btn_{model_key}")

        # ── 우: 예측 결과
        with col_right:
            section_badge("💰", "예측 결과")

            # TODO: 아래 블록을 실제 모델 예측으로 교체
            # ── 백엔드 연결 포인트 ──────────────────────────────
            # from joblib import load
            # from utils.preprocessor import parse_date, fix_floor, map_brand_grade, scale_features
            #
            # input_df = pd.DataFrame([{
            #     "시도": sido, "전용면적": area, "층": floor,
            #     "건축연도": year, "브랜드": brand
            # }])
            # input_df = map_brand_grade(input_df, col="브랜드")
            # X = scale_features(input_df, method="standard")[0]
            #
            # model = load(f"models/regression/{model_key}.pkl")
            # predicted = int(model.predict(X)[0])
            # ────────────────────────────────────────────────────

            base_price = {"서울특별시": 120000, "경기도": 50000,
                          "부산광역시": 45000,  "인천광역시": 40000}
            brand_mult = {"프리미엄": 1.3, "일반브랜드": 1.0, "공공(LH)": 0.85, "기타": 0.8}
            predicted  = int(
                base_price.get(sido, 50000)
                * (area / 84)
                * (1 + (2023 - year) * -0.005)
                * brand_mult.get(brand, 1.0)
            )

            chart_card_open()
            st.markdown(f"""
            <div style="text-align:center; padding:20px 0;">
                <div style="font-size:13px; color:#6B7280; margin-bottom:8px;">예측 거래금액</div>
                <div style="font-size:38px; font-weight:800; color:#172B4D;">{predicted:,}만원</div>
                <div style="font-size:14px; color:#9CA3AF; margin-top:8px;">≈ {predicted/10000:.2f}억원</div>
                <div style="font-size:12px; color:#D1D5DB; margin-top:6px;">
                    신뢰구간: {int(predicted*0.95):,} ~ {int(predicted*1.05):,}만원
                </div>
            </div>
            """, unsafe_allow_html=True)
            chart_card_close()

        st.markdown("<br>", unsafe_allow_html=True)

        # ── 성능 지표
        section_badge("📊", "모델 성능 지표 (전체 테스트셋 기준)")
        # TODO: DB 또는 파일에서 사전 학습된 지표 로드
        # metrics = fetch_one("SELECT mae, rmse, r2 FROM model_metrics WHERE model=%s", (model_key,))
        m = DUMMY_METRICS[model_key]
        mc1, mc2, mc3 = st.columns(3)
        mc1.markdown(stat_card(f"{m['MAE']:,}만원", "MAE",  "평균 절대 오차"), unsafe_allow_html=True)
        mc2.markdown(stat_card(f"{m['RMSE']:,}만원","RMSE", "제곱근 평균 제곱 오차"), unsafe_allow_html=True)
        mc3.markdown(stat_card(f"{m['R2']}",         "R²",   "결정계수 (1에 가까울수록 좋음)"), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── 시각화
        section_badge("📈", "시각화", color="#F97316")
        ch1, ch2 = st.columns(2)

        with ch1:
            chart_card_open()
            # TODO: 실제 y_test, y_pred 로 교체
            # from utils.visualizer import plot_prediction_vs_actual
            # fig = plot_prediction_vs_actual(y_test, y_pred)
            np.random.seed(42)
            y_true = np.random.randint(30000, 150000, 300).astype(float)
            y_pred_vals = y_true + np.random.normal(0, m["MAE"], 300)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=y_true, y=y_pred_vals, mode="markers",
                marker=dict(color="#345BCB", opacity=0.45, size=5),
                name="예측"))
            fig.add_trace(go.Scatter(
                x=[30000, 150000], y=[30000, 150000], mode="lines",
                line=dict(color="#EF4444", dash="dash", width=1.5), name="y=x"))
            fig.update_layout(
                title="예측 vs 실제", xaxis_title="실제(만원)", yaxis_title="예측(만원)",
                height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, t=36, b=0),
                xaxis=dict(showgrid=True, gridcolor="#F0F4F8"),
                yaxis=dict(showgrid=True, gridcolor="#F0F4F8"),
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            chart_card_close()

        with ch2:
            chart_card_open()
            # TODO: 실제 피처 중요도 로드
            # from utils.visualizer import plot_feature_importance
            # fig = plot_feature_importance(feature_names, importances, top_n=10)
            feats = DUMMY_IMPORTANCE[model_key]
            fig = go.Figure(go.Bar(
                x=[f[1] for f in feats],
                y=[f[0] for f in feats],
                orientation="h",
                marker_color="#345BCB",
            ))
            fig.update_layout(
                title="피처 중요도", height=300,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, t=36, b=0),
                xaxis=dict(showgrid=False),
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            chart_card_close()
