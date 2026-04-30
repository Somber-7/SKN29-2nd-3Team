# 경로: app/pages/5_분류모델.py

import streamlit as st
import sys
import os
import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.ui import (
    load_css, render_sidebar, page_header,
    section_badge, stat_card, chart_card_open, chart_card_close,
)

st.set_page_config(page_title="분류 모델", layout="wide")
load_css()
render_sidebar()

page_header("분류 모델 — 브랜드 등급 분류")

GRADE_LABELS = ["프리미엄", "일반브랜드", "공공(LH)", "기타"]
GRADE_COLORS = ["#EF4444", "#F97316", "#3B82F6", "#9CA3AF"]

# TODO: 실제 학습 성능으로 교체
# metrics = fetch_all("SELECT * FROM model_metrics WHERE task='classification'")
DUMMY_METRICS = {
    "svm":          {"Accuracy": 0.78, "Precision": 0.76, "Recall": 0.75, "F1": 0.75, "AUC": 0.87},
    "decisiontree": {"Accuracy": 0.81, "Precision": 0.80, "Recall": 0.79, "F1": 0.79, "AUC": 0.89},
    "ensemble":     {"Accuracy": 0.88, "Precision": 0.87, "Recall": 0.86, "F1": 0.86, "AUC": 0.94},
}

# 더미 혼동행렬 (4×4)
DUMMY_CM = {
    "svm":          [[320, 40, 10, 30], [55, 180, 8, 20], [12, 5, 210, 8], [35, 22, 6, 340]],
    "decisiontree": [[340, 30, 8, 22],  [40, 195, 5, 18], [10, 4, 218, 6], [28, 18, 4, 358]],
    "ensemble":     [[365, 18, 5, 12],  [20, 215, 3, 10], [ 5, 2, 228, 3], [15, 10, 2, 381]],
}

MODEL_LABELS = ["🔷 SVM", "🌿 DecisionTree", "🎯 Ensemble"]
MODEL_KEYS   = ["svm", "decisiontree", "ensemble"]

tabs = st.tabs(MODEL_LABELS)

for tab, model_key in zip(tabs, MODEL_KEYS):
    with tab:

        col_left, col_right = st.columns([1, 1.5])

        # ── 좌: 파라미터 설정
        with col_left:
            section_badge("⚙️", "모델 파라미터")

            train_ratio  = st.slider("훈련 데이터 비율", 60, 90, 80, step=5,
                                     key=f"tr_{model_key}", format="%d%%")
            class_weight = st.selectbox("class_weight", ["balanced", "None"],
                                        key=f"cw_{model_key}")

            if model_key == "svm":
                kernel = st.selectbox("커널", ["rbf", "linear", "poly"], key="svm_kernel")
                c_val  = st.slider("C (정규화)", 0.1, 10.0, 1.0, step=0.1, key="svm_c")
            elif model_key == "decisiontree":
                max_depth = st.slider("max_depth", 2, 20, 8, key="dt_depth")
            else:
                n_estimators = st.slider("n_estimators", 50, 500, 200, step=50, key="ens_n")

            st.markdown("<br>", unsafe_allow_html=True)
            run_btn = st.button("학습 실행", type="primary",
                                use_container_width=True, key=f"run_{model_key}")

        # ── 우: 분류 결과
        with col_right:
            section_badge("🏷️", "분류 결과 (샘플 예측)")

            # TODO: 실제 예측으로 교체
            # ── 백엔드 연결 포인트 ──────────────────────────────
            # from joblib import load
            # model = load(f"models/classification/{model_key}.pkl")
            # proba = model.predict_proba(X_sample)[0]
            # pred_class = GRADE_LABELS[np.argmax(proba)]
            # ────────────────────────────────────────────────────

            m = DUMMY_METRICS[model_key]
            acc = m["Accuracy"]
            # 더미 확률 (모델 정확도 기반으로 첫 클래스에 높게 배분)
            proba = [acc * 0.9, (1-acc)*0.5, (1-acc)*0.3, (1-acc)*0.2]
            proba = [p / sum(proba) for p in proba]
            pred_idx  = int(np.argmax(proba))
            pred_grade = GRADE_LABELS[pred_idx]
            pred_color = GRADE_COLORS[pred_idx]

            chart_card_open()
            st.markdown(f"""
            <div style="text-align:center; padding:12px 0 20px 0;">
                <div style="font-size:13px; color:#6B7280; margin-bottom:10px;">예측 브랜드 등급</div>
                <div style="font-size:32px; font-weight:800; color:{pred_color};">
                    {pred_grade}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # 등급별 확률 바
            for label, prob, color in zip(GRADE_LABELS, proba, GRADE_COLORS):
                st.markdown(f"""
                <div style="margin:6px 0; font-size:13px; color:#4B5563;">
                    {label}
                    <span style="float:right; font-weight:700;">{prob*100:.1f}%</span>
                </div>
                <div style="background:#E5EAF2; border-radius:6px; height:8px; margin-bottom:10px;">
                    <div style="width:{prob*100:.1f}%; background:{color};
                                border-radius:6px; height:8px;"></div>
                </div>
                """, unsafe_allow_html=True)
            chart_card_close()

        st.markdown("<br>", unsafe_allow_html=True)

        # ── 성능 지표
        section_badge("📊", "모델 성능 지표")
        # TODO: utils/metrics.py의 classification_metrics() 결과 연결
        # metrics = classification_metrics(y_test, y_pred, y_prob, average="weighted")
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.markdown(stat_card(f"{m['Accuracy']:.2f}", "Accuracy"), unsafe_allow_html=True)
        mc2.markdown(stat_card(f"{m['Precision']:.2f}", "Precision"), unsafe_allow_html=True)
        mc3.markdown(stat_card(f"{m['Recall']:.2f}",   "Recall"), unsafe_allow_html=True)
        mc4.markdown(stat_card(f"{m['F1']:.2f}",        "F1-Score"), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── 시각화
        section_badge("📈", "시각화", color="#F97316")
        ch1, ch2 = st.columns(2)

        with ch1:
            chart_card_open()
            # TODO: 실제 혼동행렬 연결
            # from utils.visualizer import plot_confusion_matrix
            # fig = plot_confusion_matrix(y_test, y_pred, labels=GRADE_LABELS)
            cm = DUMMY_CM[model_key]
            fig = ff.create_annotated_heatmap(
                z=cm,
                x=GRADE_LABELS, y=GRADE_LABELS,
                colorscale=[[0, "#EFF6FF"], [1, "#1D4ED8"]],
                showscale=False,
            )
            fig.update_layout(
                title="혼동 행렬", height=320,
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, t=40, b=0),
                xaxis=dict(title="예측"),
                yaxis=dict(title="실제", autorange="reversed"),
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            chart_card_close()

        with ch2:
            chart_card_open()
            # TODO: 실제 ROC 곡선 연결
            # from utils.visualizer import plot_roc_curve
            # fig = plot_roc_curve(y_test, y_prob, label=model_key)
            np.random.seed(0)
            fpr = np.linspace(0, 1, 100)
            tpr = np.clip(fpr + np.random.uniform(0.1, 0.25, 100)
                          * (1 - fpr) + m["AUC"] - 0.5, 0, 1)
            tpr = np.sort(tpr)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines",
                line=dict(color="#345BCB", width=2),
                name=f"AUC = {m['AUC']:.2f}"))
            fig.add_trace(go.Scatter(
                x=[0,1], y=[0,1], mode="lines",
                line=dict(color="#D1D5DB", dash="dash", width=1), name="Random"))
            fig.update_layout(
                title=f"ROC 곡선  (AUC={m['AUC']:.2f})",
                xaxis_title="FPR", yaxis_title="TPR",
                height=320, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, t=40, b=0),
                xaxis=dict(showgrid=True, gridcolor="#F0F4F8"),
                yaxis=dict(showgrid=True, gridcolor="#F0F4F8"),
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            chart_card_close()
