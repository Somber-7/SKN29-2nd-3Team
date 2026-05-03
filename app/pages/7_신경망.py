# 경로: app/pages/7_신경망.py

import streamlit as st
import sys
import os
import numpy as np
import plotly.graph_objects as go

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from utils.ui import (
    load_css, render_sidebar, page_header,
    section_badge, stat_card, chart_card_open, chart_card_close,
)

st.set_page_config(page_title="신경망 (DNN)", layout="wide")
load_css()
render_sidebar()

page_header("신경망 — DNN 모델 (PyTorch)")

tab_reg, tab_cls = st.tabs(["📊 회귀 (거래금액 예측)", "🏷️ 분류 (브랜드 등급)"])


def make_loss_curve(epochs, final_train, final_val, noise=0.04):
    """더미 학습 곡선 생성"""
    np.random.seed(42)
    x = np.arange(1, epochs + 1)
    train_loss = final_train + (1.5 - final_train) * np.exp(-x / (epochs * 0.3))
    train_loss += np.random.normal(0, noise, epochs)
    val_loss = train_loss + np.random.uniform(0.02, 0.08, epochs)
    val_loss = np.clip(val_loss, final_val, None)
    return x, train_loss, val_loss


# ══════════════════════════════
# 회귀 탭
# ══════════════════════════════
with tab_reg:
    col_left, col_right = st.columns([1, 2])

    with col_left:
        section_badge("🧠", "네트워크 구조 설정")

        n_hidden   = st.slider("은닉층 수",       1, 6,    3,     key="reg_hidden")
        n_neurons  = st.select_slider("층당 뉴런 수",
                                      options=[64, 128, 256, 512], value=256, key="reg_neurons")
        dropout    = st.slider("Dropout 비율",    0.0, 0.6, 0.3,  step=0.1, key="reg_drop")
        lr         = st.select_slider("Learning Rate",
                                      options=[0.1, 0.01, 0.001, 0.0001], value=0.001, key="reg_lr")
        batch_size = st.select_slider("Batch Size",
                                      options=[64, 128, 256, 512, 1024], value=256, key="reg_bs")
        epochs     = st.slider("Epoch",          10, 200, 50, step=10, key="reg_ep")
        use_bn     = st.checkbox("Batch Normalization 사용", value=True, key="reg_bn")
        use_es     = st.checkbox("Early Stopping 사용",     value=True, key="reg_es")

        # 네트워크 구조 미리보기
        st.markdown("<br>", unsafe_allow_html=True)
        layers = ["입력층"] + [f"Dense({n_neurons}) + ReLU" +
                               (" + BN" if use_bn else "") +
                               f" + Drop({dropout})" for _ in range(n_hidden)] + ["출력층(1)"]
        st.markdown(
            "<br>".join([f'<div style="font-size:12px;color:#6B7280;">{"→  " if i>0 else ""}{l}</div>'
                         for i, l in enumerate(layers)]),
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        train_btn = st.button("학습 시작", type="primary",
                              use_container_width=True, key="reg_train")

    with col_right:
        section_badge("📈", "학습 곡선")

        # TODO: 실제 PyTorch 학습 연결
        # ── 백엔드 연결 포인트 ──────────────────────────────
        # from models.neural_network.dnn_regressor import DNNRegressor
        # model = DNNRegressor(
        #     hidden_layers=n_hidden, neurons=n_neurons,
        #     dropout=dropout, use_bn=use_bn
        # )
        # history = model.fit(
        #     X_train, y_train,
        #     lr=lr, batch_size=batch_size, epochs=epochs,
        #     early_stopping=use_es
        # )
        # train_losses = history["train_loss"]
        # val_losses   = history["val_loss"]
        # ────────────────────────────────────────────────────

        x, train_loss, val_loss = make_loss_curve(epochs, 0.08, 0.11)

        chart_card_open()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x, y=train_loss, mode="lines",
            line=dict(color="#345BCB", width=2), name="Train Loss"))
        fig.add_trace(go.Scatter(
            x=x, y=val_loss, mode="lines",
            line=dict(color="#F97316", width=2, dash="dot"), name="Val Loss"))
        if use_es:
            best_ep = int(np.argmin(val_loss)) + 1
            fig.add_vline(x=best_ep, line_dash="dash", line_color="#10B981",
                          annotation_text=f"Early Stop (ep={best_ep})")
        fig.update_layout(
            title="학습 / 검증 손실 (MSE Loss)",
            xaxis_title="Epoch", yaxis_title="Loss",
            height=340, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=40, b=0),
            xaxis=dict(showgrid=True, gridcolor="#F0F4F8"),
            yaxis=dict(showgrid=True, gridcolor="#F0F4F8"),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        chart_card_close()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 성능 지표
    section_badge("📊", "최종 성능 지표 (Test Set)")
    # TODO: model.evaluate(X_test, y_test) 결과 연결
    # from utils.metrics import regression_metrics
    # metrics = regression_metrics(y_test, y_pred)
    mc1, mc2, mc3 = st.columns(3)
    mc1.markdown(stat_card("1,100만원", "MAE",  ""), unsafe_allow_html=True)
    mc2.markdown(stat_card("1,700만원", "RMSE", ""), unsafe_allow_html=True)
    mc3.markdown(stat_card("0.91",      "R²",   ""), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 예측 vs 실제
    section_badge("🎯", "예측 결과 시각화", color="#F97316")
    chart_card_open()
    # TODO: plot_prediction_vs_actual(y_test, y_pred) 교체
    np.random.seed(7)
    y_t = np.random.randint(30000, 150000, 300).astype(float)
    y_p = y_t + np.random.normal(0, 1100, 300)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=y_t, y=y_p, mode="markers",
        marker=dict(color="#345BCB", opacity=0.45, size=5), name="예측"))
    fig2.add_trace(go.Scatter(
        x=[30000, 150000], y=[30000, 150000], mode="lines",
        line=dict(color="#EF4444", dash="dash", width=1.5), name="y=x"))
    fig2.update_layout(
        xaxis_title="실제(만원)", yaxis_title="예측(만원)",
        height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(showgrid=True, gridcolor="#F0F4F8"),
        yaxis=dict(showgrid=True, gridcolor="#F0F4F8"),
    )
    st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
    chart_card_close()


# ══════════════════════════════
# 분류 탭
# ══════════════════════════════
with tab_cls:
    col_left, col_right = st.columns([1, 2])

    with col_left:
        section_badge("🧠", "네트워크 구조 설정")

        n_hidden_c  = st.slider("은닉층 수",      1, 6,    3,    key="cls_hidden")
        n_neurons_c = st.select_slider("층당 뉴런 수",
                                       options=[64, 128, 256, 512], value=256, key="cls_neurons")
        dropout_c   = st.slider("Dropout 비율",   0.0, 0.6, 0.3, step=0.1, key="cls_drop")
        lr_c        = st.select_slider("Learning Rate",
                                       options=[0.1, 0.01, 0.001, 0.0001], value=0.001, key="cls_lr")
        epochs_c    = st.slider("Epoch",         10, 200, 50, step=10, key="cls_ep")
        use_bn_c    = st.checkbox("Batch Normalization", value=True,  key="cls_bn")
        use_es_c    = st.checkbox("Early Stopping",      value=True,  key="cls_es")
        pos_weight  = st.slider("pos_weight (불균형 조정)", 1.0, 10.0, 3.0, step=0.5, key="cls_pw")

        st.markdown("<br>", unsafe_allow_html=True)
        train_btn_c = st.button("학습 시작", type="primary",
                                use_container_width=True, key="cls_train")

    with col_right:
        section_badge("📈", "학습 곡선")

        # TODO: DNN 분류 모델 학습 연결
        # ── 백엔드 연결 포인트 ──────────────────────────────
        # from models.neural_network.dnn_classifier import DNNClassifier
        # model = DNNClassifier(
        #     hidden_layers=n_hidden_c, neurons=n_neurons_c,
        #     dropout=dropout_c, use_bn=use_bn_c
        # )
        # criterion = torch.nn.CrossEntropyLoss()
        # history = model.fit(
        #     X_train, y_train,
        #     lr=lr_c, epochs=epochs_c,
        #     early_stopping=use_es_c
        # )
        # ────────────────────────────────────────────────────

        xc, tl_c, vl_c = make_loss_curve(epochs_c, 0.25, 0.32, noise=0.015)

        chart_card_open()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=xc, y=tl_c, mode="lines",
            line=dict(color="#345BCB", width=2), name="Train Loss"))
        fig.add_trace(go.Scatter(
            x=xc, y=vl_c, mode="lines",
            line=dict(color="#F97316", width=2, dash="dot"), name="Val Loss"))
        if use_es_c:
            best_c = int(np.argmin(vl_c)) + 1
            fig.add_vline(x=best_c, line_dash="dash", line_color="#10B981",
                          annotation_text=f"Early Stop (ep={best_c})")
        fig.update_layout(
            title="학습 / 검증 손실 (Cross Entropy)",
            xaxis_title="Epoch", yaxis_title="Loss",
            height=340, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=40, b=0),
            xaxis=dict(showgrid=True, gridcolor="#F0F4F8"),
            yaxis=dict(showgrid=True, gridcolor="#F0F4F8"),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        chart_card_close()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 성능 지표
    section_badge("📊", "최종 성능 지표 (Test Set)")
    # TODO: classification_metrics(y_test, y_pred, y_prob) 결과 연결
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.markdown(stat_card("0.90", "Accuracy",  ""), unsafe_allow_html=True)
    mc2.markdown(stat_card("0.89", "Precision", ""), unsafe_allow_html=True)
    mc3.markdown(stat_card("0.88", "Recall",    ""), unsafe_allow_html=True)
    mc4.markdown(stat_card("0.95", "AUC",       ""), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 혼동 행렬
    section_badge("🎯", "혼동 행렬", color="#F97316")
    import plotly.figure_factory as ff
    GRADE_LABELS = ["프리미엄", "일반브랜드", "공공(LH)", "기타"]
    # TODO: get_confusion_matrix(y_test, y_pred) 연결
    dummy_cm = [[370, 15, 4, 11], [18, 220, 2, 8], [4, 1, 230, 3], [12, 8, 2, 388]]
    chart_card_open()
    fig_cm = ff.create_annotated_heatmap(
        z=dummy_cm,
        x=GRADE_LABELS, y=GRADE_LABELS,
        colorscale=[[0, "#EFF6FF"], [1, "#1D4ED8"]],
        showscale=False,
    )
    fig_cm.update_layout(
        title="혼동 행렬", height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=40, b=0),
        xaxis=dict(title="예측"),
        yaxis=dict(title="실제", autorange="reversed"),
    )
    st.plotly_chart(fig_cm, use_container_width=True, config={"displayModeBar": False})
    chart_card_close()
