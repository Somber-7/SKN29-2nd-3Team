# 경로: app/pages/7_신경망.py

import streamlit as st
import sys
import os
import time
import numpy as np
import plotly.graph_objects as go

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from utils.ui import (
    load_css, render_sidebar, page_header,
    section_badge, stat_card,
)
from utils.db import load_apart_deals

st.set_page_config(page_title="신경망 (DNN)", layout="wide")
load_css()
render_sidebar()

page_header("신경망 — DNN 거래금액 예측 (PyTorch)")

import json

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
META_PATH    = os.path.join(PROJECT_ROOT, "data", "models", "dnn_regressor_meta.json")

# ── 전체 데이터 학습 결과 (고정)
if os.path.exists(META_PATH):
    with open(META_PATH, encoding="utf-8") as f:
        meta = json.load(f)

    section_badge("🏆", "전체 데이터 학습 결과 (사전 학습)")
    st.caption(f"500만 건 전체 데이터 · 은닉층 {meta['config']['hidden_layers']}개 · 뉴런 {meta['config']['neurons']} · Dropout {meta['config']['dropout']} · BN · Early Stopping")

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.markdown(stat_card(f"{meta['metrics']['MAE']:,.0f}만원",  "MAE",      ""),                                   unsafe_allow_html=True)
    mc2.markdown(stat_card(f"{meta['metrics']['RMSE']:,.0f}만원", "RMSE",     ""),                                   unsafe_allow_html=True)
    mc3.markdown(stat_card(f"{meta['metrics']['R2']:.4f}",        "R²",       ""),                                   unsafe_allow_html=True)
    mc4.markdown(stat_card(f"{meta['elapsed']:.0f}초",            "학습 시간", f"샘플 {meta['sample_size']:,}건"),   unsafe_allow_html=True)

    x = list(range(1, len(meta["train_losses"]) + 1))
    fig0 = go.Figure()
    fig0.add_trace(go.Scatter(x=x, y=meta["train_losses"], mode="lines",
                               line=dict(color="#345BCB", width=2), name="Train Loss"))
    fig0.add_trace(go.Scatter(x=x, y=meta["val_losses"],   mode="lines",
                               line=dict(color="#F97316", width=2, dash="dot"), name="Val Loss"))
    if meta["best_epoch"] < len(x):
        fig0.add_vline(x=meta["best_epoch"], line_dash="dash", line_color="#10B981",
                       annotation_text=f"Best (ep={meta['best_epoch']})")
    fig0.update_layout(
        height=260, margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Epoch", showgrid=True, gridcolor="#F0F4F8"),
        yaxis=dict(title="Loss (MSE)", showgrid=True, gridcolor="#F0F4F8"),
        legend=dict(orientation="h", y=1.05),
    )
    st.plotly_chart(fig0, use_container_width=True)

    st.divider()

@st.cache_data(show_spinner="데이터 로딩 중...", ttl=3600)
def get_df():
    import pandas as pd
    df = load_apart_deals()
    df["거래일"]  = pd.to_datetime(df["거래일"])
    df["거래금액"] = pd.to_numeric(df["거래금액"], errors="coerce")
    return df.dropna(subset=["거래금액"])


col_left, col_right = st.columns([1, 2])

# ── 좌측: 하이퍼파라미터 설정
with col_left:
    section_badge("🧠", "네트워크 구조 설정")

    n_hidden   = st.slider("은닉층 수",    1, 6,   3)
    n_neurons  = st.select_slider("층당 뉴런 수",
                                   options=[64, 128, 256, 512], value=256)
    dropout    = st.slider("Dropout 비율", 0.0, 0.5, 0.3, step=0.1)
    lr         = st.select_slider("Learning Rate",
                                   options=[0.1, 0.01, 0.001, 0.0001], value=0.001)
    batch_size = st.select_slider("Batch Size",
                                   options=[128, 256, 512, 1024], value=256)
    epochs     = st.slider("Epoch", 5, 100, 30, step=5)
    use_bn     = st.checkbox("Batch Normalization", value=True)
    use_es     = st.checkbox("Early Stopping (patience=5)", value=True)
    sample_k   = st.select_slider("학습 샘플 수",
                                   options=[10_000, 30_000, 50_000, 100_000, 200_000],
                                   value=50_000,
                                   format_func=lambda x: f"{x:,}건")

    # 네트워크 구조 미리보기
    st.markdown("<br>", unsafe_allow_html=True)
    layer_lines = ["입력층 (수치+범주형 피처)"]
    for i in range(n_hidden):
        line = f"Dense({n_neurons}) → ReLU"
        if use_bn:   line += " → BN"
        if dropout:  line += f" → Drop({dropout})"
        layer_lines.append(line)
    layer_lines.append("출력층 (거래금액, 만원)")
    st.markdown(
        "".join([
            f'<div style="font-size:12px;color:#6B7280;margin:2px 0;">{"↓  " if i>0 else "   "}{l}</div>'
            for i, l in enumerate(layer_lines)
        ]),
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    train_btn = st.button("🚀 학습 시작", type="primary", use_container_width=True)

# ── 우측: 학습 결과
with col_right:
    section_badge("📈", "학습 현황")

    if train_btn:
        df = get_df()

        # session_state 초기화
        st.session_state.pop("dnn_result", None)

        from models.regression.dnn_regressor import DNNRegressorModel

        model = DNNRegressorModel(
            hidden_layers=n_hidden,
            neurons=n_neurons,
            dropout=dropout,
            lr=lr,
            batch_size=batch_size,
            epochs=epochs,
            use_bn=use_bn,
            early_stopping=use_es,
            patience=5,
            sample_size=sample_k,
        )

        # 진행 상황 표시
        progress_bar  = st.progress(0.0, text="학습 준비 중...")
        status_text   = st.empty()
        chart_placeholder = st.empty()

        train_losses, val_losses = [], []
        t_start = time.time()

        def on_progress(epoch, total, tl, vl):
            train_losses.append(tl)
            val_losses.append(vl)
            pct = epoch / total
            elapsed = time.time() - t_start
            eta     = (elapsed / epoch) * (total - epoch) if epoch > 0 else 0
            progress_bar.progress(pct, text=f"Epoch {epoch}/{total}  |  train loss: {tl:.4f}  |  val loss: {vl:.4f}  |  ETA: {eta:.0f}s")

            # 실시간 차트 (5 epoch마다 갱신)
            if epoch % 5 == 0 or epoch == total:
                fig = go.Figure()
                x = list(range(1, len(train_losses) + 1))
                fig.add_trace(go.Scatter(x=x, y=train_losses, mode="lines",
                                         line=dict(color="#345BCB", width=2), name="Train Loss"))
                fig.add_trace(go.Scatter(x=x, y=val_losses,   mode="lines",
                                         line=dict(color="#F97316", width=2, dash="dot"), name="Val Loss"))
                fig.update_layout(
                    height=300, margin=dict(l=0, r=0, t=10, b=0),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(title="Epoch", showgrid=True, gridcolor="#F0F4F8"),
                    yaxis=dict(title="Loss (MSE)", showgrid=True, gridcolor="#F0F4F8"),
                    legend=dict(orientation="h", y=1.05),
                )
                chart_placeholder.plotly_chart(fig, use_container_width=True)

        with st.spinner(""):
            model.fit_from_dataframe(df, progress_callback=on_progress)

        progress_bar.progress(1.0, text=f"✅ 학습 완료! ({model.elapsed_:.1f}초)")

        # Early Stop 표시
        if use_es and model.best_epoch_ < epochs:
            status_text.info(f"Early Stopping: epoch {model.best_epoch_}에서 최적 모델 저장")

        # 최종 차트 (early stop 라인 포함)
        fig_final = go.Figure()
        x = list(range(1, len(model.train_losses_) + 1))
        fig_final.add_trace(go.Scatter(x=x, y=model.train_losses_, mode="lines",
                                        line=dict(color="#345BCB", width=2), name="Train Loss"))
        fig_final.add_trace(go.Scatter(x=x, y=model.val_losses_,   mode="lines",
                                        line=dict(color="#F97316", width=2, dash="dot"), name="Val Loss"))
        if use_es and model.best_epoch_ < len(x):
            fig_final.add_vline(x=model.best_epoch_, line_dash="dash", line_color="#10B981",
                                 annotation_text=f"Best (ep={model.best_epoch_})")
        fig_final.update_layout(
            height=300, margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title="Epoch", showgrid=True, gridcolor="#F0F4F8"),
            yaxis=dict(title="Loss (MSE)", showgrid=True, gridcolor="#F0F4F8"),
            legend=dict(orientation="h", y=1.05),
        )
        chart_placeholder.plotly_chart(fig_final, use_container_width=True)

        st.session_state["dnn_result"] = model

    elif "dnn_result" in st.session_state:
        model = st.session_state["dnn_result"]
        x = list(range(1, len(model.train_losses_) + 1))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=model.train_losses_, mode="lines",
                                  line=dict(color="#345BCB", width=2), name="Train Loss"))
        fig.add_trace(go.Scatter(x=x, y=model.val_losses_,   mode="lines",
                                  line=dict(color="#F97316", width=2, dash="dot"), name="Val Loss"))
        if use_es and model.best_epoch_ < len(x):
            fig.add_vline(x=model.best_epoch_, line_dash="dash", line_color="#10B981",
                           annotation_text=f"Best (ep={model.best_epoch_})")
        fig.update_layout(
            height=300, margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title="Epoch", showgrid=True, gridcolor="#F0F4F8"),
            yaxis=dict(title="Loss (MSE)", showgrid=True, gridcolor="#F0F4F8"),
            legend=dict(orientation="h", y=1.05),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("좌측에서 하이퍼파라미터를 설정하고 학습 시작 버튼을 눌러주세요.")

# ── 성능 지표 + 예측 (학습 완료 후)
if "dnn_result" in st.session_state:
    model = st.session_state["dnn_result"]
    m = model.metrics_

    st.divider()
    section_badge("📊", "최종 성능 지표 (Validation Set)")
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.markdown(stat_card(f"{m['MAE']:,.0f}만원", "MAE",  ""),  unsafe_allow_html=True)
    mc2.markdown(stat_card(f"{m['RMSE']:,.0f}만원","RMSE", ""),  unsafe_allow_html=True)
    mc3.markdown(stat_card(f"{m['R2']:.4f}",       "R²",   ""),  unsafe_allow_html=True)
    mc4.markdown(stat_card(f"{model.elapsed_:.1f}초", "학습 시간", f"샘플 {sample_k:,}건"), unsafe_allow_html=True)

    st.divider()
    section_badge("🔮", "거래금액 예측 — 내 모델 vs 사전학습 모델", color="#F97316")

    import pandas as pd
    from models.regression.dnn_regressor import DNNRegressorModel as _DNN

    MODEL_PT = os.path.join(PROJECT_ROOT, "data", "models", "dnn_regressor.pt")

    p1, p2, p3 = st.columns(3)
    with p1:
        p_area      = st.number_input("전용면적 (㎡)", 10.0, 300.0, 84.0, step=1.0)
        p_floor     = st.number_input("층", 1, 80, 15)
        p_built     = st.number_input("건축년도", 1980, 2025, 2015)
        p_region    = st.text_input("지역코드", "11680", help="강남구: 11680")
    with p2:
        p_rate      = st.number_input("기준금리", 0.0, 10.0, 3.5, step=0.25)
        p_school    = st.number_input("인근학교수", 0, 20, 3)
        p_station   = st.number_input("인근역수",  0, 20, 2)
    with p3:
        p_household = st.number_input("세대수", 0, 10000, 1000, step=50)
        p_brand     = st.radio("브랜드여부", [0, 1],
                                format_func=lambda x: "비브랜드" if x == 0 else "브랜드",
                                horizontal=True)
        p_date      = st.date_input("거래일", value=pd.Timestamp("2023-01-01"))

    if st.button("예측하기", type="primary"):
        input_df = pd.DataFrame([{
            "전용면적": p_area, "층": p_floor, "건축년도": p_built,
            "지역코드": str(p_region), "거래일": pd.Timestamp(p_date),
            "기준금리": p_rate, "인근학교수": p_school,
            "인근역수": p_station, "세대수": p_household, "브랜드여부": p_brand,
        }])

        res_col1, res_col2 = st.columns(2)

        # 내 모델 예측
        with res_col1:
            try:
                prepared = model.prepare_dataframe(input_df, need_target=False)
                pred_my  = model.predict_single(prepared[model.feature_columns])
                st.markdown(f"""
                <div style="background:#EFF6FF;border:1px solid #BFDBFE;border-radius:12px;padding:20px;text-align:center;">
                    <div style="font-size:13px;color:#6B7280;margin-bottom:6px;">내 모델 예측</div>
                    <div style="font-size:13px;color:#64748B;margin-bottom:4px;">(샘플 {sample_k:,}건 · {n_hidden}층 · {n_neurons}뉴런)</div>
                    <div style="font-size:32px;font-weight:800;color:#1D4ED8;">{pred_my:,.0f}만원</div>
                    <div style="font-size:14px;color:#9CA3AF;margin-top:4px;">≈ {pred_my/10000:.2f}억원</div>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"내 모델 오류: {e}")

        # 사전학습 모델 예측
        with res_col2:
            if os.path.exists(MODEL_PT):
                try:
                    pretrained = _DNN.load(MODEL_PT)
                    prepared2  = pretrained.prepare_dataframe(input_df, need_target=False)
                    pred_pre   = pretrained.predict_single(prepared2[pretrained.feature_columns])
                    diff       = pred_my - pred_pre
                    diff_str   = f"{'▲' if diff > 0 else '▼'} {abs(diff):,.0f}만원 {'높음' if diff > 0 else '낮음'}"
                    st.markdown(f"""
                    <div style="background:#F0FDF4;border:1px solid #BBF7D0;border-radius:12px;padding:20px;text-align:center;">
                        <div style="font-size:13px;color:#6B7280;margin-bottom:6px;">사전학습 모델 예측</div>
                        <div style="font-size:13px;color:#64748B;margin-bottom:4px;">(500만 건 · 4층 · 256뉴런)</div>
                        <div style="font-size:32px;font-weight:800;color:#15803D;">{pred_pre:,.0f}만원</div>
                        <div style="font-size:14px;color:#9CA3AF;margin-top:4px;">≈ {pred_pre/10000:.2f}억원</div>
                        <div style="font-size:13px;color:#6B7280;margin-top:6px;">내 모델 대비 {diff_str}</div>
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"사전학습 모델 오류: {e}")
            else:
                st.warning("사전학습 모델 파일이 없습니다. (scripts/train_dnn.py 실행 필요)")
