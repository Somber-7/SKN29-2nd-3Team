# 경로: app/pages/8_모델분석.py

import streamlit as st
import sys
import os
import json
import joblib
import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from utils.ui import (
    load_css, render_sidebar, page_header,
    section_badge, stat_card,
)

st.set_page_config(page_title="모델 분석", layout="wide")
load_css()
render_sidebar()

page_header("모델 분석 — 회귀 / 분류 / 군집화")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_DIR    = os.path.join(PROJECT_ROOT, "data", "models")

# ── 회귀 데이터 (기존 고정값 + DNN 메타에서 로드)
_reg_rows = [
    {"모델": "Linear",       "MAE": 8692,  "RMSE": 12169, "R²": 0.705,  "학습시간": "-",   "데이터": "-"},
    {"모델": "RandomForest", "MAE": 4106,  "RMSE": 5543,  "R²": 0.906,  "학습시간": "-",   "데이터": "-"},
    {"모델": "LightGBM",     "MAE": 4020,  "RMSE": 5387,  "R²": 0.926,  "학습시간": "-",   "데이터": "-"},
    {"모델": "XGBoost",      "MAE": 3976,  "RMSE": 5298,  "R²": 0.929,  "학습시간": "-",   "데이터": "-"},
]

_dnn_meta_path = os.path.join(MODEL_DIR, "dnn_regressor_meta.json")
if os.path.exists(_dnn_meta_path):
    with open(_dnn_meta_path, encoding="utf-8") as f:
        _dnn_meta = json.load(f)
    _reg_rows.append({
        "모델":     "DNN (PyTorch)",
        "MAE":      round(_dnn_meta["metrics"]["MAE"]),
        "RMSE":     round(_dnn_meta["metrics"]["RMSE"]),
        "R²":       round(_dnn_meta["metrics"]["R2"], 4),
        "학습시간": f"{_dnn_meta['elapsed']:.0f}s",
        "데이터":   f"{_dnn_meta['sample_size']:,}건",
    })

REG_DATA = pd.DataFrame(_reg_rows)

# ── 분류 데이터
_cls_path  = os.path.join(MODEL_DIR, "brand_grade_classifier.pkl")
_cls_model = joblib.load(_cls_path)
_cm        = _cls_model.metrics_

tab_reg, tab_cls, tab_clust = st.tabs(["📊 회귀 모델 비교", "🏷️ 분류 모델 분석", "🔵 군집화 모델 분석"])

# ══════════════════════════════════════════════════════
# 회귀 탭
# ══════════════════════════════════════════════════════
with tab_reg:
    section_badge("📋", "성능 지표 비교표")
    st.caption("Linear / RandomForest / LightGBM / XGBoost: 기존 학습 결과 | DNN: 500만 건 전체 PyTorch GPU 학습")

    best_r2 = REG_DATA["R²"].max()
    def highlight_best_reg(row):
        if row["R²"] == best_r2:
            return ["background-color:#EFF6FF; font-weight:bold"] * len(row)
        return [""] * len(row)

    st.dataframe(
        REG_DATA.style.apply(highlight_best_reg, axis=1).format(
            {"MAE": "{:,}", "RMSE": "{:,}", "R²": "{:.4f}"}, na_rep="-"
        ),
        use_container_width=True, hide_index=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    section_badge("📈", "시각화 비교", color="#F97316")
    ch1, ch2 = st.columns(2)

    n_models = len(REG_DATA)
    colors = ["#BFDBFE", "#93C5FD", "#60A5FA", "#3B82F6", "#1D4ED8"][:n_models]

    with ch1:
        fig = go.Figure(go.Bar(
            x=REG_DATA["R²"], y=REG_DATA["모델"],
            orientation="h", marker_color=colors,
            text=REG_DATA["R²"].apply(lambda v: f"{v:.4f}"),
            textposition="outside",
        ))
        fig.update_layout(
            title="모델별 R² 비교", xaxis_range=[0.6, 1.0],
            height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=40, b=0),
            xaxis=dict(showgrid=True, gridcolor="#F0F4F8"),
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with ch2:
        mae_vals  = pd.to_numeric(REG_DATA["MAE"],  errors="coerce")
        rmse_vals = pd.to_numeric(REG_DATA["RMSE"], errors="coerce")
        fig = go.Figure()
        fig.add_trace(go.Bar(name="MAE",  x=REG_DATA["모델"], y=mae_vals,  marker_color="#345BCB"))
        fig.add_trace(go.Bar(name="RMSE", x=REG_DATA["모델"], y=rmse_vals, marker_color="#F97316"))
        fig.update_layout(
            title="MAE / RMSE 비교 (만원)", barmode="group",
            height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=40, b=0),
            yaxis=dict(showgrid=True, gridcolor="#F0F4F8", tickformat=","),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ══════════════════════════════════════════════════════
# 분류 탭
# ══════════════════════════════════════════════════════
with tab_cls:
    # ── 최종 성능 stat 카드
    section_badge("📋", "최종 성능 지표")
    st.caption("브랜드명·브랜드여부 피처 제외 · 입지·가격·단지 특성만으로 4등급 분류 · XGBoost GPU · 500만 건 전체 학습")

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(stat_card(f"{_cm['Accuracy']:.4f}", "Accuracy",  "초기 0.808 → 최종"), unsafe_allow_html=True)
    c2.markdown(stat_card(f"{_cm['F1']:.4f}",       "F1 (weighted)", "초기 0.779 → 최종"), unsafe_allow_html=True)
    c3.markdown(stat_card(f"{_cm['train_rows']:,}건", "학습 데이터", ""), unsafe_allow_html=True)
    c4.markdown(stat_card(f"{_cm['test_rows']:,}건",  "테스트 데이터", ""), unsafe_allow_html=True)

    st.divider()

    # ── 튜닝 과정 — 단계별 Accuracy / F1 개선
    section_badge("📈", "단계별 튜닝 과정", color="#F97316")
    st.caption("5단계 튜닝을 통해 Accuracy 0.808 → 0.960, 일반브랜드 F1 0.396 → 0.930")

    TUNE_STEPS = ["초기 모델", "1단계\nOOM 수정", "2단계\n클래스 가중치", "3단계\n키워드 보강", "4단계\n하이퍼파라미터", "5단계\nEarly Stopping"]
    TUNE_ACC   = [0.808,       0.808,              0.800,               0.745,               0.908,               0.960]
    TUNE_F1    = [0.779,       0.779,              0.798,               None,                None,                0.960]
    TUNE_BRAND_F1 = [0.396,   0.396,              None,                0.632,               0.855,               0.930]

    ch1, ch2 = st.columns(2)
    with ch1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=TUNE_STEPS, y=TUNE_ACC,
            mode="lines+markers+text",
            name="Accuracy",
            line=dict(color="#3B82F6", width=2),
            marker=dict(size=9),
            text=[f"{v}" if v else "" for v in TUNE_ACC],
            textposition="top center",
        ))
        fig.add_trace(go.Scatter(
            x=TUNE_STEPS, y=TUNE_F1,
            mode="lines+markers+text",
            name="F1 (weighted)",
            line=dict(color="#10B981", width=2, dash="dot"),
            marker=dict(size=9),
            text=[f"{v}" if v else "" for v in TUNE_F1],
            textposition="bottom center",
            connectgaps=True,
        ))
        fig.add_hline(y=0.960, line_dash="dash", line_color="#F97316",
                      annotation_text="최종 0.960")
        fig.update_layout(
            title="Accuracy / F1 튜닝 과정",
            height=320, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=40, b=0),
            yaxis=dict(range=[0.7, 1.02], showgrid=True, gridcolor="#F0F4F8"),
            xaxis=dict(showgrid=False),
            legend=dict(orientation="h", y=1.15),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with ch2:
        # 일반브랜드 F1 개선 (데이터 있는 단계만)
        brand_steps = ["초기 모델", "3단계\n키워드 보강", "4단계\n하이퍼파라미터", "5단계\nEarly Stopping"]
        brand_vals  = [0.396,       0.632,               0.855,               0.930]
        fig2 = go.Figure(go.Bar(
            x=brand_steps, y=brand_vals,
            marker_color=["#BFDBFE", "#60A5FA", "#3B82F6", "#1D4ED8"],
            text=[f"{v}" for v in brand_vals],
            textposition="outside",
        ))
        fig2.update_layout(
            title="일반브랜드 F1 개선",
            height=320, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=40, b=0),
            yaxis=dict(range=[0, 1.1], showgrid=True, gridcolor="#F0F4F8"),
            xaxis=dict(showgrid=False),
        )
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    st.divider()

    # ── 최종 클래스별 성능
    section_badge("🎯", "최종 클래스별 성능", color="#345BCB")
    ch1, ch2 = st.columns(2)

    CLASS_DATA = pd.DataFrame([
        {"클래스": "공공(LH)",   "Precision": 0.995, "Recall": 0.999, "F1": 0.997, "Support": 3_967},
        {"클래스": "기타",       "Precision": 0.973, "Recall": 0.961, "F1": 0.967, "Support": 27_701},
        {"클래스": "일반브랜드", "Precision": 0.928, "Recall": 0.933, "F1": 0.930, "Support": 12_018},
        {"클래스": "프리미엄",   "Precision": 0.946, "Recall": 0.991, "F1": 0.968, "Support": 5_358},
    ])

    with ch1:
        st.dataframe(
            CLASS_DATA.style.format({
                "Precision": "{:.3f}", "Recall": "{:.3f}", "F1": "{:.3f}", "Support": "{:,}"
            }),
            use_container_width=True, hide_index=True,
        )

    with ch2:
        cls_colors = ["#93C5FD", "#60A5FA", "#3B82F6", "#1D4ED8"]
        fig3 = go.Figure()
        for metric, color in zip(["Precision", "Recall", "F1"], ["#3B82F6", "#10B981", "#8B5CF6"]):
            fig3.add_trace(go.Bar(
                name=metric,
                x=CLASS_DATA["클래스"],
                y=CLASS_DATA[metric],
                marker_color=color,
            ))
        fig3.update_layout(
            title="클래스별 Precision / Recall / F1",
            barmode="group",
            height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=40, b=0),
            yaxis=dict(range=[0.85, 1.02], showgrid=True, gridcolor="#F0F4F8"),
            legend=dict(orientation="h", y=1.15),
        )
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

    st.divider()

    # ── 튜닝 단계 요약
    section_badge("💡", "튜닝 단계 요약", color="#8B5CF6")
    st.markdown("""
    | 단계 | 문제 / 조치 | 결과 |
    |------|------------|------|
    | **1단계** | `시군구` OHE → OrdinalEncoder 교체 (OOM 수정) | 메모리 14.9 GiB → 정상 |
    | **2단계** | 클래스 불균형 — `sqrt` 역비율 가중치 적용 | 소수 클래스 Recall 개선 + Precision 균형 유지 |
    | **3단계** | 누락 브랜드 전수 조사 → 일반브랜드 키워드 20개 추가 | 일반브랜드 F1 0.556 → 0.632 |
    | **4단계** | `n_estimators` 300→500, `max_depth` 6→8 | Accuracy 0.745 → **0.908** |
    | **5단계** | `n_estimators=1000` + `early_stopping_rounds=30` | Accuracy **0.960**, F1 **0.960** |
    """)

    st.divider()
    section_badge("🏷️", "브랜드 등급 정의", color="#10B981")
    st.markdown("""
    | 등급 | 해당 브랜드 예시 |
    |------|----------------|
    | **프리미엄** | 래미안, 자이, 힐스테이트, 아이파크, 푸르지오, 더샵, 롯데캐슬, 아크로 등 |
    | **일반브랜드** | 위브, 하늘채, 호반, 우미, 효성, 중흥, 반도 등 |
    | **공공(LH)** | LH, 주공 |
    | **기타** | 위 3개 외 모든 단지 |
    """)

# ══════════════════════════════════════════════════════
# 군집화 탭
# ══════════════════════════════════════════════════════
with tab_clust:
    section_badge("📋", "최종 모델 성능 지표")
    st.caption("TorchKMeans GPU 전체 배치 · k=7 · 위도/경도 ×5 가중치 · n_init=10 · 500만 건 전체 학습")

    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.markdown(stat_card("0.4694", "Silhouette", "0.7↑ 강함 / 0.5~0.7 적당 / 0.3~0.5 약함"), unsafe_allow_html=True)
    col_m2.markdown(stat_card("0.8479", "Davies-Bouldin", "낮을수록 좋음 ↓"), unsafe_allow_html=True)
    col_m3.markdown(stat_card("61,227", "Calinski-Harabasz", "높을수록 좋음 ↑"), unsafe_allow_html=True)

    st.divider()
    section_badge("📈", "단계별 튜닝 과정 — Silhouette 개선", color="#F97316")
    ch1, ch2 = st.columns(2)

    with ch1:
        phase1_labels = ["#1 원본 7개", "#3 4개+가중치", "#10 +거래활성도"]
        phase1_vals   = [0.177, 0.262, 0.362]
        phase2_labels = ["MiniBatch\n최종", "GPU ×3", "GPU ×5\n(최종)", "GPU ×7", "GPU ×10"]
        phase2_vals   = [0.362, 0.3277, 0.4694, 0.5092, 0.5397]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(len(phase1_labels))), y=phase1_vals,
            mode="lines+markers+text", name="Phase 1 (MiniBatch)",
            line=dict(color="#93C5FD", width=2), marker=dict(size=8),
            text=[f"{v}" for v in phase1_vals], textposition="top center",
        ))
        fig.add_trace(go.Scatter(
            x=list(range(len(phase2_labels))), y=phase2_vals,
            mode="lines+markers+text", name="Phase 2 (TorchKMeans GPU)",
            line=dict(color="#3B82F6", width=2), marker=dict(size=8),
            text=[f"{v}" for v in phase2_vals], textposition="top center",
        ))
        fig.add_hline(y=0.4694, line_dash="dash", line_color="#10B981",
                      annotation_text="최종 선택 0.4694")
        fig.update_layout(
            title="Silhouette 개선 과정",
            height=320, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=40, b=0),
            yaxis=dict(title="Silhouette", range=[0.1, 0.65], showgrid=True, gridcolor="#F0F4F8"),
            xaxis=dict(showgrid=False),
            legend=dict(orientation="h", y=1.15),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with ch2:
        k_vals   = [5, 6, 7, 8, 9]
        k_sil    = [0.5100, 0.4321, 0.4694, 0.4671, 0.2935]
        colors_k = ["#93C5FD", "#60A5FA", "#1D4ED8", "#60A5FA", "#93C5FD"]
        fig2 = go.Figure(go.Bar(
            x=[f"k={k}" for k in k_vals], y=k_sil,
            marker_color=colors_k,
            text=[f"{v}" for v in k_sil], textposition="outside",
        ))
        fig2.add_annotation(x="k=7", y=0.4694, text="✓ 선택\n(해석 가능성)",
                             showarrow=True, arrowhead=2, ay=-40, font=dict(color="#1D4ED8"))
        fig2.update_layout(
            title="k별 Silhouette (위도/경도 ×5, n_init=10)",
            height=320, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=40, b=0),
            yaxis=dict(title="Silhouette", range=[0.2, 0.65], showgrid=True, gridcolor="#F0F4F8"),
            xaxis=dict(showgrid=False),
        )
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    st.divider()
    section_badge("🗺️", "최종 군집 해석 (k=7)", color="#345BCB")

    CLUSTER_DATA = pd.DataFrame([
        {"cluster": 2, "거래건수": 2_330_822, "비율(%)": 47, "평균평당가(만원)": 1814, "권역 해석": "서울·수도권"},
        {"cluster": 5, "거래건수":   778_648, "비율(%)": 16, "평균평당가(만원)": 1032, "권역 해석": "부산 외곽 주거지"},
        {"cluster": 3, "거래건수":   755_952, "비율(%)": 15, "평균평당가(만원)":  847, "권역 해석": "충청·전북권"},
        {"cluster": 4, "거래건수":   371_492, "비율(%)":  8, "평균평당가(만원)":  952, "권역 해석": "창원·김해 역세권 대단지"},
        {"cluster": 6, "거래건수":   267_446, "비율(%)":  5, "평균평당가(만원)":  864, "권역 해석": "전남·광주권"},
        {"cluster": 0, "거래건수":   238_234, "비율(%)":  5, "평균평당가(만원)":  668, "권역 해석": "강원·경북권"},
        {"cluster": 1, "거래건수":   183_209, "비율(%)":  4, "평균평당가(만원)":  705, "권역 해석": "전남 동부 (순천·여수·광양)"},
    ])

    cl1, cl2 = st.columns(2)
    with cl1:
        st.dataframe(
            CLUSTER_DATA.style.format({"거래건수": "{:,}", "평균평당가(만원)": "{:,}"}),
            use_container_width=True, hide_index=True,
        )

    with cl2:
        cluster_colors = ["#60A5FA","#93C5FD","#3B82F6","#2563EB","#1D4ED8","#1E40AF","#BFDBFE"]
        fig3 = go.Figure(go.Bar(
            x=CLUSTER_DATA["권역 해석"],
            y=CLUSTER_DATA["평균평당가(만원)"],
            marker_color=cluster_colors,
            text=CLUSTER_DATA["평균평당가(만원)"].apply(lambda v: f"{v:,}만"),
            textposition="outside",
        ))
        fig3.update_layout(
            title="군집별 평균 평당가",
            height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=40, b=40),
            xaxis=dict(tickangle=-20, showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="#F0F4F8", tickformat=","),
        )
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

    st.divider()
    section_badge("💡", "k=5 vs k=7 선택 근거", color="#8B5CF6")
    st.markdown("""
    | | k=5 | k=7 (선택) |
    |-|-----|-----------|
    | **Silhouette** | 0.5100 (더 높음) | 0.4694 |
    | **군집 해석** | 수도권 47% 단일 군집 — 광역적 | 부산·경남 2개 세분화, 전남 동부 별도 분리 |
    | **선택 이유** | — | 지표 0.04 낮지만 해석 가능성이 훨씬 풍부 |
    """)
    st.info("Silhouette 0.4694는 전국 500만 건 연속 분포 데이터 특성상 자연적 군집 경계가 불분명하여 이 수준이 현실적 최선. 초기 0.177 대비 **2.65배 개선**")
