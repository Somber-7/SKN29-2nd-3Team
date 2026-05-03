# 경로: app/pages/5_분류모델.py

import streamlit as st
import sys
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.figure_factory as ff

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from utils.ui import (
    load_css, render_sidebar, page_header,
    section_badge, stat_card,
)
from utils.db import fetch_all

st.set_page_config(page_title="분류 모델", layout="wide")
load_css()
render_sidebar()

page_header("분류 모델 — 브랜드 등급 분류")

GRADE_LABELS = ["기타", "공공(LH)", "일반브랜드", "프리미엄"]
GRADE_COLORS = ["#9CA3AF", "#3B82F6", "#F97316", "#EF4444"]

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "models", "brand_grade_classifier.pkl")

# 평수 → 전용면적(㎡) 변환 (1평 = 3.3058㎡, 5평~100평 1평 단위)
PYEONG_OPTIONS = {f"{p}평 ({round(p * 3.3058, 2)}㎡)": round(p * 3.3058, 2) for p in range(5, 101)}


@st.cache_data
def load_sigungu_info() -> pd.DataFrame:
    """tbl_sigungu_stats에서 시군구별 통계 로딩 (248행, 즉시 반환)."""
    rows = fetch_all(
        "SELECT sigungu AS 시군구, sido AS 시도, region_code AS 지역코드, "
        "lat_median AS 위도, lon_median AS 경도, "
        "household_total AS 세대수 "
        "FROM tbl_sigungu_stats ORDER BY region_code"
    )
    df = pd.DataFrame(rows)
    for col in ["지역코드", "위도", "경도", "세대수"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


@st.cache_resource
def load_model():
    from models.classification.brand_grade_classifier import BrandGradeClassifier
    m = BrandGradeClassifier.load(MODEL_PATH)
    # Streamlit 환경에서 GPU 크래시 방지 — 예측은 CPU로 수행
    m._model.named_steps["model"].set_params(device="cpu")
    return m


with st.spinner("모델 로딩 중..."):
    try:
        model = load_model()
        load_ok = True
    except Exception as e:
        st.error(f"모델 로드 실패: {e}")
        load_ok = False

if not load_ok:
    st.stop()

metrics = model.metrics_ or {}
cm_raw   = model.confusion_matrix_
classes  = model.classes_ or GRADE_LABELS

# ── 성능 지표
section_badge("📊", "모델 성능 지표")
mc1, mc2, mc3, mc4 = st.columns(4)
mc1.markdown(stat_card(f"{metrics.get('Accuracy',  0):.3f}", "Accuracy"),  unsafe_allow_html=True)
mc2.markdown(stat_card(f"{metrics.get('Precision', 0):.3f}", "Precision"), unsafe_allow_html=True)
mc3.markdown(stat_card(f"{metrics.get('Recall',    0):.3f}", "Recall"),    unsafe_allow_html=True)
mc4.markdown(stat_card(f"{metrics.get('F1',        0):.3f}", "F1-Score"),  unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── 학습 정보
section_badge("📋", "학습 정보")
ic1, ic2, ic3 = st.columns(3)
ic1.markdown(stat_card(f"{metrics.get('train_rows', 0):,}", "학습 데이터 수"), unsafe_allow_html=True)
ic2.markdown(stat_card(f"{metrics.get('test_rows',  0):,}", "평가 데이터 수"), unsafe_allow_html=True)
ic3.markdown(stat_card(str(metrics.get('best_iteration', '-')), "Best Iteration (Early Stop)"), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── 시각화
section_badge("📈", "시각화", color="#F97316")
ch1, ch2 = st.columns(2)

with ch1:
    if cm_raw is not None:
        cm = cm_raw.tolist()
        label_order = classes
    else:
        cm = [[0]*len(GRADE_LABELS)]*len(GRADE_LABELS)
        label_order = GRADE_LABELS

    fig = ff.create_annotated_heatmap(
        z=cm,
        x=label_order, y=label_order,
        colorscale=[[0, "#EFF6FF"], [1, "#1D4ED8"]],
        showscale=False,
    )
    fig.update_layout(
        title="혼동 행렬", height=360,
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=40, b=0),
        xaxis=dict(title="예측"),
        yaxis=dict(title="실제", autorange="reversed"),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

with ch2:
    try:
        imp_df = model.get_feature_importance_df(top_n=15)
        fig2 = go.Figure(go.Bar(
            x=imp_df["importance"],
            y=imp_df["feature"],
            orientation="h",
            marker_color="#345BCB",
        ))
        fig2.update_layout(
            title="피처 중요도 (Top 15)",
            height=360,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=40, b=0),
            xaxis=dict(showgrid=True, gridcolor="#F0F4F8"),
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
    except Exception:
        st.info("피처 중요도를 불러올 수 없습니다.")

st.markdown("<br>", unsafe_allow_html=True)

# ── 샘플 예측
section_badge("🏷️", "샘플 예측")
st.caption("아래 값을 입력하고 예측 버튼을 누르세요.")

sigungu_info = load_sigungu_info()

# ── 지역 정보 (form 밖 — 시군구 선택 즉시 위도/경도/세대수 반영)
section_badge("📍", "지역 정보", color="#6366F1")
c1, c2, c3, c4 = st.columns(4)

sido_list    = sigungu_info["시도"].unique().tolist() if not sigungu_info.empty else ["서울특별시"]
default_sido = sido_list.index("서울특별시") if "서울특별시" in sido_list else 0
with c1:
    시도 = st.selectbox("시/도", options=sido_list, index=default_sido)

filtered     = sigungu_info[sigungu_info["시도"] == 시도] if not sigungu_info.empty else sigungu_info
# 시군구명에서 시도명 접두어 제거하여 표시 (예: "서울특별시 강남구" → "강남구")
sigungu_display = [
    s.replace(시도 + " ", "", 1) if s.startswith(시도 + " ") else s
    for s in filtered["시군구"].tolist()
]
default_sg = sigungu_display.index("강남구") if "강남구" in sigungu_display and 시도 == "서울특별시" else 0
with c2:
    시군구_표시 = st.selectbox("시/군/구", options=sigungu_display, index=default_sg)

# 원래 시군구명 복원 (DB 키 기준)
시군구_원본 = filtered["시군구"].tolist()[sigungu_display.index(시군구_표시)]
_row     = filtered[filtered["시군구"] == 시군구_원본].iloc[0] if not filtered.empty else None
지역코드 = int(_row["지역코드"])        if _row is not None else 0
위도     = float(round(_row["위도"], 4)) if _row is not None else 37.5
경도     = float(round(_row["경도"], 4)) if _row is not None else 127.0
세대수   = int(_row["세대수"])           if _row is not None else 0
시군구   = 시군구_원본

with c3:
    st.text_input("지역코드", value=str(지역코드), disabled=True)
with c4:
    st.text_input("위도 / 경도", value=f"{위도} / {경도}", disabled=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── 나머지 입력 + 예측 버튼을 form으로 묶어 불필요한 재실행 방지
with st.form("predict_form"):
    # ── 거래 정보
    section_badge("💰", "거래 정보", color="#10B981")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        거래금액 = st.number_input("거래금액 (만원)", value=80000, step=1000)
    with c2:
        평수선택 = st.selectbox("전용면적", options=list(PYEONG_OPTIONS.keys()), index=20)
    with c3:
        층 = st.number_input("층", value=10, step=1)
    with c4:
        st.markdown('<p style="font-size:14px; margin-bottom:4px;">거래 시점</p>', unsafe_allow_html=True)
        _tc1, _tc2 = st.columns(2)
        거래연도 = _tc1.selectbox("연도", options=list(range(2015, 2027)), index=9, label_visibility="collapsed")
        거래월   = _tc2.selectbox("월",   options=list(range(1, 13)),      index=5, label_visibility="collapsed")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 단지 정보
    section_badge("🏢", "단지 정보", color="#F97316")
    c1, c2, c3 = st.columns(3)
    with c1:
        건물연식 = st.number_input("건물연식 (년)", value=10, step=1)
    with c2:
        st.text_input("세대수", value=f"{세대수:,}", disabled=True)
    with c3:
        기준금리 = st.number_input("기준금리 (%)", value=3.5, step=0.1)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 입지 정보
    section_badge("🗺️", "입지 정보", color="#EF4444")
    c1, c2 = st.columns(2)
    with c1:
        인근학교수 = st.number_input("인근학교수", value=3, step=1)
    with c2:
        인근역수 = st.number_input("인근역수", value=2, step=1)

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.form_submit_button("예측 실행", type="primary")
    전용면적 = PYEONG_OPTIONS[평수선택]

if predict_btn:
    with st.spinner("예측 중..."):
        try:
            sample = pd.DataFrame([{
                "거래금액": 거래금액, "전용면적": 전용면적, "층": 층,
                "건물연식": 건물연식, "기준금리": 기준금리, "위도": 위도,
                "경도": 경도, "인근학교수": 인근학교수, "인근역수": 인근역수,
                "세대수": 세대수, "거래연도": 거래연도, "거래월": 거래월,
                "지역코드": 지역코드, "시군구": 시군구,
            }])
            proba = model.predict_proba(sample[model.feature_columns])[0]
            st.session_state["pred_proba"]  = proba.tolist()
            st.session_state["pred_classes"] = classes
        except Exception as e:
            st.session_state["pred_proba"] = None
            st.session_state["pred_error"] = str(e)

if st.session_state.get("pred_proba") is not None:
    proba      = st.session_state["pred_proba"]
    classes_   = st.session_state["pred_classes"]
    pred_idx   = int(np.argmax(proba))
    pred_grade = classes_[pred_idx]
    pred_color = GRADE_COLORS[GRADE_LABELS.index(pred_grade)] if pred_grade in GRADE_LABELS else "#345BCB"

    st.markdown(f"""
    <div style="text-align:center; padding:12px 0 20px 0;">
        <div style="font-size:13px; color:#6B7280; margin-bottom:10px;">예측 브랜드 등급</div>
        <div style="font-size:36px; font-weight:800; color:{pred_color};">{pred_grade}</div>
    </div>
    """, unsafe_allow_html=True)

    for label, prob, color in zip(classes_, proba, [GRADE_COLORS[GRADE_LABELS.index(l)] if l in GRADE_LABELS else "#9CA3AF" for l in classes_]):
        st.markdown(f"""
        <div style="margin:6px 0; font-size:13px; color:#4B5563;">
            {label}<span style="float:right; font-weight:700;">{prob*100:.1f}%</span>
        </div>
        <div style="background:#E5EAF2; border-radius:6px; height:8px; margin-bottom:10px;">
            <div style="width:{prob*100:.1f}%; background:{color}; border-radius:6px; height:8px;"></div>
        </div>
        """, unsafe_allow_html=True)
elif st.session_state.get("pred_error"):
    st.error(f"예측 실패: {st.session_state['pred_error']}")
