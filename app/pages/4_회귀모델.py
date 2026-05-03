# 경로: app/pages/4_회귀모델.py

import streamlit as st
import sys
import os
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from utils.ui import (
    load_css, render_sidebar, page_header,
    section_badge,
)


# =============================================================================
# 기본 설정
# =============================================================================

st.set_page_config(page_title="회귀 모델", layout="wide")
load_css()
render_sidebar()

st.markdown("""
<style>
/* 전체 input 계열 */
div[data-baseweb="input"] input {
    background-color: #FFFFFF !important;
    color: #111827 !important;
}

div[data-baseweb="input"] {
    background-color: #FFFFFF !important;
    border: 1px solid #D1D5DB !important;
    border-radius: 8px !important;
}

/* number_input +/- 영역 */
div[data-baseweb="base-input"] {
    background-color: #FFFFFF !important;
}

/* selectbox */
div[data-baseweb="select"] > div {
    background-color: #FFFFFF !important;
    color: #111827 !important;
    border-radius: 8px !important;
}

div[data-baseweb="select"] span {
    color: #111827 !important;
}

/* radio 버튼 텍스트 */
div[role="radiogroup"] label {
    color: #111827 !important;
}

/* expander */
details {
    background-color: rgba(255, 255, 255, 0.65) !important;
    border-radius: 12px !important;
}

/* 포커스 */
div[data-baseweb="input"]:focus-within,
div[data-baseweb="select"]:focus-within {
    border-color: #345BCB !important;
    box-shadow: 0 0 0 1px #345BCB !important;
}

/* 예측 결과 카드 */
.prediction-card {
    background: #FFFFFF;
    border: 1px solid #DDE5F0;
    border-radius: 18px;
    padding: 28px 24px;
    box-shadow: 0 10px 25px rgba(15, 23, 42, 0.06);
    text-align: center;
}

.prediction-label {
    font-size: 13px;
    color: #6B7280;
    margin-bottom: 8px;
}

.prediction-value {
    font-size: 38px;
    font-weight: 800;
    color: #172B4D;
}

.prediction-sub {
    font-size: 14px;
    color: #9CA3AF;
    margin-top: 8px;
}

.model-status {
    background: #DDF7EA;
    color: #047857;
    padding: 12px 14px;
    border-radius: 8px;
    font-size: 14px;
    margin-bottom: 14px;
}

.model-status-error {
    background: #FEE2E2;
    color: #B91C1C;
    padding: 12px 14px;
    border-radius: 8px;
    font-size: 14px;
    margin-bottom: 14px;
}
</style>
""", unsafe_allow_html=True)

page_header("회귀 모델 — 거래금액 예측")


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_DIR = os.path.join(PROJECT_ROOT, "data", "models")

MODEL_CONFIG = {
    "randomforest": {
        "label": "🌲 RandomForest",
        "path": os.path.join(MODEL_DIR, "RandomForest_model.pkl"),
    },
    "lightgbm": {
        "label": "⚡ LightGBM",
        "path": os.path.join(MODEL_DIR, "LightGBM_model.pkl"),
    },
    "xgboost": {
        "label": "🚀 XGBoost",
        "path": os.path.join(MODEL_DIR, "XGBoost_model.pkl"),
    },
}

REQUIRED_FEATURES = [
    "전용면적",
    "층",
    "건축년도",
    "지역코드",
    "거래일",
    "기준금리",
    "인근학교수",
    "인근역수",
    "세대수",
    "브랜드여부",
]


# =============================================================================
# 모델 로드
# =============================================================================

@st.cache_resource
def load_models():
    loaded_models = {}

    for model_key, config in MODEL_CONFIG.items():
        model_path = config["path"]

        if not os.path.exists(model_path):
            loaded_models[model_key] = None
            continue

        loaded_models[model_key] = joblib.load(model_path)

    return loaded_models


models = load_models()


# =============================================================================
# 유틸 함수
# =============================================================================

def make_input_dataframe(
    area,
    floor,
    built_year,
    region_code,
    deal_date,
    interest_rate,
    nearby_school_count,
    nearby_station_count,
    household_count,
    brand_flag,
):
    input_df = pd.DataFrame([{
        "전용면적": float(area),
        "층": int(floor),
        "건축년도": int(built_year),
        "지역코드": str(region_code).strip(),
        "거래일": pd.to_datetime(deal_date),
        "기준금리": float(interest_rate),
        "인근학교수": int(nearby_school_count),
        "인근역수": int(nearby_station_count),
        "세대수": int(household_count),
        "브랜드여부": int(brand_flag),
    }])

    return input_df[REQUIRED_FEATURES]


def predict_price(model, input_df):
    pred = model.predict(input_df)
    return float(pred[0])


def get_model_pipeline(model):
    if hasattr(model, "model") and model.model is not None:
        return model.model

    if hasattr(model, "_model") and model._model is not None:
        return model._model

    if hasattr(model, "named_steps"):
        return model

    return None


def get_feature_names_from_pipeline(pipeline):
    if pipeline is None:
        return None

    if not hasattr(pipeline, "named_steps"):
        return None

    preprocessor = pipeline.named_steps.get("preprocessor")

    if preprocessor is None:
        return None

    if not hasattr(preprocessor, "get_feature_names_out"):
        return None

    return preprocessor.get_feature_names_out()


def get_final_estimator_from_pipeline(pipeline):
    if pipeline is None:
        return None

    if not hasattr(pipeline, "named_steps"):
        return None

    return pipeline.named_steps.get("model")


def get_feature_importance_df(model):
    pipeline = get_model_pipeline(model)
    feature_names = get_feature_names_from_pipeline(pipeline)
    final_estimator = get_final_estimator_from_pipeline(pipeline)

    if final_estimator is None:
        return None

    # LightGBM은 가능하면 gain 기준 사용
    if final_estimator.__class__.__name__ == "LGBMRegressor":
        try:
            importances = final_estimator.booster_.feature_importance(
                importance_type="gain"
            )
        except Exception:
            importances = final_estimator.feature_importances_

    elif hasattr(final_estimator, "feature_importances_"):
        importances = final_estimator.feature_importances_

    elif hasattr(final_estimator, "coef_"):
        importances = np.abs(np.ravel(final_estimator.coef_))

    else:
        return None

    if feature_names is None or len(feature_names) != len(importances):
        feature_names = [f"feature_{i}" for i in range(len(importances))]

    importance_df = pd.DataFrame({
        "피처": feature_names,
        "중요도": importances,
    })

    importance_df = importance_df.sort_values("중요도", ascending=False).reset_index(drop=True)
    return importance_df


def prettify_feature_name(name):
    name = str(name)

    if name.startswith("지역코드_"):
        return "지역코드"

    name_map = {
        "건물연식": "건물연식",
        "거래연도": "거래연도",
        "거래월": "거래월",
    }

    return name_map.get(name, name)


def group_feature_importance(importance_df, normalize=True):
    df = importance_df.copy()
    df["피처"] = df["피처"].apply(prettify_feature_name)

    grouped_df = (
        df.groupby("피처", as_index=False)["중요도"]
        .sum()
        .sort_values("중요도", ascending=False)
        .reset_index(drop=True)
    )

    if normalize:
        total = grouped_df["중요도"].sum()
        if total > 0:
            grouped_df["중요도"] = grouped_df["중요도"] / total

    return grouped_df


def plot_feature_importance(importance_df, top_n=11):
    plot_df = importance_df.sort_values("중요도", ascending=False).head(top_n)

    fig = go.Figure(go.Bar(
        x=plot_df["중요도"],
        y=plot_df["피처"],
        orientation="h",
        marker_color="#345BCB",
        text=plot_df["중요도"].apply(lambda x: f"{x:.3f}"),
        textposition="outside",
    ))

    fig.update_layout(
        title="피처 중요도",
        height=420,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=120, r=40, t=40, b=40),
        xaxis=dict(
            title="중요도",
            showgrid=True,
            gridcolor="#F0F4F8",
        ),
        yaxis=dict(
            title="피처",
            autorange="reversed",
        ),
        font=dict(size=12),
    )

    return fig


def render_model_status(model_key, model):
    label = MODEL_CONFIG[model_key]["label"]
    path = MODEL_CONFIG[model_key]["path"]

    if model is None:
        st.markdown(f"""
        <div class="model-status-error">
            ❌ {label} 모델 파일을 찾을 수 없습니다.<br>
            <span style="font-size:12px;">{path}</span>
        </div>
        """, unsafe_allow_html=True)
        return False

    st.markdown(f"""
    <div class="model-status">
        ✅ {label} 모델 로드 완료
    </div>
    """, unsafe_allow_html=True)
    return True


def get_preset_values(preset):
    if preset == "국평 신축":
        return {
            "area": 84.0,
            "floor": 15,
            "built_year": 2020,
            "nearby_school_count": 4,
            "nearby_station_count": 2,
            "household_count": 1200,
            "brand_flag": 1,
        }

    if preset == "국평 준신축":
        return {
            "area": 84.0,
            "floor": 12,
            "built_year": 2015,
            "nearby_school_count": 3,
            "nearby_station_count": 2,
            "household_count": 1000,
            "brand_flag": 1,
        }

    if preset == "소형 구축":
        return {
            "area": 59.0,
            "floor": 8,
            "built_year": 2000,
            "nearby_school_count": 2,
            "nearby_station_count": 1,
            "household_count": 700,
            "brand_flag": 0,
        }

    if preset == "대형 고급":
        return {
            "area": 135.0,
            "floor": 20,
            "built_year": 2018,
            "nearby_school_count": 4,
            "nearby_station_count": 3,
            "household_count": 1500,
            "brand_flag": 1,
        }

    return {
        "area": 84.0,
        "floor": 15,
        "built_year": 2010,
        "nearby_school_count": 3,
        "nearby_station_count": 2,
        "household_count": 1000,
        "brand_flag": 0,
    }


def render_prediction_card(predicted):
    st.markdown(f"""
    <div class="prediction-card">
        <div class="prediction-label">예측 거래금액</div>
        <div class="prediction-value">{predicted:,.0f}만원</div>
        <div class="prediction-sub">≈ {predicted / 10000:.2f}억원</div>
    </div>
    """, unsafe_allow_html=True)


def render_empty_prediction_card():
    st.markdown("""
    <div class="prediction-card">
        <div style="font-size:15px; color:#6B7280;">
            좌측에서 조건을 입력한 뒤 <b>예측하기</b> 버튼을 눌러주세요.
        </div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# 화면 구성
# =============================================================================

available_model_keys = list(MODEL_CONFIG.keys())
tabs = st.tabs([MODEL_CONFIG[key]["label"] for key in available_model_keys])

for tab, model_key in zip(tabs, available_model_keys):
    with tab:
        model = models.get(model_key)

        prediction_state_key = f"prediction_result_{model_key}"
        input_state_key = f"prediction_input_{model_key}"

        col_left, col_right = st.columns([1.1, 1.3])

        # ---------------------------------------------------------------------
        # 좌측: 입력 폼
        # ---------------------------------------------------------------------
        with col_left:
            section_badge("📋", "예측 조건 입력")
            render_model_status(model_key, model)
            with st.form(key=f"predict_form_{model_key}"):
                basic_col1, basic_col2 = st.columns(2)

                with basic_col1:
                    region_code = st.text_input(
                        "지역코드",
                        value="11680",
                        help="예: 강남구 11680, 송파구 11710",
                        key=f"region_code_{model_key}",
                    )

                    area = st.number_input(
                        "전용면적 (㎡)",
                        min_value=1.0,
                        max_value=300.0,
                        value=84.0,
                        step=1.0,
                        key=f"area_{model_key}",
                    )

                    built_year = st.number_input(
                        "건축년도",
                        min_value=1960,
                        max_value=2030,
                        value=2010,
                        step=1,
                        key=f"built_year_{model_key}",
                    )

                with basic_col2:
                    deal_date = st.date_input(
                        "거래일",
                        value=pd.Timestamp.today(),
                        key=f"deal_date_{model_key}",
                    )

                    floor = st.number_input(
                        "층",
                        min_value=-5,
                        max_value=80,
                        value=15,
                        step=1,
                        key=f"floor_{model_key}",
                    )

                    brand_flag = st.radio(
                        "브랜드여부",
                        options=[0, 1],
                        format_func=lambda x: "브랜드 아님" if x == 0 else "브랜드",
                        horizontal=True,
                        index=0,
                        key=f"brand_flag_{model_key}",
                    )

                with st.expander("고급 옵션", expanded=False):
                    adv_col1, adv_col2 = st.columns(2)

                    with adv_col1:
                        interest_rate = st.number_input(
                            "기준금리",
                            min_value=0.0,
                            max_value=10.0,
                            value=3.5,
                            step=0.25,
                            key=f"interest_rate_{model_key}",
                        )

                        nearby_school_count = st.number_input(
                            "인근학교수",
                            min_value=0,
                            max_value=100,
                            value=3,
                            step=1,
                            key=f"nearby_school_count_{model_key}",
                        )

                    with adv_col2:
                        nearby_station_count = st.number_input(
                            "인근역수",
                            min_value=0,
                            max_value=100,
                            value=2,
                            step=1,
                            key=f"nearby_station_count_{model_key}",
                        )

                        household_count = st.number_input(
                            "세대수",
                            min_value=0,
                            max_value=10000,
                            value=1000,
                            step=50,
                            key=f"household_count_{model_key}",
                        )

                predict_btn = st.form_submit_button(
                    "예측하기",
                    type="primary",
                    use_container_width=True,
                    disabled=(model is None),
                )

        # ---------------------------------------------------------------------
        # 우측: 예측 결과
        # ---------------------------------------------------------------------
        with col_right:
            section_badge("💰", "예측 결과")

            if predict_btn and model is not None:
                try:
                    input_df = make_input_dataframe(
                        area=area,
                        floor=floor,
                        built_year=built_year,
                        region_code=region_code,
                        deal_date=deal_date,
                        interest_rate=interest_rate,
                        nearby_school_count=nearby_school_count,
                        nearby_station_count=nearby_station_count,
                        household_count=household_count,
                        brand_flag=brand_flag,
                    )

                    predicted = predict_price(model, input_df)

                    st.session_state[prediction_state_key] = predicted
                    st.session_state[input_state_key] = input_df

                except Exception as e:
                    st.error("예측 중 오류가 발생했습니다.")
                    st.exception(e)

            if prediction_state_key in st.session_state:
                render_prediction_card(st.session_state[prediction_state_key])

                with st.expander("입력 데이터 확인", expanded=False):
                    st.dataframe(
                        st.session_state[input_state_key],
                        use_container_width=True,
                    )
            else:
                render_empty_prediction_card()

        st.markdown("<br>", unsafe_allow_html=True)

        # ---------------------------------------------------------------------
        # 피처 중요도
        # ---------------------------------------------------------------------
        section_badge("📈", "피처 중요도", color="#F97316")

        if model is None:
            st.warning("모델이 로드되지 않아 피처 중요도를 표시할 수 없습니다.")
        else:
            try:
                importance_df = get_feature_importance_df(model)

                if importance_df is None or importance_df.empty:
                    st.warning("이 모델에서는 피처 중요도를 가져올 수 없습니다.")
                else:
                    grouped_importance_df = group_feature_importance(
                        importance_df,
                        normalize=True,
                    )

                    display_importance_df = grouped_importance_df.copy()
                    display_importance_df["중요도"] = display_importance_df["중요도"].round(4)

                    ch1, ch2 = st.columns([1.4, 1])

                    with ch1:
                        fig = plot_feature_importance(display_importance_df, top_n=11)
                        st.plotly_chart(
                            fig,
                            use_container_width=True,
                            config={"displayModeBar": False},
                        )

                    with ch2:
                        st.dataframe(
                            display_importance_df,
                            use_container_width=True,
                            height=360,
                        )

            except Exception as e:
                st.error("피처 중요도 계산 중 오류가 발생했습니다.")
                st.exception(e)