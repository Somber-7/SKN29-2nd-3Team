# 경로: app/test.py

import streamlit as st
import pandas as pd
import numpy as np
import os
import folium
from streamlit_folium import st_folium

# 1. 페이지 기본 설정 (항상 최상단에 위치해야 함)
st.set_page_config(page_title="부동산 데이터 분석 대시보드", layout="wide")

# 2. 외부 CSS 로드 함수 정의
def load_css():
    # 현재 실행 중인 파일(test.py)의 절대 경로를 기준으로 폴더 탐색
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # app 폴더의 상위(..)로 가서 assets -> css -> style.css 파일 접근
    css_path = os.path.join(current_dir, "..", "assets", "css", "style.css")
    
    try:
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"⚠️ CSS 파일을 찾을 수 없습니다: {css_path}")

# 3. CSS 함수 실행
load_css()

# =====================================================================
# 이하 기존 대시보드 코드 작성
# =====================================================================

st.sidebar.title("부동산 통계정보")
page = st.sidebar.radio(
    "", 
    ["개요 (홈)", "지도 시각화", "가격 추이 분석", "입지 분석", "모델 비교", "모델 예측"]
)

if page == "개요 (홈)":
    st.title("대시보드 개요")
    # 상단 요약 카드 (이미지 1 참고)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="총 거래 건수", value="5,000,000", delta="기간: 2015.01 ~ 2023.04")
    with col2:
        st.metric(label="평균 거래 가격", value="7.5억", delta="전월 대비 +2%")
    with col3:
        st.metric(label="최다 거래 지역", value="서울시 강남구")
    with col4:
        st.metric(label="업데이트 일자", value="2023-04-30")
    
    st.divider()
    st.subheader("전체 통계 요약 (시각화 영역)")
    st.info("여기에 전체 거래 동향을 보여주는 메인 차트가 위치합니다.")

elif page == "지도 시각화":
    # 이미지와 똑같은 레이아웃 구성을 위해 커스텀 HTML 제목 사용
    st.markdown("""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 2rem;">
            <h1 style="margin: 0; color: #172B4D; font-size: 2.2rem; font-weight: 800;">전국 아파트 가격 시각화</h1>
            <button class="custom-btn">→ 바로가기</button>
        </div>
        <hr style="border: 1px solid #E2E8F0; margin-bottom: 2rem;">
    """, unsafe_allow_html=True)
    
    col_left, col_right = st.columns([1.2, 2.8]) # 왼쪽 검색바, 오른쪽 넓은 지도
    
    with col_left:
        st.markdown("<h4 style='color: #355CC9; font-weight: bold;'>🔍 지역 필터링</h4>", unsafe_allow_html=True)
        
        # 드릴다운 로직
        sido = st.selectbox("1단계: 시/도", ["전국", "서울특별시", "경기도", "인천광역시"])
        
        if sido != "전국":
            gugun_list = ["전체", "강남구", "서초구", "송파구"] if sido == "서울특별시" else ["전체"]
            gugun = st.selectbox("2단계: 구/군", gugun_list)
        else:
            gugun = "전체"
            
        st.markdown("<br>", unsafe_allow_html=True)
        st.button("조회하기", type="primary", use_container_width=True)

    with col_right:
        # 지역에 따른 중심 좌표 및 줌 레벨 설정
        if sido == "전국":
            center, zoom = [36.5, 127.5], 7
        elif sido == "서울특별시" and gugun == "전체":
            center, zoom = [37.5665, 126.9780], 11
        elif sido == "서울특별시" and gugun == "강남구":
            center, zoom = [37.4959, 127.0664], 13
        else:
            center, zoom = [36.5, 127.5], 7

        # Folium 지도 생성 (가장 깔끔한 cartodbpositron 테마)
        m = folium.Map(location=center, zoom_start=zoom, tiles="cartodbpositron")
        
        # 확대되었을 때(구/군 선택 시) 마커 표시 예시
        if zoom >= 11:
            folium.CircleMarker(
                location=[37.4959, 127.0664],
                radius=15,
                color="#355CC9",
                fill=True,
                fillColor="#355CC9",
                fillOpacity=0.6,
                popup="강남구 평균가: 20억"
            ).add_to(m)

        # 지도 렌더링
        st_folium(m, width=800, height=500, returned_objects=[])

elif page == "가격 추이 분석":
    st.title("가격 추이 분석")
    # 지역 선택기 (이미지 3 참고)
    selected_region = st.selectbox("지역 선택기", ["서울 전체", "강남 3구", "마용성"])
    
    st.divider()
    # 그래프 영역
    st.subheader(f"{selected_region} 가격 추이 그래프")
    chart_data = pd.DataFrame(np.random.randn(20, 3), columns=['매매가', '전세가', '거래량'])
    st.line_chart(chart_data)

elif page == "입지 분석":
    st.title("입지 분석")
    
    # 상관계수 및 평균가 카드 (이미지 4 참고)
    st.subheader("주요 입지 요인별 상관계수")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("기준금리", "0.214")
    with col2:
        st.metric("인근 역 수", "0.450")
    with col3:
        st.metric("인근 학교 수", "0.320")
        
    col4, col5 = st.columns(2)
    with col4:
        st.metric("세대수", "0.280")
    with col5:
        st.metric("건축연식", "-0.150")
        
    st.divider()
    st.subheader("입지 조건별 평균 평당가")
    col6, col7 = st.columns(2)
    with col6:
        st.info("**역세권 / 비역세권 평균 평당가 비교 차트**")
    with col7:
        st.info("**학군지 / 비학군지 평균 평당가 비교 차트**")

elif page == "모델 비교":
    st.title("예측 및 분류 모델 성능 비교")
    st.write("회귀/분류 각 영역에서 여러 머신러닝/딥러닝 모델의 성능을 나란히 비교합니다.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("거래금액 예측 (회귀 모델)")
        st.dataframe({"모델": ["RandomForest", "XGBoost", "DNN"], "RMSE": [0.12, 0.09, 0.08], "R2 Score": [0.85, 0.89, 0.92]})
    with col2:
        st.subheader("브랜드 등급 분류 (분류 모델)")
        st.dataframe({"모델": ["Logistic", "LightGBM", "DNN"], "Accuracy": [0.75, 0.88, 0.90], "F1 Score": [0.73, 0.87, 0.89]})

elif page == "모델 예측":
    st.title("맞춤형 아파트 가격 예측")
    st.write("원하시는 조건을 입력하면 학습된 AI 모델이 가격을 예측해 드립니다.")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            pred_sido = st.selectbox("지역", ["서울", "경기", "인천"])
            pred_area = st.number_input("전용면적 (㎡)", min_value=10, max_value=300, value=84)
        with col2:
            pred_floor = st.number_input("층수", min_value=1, max_value=100, value=15)
            pred_year = st.number_input("건축연도", min_value=1970, max_value=2024, value=2010)
        
        submitted = st.form_submit_button("가격 예측하기")
        if submitted:
            st.success(f"예측 결과: 선택하신 조건의 예상 거래금액은 **약 8억 5,000만 원** 입니다.")