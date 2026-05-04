# 경로: app/Home.py

import streamlit as st
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.ui import load_css

st.set_page_config(
    page_title="부동산 분석 플랫폼",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)
load_css()

with st.sidebar:
    st.markdown("""
<div style="padding: 1.5rem 0 1rem 0;">
    <div style="font-size:35px; font-weight:800; color:#172B4D;">🏢 부동산 분석</div>
    <div style="font-size:19px; color:#9CA3AF; margin-top:4px; padding-bottom:1rem; border-bottom:1px solid #E5EAF2;">Apart Deal 2015~2023</div>
</div>
""", unsafe_allow_html=True)

pg = st.navigation(
    {
        "시장 현황": [
            st.Page("pages/0_개요.py",          title="개요",          icon="🏠", default=True),
            st.Page("pages/1_가격추이분석.py", title="가격 추이 분석", icon="📈"),
            st.Page("pages/2_지도시각화.py",   title="지도 시각화",   icon="🗺️"),
        ],
        "부동산 심층 분석": [
            st.Page("pages/3_입지분석.py",    title="입지 분석",      icon="📍"),
            st.Page("pages/4_프리미엄분석.py", title="프리미엄 분석", icon="💎"),
            st.Page("pages/5_이상치분석.py",  title="이상치 분석",    icon="🔍"),
        ],
        "AI 예측 및 모델링": [
            st.Page("pages/6_회귀모델.py",  title="회귀 모델",   icon="📉"),
            st.Page("pages/7_신경망.py",    title="신경망",      icon="🧠"),
            st.Page("pages/8_분류모델.py",  title="브랜드 분류", icon="🏷️"),
            st.Page("pages/9_군집화.py",    title="지역 군집화", icon="🏘️"),
        ],
        "모델 성능 리포트": [
            st.Page("pages/10_모델분석.py", title="모델 비교 분석", icon="📊"),
        ],
    }
)
pg.run()
