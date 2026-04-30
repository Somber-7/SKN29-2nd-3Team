# 경로: app/Home.py

import streamlit as st
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from utils.ui import load_css, render_sidebar, page_header, section_badge, stat_card, chart_card_open, chart_card_close

st.set_page_config(
    page_title="부동산 분석 플랫폼",
    layout="wide",
    initial_sidebar_state="expanded",
)
load_css()
render_sidebar()

# ── 페이지 헤더
page_header("개요")
 
# ── 상단 통계 카드 4개
section_badge("📊", "데이터 요약")

c1, c2, c3, c4 = st.columns(4)
cards = [
    ("5,000,000건", "총 거래 건수", "2015.01 ~ 2023.04"),
    ("준비 중",     "평균 거래 가격", ""),
    ("준비 중",     "최다 거래 지역", ""),
    ("준비 중",     "데이터 기준일",  ""),
]
for col, (val, lbl, sub) in zip([c1, c2, c3, c4], cards):
    col.markdown(stat_card(val, lbl, sub), unsafe_allow_html=True)

# ── 하단 메인 차트
section_badge("📈", "전체 거래 추이", color="#F97316")
chart_card_open()
st.info("전국 월별 거래량 추이 차트가 여기에 위치합니다. (데이터 연동 후 교체)")
chart_card_close()
