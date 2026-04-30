# 경로: app/pages/1_가격추이분석.py

import streamlit as st
import sys
import os
import pandas as pd
import numpy as np
 
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.ui import load_css, render_sidebar, page_header, section_badge, chart_card_open, chart_card_close

st.set_page_config(page_title="가격 추이 분석", layout="wide")
load_css()
render_sidebar()

page_header("가격 추이 분석")

# ── 지역 선택기
section_badge("📍", "지역 선택")
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    sido = st.selectbox("시/도", ["전국", "서울", "경기", "인천", "부산", "대구", "광주", "대전"])
with col2:
    gugun_map = {
        "서울": ["전체", "강남구", "서초구", "송파구", "마포구", "용산구", "성동구"],
        "경기": ["전체", "수원시", "성남시", "용인시", "고양시"],
        "인천": ["전체", "연수구", "남동구", "부평구"],
        "부산": ["전체", "해운대구", "수영구", "남구"],
    }
    gugun = st.selectbox("구/군", gugun_map.get(sido, ["전체"]))
with col3:
    freq = st.selectbox("집계 단위", ["월별", "분기별", "연별"])

# ── 가격 추이 그래프
section_badge("📈", "가격 추이 그래프", color="#F97316")
chart_card_open()
dummy = pd.DataFrame(
    np.random.randint(30000, 80000, size=(36, 2)),
    columns=["매매가(만원)", "전세가(만원)"]
)
st.line_chart(dummy, use_container_width=True, height=350)
chart_card_close()
