# 경로: app/pages/2_입지분석.py

import streamlit as st
import sys
import os
 
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.ui import load_css, render_sidebar, page_header, section_badge, info_card, chart_card_open, chart_card_close

st.set_page_config(page_title="입지 분석", layout="wide")
load_css()
render_sidebar()

page_header("입지 분석")

# ── 상관계수 카드 — 3개
section_badge("📐", "주요 입지 요인 상관계수")
c1, c2, c3 = st.columns(3)
for col, (lbl, val, sub) in zip([c1, c2, c3], [
    ("기준금리",    "0.214", "거래금액 기준"),
    ("인근 역 수",  "0.450", "반경 1km"),
    ("인근 학교 수","0.320", "반경 500m"),
]):
    col.markdown(info_card(lbl, val, sub), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── 상관계수 카드 — 2개
c4, c5, _ = st.columns([1, 1, 1])
for col, (lbl, val, sub) in zip([c4, c5], [
    ("세대수",   "0.280", "단지 규모"),
    ("건축연식", "-0.150", "노후도 역상관"),
]):
    col.markdown(info_card(lbl, val, sub), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── 비교 차트 카드 — 2개
section_badge("🏙️", "입지 조건별 평균 평당가", color="#10B981")
co1, co2 = st.columns(2)
with co1:
    chart_card_open()
    st.markdown("**역세권 / 비역세권 평균 평당가**")
    st.info("bar chart 교체 예정 — plot_price_by_brand(df, ...)")
    chart_card_close()
with co2:
    chart_card_open()
    st.markdown("**학군지 / 비학군지 평균 평당가**")
    st.info("bar chart 교체 예정 — plot_price_by_brand(df, ...)")
    chart_card_close()
