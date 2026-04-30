# 경로: app/utils/ui.py

import streamlit as st
import os
 

def load_css():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    css_path = os.path.join(current_dir, "..", "..", "assets", "css", "style.css")
    try:
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"CSS 파일을 찾을 수 없습니다: {css_path}")


def render_sidebar():
    """모든 페이지 공통 사이드바 — 로고 + 커스텀 네비게이션"""
    st.sidebar.markdown("""
    <div style="padding: 8px 0 16px 0;">
        <div style="font-size:35px; font-weight:800; color:#172B4D;">🏢 부동산 분석</div>
        <div style="font-size:19px; color:#9CA3AF; margin-top:4px;">Apart Deal 2015~2023</div>
    </div>
    """, unsafe_allow_html=True)
    st.sidebar.divider()

    with st.sidebar:
        st.page_link("Home.py",                 label="🏠  개요")
        st.page_link("pages/1_가격추이분석.py", label="📈  가격 추이 분석")
        st.page_link("pages/2_입지분석.py",     label="📍  입지 분석")
        st.page_link("pages/3_지도시각화.py",   label="🗺️  지도 시각화")
        st.page_link("pages/4_회귀모델.py",     label="📊  회귀 모델")
        st.page_link("pages/5_분류모델.py",     label="🏷️  분류 모델")
        st.page_link("pages/6_군집화.py",       label="🔵  군집화")
        st.page_link("pages/7_신경망.py",       label="🧠  신경망 (DNN)")
        st.page_link("pages/8_모델비교.py",     label="📋  모델 비교")


def page_header(title: str):
    """REB 스타일 상단 헤더 — 제목만"""
    st.markdown(f"""
    <div class="page-header">
        <h1>{title}</h1>
    </div>
    <hr class="header-divider">
    """, unsafe_allow_html=True)


def section_badge(icon: str, title: str, color: str = "#345BCB"):
    """컬러 원형 배지 + 섹션명 + 구분선"""
    st.markdown(f"""
    <div class="section-header">
        <div class="section-badge" style="background-color:{color};">{icon}</div>
        <span class="section-title">{title}</span>
        <div class="section-divider"></div>
    </div>
    """, unsafe_allow_html=True)


def stat_card(value: str, label: str, sub: str = "") -> str:
    """통계 카드 HTML 반환 — st.columns 내 st.markdown으로 사용"""
    sub_html = f'<div class="stat-sub">{sub}</div>' if sub else ""
    return f"""
    <div class="stat-card">
        <div class="stat-value">{value}</div>
        <div class="stat-label">{label}</div>
        {sub_html}
    </div>
    """


def info_card(label: str, value: str, sub: str = "") -> str:
    """입지 분석 상관계수 카드 HTML 반환"""
    sub_html = f'<div class="card-sub">{sub}</div>' if sub else ""
    return f"""
    <div class="info-card">
        <div class="card-label">{label}</div>
        <div class="card-value">{value}</div>
        {sub_html}
    </div>
    """


def chart_card_open():
    """차트 래퍼 박스 시작 — chart_card_close()와 쌍으로 사용"""
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)


def chart_card_close():
    """차트 래퍼 박스 종료"""
    st.markdown('</div>', unsafe_allow_html=True)
