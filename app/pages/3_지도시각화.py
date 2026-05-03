# 경로: app/pages/3_지도시각화.py

import streamlit as st
import sys
import os
import requests
import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from utils.ui import load_css, render_sidebar, page_header, section_badge, info_card

st.set_page_config(page_title="지도 시각화", layout="wide")
load_css()
render_sidebar()

page_header("지도 시각화")

# ── 17개 시/도 더미 평균 평당가 (만원/3.3㎡)
PRICE_DATA = {
    "서울특별시":      85000,
    "경기도":          42000,
    "부산광역시":      38000,
    "인천광역시":      35000,
    "제주특별자치도":  35000,
    "세종특별자치시":  32000,
    "대구광역시":      30000,
    "울산광역시":      28000,
    "대전광역시":      27000,
    "광주광역시":      25000,
    "경상남도":        20000,
    "강원특별자치도":  18000,
    "충청남도":        17000,
    "충청북도":        16000,
    "경상북도":        16000,
    "전북특별자치도":  15000,
    "전라남도":        14000,
}

# 시/도별 중심 좌표 (Scattergeo 마커용)
SIDO_CENTER = {
    "서울특별시":      (37.5665, 126.9780),
    "경기도":          (37.4138, 127.5183),
    "부산광역시":      (35.1796, 129.0756),
    "인천광역시":      (37.4563, 126.7052),
    "제주특별자치도":  (33.4890, 126.4983),
    "세종특별자치시":  (36.4801, 127.2890),
    "대구광역시":      (35.8714, 128.6014),
    "울산광역시":      (35.5384, 129.3114),
    "대전광역시":      (36.3504, 127.3845),
    "광주광역시":      (35.1595, 126.8526),
    "경상남도":        (35.4606, 128.2132),
    "강원특별자치도":  (37.8228, 128.1555),
    "충청남도":        (36.5184, 126.8000),
    "충청북도":        (36.6358, 127.4914),
    "경상북도":        (36.4919, 128.8889),
    "전북특별자치도":  (35.7175, 127.1530),
    "전라남도":        (34.8161, 126.4630),
}

SIDO_LIST = list(PRICE_DATA.keys())

# ── GeoJSON 로드 (southkorea-maps 공개 데이터, 캐싱)
@st.cache_data(show_spinner="지도 데이터 로딩 중...")
def load_geojson():
    url = (
        "https://raw.githubusercontent.com/southkorea/southkorea-maps"
        "/master/kostat/2018/json/skorea-provinces-2018-geo.json"
    )
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"GeoJSON 로드 실패: {e}")
        return None


def price_to_color(price: int) -> str:
    """가격 3구간 → 색상 (싼거=파랑, 중간=초록, 비싼거=빨강)"""
    if price >= 50000:
        return "#EF4444"   # 빨강 — 비싼거
    elif price >= 25000:
        return "#4ADE80"   # 초록 — 중간거
    else:
        return "#60A5FA"   # 파랑 — 싼거


def build_map(geojson, selected_sido: str = "전국", search: str = "") -> go.Figure:
    df = pd.DataFrame([
        {
            "sido":  k,
            "price": v,
            "label": f"<b>{k}</b><br>평균 평당가: {v:,}만원",
            "lat":   SIDO_CENTER[k][0],
            "lon":   SIDO_CENTER[k][1],
        }
        for k, v in PRICE_DATA.items()
    ])

    # ① Choropleth — 시/도 면 색 채우기
    choropleth = go.Choropleth(
        geojson=geojson,
        locations=df["sido"],
        z=df["price"],
        featureidkey="properties.name",
        colorscale=[
            [0.00, "#BFDBFE"],  # 연파랑 — 저가
            [0.35, "#60A5FA"],  # 파랑
            [0.60, "#4ADE80"],  # 초록 — 중간
            [0.80, "#FB923C"],  # 주황
            [1.00, "#EF4444"],  # 빨강 — 고가
        ],
        zmin=min(PRICE_DATA.values()),
        zmax=max(PRICE_DATA.values()),
        text=df["label"],
        hovertemplate="%{text}<extra></extra>",
        marker_line_color="#FFFFFF",
        marker_line_width=1.5,
        showscale=True,
        colorbar=dict(
            title=dict(text="만원/3.3㎡", side="right"),
            thickness=14,
            len=0.65,
            x=1.01,
            tickfont=dict(size=11),
        ),
        name="",
    )

    traces = [choropleth]

    # ② 선택된 시/도 강조 (노란 테두리 오버레이)
    if selected_sido != "전국":
        sel = df[df["sido"] == selected_sido]
        highlight = go.Choropleth(
            geojson=geojson,
            locations=sel["sido"],
            z=sel["price"],
            featureidkey="properties.name",
            colorscale=[[0, "#FDE047"], [1, "#FDE047"]],
            zmin=0, zmax=1,
            showscale=False,
            marker_line_color="#172B4D",
            marker_line_width=3.5,
            hoverinfo="skip",
            name="",
        )
        traces.append(highlight)

    # ③ Scattergeo — 가격 원 마커
    marker_df = df if selected_sido == "전국" else df[df["sido"] == selected_sido]
    scatter = go.Scattergeo(
        lat=marker_df["lat"],
        lon=marker_df["lon"],
        text=marker_df["label"],
        hovertemplate="%{text}<extra></extra>",
        mode="markers",
        marker=dict(
            size=[max(10, v // 4000) for v in marker_df["price"]],
            color=[price_to_color(v) for v in marker_df["price"]],
            line=dict(color="#FFFFFF", width=1.5),
            opacity=0.85,
        ),
        name="",
    )
    traces.append(scatter)

    fig = go.Figure(data=traces)
    fig.update_geos(
        visible=False,           # 배경 타일 제거 (순수 SVG)
        fitbounds="locations",   # 한국 범위 자동 맞춤
        bgcolor="rgba(0,0,0,0)",
        showframe=False,
        showcoastlines=False,
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        geo=dict(bgcolor="rgba(0,0,0,0)"),
        showlegend=False,
        height=720,
    )
    return fig


# ══════════════════════════════════════════
# 페이지 렌더링
# ══════════════════════════════════════════

# ── 상단 검색창 (전체 너비)
search = st.text_input(
    label="아파트명 검색창",
    placeholder="🔍  아파트명을 입력하세요  (예: 래미안, 힐스테이트, 자이...)",
    label_visibility="collapsed",
)

st.markdown("<br>", unsafe_allow_html=True)

col_left, col_right = st.columns([1, 2])

# ── 좌측 패널
with col_left:
    section_badge("📍", "지역 선택")

    sido = st.selectbox("① 시/도", ["전국"] + SIDO_LIST, label_visibility="collapsed",
                        format_func=lambda x: f"🗺  {x}")

    # 드릴다운 구/군 (현재 더미 — 추후 연동)
    from utils.ui import load_css  # noqa — already loaded
    GUGUN_MAP = {
        "서울특별시": ["강남구","서초구","송파구","마포구","용산구","성동구","영등포구","노원구"],
        "경기도":     ["수원시","성남시","용인시","고양시","화성시","부천시"],
        "부산광역시": ["해운대구","수영구","남구","동래구"],
        "인천광역시": ["연수구","남동구","부평구","서구"],
    }
    gugun_list = GUGUN_MAP.get(sido, [])
    gugun = st.selectbox(
        "② 시/군/구",
        ["전체"] + gugun_list,
        disabled=(sido == "전국" or not gugun_list),
        label_visibility="collapsed",
        format_func=lambda x: f"🏘  {x}",
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 선택 지역 요약 카드
    if sido != "전국":
        price = PRICE_DATA.get(sido, 0)
        tier = "🔴 고가" if price >= 50000 else ("🟢 중간" if price >= 25000 else "🔵 저가")
        st.markdown(
            info_card(sido, f"{price:,}만원", f"평균 평당가 · {tier}"),
            unsafe_allow_html=True,
        )
    else:
        avg = int(sum(PRICE_DATA.values()) / len(PRICE_DATA))
        st.markdown(
            info_card("전국 평균", f"{avg:,}만원", "17개 시/도 더미 기준"),
            unsafe_allow_html=True,
        )

    # ── 가격 범례
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:13px; color:#6B7280; line-height:2;">
        <div>🔴 &nbsp;비싼거 &nbsp;(5억↑ / 3.3㎡)</div>
        <div>🟢 &nbsp;중간거 &nbsp;(2.5억~5억)</div>
        <div>🔵 &nbsp;싼거 &nbsp;&nbsp;&nbsp;(2.5억↓)</div>
    </div>
    """, unsafe_allow_html=True)

# ── 우측 지도
with col_right:
    geojson = load_geojson()
    if geojson:
        fig = build_map(geojson, selected_sido=sido, search=search)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    else:
        st.warning("지도 데이터를 불러올 수 없습니다. 인터넷 연결을 확인하세요.")
