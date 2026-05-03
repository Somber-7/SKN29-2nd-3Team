# 경로: app/pages/3_지도시각화.py

import streamlit as st
import sys
import os
import requests
import pandas as pd
import folium
from folium.features import GeoJsonTooltip
from streamlit_folium import st_folium

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from utils.ui import load_css, render_sidebar, page_header, section_badge, info_card
from utils.db import fetch_all

st.set_page_config(page_title="지도 시각화", layout="wide")
load_css()
render_sidebar()

page_header("지도 시각화")

# GeoJSON 통계청 코드 → 시/도명
GEO_SIDO_CODE = {
    "11": "서울특별시",    "21": "부산광역시",    "22": "대구광역시",
    "23": "인천광역시",    "24": "광주광역시",     "25": "대전광역시",
    "26": "울산광역시",    "29": "세종특별자치시", "31": "경기도",
    "32": "강원특별자치도", "33": "충청북도",      "34": "충청남도",
    "35": "전북특별자치도", "36": "전라남도",      "37": "경상북도",
    "38": "경상남도",      "39": "제주특별자치도",
}
SIDO_NAME_TO_GEO = {v: k for k, v in GEO_SIDO_CODE.items()}
SIDO_LIST = list(GEO_SIDO_CODE.values())


def price_color(price: float, vmin: float, vmax: float) -> str:
    if vmax == vmin:
        return "#60A5FA"
    ratio = (price - vmin) / (vmax - vmin)
    if ratio >= 0.80:
        return "#EF4444"
    elif ratio >= 0.60:
        return "#FB923C"
    elif ratio >= 0.40:
        return "#FACC15"
    elif ratio >= 0.20:
        return "#4ADE80"
    return "#60A5FA"


@st.cache_data(show_spinner="시/도 지도 로딩 중...")
def load_sido_geojson():
    r = requests.get(
        "https://raw.githubusercontent.com/southkorea/southkorea-maps"
        "/master/kostat/2018/json/skorea-provinces-2018-geo.json",
        timeout=10,
    )
    r.raise_for_status()
    return r.json()


@st.cache_data(show_spinner="구/군 지도 로딩 중...")
def load_sigungu_geojson():
    r = requests.get(
        "https://raw.githubusercontent.com/southkorea/southkorea-maps"
        "/master/kostat/2018/json/skorea-municipalities-2018-geo.json",
        timeout=15,
    )
    r.raise_for_status()
    return r.json()


@st.cache_data(show_spinner="평당가 데이터 로딩 중...", ttl=3600)
def load_sigungu_stats() -> pd.DataFrame:
    rows = fetch_all(
        "SELECT sigungu, sido, region_code, geo_code, avg_price_per_pyeong "
        "FROM tbl_sigungu_stats WHERE geo_code IS NOT NULL ORDER BY region_code"
    )
    return pd.DataFrame(rows)


def build_sido_map(sido_geojson: dict, stats_df: pd.DataFrame, selected_sido: str = "") -> folium.Map:
    sido_df = (
        stats_df.groupby("sido", as_index=False)["avg_price_per_pyeong"]
        .mean().round(0)
    )
    sido_df["avg_price_per_pyeong"] = sido_df["avg_price_per_pyeong"].astype(int)
    sido_df["geo_code"] = sido_df["sido"].map(SIDO_NAME_TO_GEO)
    sido_df = sido_df.dropna(subset=["geo_code"])

    price_map = dict(zip(sido_df["geo_code"], sido_df["avg_price_per_pyeong"]))
    name_map  = dict(zip(sido_df["geo_code"], sido_df["sido"]))
    vmin = int(sido_df["avg_price_per_pyeong"].min())
    vmax = int(sido_df["avg_price_per_pyeong"].max())

    for f in sido_geojson["features"]:
        code = f["properties"]["code"]
        f["properties"]["price"] = price_map.get(code, 0)
        f["properties"]["name"]  = name_map.get(code, "")

    m = folium.Map(
        location=[36.5, 127.8],
        zoom_start=7,
        tiles=None,  # 배경 지도 없음
        zoom_control=True,
        scrollWheelZoom=True,
    )
    # 흰 배경
    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}{r}.png",
        attr="CartoDB",
        name="base",
        opacity=0.0,  # 타일 완전 투명
    ).add_to(m)

    def style_fn(feature):
        code  = feature["properties"]["code"]
        price = feature["properties"]["price"]
        is_selected = name_map.get(code, "") == selected_sido
        return {
            "fillColor":   price_color(price, vmin, vmax),
            "fillOpacity": 0.85,
            "color":       "#F59E0B" if is_selected else "#94A3B8",
            "weight":      3.5      if is_selected else 1.0,
        }

    def highlight_fn(_feature):
        return {
            "fillOpacity": 1.0,
            "color":       "#1E293B",
            "weight":      2.5,
        }

    folium.GeoJson(
        sido_geojson,
        style_function=style_fn,
        highlight_function=highlight_fn,
        tooltip=GeoJsonTooltip(
            fields=["name", "price"],
            aliases=["시/도:", "평균 평당가 (만원/평):"],
            localize=True,
            sticky=True,
            labels=True,
            style=(
                "background-color:#1E293B; color:#F8FAFC; "
                "font-size:13px; border:1px solid #94A3B8; "
                "border-radius:4px; padding:6px 10px;"
            ),
        ),
    ).add_to(m)

    return m


def build_sigungu_map(full_geo: dict, sido_stats: pd.DataFrame, geo_prefix: str, selected_gugun: str = "") -> folium.Map:
    features = [f for f in full_geo["features"] if f["properties"]["code"].startswith(geo_prefix)]
    filtered_geo = {"type": "FeatureCollection", "features": features}

    price_map = dict(zip(sido_stats["geo_code"], sido_stats["avg_price_per_pyeong"]))
    name_map  = dict(zip(sido_stats["geo_code"], sido_stats["sigungu"]))

    prices = [p for p in price_map.values() if p]
    vmin = int(min(prices)) if prices else 0
    vmax = int(max(prices)) if prices else 1

    for f in filtered_geo["features"]:
        code = f["properties"]["code"]
        full_name  = name_map.get(code, "")
        short_name = full_name.split()[-1] if full_name else ""
        f["properties"]["price"]      = price_map.get(code, 0)
        f["properties"]["name"]       = full_name
        f["properties"]["short_name"] = short_name

    lats, lons = [], []
    for f in filtered_geo["features"]:
        coords    = f["geometry"]["coordinates"]
        geom_type = f["geometry"]["type"]
        if geom_type == "Polygon":
            for pt in coords[0]:
                lons.append(pt[0]); lats.append(pt[1])
        elif geom_type == "MultiPolygon":
            for poly in coords:
                for pt in poly[0]:
                    lons.append(pt[0]); lats.append(pt[1])

    center = [(min(lats)+max(lats))/2, (min(lons)+max(lons))/2] if lats else [36.5, 127.8]

    m = folium.Map(
        location=center,
        zoom_start=10,
        tiles=None,
        zoom_control=True,
        scrollWheelZoom=True,
    )
    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}{r}.png",
        attr="CartoDB",
        name="base",
        opacity=0.0,
    ).add_to(m)

    def style_fn(feature):
        price = feature["properties"]["price"]
        short = feature["properties"]["short_name"]
        is_selected = short == selected_gugun and selected_gugun != "전체"
        return {
            "fillColor":   price_color(price, vmin, vmax),
            "fillOpacity": 0.85,
            "color":       "#F59E0B" if is_selected else "#94A3B8",
            "weight":      3.5      if is_selected else 0.8,
        }

    def highlight_fn(_feature):
        return {
            "fillOpacity": 1.0,
            "color":       "#1E293B",
            "weight":      2.5,
        }

    folium.GeoJson(
        filtered_geo,
        style_function=style_fn,
        highlight_function=highlight_fn,
        tooltip=GeoJsonTooltip(
            fields=["name", "price"],
            aliases=["구/군:", "평균 평당가 (만원/평):"],
            localize=True,
            sticky=True,
            labels=True,
            style=(
                "background-color:#1E293B; color:#F8FAFC; "
                "font-size:13px; border:1px solid #94A3B8; "
                "border-radius:4px; padding:6px 10px;"
            ),
        ),
    ).add_to(m)

    return m


# ══════════════════════════════════════════
# 세션 상태 초기화
# ══════════════════════════════════════════
if "map_sido" not in st.session_state:
    st.session_state.map_sido = "전국"
if "map_gugun" not in st.session_state:
    st.session_state.map_gugun = "전체"
if "last_click_coord" not in st.session_state:
    st.session_state.last_click_coord = None

stats_df = load_sigungu_stats()

# ══════════════════════════════════════════
# 섹션 1: 지역 선택
# ══════════════════════════════════════════
section_badge("📍", "지역 선택")
sel_col1, sel_col2, sel_col3 = st.columns([2, 2, 1])

with sel_col1:
    sido_options = ["전국"] + SIDO_LIST
    sido_idx = sido_options.index(st.session_state.map_sido) if st.session_state.map_sido in sido_options else 0
    sido = st.selectbox(
        "시/도 선택",
        sido_options,
        index=sido_idx,
        format_func=lambda x: f"🗺  {x}",
        key="sido_select",
    )
    if sido != st.session_state.map_sido:
        st.session_state.map_sido = sido
        st.session_state.map_gugun = "전체"
        st.rerun()

with sel_col2:
    gugun_options = ["전체"]
    if sido != "전국":
        sido_stats_sel = stats_df[stats_df["sido"] == sido]
        gugun_options += sorted(
            row["sigungu"].split()[-1]
            for _, row in sido_stats_sel.iterrows()
            if row["geo_code"]
        )

    gugun_idx = gugun_options.index(st.session_state.map_gugun) if st.session_state.map_gugun in gugun_options else 0
    gugun = st.selectbox(
        "구/군 선택",
        gugun_options,
        index=gugun_idx,
        disabled=(sido == "전국"),
        format_func=lambda x: f"🏘  {x}",
        key="gugun_select",
    )
    if gugun != st.session_state.map_gugun:
        st.session_state.map_gugun = gugun
        st.rerun()

with sel_col3:
    st.markdown("<br>", unsafe_allow_html=True)
    if sido != "전국":
        if st.button("← 전국 보기", use_container_width=True):
            st.session_state.map_sido = "전국"
            st.session_state.map_gugun = "전체"
            st.rerun()

st.divider()

# ══════════════════════════════════════════
# 섹션 2: 요약 + 범례
# ══════════════════════════════════════════
info_col1, info_col2 = st.columns([1, 2])

with info_col1:
    section_badge("📊", "선택 지역 요약")
    if sido != "전국":
        sido_stats_f = stats_df[stats_df["sido"] == sido]
        price = int(sido_stats_f["avg_price_per_pyeong"].mean().round(0))
        all_sido_prices = stats_df.groupby("sido")["avg_price_per_pyeong"].mean()
        vmin_s, vmax_s = int(all_sido_prices.min()), int(all_sido_prices.max())
        ratio = (price - vmin_s) / max(vmax_s - vmin_s, 1)
        tier = "🔴 고가" if ratio >= 0.7 else ("🟢 중간" if ratio >= 0.3 else "🔵 저가")
        st.markdown(info_card(sido, f"{price:,}만원/평", f"실거래 평균 · {tier}"), unsafe_allow_html=True)
    else:
        avg = int(stats_df["avg_price_per_pyeong"].mean().round(0))
        st.markdown(info_card("전국 평균", f"{avg:,}만원/평", "실거래 평균 (2015~2023)"), unsafe_allow_html=True)

with info_col2:
    section_badge("🎨", "가격 범례")
    st.markdown("""
    <div style="display:flex; gap:20px; font-size:14px; color:#1E293B; padding-top:8px; flex-wrap:wrap;">
        <div><span style="display:inline-block;width:16px;height:16px;background:#EF4444;border-radius:2px;vertical-align:middle;margin-right:5px;border:1px solid #cbd5e1;"></span><b>최고가</b> 상위 20%</div>
        <div><span style="display:inline-block;width:16px;height:16px;background:#FB923C;border-radius:2px;vertical-align:middle;margin-right:5px;border:1px solid #cbd5e1;"></span><b>고가</b></div>
        <div><span style="display:inline-block;width:16px;height:16px;background:#FACC15;border-radius:2px;vertical-align:middle;margin-right:5px;border:1px solid #cbd5e1;"></span><b>중간</b></div>
        <div><span style="display:inline-block;width:16px;height:16px;background:#4ADE80;border-radius:2px;vertical-align:middle;margin-right:5px;border:1px solid #cbd5e1;"></span><b>저가</b></div>
        <div><span style="display:inline-block;width:16px;height:16px;background:#60A5FA;border-radius:2px;vertical-align:middle;margin-right:5px;border:1px solid #cbd5e1;"></span><b>최저가</b> 하위 20%</div>
    </div>
    <div style="font-size:12px; color:#64748B; margin-top:8px;">
        ※ 2015~2023 실거래가 기반 평균 평당가 (만원/3.3㎡) &nbsp;|&nbsp; 지도 클릭 시 해당 시/도로 드릴다운
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ══════════════════════════════════════════
# 섹션 3: 지도
# ══════════════════════════════════════════
map_title = "전국 아파트 평균 평당가" if sido == "전국" else f"{sido} 구/군별 평균 평당가"
section_badge("🗺️", map_title)

def point_in_polygon(lat: float, lng: float, ring: list) -> bool:
    """Ray casting — ring은 [lon, lat] 형식의 좌표 리스트"""
    inside = False
    x, y = lng, lat
    n = len(ring)
    j = n - 1
    for i in range(n):
        xi, yi = ring[i][0], ring[i][1]
        xj, yj = ring[j][0], ring[j][1]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def find_feature_at(geojson: dict, lat: float, lng: float) -> dict | None:
    """클릭 좌표가 속하는 GeoJSON feature 반환"""
    for f in geojson["features"]:
        geom = f["geometry"]
        rings = []
        if geom["type"] == "Polygon":
            rings = [geom["coordinates"][0]]
        elif geom["type"] == "MultiPolygon":
            rings = [poly[0] for poly in geom["coordinates"]]
        if any(point_in_polygon(lat, lng, ring) for ring in rings):
            return f
    return None


try:
    if sido == "전국":
        sido_geojson = load_sido_geojson()
        # geo_code → sido명 역매핑 (GeoJSON code 앞 2자리 기준)
        sido_code_to_name = {}
        sido_df_tmp = stats_df.groupby("sido", as_index=False)["avg_price_per_pyeong"].mean()
        sido_df_tmp["geo_code"] = sido_df_tmp["sido"].map(SIDO_NAME_TO_GEO)
        for _, r in sido_df_tmp.dropna(subset=["geo_code"]).iterrows():
            sido_code_to_name[r["geo_code"]] = r["sido"]

        m = build_sido_map(sido_geojson, stats_df, selected_sido=sido)
        result = st_folium(m, width="100%", height=620, returned_objects=["last_clicked"])

        st.write("▶ result 전체:", result)

        last = result.get("last_clicked") if result else None
        if last and last.get("lat") is not None:
            coord_key = (round(last["lat"], 5), round(last["lng"], 5))
            if coord_key != st.session_state.last_click_coord:
                st.session_state.last_click_coord = coord_key
                lat, lng = last["lat"], last["lng"]
                f = find_feature_at(sido_geojson, lat, lng)
                st.write("▶ find_feature_at 결과:", f["properties"] if f else None)
                st.write("▶ sido_code_to_name:", sido_code_to_name)
                if f:
                    code = f["properties"].get("code", "")
                    clicked_name = sido_code_to_name.get(code, "")
                    st.write("▶ code:", code, "/ clicked_name:", clicked_name)
                    if clicked_name in SIDO_LIST:
                        st.session_state.map_sido = clicked_name
                        st.session_state.map_gugun = "전체"
                        st.rerun()

    else:
        sido_stats_map = stats_df[stats_df["sido"] == sido]
        geo_prefix = SIDO_NAME_TO_GEO[sido]
        sigungu_geo_full = load_sigungu_geojson()
        # geo_code → sigungu 단축명 매핑
        gugun_code_to_short = {
            row["geo_code"]: row["sigungu"].split()[-1]
            for _, row in sido_stats_map.iterrows() if row["geo_code"]
        }

        m = build_sigungu_map(sigungu_geo_full, sido_stats_map, geo_prefix, selected_gugun=gugun)
        result = st_folium(m, width="100%", height=620, returned_objects=["last_clicked"])

        last = result.get("last_clicked") if result else None
        if last and last.get("lat") is not None:
            coord_key = (round(last["lat"], 5), round(last["lng"], 5))
            if coord_key != st.session_state.last_click_coord:
                st.session_state.last_click_coord = coord_key
                lat, lng = last["lat"], last["lng"]
                features_filtered = [
                    f for f in sigungu_geo_full["features"]
                    if f["properties"]["code"].startswith(geo_prefix)
                ]
                f = find_feature_at({"type": "FeatureCollection", "features": features_filtered}, lat, lng)
                if f:
                    code = f["properties"].get("code", "")
                    short_name = gugun_code_to_short.get(code, "")
                    if short_name and short_name in gugun_options:
                        st.session_state.map_gugun = short_name
                        st.rerun()

except requests.RequestException as e:
    st.error(f"지도 데이터 로드 실패. 인터넷 연결을 확인하세요.\n{e}")
except Exception as e:
    st.error(f"오류가 발생했습니다: {e}")
