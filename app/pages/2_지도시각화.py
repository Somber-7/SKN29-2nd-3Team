# 경로: app/pages/3_지도시각화.py

import streamlit as st
import sys
import os
import json
import requests
import pandas as pd
import folium
from folium.features import GeoJsonTooltip
from streamlit_folium import st_folium

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from utils.ui import page_header, section_badge, info_card
from utils.db import fetch_all

page_header("지도 시각화")

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
    if ratio >= 0.80: return "#EF4444"
    if ratio >= 0.60: return "#FB923C"
    if ratio >= 0.40: return "#FACC15"
    if ratio >= 0.20: return "#4ADE80"
    return "#60A5FA"


def pip(lat: float, lng: float, ring: list) -> bool:
    inside = False
    x, y = lng, lat
    j = len(ring) - 1
    for i, (xi, yi) in enumerate(ring):
        xj, yj = ring[j]
        if ((yi > y) != (yj > y)) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def find_feature(geojson: dict, lat: float, lng: float):
    for f in geojson["features"]:
        geom  = f["geometry"]
        rings = ([geom["coordinates"][0]] if geom["type"] == "Polygon"
                 else [p[0] for p in geom["coordinates"]])
        if any(pip(lat, lng, r) for r in rings):
            return f
    return None


@st.cache_data(show_spinner="시/도 지도 로딩 중...")
def load_sido_geojson():
    r = requests.get(
        "https://raw.githubusercontent.com/southkorea/southkorea-maps"
        "/master/kostat/2018/json/skorea-provinces-2018-geo.json", timeout=10)
    r.raise_for_status()
    return r.json()


@st.cache_data(show_spinner="구/군 지도 로딩 중...")
def load_sigungu_geojson():
    r = requests.get(
        "https://raw.githubusercontent.com/southkorea/southkorea-maps"
        "/master/kostat/2018/json/skorea-municipalities-2018-geo.json", timeout=15)
    r.raise_for_status()
    return r.json()


@st.cache_data(show_spinner="평당가 데이터 로딩 중...", ttl=3600)
def load_sigungu_stats() -> pd.DataFrame:
    rows = fetch_all(
        "SELECT sigungu, sido, region_code, geo_code, avg_price_per_pyeong "
        "FROM tbl_sigungu_stats WHERE geo_code IS NOT NULL ORDER BY region_code"
    )
    return pd.DataFrame(rows)


# ══════════════════════════════════════════
# 세션 상태 초기화
# ══════════════════════════════════════════
if "map_sido" not in st.session_state:
    st.session_state.map_sido = "전국"
if "last_click" not in st.session_state:
    st.session_state.last_click = ""

# session_state 타입 오염 방지
if not isinstance(st.session_state.map_sido, str) or st.session_state.map_sido not in ["전국"] + SIDO_LIST:
    st.session_state.map_sido = "전국"
if not isinstance(st.session_state.last_click, str):
    st.session_state.last_click = ""

stats_df = load_sigungu_stats()

sido = st.session_state.map_sido

# ══════════════════════════════════════════
# 섹션 1: 지역 선택
# ══════════════════════════════════════════
section_badge("📍", "지역 선택")
sel_col1, sel_col2 = st.columns([3, 1])

with sel_col1:
    sido_options = ["전국"] + SIDO_LIST
    new_sido = st.selectbox(
        "시/도 선택", sido_options,
        index=sido_options.index(sido) if sido in sido_options else 0,
        format_func=lambda x: f"🗺  {x}",
    )
    if new_sido != sido:
        st.session_state.map_sido  = new_sido
        st.session_state.last_click = ""
        st.rerun()

with sel_col2:
    st.markdown("<br>", unsafe_allow_html=True)
    if sido != "전국":
        if st.button("← 전국 보기", use_container_width=True):
            st.session_state.map_sido  = "전국"
            st.session_state.last_click = ""
            st.rerun()

st.divider()

# ══════════════════════════════════════════
# 섹션 2: 요약 + 범례
# ══════════════════════════════════════════
info_col1, info_col2 = st.columns([1, 2])

with info_col1:
    section_badge("📊", "선택 지역 요약")
    if sido != "전국":
        sido_stats_f   = stats_df[stats_df["sido"] == sido]
        price          = int(sido_stats_f["avg_price_per_pyeong"].mean().round(0))
        all_prices     = stats_df.groupby("sido")["avg_price_per_pyeong"].mean()
        vmin_s, vmax_s = int(all_prices.min()), int(all_prices.max())
        ratio = (price - vmin_s) / max(vmax_s - vmin_s, 1)
        tier  = "🔴 고가" if ratio >= 0.7 else ("🟢 중간" if ratio >= 0.3 else "🔵 저가")
        st.markdown(info_card(sido, f"{price:,}만원/평", f"실거래 평균 · {tier}"), unsafe_allow_html=True)
    else:
        avg = int(stats_df["avg_price_per_pyeong"].mean().round(0))
        st.markdown(info_card("전국 평균", f"{avg:,}만원/평", "실거래 평균 (2015~2023)"), unsafe_allow_html=True)

with info_col2:
    section_badge("🎨", "가격 범례")
    st.markdown("""
    <div style="display:flex;gap:20px;font-size:14px;color:#1E293B;padding-top:8px;flex-wrap:wrap;">
        <div><span style="display:inline-block;width:16px;height:16px;background:#EF4444;border-radius:2px;vertical-align:middle;margin-right:5px;border:1px solid #cbd5e1;"></span><b>최고가</b> 상위 20%</div>
        <div><span style="display:inline-block;width:16px;height:16px;background:#FB923C;border-radius:2px;vertical-align:middle;margin-right:5px;border:1px solid #cbd5e1;"></span><b>고가</b></div>
        <div><span style="display:inline-block;width:16px;height:16px;background:#FACC15;border-radius:2px;vertical-align:middle;margin-right:5px;border:1px solid #cbd5e1;"></span><b>중간</b></div>
        <div><span style="display:inline-block;width:16px;height:16px;background:#4ADE80;border-radius:2px;vertical-align:middle;margin-right:5px;border:1px solid #cbd5e1;"></span><b>저가</b></div>
        <div><span style="display:inline-block;width:16px;height:16px;background:#60A5FA;border-radius:2px;vertical-align:middle;margin-right:5px;border:1px solid #cbd5e1;"></span><b>최저가</b> 하위 20%</div>
    </div>
    <div style="font-size:12px;color:#64748B;margin-top:8px;">
        ※ 2015~2023 실거래가 기반 평균 평당가 (만원/3.3㎡) &nbsp;|&nbsp; 지도 클릭 시 해당 시/도로 드릴다운
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ══════════════════════════════════════════
# 섹션 3: 지도
# ══════════════════════════════════════════
map_title = "전국 아파트 평균 평당가" if sido == "전국" else f"{sido} 구/군별 평균 평당가"
section_badge("🗺️", map_title)

try:
    if sido == "전국":
        sido_geojson = load_sido_geojson()
        geo = json.loads(json.dumps(sido_geojson))

        sido_df = stats_df.groupby("sido", as_index=False)["avg_price_per_pyeong"].mean().round(0)
        sido_df["avg_price_per_pyeong"] = sido_df["avg_price_per_pyeong"].astype(int)
        sido_df["geo_code"] = sido_df["sido"].map(SIDO_NAME_TO_GEO)
        sido_df = sido_df.dropna(subset=["geo_code"])
        price_map = dict(zip(sido_df["geo_code"], sido_df["avg_price_per_pyeong"]))
        name_map  = dict(zip(sido_df["geo_code"], sido_df["sido"]))
        vmin = int(sido_df["avg_price_per_pyeong"].min())
        vmax = int(sido_df["avg_price_per_pyeong"].max())

        for f in geo["features"]:
            code = f["properties"]["code"]
            f["properties"]["price"]     = price_map.get(code, 0)
            f["properties"]["sido_name"] = name_map.get(code, "")

        m = folium.Map(location=[36.5, 127.8], zoom_start=7, tiles=None,
                       zoom_control=True, scrollWheelZoom=True)
        folium.GeoJson(
            geo,
            style_function=lambda feature: {
                "fillColor":   price_color(feature["properties"]["price"], vmin, vmax),
                "fillOpacity": 0.85,
                "color":       "#94A3B8",
                "weight":      1.0,
            },
            highlight_function=lambda _f: {"fillOpacity": 1.0, "color": "#1E293B", "weight": 2.5},
            tooltip=GeoJsonTooltip(
                fields=["sido_name", "price"], aliases=["시/도:", "평균 평당가 (만원/평):"],
                localize=True, sticky=True, labels=True,
                style="background-color:#1E293B;color:#F8FAFC;font-size:13px;border:1px solid #94A3B8;border-radius:4px;padding:6px 10px;",
            ),
        ).add_to(m)

        result = st_folium(m, width="100%", height=620,
                           returned_objects=["last_clicked"],
                           key="folium_sido_map")

        # last_clicked 안전 추출
        clicked = None
        try:
            raw = result if isinstance(result, dict) else {}
            lc = raw.get("last_clicked")
            if isinstance(lc, dict) and "lat" in lc and "lng" in lc:
                clicked = lc
        except Exception:
            pass

        if clicked is not None:
            lat, lng = clicked["lat"], clicked["lng"]
            click_str = f"{round(lat, 4)},{round(lng, 4)}"
            if click_str != st.session_state.last_click:
                st.session_state.last_click = click_str
                feat = find_feature(geo, lat, lng)
                if feat:
                    name = feat["properties"].get("sido_name", "")
                    if name in SIDO_LIST:
                        st.session_state.map_sido = name
                        st.rerun()

    else:
        sigungu_geo_full = load_sigungu_geojson()
        sido_stats_map = stats_df[stats_df["sido"] == sido]
        geo_prefix = SIDO_NAME_TO_GEO[sido]

        features = [f for f in sigungu_geo_full["features"]
                    if f["properties"]["code"].startswith(geo_prefix)]
        geo = {"type": "FeatureCollection", "features": json.loads(json.dumps(features))}

        price_map = dict(zip(sido_stats_map["geo_code"], sido_stats_map["avg_price_per_pyeong"]))
        name_map  = dict(zip(sido_stats_map["geo_code"], sido_stats_map["sigungu"]))
        prices = list(price_map.values())
        vmin = int(min(prices)) if prices else 0
        vmax = int(max(prices)) if prices else 1

        for f in geo["features"]:
            code      = f["properties"]["code"]
            full_name = name_map.get(code, "")
            f["properties"]["price"]      = price_map.get(code, 0)
            f["properties"]["gugun_name"] = full_name
            f["properties"]["short_name"] = full_name.split()[-1] if full_name else ""

        lats, lons = [], []
        for f in geo["features"]:
            coords = f["geometry"]["coordinates"]
            gtype  = f["geometry"]["type"]
            if gtype == "Polygon":
                for pt in coords[0]: lons.append(pt[0]); lats.append(pt[1])
            elif gtype == "MultiPolygon":
                for poly in coords:
                    for pt in poly[0]: lons.append(pt[0]); lats.append(pt[1])
        if lats:
            center = [(min(lats)+max(lats))/2, (min(lons)+max(lons))/2]
            span = max(max(lats)-min(lats), max(lons)-min(lons))
            zoom = 11 if span < 0.45 else (10 if span < 0.95 else (9 if span < 2.0 else 8))
        else:
            center, zoom = [36.5, 127.8], 10

        m = folium.Map(location=center, zoom_start=zoom, tiles=None,
                       zoom_control=True, scrollWheelZoom=True)
        folium.GeoJson(
            geo,
            style_function=lambda feature: {
                "fillColor":   price_color(feature["properties"]["price"], vmin, vmax),
                "fillOpacity": 0.85,
                "color":       "#94A3B8",
                "weight":      0.8,
            },
            highlight_function=lambda _f: {"fillOpacity": 1.0, "color": "#1E293B", "weight": 2.5},
            tooltip=GeoJsonTooltip(
                fields=["gugun_name", "price"], aliases=["구/군:", "평균 평당가 (만원/평):"],
                localize=True, sticky=True, labels=True,
                style="background-color:#1E293B;color:#F8FAFC;font-size:13px;border:1px solid #94A3B8;border-radius:4px;padding:6px 10px;",
            ),
        ).add_to(m)

        st_folium(m, width="100%", height=620,
                  returned_objects=[],
                  key=f"folium_sigungu_{sido}")

except requests.RequestException as e:
    st.error(f"지도 데이터 로드 실패. 인터넷 연결을 확인하세요.\n{e}")
except Exception as e:
    st.error(f"오류가 발생했습니다: {e}")
