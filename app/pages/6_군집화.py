# 경로: app/pages/6_군집화.py

import streamlit as st
import sys
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit.components.v1 as components

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from utils.ui import (
    load_css, render_sidebar, page_header,
    section_badge, stat_card,
)

st.set_page_config(page_title="군집화", layout="wide")
load_css()
render_sidebar()

page_header("군집화 — 지역 군집 분석")

CLUSTER_COLORS = [
    "#345BCB", "#EF4444", "#10B981", "#F97316",
    "#8B5CF6", "#EC4899", "#14B8A6", "#F59E0B",
]

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "models", "torch_kmeans_clustering.pkl")
KAKAO_JS_KEY = "f1eb550dcaed1694093d057a8d9019fd"

# cluster 인덱스 → 권역 해석 (군집화 분석 결과 기준)
CLUSTER_REGION = {
    0: "강원·경북권",
    1: "전남 동부 (순천·여수·광양)",
    2: "서울·수도권",
    3: "충청·전북권",
    4: "창원·김해 역세권 대단지",
    5: "부산 외곽 주거지",
    6: "전남·광주권",
}


@st.cache_resource
def load_model():
    from models.clustering.torch_kmeans_models import TorchKMeansLocationClusterModel
    m = TorchKMeansLocationClusterModel.load(MODEL_PATH)
    # Streamlit 환경에서 GPU 크래시 방지
    m.device = "cpu"
    return m


with st.spinner("모델 로딩 중..."):
    try:
        model = load_model()
        load_ok = True
    except Exception as e:
        st.error(f"모델 로드 실패: {e}")
        load_ok = False

if not load_ok:
    st.stop()

metrics    = model.metrics_ or {}
n_clusters = model.n_clusters
labels_all = model._labels
centroids  = model.centroids_
feature_cols = model.feature_cols
scaler       = model._scaler

# ── 성능 지표
section_badge("📐", "군집 평가 지표")
mc1, mc2, mc3, mc4 = st.columns(4)
mc1.markdown(stat_card(str(n_clusters),                                 "군집 수"),                               unsafe_allow_html=True)
mc2.markdown(stat_card(f"{metrics.get('Silhouette', 0):.4f}",          "Silhouette Score", "1에 가까울수록 좋음"), unsafe_allow_html=True)
mc3.markdown(stat_card(f"{metrics.get('Davies-Bouldin', 0):.4f}",      "Davies-Bouldin",   "0에 가까울수록 좋음"), unsafe_allow_html=True)
mc4.markdown(stat_card(f"{metrics.get('Calinski-Harabasz', 0):,.0f}",  "Calinski-Harabasz","클수록 좋음"),         unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── 군집 특성 시각화 (레이더 + 히트맵)
section_badge("📊", "군집 특성 시각화")

# 중심점 원본값 복원 (스케일러 역변환)
_cuw = centroids.copy()
for col, w in model.feature_weights.items():
    if col in feature_cols:
        _cuw[:, feature_cols.index(col)] /= w
_c_orig = scaler.inverse_transform(_cuw)
_radar_df = pd.DataFrame(_c_orig, columns=feature_cols)

# 피처별 0~1 정규화 (레이더·히트맵 공용)
_norm_df = _radar_df.copy()
for col in feature_cols:
    col_min, col_max = _norm_df[col].min(), _norm_df[col].max()
    if col_max > col_min:
        _norm_df[col] = (_norm_df[col] - col_min) / (col_max - col_min)
    else:
        _norm_df[col] = 0.5

# 피처 표시명
FEAT_LABELS = {
    "위도": "위도", "경도": "경도",
    "평당가": "평당가", "건물연식": "건물연식", "거래활성도": "거래활성도",
}
radar_cats = [FEAT_LABELS.get(c, c) for c in feature_cols]
cluster_labels = [f"군집 {i+1}" for i in range(n_clusters)]

viz_col1, viz_col2 = st.columns(2)

# ── 레이더 차트
with viz_col1:
    fig_radar = go.Figure()
    for i in range(n_clusters):
        vals = _norm_df.iloc[i].tolist()
        fig_radar.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=radar_cats + [radar_cats[0]],
            fill="toself",
            fillcolor=CLUSTER_COLORS[i % len(CLUSTER_COLORS)],
            line=dict(color=CLUSTER_COLORS[i % len(CLUSTER_COLORS)], width=2),
            opacity=0.4,
            name=cluster_labels[i],
        ))
    fig_radar.update_layout(
        title="군집별 피처 프로필 (정규화)",
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], showticklabels=False),
            bgcolor="rgba(0,0,0,0)",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        height=420,
        margin=dict(l=40, r=40, t=50, b=20),
        legend=dict(orientation="h", y=-0.1, x=0.5, xanchor="center"),
        showlegend=True,
    )
    st.plotly_chart(fig_radar, use_container_width=True, config={"displayModeBar": False})

# ── 히트맵
with viz_col2:
    heat_z = _norm_df[feature_cols].values.tolist()
    fig_heat = go.Figure(go.Heatmap(
        z=heat_z,
        x=radar_cats,
        y=cluster_labels,
        colorscale=[[0, "#EFF6FF"], [0.5, "#60A5FA"], [1, "#1D4ED8"]],
        showscale=True,
        text=[[f"{v:.2f}" for v in row] for row in heat_z],
        texttemplate="%{text}",
        textfont=dict(size=11),
    ))
    fig_heat.update_layout(
        title="군집 × 피처 히트맵 (정규화)",
        height=420,
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=50, b=20),
        xaxis=dict(side="bottom"),
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig_heat, use_container_width=True, config={"displayModeBar": False})

st.markdown("<br>", unsafe_allow_html=True)

# ── 군집 중심 지도 (카카오맵)
if "위도" in feature_cols and "경도" in feature_cols:
    section_badge("🗺️", "군집 중심 지도", color="#10B981")

    lat_idx = feature_cols.index("위도")
    lon_idx = feature_cols.index("경도")

    centroids_unweighted = centroids.copy()
    centroids_unweighted[:, lat_idx] /= model.feature_weights.get("위도", 1.0)
    centroids_unweighted[:, lon_idx] /= model.feature_weights.get("경도", 1.0)
    centroids_orig = scaler.inverse_transform(centroids_unweighted)
    center_lats = centroids_orig[:, lat_idx]
    center_lons = centroids_orig[:, lon_idx]

    markers_js = ""
    for i in range(n_clusters):
        color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
        lat = float(center_lats[i])
        lon = float(center_lons[i])
        label = f"군집 {i+1}"
        markers_js += f"""
        (function() {{
            var pos = new kakao.maps.LatLng({lat}, {lon});
            var circle = new kakao.maps.Circle({{
                center: pos,
                radius: 40000,
                strokeWeight: 2,
                strokeColor: '{color}',
                strokeOpacity: 1,
                fillColor: '{color}',
                fillOpacity: 0.7,
            }});
            circle.setMap(map);
            var overlay = new kakao.maps.CustomOverlay({{
                position: pos,
                content: '<div style="padding:4px 8px; background:{color}; color:#fff; font-size:12px; font-weight:700; border-radius:4px; white-space:nowrap;">{label}</div>',
                yAnchor: 2.6,
            }});
            overlay.setMap(map);
        }})();
        """

    center_lat = float(center_lats.mean())
    center_lon = float(center_lons.mean())

    kakao_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{ margin: 0; padding: 0; }}
            #map {{ width: 100%; height: 480px; }}
        </style>
    </head>
    <body>
        <div id="map"></div>
        <script type="text/javascript"
            src="//dapi.kakao.com/v2/maps/sdk.js?appkey={KAKAO_JS_KEY}">
        </script>
        <script>
            var container = document.getElementById('map');
            var options = {{
                center: new kakao.maps.LatLng({center_lat}, {center_lon}),
                level: 13,
            }};
            var map = new kakao.maps.Map(container, options);
            {markers_js}
        </script>
    </body>
    </html>
    """

    _, map_col, _ = st.columns([1, 3, 1])
    with map_col:
        components.html(kakao_html, height=500)
    st.markdown("<br>", unsafe_allow_html=True)

# ── 군집별 피처 평균
section_badge("📋", "군집별 피처 평균", color="#F97316")

centroids_unweighted_all = centroids.copy()
for col, w in model.feature_weights.items():
    if col in feature_cols:
        centroids_unweighted_all[:, feature_cols.index(col)] /= w
centroids_orig_all = scaler.inverse_transform(centroids_unweighted_all)

summary_df = pd.DataFrame(centroids_orig_all, columns=feature_cols)
summary_df.index = [f"군집 {i+1}" for i in range(n_clusters)]

if "평당가" in summary_df.columns:
    summary_df["평당가(만원/평)"] = np.expm1(summary_df["평당가"]).round(0).astype(int)
    summary_df = summary_df.drop(columns=["평당가"])
if "건물연식" in summary_df.columns:
    summary_df["건물연식"] = summary_df["건물연식"].round(1)
for col in ["위도", "경도"]:
    if col in summary_df.columns:
        summary_df[col] = summary_df[col].round(4)

# 군집 인덱스 기준 권역명 삽입
summary_df.insert(0, "권역", [
    CLUSTER_REGION.get(i, "기타") for i in range(n_clusters)
])

# 거래활성도: 매우 작은 비율값이므로 소수점 6자리 표시
float_cols = summary_df.select_dtypes("float").columns
fmt = {c: ("{:.6f}" if c == "거래활성도" else "{:,.2f}") for c in float_cols}

st.dataframe(summary_df.style.format(fmt), use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── 군집 크기 바차트
section_badge("📊", "군집별 데이터 비율")
cluster_counts = pd.Series(labels_all).value_counts().sort_index()

fig3 = go.Figure(go.Bar(
    x=[f"군집 {i+1}" for i in cluster_counts.index],
    y=cluster_counts.values,
    marker_color=[CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i in cluster_counts.index],
    text=[f"{v:,}건" for v in cluster_counts.values],
    textposition="outside",
))
fig3.update_layout(
    height=280,
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=0, r=0, t=10, b=0),
    yaxis=dict(showgrid=True, gridcolor="#F0F4F8"),
)
st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})
