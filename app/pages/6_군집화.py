# 경로: app/pages/6_군집화.py

import streamlit as st
import sys
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit.components.v1 as components
from sklearn.decomposition import PCA

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

# ── 군집 결과 시각화 (PCA 2D)
section_badge("📊", "군집 결과 시각화 (PCA 2D)")

pca = PCA(n_components=2, random_state=42)
centroids_2d = pca.fit_transform(centroids)

fig = go.Figure()
for i in range(n_clusters):
    fig.add_trace(go.Scatter(
        x=[centroids_2d[i, 0]],
        y=[centroids_2d[i, 1]],
        mode="markers+text",
        marker=dict(size=20, color=CLUSTER_COLORS[i % len(CLUSTER_COLORS)], symbol="star"),
        text=[f"군집 {i+1}"],
        textposition="top center",
        name=f"군집 {i+1} 중심",
    ))
fig.update_layout(
    title=f"군집 중심점 (PCA 2D, k={n_clusters})",
    height=400,
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=0, r=0, t=40, b=0),
    xaxis=dict(showgrid=True, gridcolor="#F0F4F8", title="PC1"),
    yaxis=dict(showgrid=True, gridcolor="#F0F4F8", title="PC2"),
)
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

st.markdown("<br>", unsafe_allow_html=True)

# ── 군집 중심 지도 (Folium)
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

    m = folium.Map(
        location=[float(center_lats.mean()), float(center_lons.mean())],
        zoom_start=6,
        tiles="CartoDB positron",
    )
    for i in range(n_clusters):
        color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
        folium.CircleMarker(
            location=[float(center_lats[i]), float(center_lons[i])],
            radius=18,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.8,
            popup=folium.Popup(f"<b>군집 {i+1}</b>", max_width=120),
            tooltip=f"군집 {i+1}",
        ).add_to(m)
        folium.Marker(
            location=[float(center_lats[i]), float(center_lons[i])],
            icon=folium.DivIcon(
                html=f'<div style="font-size:12px; font-weight:700; color:{color}; white-space:nowrap; margin-top:-28px; margin-left:22px;">군집 {i+1}</div>',
                icon_size=(80, 20),
            ),
        ).add_to(m)

    st_folium(m, width="100%", height=480, returned_objects=[])
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

st.dataframe(summary_df.style.format({
    c: "{:,.2f}" for c in summary_df.select_dtypes("float").columns
}), use_container_width=True)

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
