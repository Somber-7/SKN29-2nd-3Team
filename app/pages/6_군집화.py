# 경로: app/pages/6_군집화.py

import streamlit as st
import sys
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.ui import (
    load_css, render_sidebar, page_header,
    section_badge, stat_card, chart_card_open, chart_card_close,
)

st.set_page_config(page_title="군집화", layout="wide")
load_css()
render_sidebar()

page_header("군집화 — 단지 / 지역 군집 분석")

# ── 더미 데이터 생성 (실제 연결 시 DB 로드로 교체)
@st.cache_data
def load_dummy_data(n=500):
    # TODO: 실제 데이터 연결
    # ── 백엔드 연결 포인트 ──────────────────────────────
    # from utils.db import fetch_all
    # rows = fetch_all("""
    #     SELECT 전용면적, 층, 건축연도, 거래금액
    #     FROM apart_deal
    #     WHERE 시도 = %s
    #     LIMIT 5000
    # """, (sido,))
    # df = pd.DataFrame(rows)
    # df = parse_price(df); df = fix_floor(df)
    # ────────────────────────────────────────────────────
    np.random.seed(42)
    df = pd.DataFrame({
        "전용면적": np.random.normal(84,  30, n).clip(20, 200),
        "층":       np.random.normal(12,   6, n).clip(1, 50),
        "건축연도": np.random.normal(2005, 10, n).clip(1980, 2023),
        "거래금액": np.random.normal(60000, 25000, n).clip(10000, 200000),
    })
    return df

df_raw = load_dummy_data()

# ── 전처리 (스케일링)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_raw)

# ── PCA 2D 축소 (시각화용)
pca = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(X_scaled)

MODEL_LABELS = ["🔵 KMeans", "🟡 DBSCAN", "🟢 Agglomerative"]
tabs = st.tabs(MODEL_LABELS)

CLUSTER_COLORS = [
    "#345BCB","#EF4444","#10B981","#F97316",
    "#8B5CF6","#EC4899","#14B8A6","#F59E0B",
]

# ══════════════════════════════
# KMeans
# ══════════════════════════════
with tabs[0]:
    col_left, col_right = st.columns([1, 2])

    with col_left:
        section_badge("⚙️", "KMeans 파라미터")
        n_clusters = st.slider("군집 수 (k)", 2, 10, 4, key="km_k")
        init_method = st.selectbox("초기화 방법", ["k-means++", "random"], key="km_init")
        max_iter = st.slider("최대 반복 횟수", 100, 500, 300, step=50, key="km_iter")
        run_btn = st.button("군집화 실행", type="primary",
                            use_container_width=True, key="run_km")

    with col_right:
        section_badge("📊", "군집 결과 시각화")

        # TODO: 실제 KMeans 모델로 교체
        # from models.clustering.kmeans import KMeansModel
        # model = KMeansModel(n_clusters=n_clusters)
        # model.fit(X_scaled)
        # labels = model._labels
        km = KMeans(n_clusters=n_clusters, init=init_method,
                    max_iter=max_iter, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)

        chart_card_open()
        fig = go.Figure()
        for i in range(n_clusters):
            mask = labels == i
            fig.add_trace(go.Scatter(
                x=X_2d[mask, 0], y=X_2d[mask, 1],
                mode="markers",
                marker=dict(size=5, color=CLUSTER_COLORS[i % len(CLUSTER_COLORS)], opacity=0.6),
                name=f"군집 {i+1}",
            ))
        fig.update_layout(
            title="KMeans 군집 결과 (PCA 2D)",
            height=380, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=40, b=0),
            xaxis=dict(showgrid=True, gridcolor="#F0F4F8", title="PC1"),
            yaxis=dict(showgrid=True, gridcolor="#F0F4F8", title="PC2"),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        chart_card_close()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 성능 지표
    section_badge("📐", "군집 평가 지표")
    # TODO: utils/metrics.py의 clustering_metrics() 연결
    # from utils.metrics import clustering_metrics
    # metrics = clustering_metrics(X_scaled, labels)
    from sklearn.metrics import silhouette_score, davies_bouldin_score
    sil = silhouette_score(X_scaled, labels)
    dbi = davies_bouldin_score(X_scaled, labels)

    mc1, mc2, mc3 = st.columns(3)
    mc1.markdown(stat_card(f"{sil:.3f}", "Silhouette Score", "1에 가까울수록 좋음"), unsafe_allow_html=True)
    mc2.markdown(stat_card(f"{dbi:.3f}", "Davies-Bouldin",   "0에 가까울수록 좋음"), unsafe_allow_html=True)
    mc3.markdown(stat_card(str(n_clusters), "군집 수", ""), unsafe_allow_html=True)

    # ── 군집별 평균 가격
    st.markdown("<br>", unsafe_allow_html=True)
    section_badge("💰", "군집별 평균 거래금액", color="#F97316")
    df_raw["군집"] = [f"군집 {l+1}" for l in labels]
    group_mean = df_raw.groupby("군집")["거래금액"].mean().reset_index()

    chart_card_open()
    fig2 = go.Figure(go.Bar(
        x=group_mean["군집"],
        y=group_mean["거래금액"],
        marker_color=CLUSTER_COLORS[:n_clusters],
        text=group_mean["거래금액"].apply(lambda v: f"{int(v):,}만원"),
        textposition="outside",
    ))
    fig2.update_layout(
        height=280, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=10, b=0),
        yaxis=dict(showgrid=True, gridcolor="#F0F4F8"),
    )
    st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
    chart_card_close()

# ══════════════════════════════
# DBSCAN
# ══════════════════════════════
with tabs[1]:
    col_left, col_right = st.columns([1, 2])

    with col_left:
        section_badge("⚙️", "DBSCAN 파라미터")
        eps = st.slider("eps (반경)", 0.1, 2.0, 0.5, step=0.05, key="db_eps")
        min_samples = st.slider("min_samples", 3, 20, 5, key="db_min")
        st.info("💡 라벨 -1 = 노이즈 포인트")
        run_btn2 = st.button("군집화 실행", type="primary",
                             use_container_width=True, key="run_db")

    with col_right:
        section_badge("📊", "군집 결과 시각화")

        # TODO: models/clustering/dbscan.py 연결
        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels_db = db.fit_predict(X_scaled)
        n_clusters_db = len(set(labels_db)) - (1 if -1 in labels_db else 0)
        noise_cnt = int((labels_db == -1).sum())

        chart_card_open()
        fig = go.Figure()
        unique_labels = sorted(set(labels_db))
        for lbl in unique_labels:
            mask = labels_db == lbl
            name  = f"노이즈 ({noise_cnt}개)" if lbl == -1 else f"군집 {lbl+1}"
            color = "#D1D5DB" if lbl == -1 else CLUSTER_COLORS[lbl % len(CLUSTER_COLORS)]
            fig.add_trace(go.Scatter(
                x=X_2d[mask, 0], y=X_2d[mask, 1],
                mode="markers",
                marker=dict(size=5, color=color, opacity=0.6),
                name=name,
            ))
        fig.update_layout(
            title=f"DBSCAN 결과 (군집 {n_clusters_db}개, 노이즈 {noise_cnt}개)",
            height=380, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=40, b=0),
            xaxis=dict(showgrid=True, gridcolor="#F0F4F8", title="PC1"),
            yaxis=dict(showgrid=True, gridcolor="#F0F4F8", title="PC2"),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        chart_card_close()

    st.markdown("<br>", unsafe_allow_html=True)
    section_badge("📐", "군집 평가 지표")
    mc1, mc2, mc3 = st.columns(3)
    mc1.markdown(stat_card(str(n_clusters_db), "발견된 군집 수", ""), unsafe_allow_html=True)
    mc2.markdown(stat_card(str(noise_cnt),     "노이즈 포인트",  "라벨 -1"), unsafe_allow_html=True)
    if n_clusters_db > 1:
        valid_mask = labels_db != -1
        sil_db = silhouette_score(X_scaled[valid_mask], labels_db[valid_mask])
        mc3.markdown(stat_card(f"{sil_db:.3f}", "Silhouette Score", "노이즈 제외"), unsafe_allow_html=True)
    else:
        mc3.markdown(stat_card("N/A", "Silhouette Score", "군집 2개 이상 필요"), unsafe_allow_html=True)

# ══════════════════════════════
# Agglomerative
# ══════════════════════════════
with tabs[2]:
    col_left, col_right = st.columns([1, 2])

    with col_left:
        section_badge("⚙️", "Agglomerative 파라미터")
        n_clusters_ag = st.slider("군집 수", 2, 10, 4, key="ag_k")
        linkage = st.selectbox("Linkage", ["ward", "complete", "average", "single"], key="ag_link")
        run_btn3 = st.button("군집화 실행", type="primary",
                             use_container_width=True, key="run_ag")

    with col_right:
        section_badge("📊", "군집 결과 시각화")

        # TODO: models/clustering/agglomerative.py 연결
        ag = AgglomerativeClustering(n_clusters=n_clusters_ag, linkage=linkage)
        labels_ag = ag.fit_predict(X_scaled)

        chart_card_open()
        fig = go.Figure()
        for i in range(n_clusters_ag):
            mask = labels_ag == i
            fig.add_trace(go.Scatter(
                x=X_2d[mask, 0], y=X_2d[mask, 1],
                mode="markers",
                marker=dict(size=5, color=CLUSTER_COLORS[i % len(CLUSTER_COLORS)], opacity=0.6),
                name=f"군집 {i+1}",
            ))
        fig.update_layout(
            title=f"Agglomerative 결과 (linkage={linkage})",
            height=380, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=40, b=0),
            xaxis=dict(showgrid=True, gridcolor="#F0F4F8", title="PC1"),
            yaxis=dict(showgrid=True, gridcolor="#F0F4F8", title="PC2"),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        chart_card_close()

    st.markdown("<br>", unsafe_allow_html=True)
    section_badge("📐", "군집 평가 지표")
    sil_ag = silhouette_score(X_scaled, labels_ag)
    dbi_ag = davies_bouldin_score(X_scaled, labels_ag)
    mc1, mc2, mc3 = st.columns(3)
    mc1.markdown(stat_card(f"{sil_ag:.3f}", "Silhouette Score", ""), unsafe_allow_html=True)
    mc2.markdown(stat_card(f"{dbi_ag:.3f}", "Davies-Bouldin",   ""), unsafe_allow_html=True)
    mc3.markdown(stat_card(str(n_clusters_ag), "군집 수", ""), unsafe_allow_html=True)
