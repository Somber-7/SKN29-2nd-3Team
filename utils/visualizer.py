"""
utils/visualizer.py — Plotly 기반 시각화 유틸리티

모든 함수는 plotly.graph_objects.Figure 객체를 반환합니다.
Streamlit에서는 st.plotly_chart(fig) 로 출력하세요.

한글 폰트(Malgun Gothic)를 기본으로 설정합니다.

사용 예시:
    import streamlit as st
    from utils.visualizer import plot_confusion_matrix, plot_feature_importance

    st.plotly_chart(plot_confusion_matrix(y_true, y_pred, labels=["정상", "이상"]))
    st.plotly_chart(plot_feature_importance(feature_names, importances))
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc, confusion_matrix

matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False


# ── 분류 ──────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, labels=None) -> go.Figure:
    """혼동 행렬(Confusion Matrix)을 히트맵으로 시각화합니다.

    Args:
        y_true: 실제 정답 레이블
        y_pred: 모델 예측 레이블
        labels: 축 레이블 리스트 (예: ["정상", "이상"]).
                None이면 숫자 인덱스로 표시됩니다.

    Returns:
        Plotly Figure. 행은 실제값, 열은 예측값, 셀 값은 샘플 수.
    """
    cm = confusion_matrix(y_true, y_pred)
    fig = px.imshow(
        cm,
        text_auto=True,
        color_continuous_scale="Blues",
        x=labels,
        y=labels,
        labels={"x": "예측값", "y": "실제값", "color": "Count"},
        title="혼동 행렬 (Confusion Matrix)",
    )
    return fig


def plot_roc_curve(y_true, y_prob, label="Model") -> go.Figure:
    """ROC 곡선과 AUC 값을 시각화합니다. (이진 분류 전용)

    Args:
        y_true: 실제 이진 레이블 (0 또는 1)
        y_prob: 양성 클래스(1)에 대한 예측 확률 (1D array)
        label:  범례에 표시할 모델 이름

    Returns:
        Plotly Figure. 대각선 점선은 랜덤 분류기(AUC=0.5) 기준선.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"{label} (AUC={roc_auc:.3f})", mode="lines"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Random", mode="lines",
                             line=dict(dash="dash", color="gray")))
    fig.update_layout(
        title="ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
    )
    return fig


# ── 회귀 ──────────────────────────────────────────────

def plot_prediction_vs_actual(y_true, y_pred) -> go.Figure:
    """예측값과 실제값의 산점도를 그립니다.

    점이 빨간 점선(완벽한 예측선)에 가까울수록 모델 성능이 좋습니다.

    Args:
        y_true: 실제 정답 값 (array-like)
        y_pred: 모델 예측 값 (array-like)

    Returns:
        Plotly Figure.
    """
    fig = px.scatter(
        x=y_true, y=y_pred,
        labels={"x": "실제값", "y": "예측값"},
        title="예측값 vs 실제값",
    )
    min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val], y=[min_val, max_val],
        mode="lines", name="완벽한 예측", line=dict(dash="dash", color="red")
    ))
    return fig


def plot_residuals(y_true, y_pred) -> go.Figure:
    """잔차 플롯(Residual Plot)을 그립니다.

    잔차 = 실제값 - 예측값. 점들이 y=0 주변에 무작위로 분포하면 좋은 모델입니다.
    패턴이 보이면 모델이 특정 구간의 오차를 체계적으로 범하고 있다는 신호입니다.

    Args:
        y_true: 실제 정답 값 (array-like)
        y_pred: 모델 예측 값 (array-like)

    Returns:
        Plotly Figure. x축은 예측값, y축은 잔차.
    """
    residuals = np.array(y_true) - np.array(y_pred)
    fig = px.scatter(
        x=y_pred, y=residuals,
        labels={"x": "예측값", "y": "잔차"},
        title="잔차 플롯 (Residual Plot)",
    )
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    return fig


# ── 공통 ──────────────────────────────────────────────

def plot_feature_importance(feature_names: list, importances: np.ndarray, top_n: int = 20) -> go.Figure:
    """피처 중요도를 내림차순 가로 막대 차트로 시각화합니다.

    RandomForest, LightGBM 등 feature_importances_ 속성을 제공하는 모델에 사용합니다.
    models/base.py의 get_feature_importance()와 함께 사용하세요.

    Args:
        feature_names: 피처 이름 리스트 (학습 데이터의 컬럼명)
        importances:   모델의 feature_importances_ 배열
        top_n:         상위 몇 개 피처를 표시할지 (기본 20)

    Returns:
        Plotly Figure. 중요도가 높은 피처가 상단에 위치합니다.
    """
    idx = np.argsort(importances)[::-1][:top_n]
    fig = px.bar(
        x=importances[idx],
        y=[feature_names[i] for i in idx],
        orientation="h",
        labels={"x": "중요도", "y": "피처"},
        title=f"피처 중요도 (Top {top_n})",
    )
    fig.update_layout(yaxis=dict(autorange="reversed"))
    return fig


def plot_learning_curve(train_scores: list, val_scores: list, metric_name: str = "Score") -> go.Figure:
    """에포크별 학습/검증 성능 곡선을 그립니다.

    PyTorch DNN 또는 sklearn 학습 루프의 기록 리스트를 그대로 전달하면 됩니다.
    두 곡선의 격차가 크면 과적합, 둘 다 낮으면 과소적합을 의심할 수 있습니다.

    Args:
        train_scores: 에포크별 학습 점수 리스트 (loss 또는 accuracy)
        val_scores:   에포크별 검증 점수 리스트 (train_scores와 길이 동일)
        metric_name:  y축 레이블 (예: "Loss", "Accuracy", "F1")

    Returns:
        Plotly Figure.
    """
    epochs = list(range(1, len(train_scores) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=train_scores, name="Train", mode="lines+markers"))
    fig.add_trace(go.Scatter(x=epochs, y=val_scores, name="Validation", mode="lines+markers"))
    fig.update_layout(title="학습 곡선", xaxis_title="Epoch", yaxis_title=metric_name)
    return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """수치형 컬럼 간 피어슨 상관계수 히트맵을 그립니다.

    수치형 컬럼만 자동으로 선택되므로 범주형 컬럼이 포함된 DataFrame을 그대로 전달해도 됩니다.
    1에 가까우면 양의 상관, -1에 가까우면 음의 상관, 0이면 상관 없음.

    Args:
        df: 분석할 DataFrame (수치형 컬럼 2개 이상 필요)

    Returns:
        Plotly Figure. 색상은 파란색(음의 상관) ↔ 빨간색(양의 상관).
    """
    corr = df.select_dtypes(include="number").corr()
    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title="상관관계 히트맵",
    )
    return fig


def plot_distribution(series: pd.Series, title: str = "") -> go.Figure:
    """단일 컬럼의 분포를 히스토그램 + 박스플롯으로 시각화합니다.

    데이터의 왜도, 이상치, 분포 형태를 한눈에 파악할 수 있습니다.

    Args:
        series: 시각화할 pandas Series
        title:  차트 제목. 빈 문자열이면 "{컬럼명} 분포"로 자동 설정.

    Returns:
        Plotly Figure. 상단 히스토그램, 하단 박스플롯.
    """
    fig = px.histogram(series, nbins=30, title=title or f"{series.name} 분포", marginal="box")
    return fig


def plot_clusters_2d(X: np.ndarray, labels: np.ndarray, title: str = "군집화 결과") -> go.Figure:
    """2D 공간에서 군집 결과를 색상으로 구분해 산점도로 시각화합니다.

    3차원 이상의 데이터는 PCA로 2차원 축소 후 전달하세요.
    DBSCAN의 노이즈 포인트(label=-1)는 별도 색상으로 표시됩니다.

    Args:
        X:      2D 피처 배열 (shape: [n_samples, 2])
        labels: 군집 레이블 배열 (shape: [n_samples,])
        title:  차트 제목

    Returns:
        Plotly Figure. 군집별로 다른 색상의 점으로 표시됩니다.
    """
    fig = px.scatter(
        x=X[:, 0], y=X[:, 1],
        color=labels.astype(str),
        labels={"x": "Feature 1", "y": "Feature 2", "color": "Cluster"},
        title=title,
    )
    return fig


# ── Apart Deal 전용 ───────────────────────────────────

def plot_price_trend(df: pd.DataFrame,
                     date_col: str = "거래일",
                     price_col: str = "거래금액",
                     freq: str = "M") -> go.Figure:
    """월별 또는 연별 평균 거래금액 추이를 선 그래프로 시각화합니다.

    parse_date() 호출 없이 원본 거래일 컬럼을 직접 사용합니다.

    Args:
        df:        거래 데이터 DataFrame
        date_col:  날짜 컬럼명 (기본값: "거래일")
        price_col: 거래금액 컬럼명 (기본값: "거래금액", 숫자형이어야 함)
        freq:      집계 주기. "M" = 월별, "Q" = 분기별, "Y" = 연별

    Returns:
        Plotly Figure. x축은 기간, y축은 평균 거래금액(만원).
    """
    tmp = df[[date_col, price_col]].copy()
    tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
    tmp[price_col] = pd.to_numeric(tmp[price_col].astype(str).str.replace(",", ""), errors="coerce")
    trend = tmp.set_index(date_col)[price_col].resample(freq).mean().dropna().reset_index()

    freq_label = {"M": "월별", "Q": "분기별", "Y": "연별"}.get(freq, freq)
    fig = px.line(
        trend, x=date_col, y=price_col,
        title=f"{freq_label} 평균 거래금액 추이",
        labels={date_col: "기간", price_col: "평균 거래금액 (만원)"},
        markers=True,
    )
    return fig


def plot_price_map(df: pd.DataFrame,
                   lat_col: str = "위도",
                   lon_col: str = "경도",
                   price_col: str = "거래금액",
                   hover_col: str = "아파트") -> go.Figure:
    """위도/경도 기반으로 거래금액을 지도 위에 산점도로 시각화합니다.

    거래금액이 높을수록 진한 색으로 표시됩니다.
    Plotly의 mapbox를 사용하며 인터넷 연결 없이도 기본 지도가 표시됩니다.

    Args:
        df:        거래 데이터 DataFrame
        lat_col:   위도 컬럼명 (기본값: "위도")
        lon_col:   경도 컬럼명 (기본값: "경도")
        price_col: 색상 기준 컬럼명 (기본값: "거래금액", 숫자형이어야 함)
        hover_col: 마우스 오버 시 표시할 컬럼명 (기본값: "아파트")

    Returns:
        Plotly Figure. 지도 위 산점도.
    """
    tmp = df[[lat_col, lon_col, price_col, hover_col]].copy()
    tmp[price_col] = pd.to_numeric(tmp[price_col].astype(str).str.replace(",", ""), errors="coerce")
    tmp = tmp.dropna(subset=[lat_col, lon_col, price_col])

    fig = px.scatter_mapbox(
        tmp,
        lat=lat_col, lon=lon_col,
        color=price_col,
        hover_name=hover_col,
        color_continuous_scale="Viridis",
        zoom=10,
        title="지역별 거래금액 분포",
        labels={price_col: "거래금액 (만원)"},
    )
    fig.update_layout(mapbox_style="open-street-map")
    return fig


def plot_price_by_brand(df: pd.DataFrame,
                        brand_col: str = "brand_grade",
                        price_col: str = "거래금액") -> go.Figure:
    """브랜드 등급별 거래금액 분포를 박스플롯으로 비교합니다.

    map_brand_grade()로 생성한 brand_grade 컬럼과 함께 사용하세요.

    Args:
        df:        거래 데이터 DataFrame
        brand_col: 브랜드 등급 컬럼명 (기본값: "brand_grade")
        price_col: 거래금액 컬럼명 (기본값: "거래금액", 숫자형이어야 함)

    Returns:
        Plotly Figure. 등급별 중앙값/사분위수/이상치 비교 박스플롯.
    """
    tmp = df[[brand_col, price_col]].copy()
    tmp[price_col] = pd.to_numeric(tmp[price_col].astype(str).str.replace(",", ""), errors="coerce")
    tmp = tmp.dropna()

    order = ["프리미엄", "일반브랜드", "공공(LH)", "기타"]
    existing = [g for g in order if g in tmp[brand_col].unique()]

    fig = px.box(
        tmp, x=brand_col, y=price_col,
        category_orders={brand_col: existing},
        title="브랜드 등급별 거래금액 분포",
        labels={brand_col: "브랜드 등급", price_col: "거래금액 (만원)"},
    )
    return fig
