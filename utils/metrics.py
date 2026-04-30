"""
utils/metrics.py — 모델 평가지표 계산 유틸리티

회귀 / 분류 / 군집화 세 영역의 평가지표를 딕셔너리로 반환합니다.
반환된 딕셔너리는 Streamlit의 st.metric() 또는 st.dataframe()으로 바로 출력할 수 있습니다.

사용 예시:
    from utils.metrics import regression_metrics, classification_metrics, clustering_metrics

    # 회귀
    metrics = regression_metrics(y_test, y_pred)

    # 분류 (확률값 있을 때 ROC-AUC 자동 계산)
    metrics = classification_metrics(y_test, y_pred, y_prob=model.predict_proba(X_test))

    # 군집화
    metrics = clustering_metrics(X, cluster_labels)
"""

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)


# ── 회귀 ──────────────────────────────────────────────

def regression_metrics(y_true, y_pred) -> dict:
    """회귀 모델의 주요 평가지표를 계산합니다.

    거래금액처럼 절대 스케일이 큰 경우 MAE/RMSE 단독보다 MAPE를 함께 보는 것을 권장합니다.

    Args:
        y_true: 실제 정답 값 (array-like)
        y_pred: 모델 예측 값 (array-like)

    Returns:
        {"MAE": float, "MSE": float, "RMSE": float, "MAPE": float, "R2": float}
        - MAE  : 평균 절대 오차 (낮을수록 좋음)
        - MSE  : 평균 제곱 오차 (낮을수록 좋음)
        - RMSE : 평균 제곱근 오차 (낮을수록 좋음, MAE보다 이상치에 민감)
        - MAPE : 평균 절대 퍼센트 오차 (낮을수록 좋음, 단위 무관 비교 가능)
        - R2   : 결정계수 (1에 가까울수록 좋음, 음수도 가능)
    """
    return {
        "MAE":  mean_absolute_error(y_true, y_pred),
        "MSE":  mean_squared_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAPE": mape(y_true, y_pred),
        "R2":   r2_score(y_true, y_pred),
    }


def mape(y_true, y_pred) -> float:
    """평균 절대 퍼센트 오차(MAPE)를 계산합니다.

    거래금액처럼 절대값이 크고 범위가 넓은 타깃에서 MAE보다 직관적인 지표입니다.
    예: MAPE=0.05 → 평균 5% 오차.
    y_true에 0이 포함된 경우 해당 행은 자동으로 제외합니다.

    Args:
        y_true: 실제 정답 값 (array-like, 0이 있으면 해당 행 제외)
        y_pred: 모델 예측 값 (array-like)

    Returns:
        MAPE 값 (float). 0에 가까울수록 좋음.
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))


# ── 분류 ──────────────────────────────────────────────

def classification_metrics(y_true, y_pred, y_prob=None, average="weighted") -> dict:
    """분류 모델의 주요 평가지표를 계산합니다.

    Args:
        y_true:  실제 정답 레이블 (array-like)
        y_pred:  모델 예측 레이블 (array-like)
        y_prob:  클래스별 예측 확률 (array-like, shape: [n_samples, n_classes]).
                 None이면 ROC-AUC를 계산하지 않습니다.
        average: 다중 클래스 평균 방식. "weighted" | "macro" | "micro"
                 - weighted : 클래스 불균형 고려 (기본값, 권장)
                 - macro    : 클래스별 단순 평균
                 - micro    : 전체 TP/FP/FN 합산 후 계산

    Returns:
        {"Accuracy": float, "Precision": float, "Recall": float, "F1": float}
        y_prob 제공 시 "ROC-AUC" 키 추가.
        모든 값은 0~1 범위 (높을수록 좋음).
    """
    metrics = {
        "Accuracy":  accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "Recall":    recall_score(y_true, y_pred, average=average, zero_division=0),
        "F1":        f1_score(y_true, y_pred, average=average, zero_division=0),
    }
    if y_prob is not None:
        try:
            # 이진 분류: 양성 클래스(index 1)의 확률만 사용
            if y_prob.ndim == 2 and y_prob.shape[1] == 2:
                metrics["ROC-AUC"] = roc_auc_score(y_true, y_prob[:, 1])
            else:
                # 다중 분류: OvR(One-vs-Rest) 방식
                metrics["ROC-AUC"] = roc_auc_score(
                    y_true, y_prob, multi_class="ovr", average=average
                )
        except ValueError:
            # 클래스 수 불일치 등으로 계산 불가한 경우 무시
            pass
    return metrics


def get_confusion_matrix(y_true, y_pred) -> np.ndarray:
    """혼동 행렬(Confusion Matrix)을 numpy 배열로 반환합니다.

    Args:
        y_true: 실제 정답 레이블
        y_pred: 모델 예측 레이블

    Returns:
        shape (n_classes, n_classes)의 2D 배열.
        행(row)은 실제값, 열(col)은 예측값.
        visualizer.plot_confusion_matrix()에 바로 전달할 수 있습니다.
    """
    return confusion_matrix(y_true, y_pred)


# ── 군집화 ────────────────────────────────────────────

def clustering_metrics(X, labels) -> dict:
    """군집화 모델의 내부 평가지표를 계산합니다. (정답 레이블 불필요)

    Args:
        X:      원본 피처 데이터 (array-like, shape: [n_samples, n_features])
        labels: 군집 레이블 배열. DBSCAN의 노이즈 포인트(-1)는 자동 제외됩니다.

    Returns:
        {"Silhouette": float, "Davies-Bouldin": float, "Calinski-Harabasz": float}
        - Silhouette        : -1~1, 1에 가까울수록 군집이 잘 분리됨
        - Davies-Bouldin    : 낮을수록 군집 내 응집도/분리도가 좋음
        - Calinski-Harabasz : 높을수록 군집이 밀집되고 잘 분리됨
        군집 수가 2개 미만이면 {"error": str} 반환.
    """
    metrics = {}
    # DBSCAN의 노이즈(-1)를 제외한 실제 군집 수 계산
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    if n_clusters < 2:
        return {"error": "군집 수가 2개 미만이어서 평가 불가"}

    metrics["Silhouette"]        = silhouette_score(X, labels)
    metrics["Davies-Bouldin"]    = davies_bouldin_score(X, labels)
    metrics["Calinski-Harabasz"] = calinski_harabasz_score(X, labels)
    return metrics
