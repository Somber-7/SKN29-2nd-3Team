"""
models/clustering/location_cluster_models.py — 아파트 지역/입지 군집화 모델

아파트 실거래가 데이터를 기반으로 입지 특성이 비슷한 거래/단지를 군집화합니다.
500만 건 전체 데이터 기준으로 동작합니다.

sklearn MiniBatchKMeans 사용
----------------------------
전체 데이터를 한 번에 처리하는 대신 미니배치 방식으로 학습합니다.
메모리 효율적이고 속도가 빠릅니다. (수 분 이내)

사용 예시
---------
    from utils.db import load_apart_deals
    from models.clustering.location_cluster_models import KMeansLocationClusterModel

    df = load_apart_deals()
    model = KMeansLocationClusterModel(n_clusters=5)
    model.fit_from_dataframe(df)

    summary = model.summarize_clusters(df)
    print(summary)
"""

from __future__ import annotations

from typing import Optional, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

from models.base import BaseClusterModel


# ── 평가 지표 ───────────────────────────────────────────────────

def _clustering_metrics(X: np.ndarray, labels: np.ndarray) -> dict:
    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

    n_clusters = int(len(set(labels)))
    metrics = {
        "n_samples":         int(len(labels)),
        "n_clusters":        n_clusters,
        "Silhouette":        np.nan,
        "Davies-Bouldin":    np.nan,
        "Calinski-Harabasz": np.nan,
    }
    if n_clusters >= 2 and len(X) > n_clusters:
        sample_size = min(50_000, len(X))
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X), sample_size, replace=False)
        metrics["Silhouette"]        = float(silhouette_score(X[idx], labels[idx]))
        metrics["Davies-Bouldin"]    = float(davies_bouldin_score(X[idx], labels[idx]))
        metrics["Calinski-Harabasz"] = float(calinski_harabasz_score(X[idx], labels[idx]))
    return metrics


# ── 모델 ───────────────────────────────────────────────────────

class KMeansLocationClusterModel(BaseClusterModel):
    """sklearn MiniBatchKMeans 기반 아파트 지역/입지 군집화 모델.

    전체 500만 건을 미니배치로 처리하여 메모리 효율적으로 학습합니다.

    파생 피처:
        평당가   = 거래금액 / (전용면적 / 3.3)
        건물연식 = current_year - 건축년도
    """

    def __init__(
        self,
        n_clusters: int = 5,
        max_iter: int = 100,
        batch_size: int = 10_000,
        current_year: int = 2026,
        random_state: int = 42,
        feature_cols: Optional[list[str]] = None,
    ):
        super().__init__(name="KMeansLocationCluster(MiniBatch)")
        self.n_clusters   = n_clusters
        self.max_iter     = max_iter
        self.batch_size   = batch_size
        self.current_year = current_year
        self.random_state = random_state
        self.feature_cols = feature_cols or [
            "위도", "경도", "평당가", "건물연식",
            "인근학교수", "인근역수", "세대수",
        ]
        self._scaler: Optional[StandardScaler] = None
        self.fitted_feature_index_: Optional[pd.Index] = None
        self.metrics_: Optional[dict] = None

    @staticmethod
    def _clean_numeric(series: pd.Series) -> pd.Series:
        return pd.to_numeric(
            series.astype(str).str.strip().str.replace(",", "", regex=False),
            errors="coerce",
        )

    def _validate_columns(self, df: pd.DataFrame, required: Iterable[str]) -> None:
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"필수 컬럼 없음: {missing}")

    def prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """원본 DataFrame을 군집화용 피처 DataFrame으로 변환합니다."""
        required = ["위도", "경도", "거래금액", "전용면적", "건축년도"]
        self._validate_columns(df, required)

        extra = ["인근학교수", "인근역수", "세대수", "시군구", "법정동", "아파트", "지역코드"]
        use_cols = list(dict.fromkeys(required + [c for c in extra if c in df.columns]))
        data = df[use_cols].copy()

        for col in ["위도", "경도", "거래금액", "전용면적", "건축년도",
                    "인근학교수", "인근역수", "세대수"]:
            if col not in data.columns:
                data[col] = 0
            data[col] = self._clean_numeric(data[col])

        data["평당가"]   = data["거래금액"] / (data["전용면적"] / 3.3)
        data["건물연식"] = self.current_year - data["건축년도"]

        data = data.replace([np.inf, -np.inf], np.nan)
        data = data.dropna(subset=self.feature_cols)
        data = data[(data["전용면적"] > 0) & (data["거래금액"] > 0)]

        return data

    def _prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        self._validate_columns(data, self.feature_cols)
        X = data[self.feature_cols].copy()
        for col in self.feature_cols:
            X[col] = self._clean_numeric(X[col])
        return X.replace([np.inf, -np.inf], np.nan).dropna(subset=self.feature_cols)

    def fit(self, X: pd.DataFrame) -> "KMeansLocationClusterModel":
        """피처 DataFrame으로 MiniBatchKMeans를 학습합니다."""
        X_feat = self._prepare_features(X)

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X_feat)

        print(f"[{self.name}] 학습 시작 — n={len(X_scaled):,}, k={self.n_clusters}, batch={self.batch_size:,}")

        self._model = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            max_iter=self.max_iter,
            batch_size=self.batch_size,
            random_state=self.random_state,
            n_init=3,
            verbose=0,
        )
        self._model.fit(X_scaled)

        self._labels = self._model.labels_
        self.fitted_feature_index_ = X_feat.index
        self.metrics_ = _clustering_metrics(X_scaled, self._labels)
        return self

    def fit_from_dataframe(self, df: pd.DataFrame) -> "KMeansLocationClusterModel":
        """원본 DataFrame을 받아 군집화 모델을 학습합니다."""
        data = self.prepare_dataframe(df)
        return self.fit(data[self.feature_cols])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """학습된 모델로 새 데이터의 군집 레이블을 반환합니다."""
        if self._labels is None:
            raise RuntimeError(f"[{self.name}] fit()을 먼저 호출하세요.")

        X_feat   = self._prepare_features(X)
        X_scaled = self._scaler.transform(X_feat)
        return self._model.predict(X_scaled)

    def evaluate(self, X: pd.DataFrame) -> dict:
        if self._labels is None:
            raise RuntimeError(f"[{self.name}] fit()을 먼저 호출하세요.")
        X_feat   = self._prepare_features(X)
        X_scaled = self._scaler.transform(X_feat)
        if len(X_scaled) == len(self._labels):
            labels = self._labels
        else:
            labels = self.predict(X)
        return _clustering_metrics(X_scaled, labels)

    def add_cluster_labels(self, df: pd.DataFrame, label_col: str = "cluster") -> pd.DataFrame:
        """원본 DataFrame에 군집 라벨 컬럼을 추가해 반환합니다."""
        if self._labels is None:
            raise RuntimeError(f"[{self.name}] fit()을 먼저 호출하세요.")
        result   = df.copy()
        prepared = self.prepare_dataframe(result)
        labels   = self.predict(prepared[self.feature_cols])
        result.loc[prepared.index, label_col] = labels
        return result

    def summarize_clusters(self, df: pd.DataFrame, label_col: str = "cluster") -> pd.DataFrame:
        """군집별 요약 통계를 반환합니다."""
        labeled = self.add_cluster_labels(df, label_col=label_col)
        data    = self.prepare_dataframe(labeled)
        data[label_col] = labeled.loc[data.index, label_col]
        data = data.dropna(subset=[label_col])
        data[label_col] = data[label_col].astype(int)

        summary = data.groupby(label_col).agg(
            거래건수      =("거래금액",   "count"),
            평균거래금액  =("거래금액",   "mean"),
            중앙거래금액  =("거래금액",   "median"),
            평균평당가    =("평당가",     "mean"),
            중앙평당가    =("평당가",     "median"),
            평균전용면적  =("전용면적",   "mean"),
            평균건물연식  =("건물연식",   "mean"),
            평균인근학교수=("인근학교수", "mean"),
            평균인근역수  =("인근역수",   "mean"),
            평균세대수    =("세대수",     "mean"),
            중심위도      =("위도",       "mean"),
            중심경도      =("경도",       "mean"),
        )
        return summary.reset_index().sort_values("거래건수", ascending=False)

    def get_params(self) -> dict:
        return {
            "n_clusters":   self.n_clusters,
            "max_iter":     self.max_iter,
            "batch_size":   self.batch_size,
            "feature_cols": self.feature_cols,
        }

    def save(self, path: str) -> None:
        if self._labels is None:
            raise RuntimeError(f"[{self.name}] fit()을 먼저 호출하세요.")
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "KMeansLocationClusterModel":
        loaded = joblib.load(path)
        if not isinstance(loaded, cls):
            raise TypeError(f"불러온 객체가 {cls.__name__}이 아닙니다.")
        return loaded


# ── 최적 k 탐색 ────────────────────────────────────────────────

def find_best_k(
    df: pd.DataFrame,
    k_values: Iterable[int] = range(2, 11),
    random_state: int = 42,
) -> pd.DataFrame:
    """여러 k 값에 대해 KMeans 지표를 비교합니다. (Elbow 탐색용)

    Silhouette은 높을수록, Davies-Bouldin은 낮을수록 좋습니다.
    """
    rows = []
    for k in k_values:
        model = KMeansLocationClusterModel(n_clusters=int(k), random_state=random_state)
        try:
            model.fit_from_dataframe(df)
            rows.append({"k": int(k), **(model.metrics_ or {})})
        except Exception as exc:
            rows.append({"k": int(k), "error": str(exc)})
    return pd.DataFrame(rows)
