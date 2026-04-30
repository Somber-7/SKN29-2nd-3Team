"""
models/clustering/location_cluster_models.py — 아파트 지역/입지 군집화 모델

아파트 실거래가 데이터를 기반으로 입지 특성이 비슷한 거래/단지를 군집화합니다.
BaseClusterModel 상속 구조에 맞춰 K-Means, DBSCAN 모델을 제공합니다.


주요 기능
---------
- 위도/경도, 평당가, 건물연식, 인근학교수, 인근역수, 세대수 기반 군집화
- 원본 DataFrame 편의 학습: fit_from_dataframe(df)
- 군집 라벨이 붙은 DataFrame 반환: add_cluster_labels(df)
- 군집별 요약 통계 반환: summarize_clusters(df)
- K-Means 최적 k 탐색 보조: find_best_kmeans_k(df)

사용 예시
---------
import pandas as pd
from models.clustering.location_cluster_models import KMeansLocationClusterModel

df = pd.read_csv("Apart Deal_6.csv", encoding="cp949")

model = KMeansLocationClusterModel(n_clusters=5, sample_size=100_000)
model.fit_from_dataframe(df)

labeled_df = model.add_cluster_labels(df)
summary_df = model.summarize_clusters(df)
print(summary_df)
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Optional, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from models.base import BaseClusterModel


# -----------------------------------------------------------------------------
# 공통 지표 함수
# -----------------------------------------------------------------------------

def _clustering_metrics(X: pd.DataFrame | np.ndarray, labels: np.ndarray) -> dict[str, float | int]:
    """군집화 평가 지표를 계산합니다.

    DBSCAN의 noise 라벨(-1)은 평가에서 제외합니다.
    군집 수가 2개 미만이면 Silhouette 등 지표를 계산할 수 없으므로 NaN을 반환합니다.
    """
    X_arr = np.asarray(X)
    labels_arr = np.asarray(labels)

    valid_mask = labels_arr != -1
    X_valid = X_arr[valid_mask]
    labels_valid = labels_arr[valid_mask]

    n_samples = int(len(labels_arr))
    n_noise = int(np.sum(labels_arr == -1))
    n_clusters = int(len(set(labels_valid)))

    metrics: dict[str, float | int] = {
        "n_samples": n_samples,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "noise_ratio": float(n_noise / n_samples) if n_samples > 0 else np.nan,
        "Silhouette": np.nan,
        "Davies-Bouldin": np.nan,
        "Calinski-Harabasz": np.nan,
    }

    if n_clusters < 2 or len(X_valid) <= n_clusters:
        return metrics

    metrics["Silhouette"] = float(silhouette_score(X_valid, labels_valid))
    metrics["Davies-Bouldin"] = float(davies_bouldin_score(X_valid, labels_valid))
    metrics["Calinski-Harabasz"] = float(calinski_harabasz_score(X_valid, labels_valid))
    return metrics


# -----------------------------------------------------------------------------
# 공통 베이스 군집화 클래스
# -----------------------------------------------------------------------------

class ApartmentLocationClusterBase(BaseClusterModel):
    """아파트 지역/입지 군집화 모델의 공통 베이스 클래스.

    하위 클래스는 `_create_clusterer()`만 구현하면 됩니다.
    """

    def __init__(
        self,
        name: str,
        current_year: int = 2026,
        sample_size: Optional[int] = 100_000,
        random_state: int = 42,
        feature_cols: Optional[list[str]] = None,
        cluster_params: Optional[dict] = None,
    ):
        super().__init__(name=name)
        self.current_year = current_year
        self.sample_size = sample_size
        self.random_state = random_state
        self.cluster_params = cluster_params or {}
        self.feature_cols = feature_cols or [
            "위도",
            "경도",
            "평당가",
            "건물연식",
            "인근학교수",
            "인근역수",
            "세대수",
        ]
        self.fitted_feature_index_: Optional[pd.Index] = None
        self.metrics_: Optional[dict[str, float | int]] = None

    @property
    def is_fitted(self) -> bool:
        """군집화 학습 완료 여부를 반환합니다."""
        return self._labels is not None

    def _check_fitted(self) -> None:
        """fit() 호출 전 predict/evaluate 사용 시 RuntimeError를 발생시킵니다."""
        if self._labels is None or self._model is None:
            raise RuntimeError(f"[{self.name}] 모델이 아직 학습되지 않았습니다. fit()을 먼저 호출하세요.")

    @staticmethod
    def _clean_numeric(series: pd.Series) -> pd.Series:
        """쉼표가 포함된 문자열 숫자를 안전하게 숫자형으로 변환합니다."""
        return pd.to_numeric(
            series.astype(str).str.strip().str.replace(",", "", regex=False),
            errors="coerce",
        )

    @staticmethod
    def _parse_date(series: pd.Series) -> pd.Series:
        """거래일 컬럼을 날짜형으로 변환합니다."""
        return pd.to_datetime(series, errors="coerce")

    def _validate_columns(self, df: pd.DataFrame, required_cols: Iterable[str]) -> None:
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"필수 컬럼이 없습니다: {missing}")

    def prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """원본 DataFrame을 군집화용 DataFrame으로 변환합니다.

        생성 피처
        ---------
        - 평당가 = 거래금액 / (전용면적 / 3.3)
        - 건물연식 = current_year - 건축년도
        """
        required = ["위도", "경도", "거래금액", "전용면적", "건축년도"]
        self._validate_columns(df, required)

        use_cols = list(dict.fromkeys(required + ["인근학교수", "인근역수", "세대수", "시군구", "법정동", "아파트", "지역코드"]))
        use_cols = [col for col in use_cols if col in df.columns]
        data = df[use_cols].copy()

        for col in ["위도", "경도", "거래금액", "전용면적", "건축년도", "인근학교수", "인근역수", "세대수"]:
            if col not in data.columns:
                data[col] = 0
            data[col] = self._clean_numeric(data[col])

        data["평당가"] = data["거래금액"] / (data["전용면적"] / 3.3)
        data["건물연식"] = self.current_year - data["건축년도"]

        # 비정상 값 제거
        data = data.replace([np.inf, -np.inf], np.nan)
        data = data.dropna(subset=self.feature_cols)
        data = data[data["전용면적"] > 0]
        data = data[data["거래금액"] > 0]

        return data

    def prepare_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """피처 DataFrame을 군집화 입력 형태로 정리합니다.

        이미 `평당가`, `건물연식`이 있으면 그대로 사용하고,
        없으면 원본 컬럼인 `거래금액`, `전용면적`, `건축년도`에서 생성합니다.
        """
        X = X.copy()

        if "평당가" not in X.columns:
            self._validate_columns(X, ["거래금액", "전용면적"])
            X["거래금액"] = self._clean_numeric(X["거래금액"])
            X["전용면적"] = self._clean_numeric(X["전용면적"])
            X["평당가"] = X["거래금액"] / (X["전용면적"] / 3.3)

        if "건물연식" not in X.columns:
            self._validate_columns(X, ["건축년도"])
            X["건축년도"] = self._clean_numeric(X["건축년도"])
            X["건물연식"] = self.current_year - X["건축년도"]

        for col in ["위도", "경도", "인근학교수", "인근역수", "세대수"]:
            if col not in X.columns:
                X[col] = 0

        self._validate_columns(X, self.feature_cols)

        for col in self.feature_cols:
            X[col] = self._clean_numeric(X[col])

        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.dropna(subset=self.feature_cols)
        return X[self.feature_cols]

    def _sample_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """대용량 데이터에서 군집화 계산량을 줄이기 위해 표본을 추출합니다."""
        if self.sample_size is not None and len(X) > self.sample_size:
            return X.sample(n=self.sample_size, random_state=self.random_state)
        return X

    @abstractmethod
    def _create_clusterer(self):
        """하위 클래스에서 실제 군집화 모델 객체를 생성합니다."""

    def _build_pipeline(self) -> Pipeline:
        """스케일링 + 군집화 파이프라인을 생성합니다."""
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("cluster", self._create_clusterer()),
            ]
        )

    def fit(self, X: pd.DataFrame, **kwargs) -> "ApartmentLocationClusterBase":
        """BaseClusterModel 표준 인터페이스에 맞춰 군집화 모델을 학습합니다."""
        X_prepared = self.prepare_features(X)
        X_fit = self._sample_features(X_prepared)

        old_params = self.cluster_params
        self.cluster_params = {**self.cluster_params, **kwargs}
        self._model = self._build_pipeline()
        self.cluster_params = old_params

        labels = self._model.fit_predict(X_fit)
        self._labels = np.asarray(labels)
        self.fitted_feature_index_ = X_fit.index
        self.metrics_ = self.evaluate(X_fit)
        return self

    def fit_from_dataframe(self, df: pd.DataFrame) -> "ApartmentLocationClusterBase":
        """원본 DataFrame을 받아 군집화 모델을 학습합니다."""
        data = self.prepare_dataframe(df)
        return self.fit(data[self.feature_cols])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """새 데이터에 대한 군집 라벨을 예측합니다.

        K-Means처럼 predict()를 지원하는 모델에서만 동작합니다.
        DBSCAN은 새 데이터 예측을 지원하지 않으므로 하위 클래스에서 NotImplementedError를 발생시킵니다.
        """
        self._check_fitted()
        X_prepared = self.prepare_features(X)
        if not hasattr(self._model, "predict"):
            raise NotImplementedError(f"[{self.name}] 모델은 새 데이터 predict()를 지원하지 않습니다.")
        return self._model.predict(X_prepared)

    def evaluate(self, X: pd.DataFrame) -> dict[str, float | int]:
        """군집화 결과의 품질을 평가합니다."""
        if self._labels is None:
            raise RuntimeError(f"[{self.name}] 모델이 아직 학습되지 않았습니다. fit()을 먼저 호출하세요.")

        X_prepared = self.prepare_features(X)

        # evaluate에는 fit에 사용된 표본 X가 들어오는 것을 권장합니다.
        # 행 수가 다르면 현재 모델로 라벨을 다시 얻을 수 있는 경우에만 재계산합니다.
        if len(X_prepared) == len(self._labels):
            labels = self._labels
        elif hasattr(self._model, "predict"):
            labels = self._model.predict(X_prepared)
        else:
            raise ValueError("DBSCAN처럼 predict가 없는 모델은 fit에 사용한 X로만 evaluate할 수 있습니다.")

        scaled_X = self._model.named_steps["scaler"].transform(X_prepared)
        return _clustering_metrics(scaled_X, labels)

    def add_cluster_labels(
        self,
        df: pd.DataFrame,
        label_col: str = "cluster",
        fitted_only_for_no_predict: bool = True,
    ) -> pd.DataFrame:
        """원본 DataFrame에 군집 라벨 컬럼을 추가해 반환합니다.

        K-Means는 전체 df에 대해 새로 라벨을 예측합니다.
        DBSCAN처럼 predict가 없는 모델은 기본적으로 fit에 사용된 표본 행에만 라벨을 붙입니다.
        """
        self._check_fitted()
        result = df.copy()

        if hasattr(self._model, "predict"):
            prepared = self.prepare_dataframe(result)
            labels = self.predict(prepared[self.feature_cols])
            result.loc[prepared.index, label_col] = labels
            return result

        if fitted_only_for_no_predict:
            result[label_col] = np.nan
            if self.fitted_feature_index_ is None:
                return result
            result.loc[self.fitted_feature_index_, label_col] = self._labels
            return result

        raise NotImplementedError(f"[{self.name}] 모델은 전체 데이터에 새 라벨을 예측할 수 없습니다.")

    def summarize_clusters(self, df: pd.DataFrame, label_col: str = "cluster") -> pd.DataFrame:
        """군집별 요약 통계를 반환합니다."""
        labeled = self.add_cluster_labels(df, label_col=label_col)
        data = self.prepare_dataframe(labeled)
        data[label_col] = labeled.loc[data.index, label_col]
        data = data.dropna(subset=[label_col])
        data[label_col] = data[label_col].astype(int)

        agg_dict = {
            "거래금액": ["count", "mean", "median"],
            "평당가": ["mean", "median"],
            "전용면적": "mean",
            "건물연식": "mean",
            "인근학교수": "mean",
            "인근역수": "mean",
            "세대수": "mean",
            "위도": "mean",
            "경도": "mean",
        }

        summary = data.groupby(label_col).agg(agg_dict)
        summary.columns = ["_".join(col).strip("_") for col in summary.columns]
        summary = summary.rename(
            columns={
                "거래금액_count": "거래건수",
                "거래금액_mean": "평균거래금액",
                "거래금액_median": "중앙거래금액",
                "평당가_mean": "평균평당가",
                "평당가_median": "중앙평당가",
                "전용면적_mean": "평균전용면적",
                "건물연식_mean": "평균건물연식",
                "인근학교수_mean": "평균인근학교수",
                "인근역수_mean": "평균인근역수",
                "세대수_mean": "평균세대수",
                "위도_mean": "중심위도",
                "경도_mean": "중심경도",
            }
        )
        return summary.reset_index().sort_values("거래건수", ascending=False)

    def save(self, path: str) -> None:
        """학습된 군집화 모델 객체 전체를 저장합니다."""
        self._check_fitted()
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str):
        """저장된 모델 객체를 불러옵니다."""
        loaded = joblib.load(path)
        if not isinstance(loaded, cls):
            raise TypeError(f"불러온 객체가 {cls.__name__}이 아닙니다.")
        return loaded


# -----------------------------------------------------------------------------
# 개별 군집화 모델 클래스
# -----------------------------------------------------------------------------

class KMeansLocationClusterModel(ApartmentLocationClusterBase):
    """K-Means 기반 지역/입지 군집화 모델.

    새 데이터에 대한 predict()가 가능하므로 Streamlit 지도에서 전체 데이터 라벨링에 쓰기 좋습니다.
    """

    def __init__(self, n_clusters: int = 5, **kwargs):
        cluster_params = kwargs.pop("cluster_params", {}) or {}
        default_params = {
            "n_clusters": n_clusters,
            "n_init": 10,
            "random_state": kwargs.get("random_state", 42),
        }
        default_params.update(cluster_params)
        super().__init__(name="KMeansLocationCluster", cluster_params=default_params, **kwargs)

    def _create_clusterer(self):
        return KMeans(**self.cluster_params)

    @property
    def inertia_(self) -> float | None:
        """K-Means inertia 값을 반환합니다."""
        if self._model is None:
            return None
        clusterer = self._model.named_steps.get("cluster")
        return float(clusterer.inertia_) if hasattr(clusterer, "inertia_") else None


class DBSCANLocationClusterModel(ApartmentLocationClusterBase):
    """DBSCAN 기반 지역/입지 군집화 모델.

    밀도 기반 군집화라 이상 지역/노이즈 탐지에 유용합니다.
    단, sklearn DBSCAN은 새 데이터 predict()를 지원하지 않습니다.
    """

    def __init__(self, eps: float = 0.8, min_samples: int = 10, **kwargs):
        cluster_params = kwargs.pop("cluster_params", {}) or {}
        default_params = {
            "eps": eps,
            "min_samples": min_samples,
            "n_jobs": -1,
        }
        default_params.update(cluster_params)
        super().__init__(name="DBSCANLocationCluster", cluster_params=default_params, **kwargs)

    def _create_clusterer(self):
        return DBSCAN(**self.cluster_params)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError("DBSCAN은 sklearn 기본 구현에서 새 데이터 predict()를 지원하지 않습니다.")


class AgglomerativeLocationClusterModel(ApartmentLocationClusterBase):
    """계층적 군집화 기반 지역/입지 군집화 모델.

    데이터가 매우 크면 느릴 수 있으므로 sample_size를 작게 두는 것을 권장합니다.
    sklearn AgglomerativeClustering은 새 데이터 predict()를 지원하지 않습니다.
    """

    def __init__(self, n_clusters: int = 5, linkage: str = "ward", **kwargs):
        cluster_params = kwargs.pop("cluster_params", {}) or {}
        default_params = {
            "n_clusters": n_clusters,
            "linkage": linkage,
        }
        default_params.update(cluster_params)
        super().__init__(name="AgglomerativeLocationCluster", cluster_params=default_params, **kwargs)

    def _create_clusterer(self):
        return AgglomerativeClustering(**self.cluster_params)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError("AgglomerativeClustering은 sklearn 기본 구현에서 새 데이터 predict()를 지원하지 않습니다.")


# -----------------------------------------------------------------------------
# 모델 선택 보조 함수
# -----------------------------------------------------------------------------

def find_best_kmeans_k(
    df: pd.DataFrame,
    k_values: Iterable[int] = range(2, 11),
    sample_size: Optional[int] = 100_000,
    random_state: int = 42,
) -> pd.DataFrame:
    """여러 k 값에 대해 K-Means 군집화 지표를 비교합니다.

    Silhouette은 높을수록 좋고, Davies-Bouldin은 낮을수록 좋습니다.
    """
    rows: list[dict] = []

    for k in k_values:
        model = KMeansLocationClusterModel(
            n_clusters=int(k),
            sample_size=sample_size,
            random_state=random_state,
        )
        try:
            model.fit_from_dataframe(df)
            row = {
                "k": int(k),
                "inertia": model.inertia_,
                **(model.metrics_ or {}),
                "error": None,
            }
        except Exception as exc:
            row = {
                "k": int(k),
                "inertia": np.nan,
                "n_samples": 0,
                "n_clusters": 0,
                "n_noise": 0,
                "noise_ratio": np.nan,
                "Silhouette": np.nan,
                "Davies-Bouldin": np.nan,
                "Calinski-Harabasz": np.nan,
                "error": str(exc),
            }
        rows.append(row)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    # 프로젝트 루트에서 실행하는 예시:
    # python -m models.clustering.location_cluster_models
    df = pd.read_csv("Apart Deal_6.csv", encoding="cp949")

    model = KMeansLocationClusterModel(n_clusters=5, sample_size=100_000)
    model.fit_from_dataframe(df)

    print(model.metrics_)
    print(model.summarize_clusters(df).head())
