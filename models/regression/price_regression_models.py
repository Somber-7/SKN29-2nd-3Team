"""
models/regression/price_regression_models.py — 아파트 거래금액 회귀 모델 모음

이 파일은 아파트 실거래가 데이터의 `거래금액`을 예측하기 위한 회귀 모델들을
BaseModel 상속 구조로 구현한 파일입니다.

구현 모델
---------
1. LinearRegressionPriceModel
2. RandomForestPriceModel
3. LightGBMPriceModel
4. XGBoostPriceModel

공통 특징
---------
- BaseModel 인터페이스 준수: fit(X_train, y_train), predict(X), evaluate(X_test, y_test)
- 원본 DataFrame 편의 학습: fit_from_dataframe(df)
- 공통 전처리:
    - 거래금액, 전용면적, 층 등 숫자형 변환
    - 건축년도 -> 건물연식 생성
    - 거래일 -> 거래연도, 거래월 생성
    - 지역코드 원-핫 인코딩
- permutation importance 지원
- 여러 모델 성능 비교 함수 compare_regression_models 제공

사용 예시
---------
import pandas as pd
from models.regression.price_regression_models import (
    LinearRegressionPriceModel,
    RandomForestPriceModel,
    LightGBMPriceModel,
    XGBoostPriceModel,
    compare_regression_models,
)

df = pd.read_csv("Apart Deal_6.csv")

models = [
    LinearRegressionPriceModel(sample_size=100_000),
    RandomForestPriceModel(sample_size=100_000),
    LightGBMPriceModel(sample_size=100_000),
    XGBoostPriceModel(sample_size=100_000),
]

result_df = compare_regression_models(models, df)
print(result_df)
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Optional, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance as sklearn_permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from models.base import BaseModel


# -----------------------------------------------------------------------------
# 지표 함수
# -----------------------------------------------------------------------------

def _regression_metrics(y_true, y_pred) -> dict[str, float]:
    """utils.metrics.regression_metrics가 없어도 동작하도록 둔 회귀 지표 함수입니다."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    nonzero_mask = y_true != 0
    if nonzero_mask.any():
        mape = np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])) * 100
    else:
        mape = np.nan

    return {
        "MAE": float(mae),
        "MSE": float(mse),
        "RMSE": float(rmse),
        "MAPE": float(mape) if not np.isnan(mape) else np.nan,
        "R2": float(r2),
    }


# -----------------------------------------------------------------------------
# 공통 베이스 회귀 클래스
# -----------------------------------------------------------------------------

class ApartmentPriceRegressionBase(BaseModel):
    """아파트 거래금액 회귀 모델들의 공통 베이스 클래스.

    하위 클래스는 `_create_estimator()`만 구현하면 됩니다.
    전처리, 학습, 예측, 평가, 저장/로드, permutation importance는 이 클래스에서 공통 처리합니다.
    """

    def __init__(
        self,
        name: str,
        target_col: str = "거래금액",
        current_year: int = 2026,
        sample_size: Optional[int] = 100_000,
        test_size: float = 0.2,
        random_state: int = 42,
        numeric_cols: Optional[list[str]] = None,
        categorical_cols: Optional[list[str]] = None,
        scale_numeric: bool = False,
        estimator_params: Optional[dict] = None,
    ):
        super().__init__(name=name)
        self.target_col = target_col
        self.current_year = current_year
        self.sample_size = sample_size
        self.test_size = test_size
        self.random_state = random_state
        self.scale_numeric = scale_numeric
        self.estimator_params = estimator_params or {}

        self.numeric_cols = numeric_cols or [
            "전용면적",
            "층",
            "건물연식",
            "기준금리",
            "위도",
            "경도",
            "인근학교수",
            "인근역수",
            "세대수",
            "브랜드여부",
            "거래연도",
            "거래월",
        ]
        self.categorical_cols = categorical_cols or ["지역코드"]

        self.feature_cols_: Optional[list[str]] = None
        self.metrics_: Optional[dict[str, float]] = None

    @property
    def feature_columns(self) -> list[str]:
        """모델에 투입되는 원본 피처 컬럼 목록입니다."""
        return self.numeric_cols + self.categorical_cols

    @property
    def model(self) -> Optional[Pipeline]:
        """기존 코드와 호환되도록 `_model`을 `model` 이름으로도 접근합니다."""
        return self._model

    def _required_input_cols(self) -> list[str]:
        """원본 DataFrame 기준 필요한 입력 컬럼 목록입니다."""
        required = [
            "전용면적",
            "층",
            "건축년도",
            "지역코드",
            "거래일",
        ]

        # 추가 컬럼은 프로젝트에서 직접 만든 컬럼이므로, 없는 경우 0으로 채울 수 있게 별도 처리합니다.
        optional = [
            "기준금리",
            "위도",
            "경도",
            "인근학교수",
            "인근역수",
            "세대수",
            "브랜드여부",
        ]
        return required + optional

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

    def prepare_dataframe(self, df: pd.DataFrame, need_target: bool = True) -> pd.DataFrame:
        """원본 DataFrame을 모델 학습/예측용 DataFrame으로 변환합니다.

        Args:
            df: 원본 데이터프레임
            need_target: True이면 타깃 컬럼까지 정리합니다.

        Returns:
            정리된 데이터프레임
        """
        base_required = ["전용면적", "층", "건축년도", "지역코드", "거래일"]
        if need_target:
            base_required.append(self.target_col)
        self._validate_columns(df, base_required)

        use_cols = [col for col in self._required_input_cols() if col in df.columns]
        if need_target:
            use_cols.append(self.target_col)

        data = df[use_cols].copy()

        # 프로젝트에서 추가한 컬럼이 아직 없는 상황에서도 테스트 가능하도록 기본값을 채웁니다.
        for col in ["기준금리", "위도", "경도", "인근학교수", "인근역수", "세대수", "브랜드여부"]:
            if col not in data.columns:
                data[col] = 0

        numeric_source_cols = [
            "전용면적",
            "층",
            "건축년도",
            "기준금리",
            "위도",
            "경도",
            "인근학교수",
            "인근역수",
            "세대수",
            "브랜드여부",
        ]
        if need_target:
            numeric_source_cols.append(self.target_col)

        for col in numeric_source_cols:
            data[col] = self._clean_numeric(data[col])

        data["지역코드"] = data["지역코드"].astype(str).str.strip()

        거래일 = self._parse_date(data["거래일"])
        data["거래연도"] = 거래일.dt.year
        data["거래월"] = 거래일.dt.month

        data["건물연식"] = self.current_year - data["건축년도"]
        data = data.drop(columns=["건축년도", "거래일"])

        subset = self.feature_columns.copy()
        if need_target:
            subset.append(self.target_col)
        data = data.dropna(subset=subset)

        return data

    def prepare_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """피처 DataFrame을 모델 입력 형태로 정리합니다.

        이미 파생 피처가 있으면 그대로 사용하고, 없으면 원본 컬럼에서 생성합니다.
        """
        X = X.copy()

        if "건물연식" not in X.columns:
            if "건축년도" not in X.columns:
                raise ValueError("`건물연식` 또는 `건축년도` 컬럼이 필요합니다.")
            X["건축년도"] = self._clean_numeric(X["건축년도"])
            X["건물연식"] = self.current_year - X["건축년도"]

        if "거래연도" not in X.columns or "거래월" not in X.columns:
            if "거래일" not in X.columns:
                raise ValueError("`거래연도`/`거래월` 또는 `거래일` 컬럼이 필요합니다.")
            거래일 = self._parse_date(X["거래일"])
            X["거래연도"] = 거래일.dt.year
            X["거래월"] = 거래일.dt.month

        for col in ["기준금리", "위도", "경도", "인근학교수", "인근역수", "세대수", "브랜드여부"]:
            if col not in X.columns:
                X[col] = 0

        self._validate_columns(X, self.feature_columns)

        for col in self.numeric_cols:
            X[col] = self._clean_numeric(X[col])

        for col in self.categorical_cols:
            X[col] = X[col].astype(str).str.strip()

        X = X.dropna(subset=self.feature_columns)
        return X[self.feature_columns]

    def _build_preprocessor(self) -> ColumnTransformer:
        """숫자형/범주형 전처리기를 생성합니다."""
        numeric_transformer = StandardScaler() if self.scale_numeric else "passthrough"

        return ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, self.numeric_cols),
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                    self.categorical_cols,
                ),
            ],
            remainder="drop",
            sparse_threshold=0.3,
            verbose_feature_names_out=False,
        )

    @abstractmethod
    def _create_estimator(self):
        """하위 클래스에서 실제 회귀 모델 객체를 생성합니다."""

    def _build_pipeline(self) -> Pipeline:
        """전처리 + 회귀 모델 파이프라인을 생성합니다."""
        return Pipeline(
            steps=[
                ("preprocessor", self._build_preprocessor()),
                ("model", self._create_estimator()),
            ]
        )

    def fit(self, X_train: pd.DataFrame, y_train, **kwargs) -> "ApartmentPriceRegressionBase":
        """BaseModel 표준 인터페이스에 맞춰 모델을 학습합니다."""
        X_train = self.prepare_features(X_train)
        y_train = self._clean_numeric(pd.Series(y_train, index=X_train.index)).dropna()
        X_train = X_train.loc[y_train.index]

        old_params = self.estimator_params
        self.estimator_params = {**self.estimator_params, **kwargs}
        self._model = self._build_pipeline()
        self.estimator_params = old_params

        self._model.fit(X_train, y_train)
        self._is_trained = True
        self.feature_cols_ = list(X_train.columns)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """입력 데이터에 대한 거래금액 예측값을 반환합니다."""
        self._check_trained()
        X = self.prepare_features(X)
        return self._model.predict(X)

    def predict_series(self, X: pd.DataFrame, name: str = "예측거래금액") -> pd.Series:
        """예측 결과를 인덱스가 유지된 Series로 반환합니다."""
        self._check_trained()
        prepared_X = self.prepare_features(X)
        pred = self._model.predict(prepared_X)
        return pd.Series(pred, index=prepared_X.index, name=name)

    def evaluate(self, X_test: pd.DataFrame, y_test) -> dict[str, float]:
        """테스트 데이터로 회귀 성능을 평가합니다."""
        self._check_trained()
        X_test = self.prepare_features(X_test)
        y_test = self._clean_numeric(pd.Series(y_test, index=X_test.index)).dropna()
        X_test = X_test.loc[y_test.index]
        pred = self._model.predict(X_test)

        metrics = _regression_metrics(y_test, pred)
        metrics["rows"] = int(len(X_test))
        return metrics

    def fit_from_dataframe(self, df: pd.DataFrame) -> dict[str, float]:
        """원본 DataFrame을 받아 train/test split 후 학습하고 성능 지표를 반환합니다."""
        data = self.prepare_dataframe(df, need_target=True)
        X = data[self.feature_columns]
        y = data[self.target_col]

        if self.sample_size is not None and len(X) > self.sample_size:
            sample_idx = X.sample(n=self.sample_size, random_state=self.random_state).index
            X = X.loc[sample_idx]
            y = y.loc[sample_idx]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
        )

        self.fit(X_train, y_train)
        metrics = self.evaluate(X_test, y_test)
        metrics["train_rows"] = int(len(X_train))
        metrics["test_rows"] = int(len(X_test))
        self.metrics_ = metrics
        return metrics

    def evaluate_dataframe(self, df: pd.DataFrame) -> dict[str, float]:
        """타깃 컬럼이 있는 원본 DataFrame으로 모델 성능을 평가합니다."""
        data = self.prepare_dataframe(df, need_target=True)
        X = data[self.feature_columns]
        y = data[self.target_col]
        return self.evaluate(X, y)

    def permutation_importance(
        self,
        df: pd.DataFrame,
        n_samples: int = 10_000,
        n_repeats: int = 5,
        scoring: str = "r2",
        n_jobs: int = 1,
    ) -> pd.DataFrame:
        """원본 피처 단위 permutation importance를 계산합니다."""
        self._check_trained()
        data = self.prepare_dataframe(df, need_target=True)
        X = data[self.feature_columns]
        y = data[self.target_col]

        if len(X) > n_samples:
            X = X.sample(n=n_samples, random_state=self.random_state)
            y = y.loc[X.index]

        result = sklearn_permutation_importance(
            self._model,
            X,
            y,
            n_repeats=n_repeats,
            random_state=self.random_state,
            scoring=scoring,
            n_jobs=n_jobs,
        )

        return (
            pd.DataFrame(
                {
                    "feature": X.columns,
                    "importance_mean": result.importances_mean,
                    "importance_std": result.importances_std,
                }
            )
            .sort_values("importance_mean", ascending=False)
            .reset_index(drop=True)
        )

    def get_feature_importance(self) -> np.ndarray | None:
        """최종 모델이 feature_importances_를 지원하면 반환합니다."""
        self._check_trained()
        final_model = self._model.named_steps.get("model")
        if hasattr(final_model, "feature_importances_"):
            return final_model.feature_importances_
        if hasattr(final_model, "coef_"):
            return np.ravel(final_model.coef_)
        return None

    def save(self, path: str) -> None:
        """학습된 모델 객체 전체를 저장합니다."""
        self._check_trained()
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str):
        """저장된 모델 객체를 불러옵니다."""
        loaded = joblib.load(path)
        if not isinstance(loaded, cls):
            raise TypeError(f"불러온 객체가 {cls.__name__}이 아닙니다.")
        return loaded


# -----------------------------------------------------------------------------
# 개별 회귀 모델 클래스
# -----------------------------------------------------------------------------

class LinearRegressionPriceModel(ApartmentPriceRegressionBase):
    """선형 회귀 기반 거래금액 예측 모델.

    기준 모델(baseline)로 사용하기 좋습니다.
    숫자형 피처는 StandardScaler로 스케일링합니다.
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("scale_numeric", True)
        super().__init__(name="LinearRegression", **kwargs)

    def _create_estimator(self):
        return LinearRegression(**self.estimator_params)


class RandomForestPriceModel(ApartmentPriceRegressionBase):
    """RandomForestRegressor 기반 거래금액 예측 모델."""

    def __init__(self, **kwargs):
        estimator_params = kwargs.pop("estimator_params", {}) or {}
        default_params = {
            "n_estimators": 100,
            "max_depth": 20,
            "min_samples_leaf": 2,
            "n_jobs": -1,
            "random_state": kwargs.get("random_state", 42),
        }
        default_params.update(estimator_params)
        super().__init__(name="RandomForest", estimator_params=default_params, **kwargs)

    def _create_estimator(self):
        return RandomForestRegressor(**self.estimator_params)


class LightGBMPriceModel(ApartmentPriceRegressionBase):
    """LightGBM 기반 거래금액 예측 모델.

    lightgbm 패키지가 설치되어 있어야 합니다.
    """

    def __init__(self, **kwargs):
        estimator_params = kwargs.pop("estimator_params", {}) or {}
        default_params = {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "num_leaves": 64,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": kwargs.get("random_state", 42),
            "n_jobs": -1,
            "verbose": -1,
        }
        default_params.update(estimator_params)
        super().__init__(name="LightGBM", estimator_params=default_params, **kwargs)

    def _create_estimator(self):
        try:
            from lightgbm import LGBMRegressor
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "LightGBMPriceModel을 사용하려면 `pip install lightgbm`이 필요합니다."
            ) from exc
        return LGBMRegressor(**self.estimator_params)


class XGBoostPriceModel(ApartmentPriceRegressionBase):
    """XGBoost 기반 거래금액 예측 모델.

    xgboost 패키지가 설치되어 있어야 합니다.
    """

    def __init__(self, **kwargs):
        estimator_params = kwargs.pop("estimator_params", {}) or {}
        default_params = {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 8,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "reg:squarederror",
            "random_state": kwargs.get("random_state", 42),
            "n_jobs": -1,
        }
        default_params.update(estimator_params)
        super().__init__(name="XGBoost", estimator_params=default_params, **kwargs)

    def _create_estimator(self):
        try:
            from xgboost import XGBRegressor
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "XGBoostPriceModel을 사용하려면 `pip install xgboost`가 필요합니다."
            ) from exc
        return XGBRegressor(**self.estimator_params)


# -----------------------------------------------------------------------------
# 모델 비교 유틸 함수
# -----------------------------------------------------------------------------

def compare_regression_models(
    models: list[ApartmentPriceRegressionBase],
    df: pd.DataFrame,
    sort_by: str = "RMSE",
) -> pd.DataFrame:
    """여러 회귀 모델을 같은 데이터로 학습하고 성능을 비교합니다.

    Args:
        models: ApartmentPriceRegressionBase를 상속한 모델 객체 리스트
        df: 원본 데이터프레임
        sort_by: 정렬 기준. 보통 MAE, RMSE는 낮을수록 좋고 R2는 높을수록 좋습니다.

    Returns:
        모델별 성능 비교 DataFrame
    """
    rows: list[dict] = []

    for model in models:
        try:
            metrics = model.fit_from_dataframe(df)
            row = {"model": model.name, **metrics, "error": None}
        except Exception as exc:
            row = {
                "model": getattr(model, "name", model.__class__.__name__),
                "MAE": np.nan,
                "MSE": np.nan,
                "RMSE": np.nan,
                "MAPE": np.nan,
                "R2": np.nan,
                "rows": 0,
                "train_rows": 0,
                "test_rows": 0,
                "error": str(exc),
            }
        rows.append(row)

    result = pd.DataFrame(rows)

    if sort_by in result.columns:
        ascending = sort_by.upper() != "R2"
        result = result.sort_values(sort_by, ascending=ascending, na_position="last")

    return result.reset_index(drop=True)


if __name__ == "__main__":
    # 프로젝트 루트에서 실행하는 예시:
    # python -m models.regression.price_regression_models
    df = pd.read_csv("Apart Deal_6.csv")

    models = [
        LinearRegressionPriceModel(sample_size=100_000),
        RandomForestPriceModel(sample_size=100_000),
        LightGBMPriceModel(sample_size=100_000),
        XGBoostPriceModel(sample_size=100_000),
    ]

    result_df = compare_regression_models(models, df)
    print(result_df)
