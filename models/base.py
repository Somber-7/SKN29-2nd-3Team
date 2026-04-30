"""
models/base.py — 모델 공통 추상 베이스 클래스

모든 모델은 이 클래스를 상속하여 일관된 인터페이스를 유지합니다.
- 지도학습 모델  → BaseModel 상속
- 군집화 모델    → BaseClusterModel 상속

구현 예시:
    from models.base import BaseModel
    from utils.metrics import regression_metrics

    class LinearRegressionModel(BaseModel):
        def __init__(self):
            super().__init__(name="LinearRegression")

        def fit(self, X_train, y_train, **kwargs):
            from sklearn.linear_model import LinearRegression
            self._model = LinearRegression(**kwargs)
            self._model.fit(X_train, y_train)
            self._is_trained = True
            return self

        def predict(self, X):
            self._check_trained()
            return self._model.predict(X)

        def evaluate(self, X_test, y_test):
            self._check_trained()
            return regression_metrics(y_test, self.predict(X_test))
"""

from abc import ABC, abstractmethod
from typing import Any
import numpy as np


class BaseModel(ABC):
    """지도학습 모델(회귀/분류)의 공통 추상 클래스.

    하위 클래스는 fit(), predict(), evaluate() 세 메서드를 반드시 구현해야 합니다.
    predict_proba(), get_feature_importance()는 해당 기능을 지원하는 모델에서만 오버라이드하세요.

    Attributes:
        name:        모델 식별 이름 (Streamlit UI 표시 등에 사용)
        _model:      실제 sklearn/PyTorch 모델 객체 (fit() 후 설정)
        _is_trained: 학습 완료 여부 (fit() 호출 시 True로 변경)
    """

    def __init__(self, name: str):
        """
        Args:
            name: 모델 식별 이름 (예: "RandomForest", "LightGBM")
        """
        self.name = name
        self._model = None
        self._is_trained = False

    # ── 필수 구현 ─────────────────────────────────────

    @abstractmethod
    def fit(self, X_train, y_train, **kwargs) -> "BaseModel":
        """모델을 학습합니다. 메서드 체이닝을 위해 self를 반환합니다.

        구현 시 반드시 self._is_trained = True 를 설정하세요.

        Args:
            X_train: 학습 피처 데이터
            y_train: 학습 타깃 데이터
            **kwargs: 모델별 추가 파라미터

        Returns:
            self (메서드 체이닝 가능: model.fit(X, y).evaluate(X_te, y_te))
        """

    @abstractmethod
    def predict(self, X) -> np.ndarray:
        """입력 데이터에 대한 예측값을 반환합니다.

        미학습 상태면 _check_trained()가 RuntimeError를 발생시킵니다.

        Args:
            X: 예측할 피처 데이터

        Returns:
            예측값 numpy 배열
        """

    @abstractmethod
    def evaluate(self, X_test, y_test) -> dict:
        """테스트 데이터로 모델 성능을 평가합니다.

        utils/metrics.py의 함수를 사용하여 지표를 계산하고 딕셔너리로 반환하세요.

        Args:
            X_test: 테스트 피처 데이터
            y_test: 테스트 타깃 데이터

        Returns:
            평가지표 딕셔너리 (예: {"MAE": 0.5, "R2": 0.92})
        """

    # ── 선택 구현 (기본값 제공) ────────────────────────

    def predict_proba(self, X) -> np.ndarray | None:
        """클래스별 예측 확률을 반환합니다. (분류 모델에서 오버라이드)

        ROC-AUC 계산이나 Soft Voting에 필요합니다.
        확률을 지원하지 않는 모델(예: LinearSVC)은 None을 반환합니다.

        Args:
            X: 예측할 피처 데이터

        Returns:
            shape (n_samples, n_classes)의 확률 배열, 또는 None
        """
        return None

    def get_params(self) -> dict:
        """현재 모델의 하이퍼파라미터를 딕셔너리로 반환합니다.

        sklearn 모델은 get_params()를 자동으로 호출합니다.
        PyTorch 등 커스텀 모델은 이 메서드를 오버라이드하세요.

        Returns:
            하이퍼파라미터 딕셔너리. 미학습 시 빈 딕셔너리.
        """
        if self._model is not None and hasattr(self._model, "get_params"):
            return self._model.get_params()
        return {}

    def get_feature_importance(self) -> np.ndarray | None:
        """피처 중요도 배열을 반환합니다.

        feature_importances_ 속성을 가진 모델(RandomForest, LightGBM 등)에서 자동 동작합니다.
        선형 모델(계수 기반)은 이 메서드를 오버라이드하여 coef_를 반환하세요.

        Returns:
            피처 중요도 numpy 배열 (학습 피처 순서와 동일), 또는 None
        """
        if self._model is not None and hasattr(self._model, "feature_importances_"):
            return self._model.feature_importances_
        return None

    # ── 공통 유틸 ─────────────────────────────────────

    @property
    def is_trained(self) -> bool:
        """모델 학습 완료 여부를 반환합니다."""
        return self._is_trained

    def _check_trained(self):
        """학습되지 않은 상태에서 predict/evaluate 호출 시 RuntimeError를 발생시킵니다.

        모든 predict(), evaluate() 구현의 첫 줄에 호출하세요.
        """
        if not self._is_trained:
            raise RuntimeError(f"[{self.name}] 모델이 아직 학습되지 않았습니다. fit()을 먼저 호출하세요.")

    def __repr__(self) -> str:
        status = "trained" if self._is_trained else "untrained"
        return f"{self.__class__.__name__}(name={self.name!r}, status={status})"


class BaseClusterModel(ABC):
    """군집화 모델의 공통 추상 클래스. (타깃 레이블 y 없음)

    하위 클래스는 fit(), predict(), evaluate() 세 메서드를 반드시 구현해야 합니다.

    Attributes:
        name:    모델 식별 이름
        _model:  실제 sklearn 모델 객체
        _labels: fit() 후 할당된 군집 레이블 배열
    """

    def __init__(self, name: str):
        """
        Args:
            name: 모델 식별 이름 (예: "KMeans", "DBSCAN")
        """
        self.name = name
        self._model = None
        self._labels: np.ndarray | None = None

    @abstractmethod
    def fit(self, X) -> "BaseClusterModel":
        """데이터를 군집화하고 각 샘플의 군집 레이블을 self._labels에 저장합니다.

        Args:
            X: 군집화할 피처 데이터

        Returns:
            self
        """

    @abstractmethod
    def predict(self, X) -> np.ndarray:
        """새 데이터에 대한 군집 레이블을 예측합니다.

        DBSCAN처럼 새 데이터 예측을 지원하지 않는 모델은 NotImplementedError를 발생시키세요.

        Args:
            X: 예측할 피처 데이터

        Returns:
            군집 레이블 배열
        """

    @abstractmethod
    def evaluate(self, X) -> dict:
        """군집화 결과의 품질을 평가합니다.

        utils/metrics.py의 clustering_metrics()를 사용하여 지표를 반환하세요.

        Args:
            X: 평가에 사용할 피처 데이터 (fit()에 사용한 데이터)

        Returns:
            평가지표 딕셔너리 (Silhouette, Davies-Bouldin, Calinski-Harabasz)
        """

    @property
    def labels(self) -> np.ndarray | None:
        """fit() 후 생성된 군집 레이블 배열을 반환합니다. fit() 전이면 None."""
        return self._labels

    def get_params(self) -> dict:
        """현재 모델의 하이퍼파라미터를 딕셔너리로 반환합니다.

        Returns:
            하이퍼파라미터 딕셔너리. 모델이 없으면 빈 딕셔너리.
        """
        if self._model is not None and hasattr(self._model, "get_params"):
            return self._model.get_params()
        return {}

    def __repr__(self) -> str:
        fitted = self._labels is not None
        return f"{self.__class__.__name__}(name={self.name!r}, fitted={fitted})"
