"""
브랜드 등급 다중 분류 모델.

이 모듈은 아파트 거래 데이터에서 브랜드명을 4개 등급으로 변환한 뒤,
입지/가격/단지 특성으로 브랜드 등급을 예측하는 분류 모델입니다.

등급 예시
---------
- 프리미엄
- 일반브랜드
- 공공(LH)
- 기타

주의
----
`브랜드`, `브랜드여부`는 타깃 생성에 직접 사용되므로 기본 피처에서는 제외합니다.
아파트명에 래미안/자이/푸르지오 같은 브랜드명이 포함될 수 있으므로,
기본 설정에서는 `아파트` 컬럼도 피처에서 제외합니다.

프로젝트 배치 위치 예시
----------------------
models/classification/brand_grade_classifier.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from lightgbm import LGBMClassifier
except ModuleNotFoundError:  # lightgbm이 설치되지 않은 환경에서도 RandomForest는 사용 가능하게 처리
    LGBMClassifier = None

from models.base import BaseModel


PREMIUM_KEYWORDS = [
    "래미안",
    "자이",
    "힐스테이트",
    "아이파크",
    "푸르지오",
    "e편한세상",
    "이편한세상",
    "더샵",
    "롯데캐슬",
    "아크로",
    "SK뷰",
    "SK VIEW",
]

PUBLIC_KEYWORDS = [
    "LH",
    "주공",
]

GENERAL_BRAND_KEYWORDS = [
    "위브",
    "하늘채",
    "베르디움",
    "린",
    "해링턴",
    "스위첸",
    "데시앙",
    "노블랜드",
    "스타힐스",
    "파밀리에",
    "어울림",
    "블루밍",
    "S-클래스",
    "한신",
    "부영",
    "한라",
    "풍림",
    "쌍용",
    "벽산",
    "동아",
    "두산",
    "금호",
    "코오롱",
    "호반",
    "우미",
    "효성",
    "대방",
    "서희",
    "중흥",
    "반도",
    "KCC",
]

GRADE_ORDER = ["기타", "공공(LH)", "일반브랜드", "프리미엄"]


@dataclass
class BrandGradeClassifier(BaseModel):
    """아파트 브랜드 등급 다중 분류 모델.

    BaseModel 표준 인터페이스:
    - fit(X_train, y_train)
    - predict(X)
    - evaluate(X_test, y_test)

    편의 메서드:
    - fit_from_dataframe(df)
    - prepare_dataframe(df)
    - predict_dataframe(df)
    - classification_report_dataframe(df)
    - save(path), load(path)
    """

    model_type: str = "lightgbm"  # "lightgbm" 또는 "random_forest"
    target_col: str = "brand_grade"
    current_year: int = 2026
    sample_size: Optional[int] = 100_000
    test_size: float = 0.2
    random_state: int = 42

    numeric_cols: list[str] = field(
        default_factory=lambda: [
            "거래금액",
            "전용면적",
            "층",
            "건물연식",
            "기준금리",
            "위도",
            "경도",
            "인근학교수",
            "인근역수",
            "세대수",
            "거래연도",
            "거래월",
        ]
    )
    categorical_cols: list[str] = field(default_factory=lambda: ["지역코드", "시군구"])

    metrics_: Optional[dict[str, float]] = field(default=None, init=False)
    classes_: Optional[list[str]] = field(default=None, init=False)
    feature_columns_: Optional[list[str]] = field(default=None, init=False)
    confusion_matrix_: Optional[np.ndarray] = field(default=None, init=False)

    def __post_init__(self) -> None:
        super().__init__(name=f"BrandGradeClassifier({self.model_type})")

    @property
    def feature_columns(self) -> list[str]:
        """모델 입력 피처 컬럼 목록을 반환합니다."""
        return self.numeric_cols + self.categorical_cols

    @staticmethod
    def _clean_numeric(series: pd.Series) -> pd.Series:
        """쉼표가 포함된 문자열 숫자를 안전하게 숫자형으로 변환합니다."""
        return pd.to_numeric(
            series.astype(str).str.strip().str.replace(",", "", regex=False),
            errors="coerce",
        )

    @staticmethod
    def make_brand_grade(brand: object) -> str:
        """브랜드명을 4개 등급으로 변환합니다."""
        if pd.isna(brand):
            return "기타"

        text = str(brand).strip()
        if text == "" or text == "기타":
            return "기타"

        upper_text = text.upper()

        if any(keyword.upper() in upper_text for keyword in PUBLIC_KEYWORDS):
            return "공공(LH)"

        if any(keyword.upper() in upper_text for keyword in PREMIUM_KEYWORDS):
            return "프리미엄"

        if any(keyword.upper() in upper_text for keyword in GENERAL_BRAND_KEYWORDS):
            return "일반브랜드"

        # 브랜드여부가 1이어도 구체 브랜드명이 가이드에 없으면 일반브랜드로 해석할 수 있지만,
        # 이 함수은 브랜드명만 받으므로 보수적으로 기타 처리합니다.
        return "기타"

    def _required_cols(self, need_target: bool = True) -> list[str]:
        """원본 데이터에 필요한 컬럼 목록을 반환합니다."""
        cols = [
            "지역코드",
            "시군구",
            "거래일",
            "건축년도",
            "층",
            "전용면적",
            "거래금액",
            "기준금리",
            "위도",
            "경도",
            "인근학교수",
            "인근역수",
            "세대수",
        ]

        if need_target and self.target_col not in cols:
            cols.append("브랜드")

        return cols

    def _validate_columns(self, df: pd.DataFrame, need_target: bool) -> None:
        required = self._required_cols(need_target=need_target)
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"필수 컬럼이 없습니다: {missing}")

    def prepare_dataframe(self, df: pd.DataFrame, need_target: bool = True) -> pd.DataFrame:
        """원본 데이터프레임을 모델 입력 형태로 변환합니다.

        Parameters
        ----------
        df:
            원본 아파트 거래 데이터
        need_target:
            True이면 `브랜드` 또는 기존 `brand_grade`를 사용해 타깃을 생성합니다.
            False이면 예측용 데이터로 간주하고 타깃 없이 전처리합니다.
        """
        # 사용자가 이미 brand_grade를 만든 경우 `브랜드` 없이도 학습 가능하게 처리
        if need_target and self.target_col in df.columns:
            required = [col for col in self._required_cols(need_target=False) if col not in df.columns]
            if required:
                raise ValueError(f"필수 컬럼이 없습니다: {required}")
        else:
            self._validate_columns(df, need_target=need_target)

        base_cols = [
            "지역코드",
            "시군구",
            "거래일",
            "건축년도",
            "층",
            "전용면적",
            "거래금액",
            "기준금리",
            "위도",
            "경도",
            "인근학교수",
            "인근역수",
            "세대수",
        ]

        use_cols = base_cols.copy()
        if need_target:
            if self.target_col in df.columns:
                use_cols.append(self.target_col)
            else:
                use_cols.append("브랜드")

        data = df[use_cols].copy()

        numeric_source_cols = [
            "건축년도",
            "층",
            "전용면적",
            "거래금액",
            "기준금리",
            "위도",
            "경도",
            "인근학교수",
            "인근역수",
            "세대수",
        ]

        for col in numeric_source_cols:
            data[col] = self._clean_numeric(data[col])

        data["지역코드"] = data["지역코드"].astype(str).str.strip()
        data["시군구"] = data["시군구"].astype(str).str.strip()

        parsed_date = pd.to_datetime(data["거래일"], errors="coerce")
        data["거래연도"] = parsed_date.dt.year
        data["거래월"] = parsed_date.dt.month
        data["건물연식"] = self.current_year - data["건축년도"]

        if need_target:
            if self.target_col not in data.columns:
                data[self.target_col] = data["브랜드"].apply(self.make_brand_grade)
            data[self.target_col] = pd.Categorical(
                data[self.target_col].astype(str),
                categories=GRADE_ORDER,
                ordered=True,
            ).astype(str)

        drop_cols = ["거래일", "건축년도"]
        if "브랜드" in data.columns:
            drop_cols.append("브랜드")
        data = data.drop(columns=drop_cols)

        subset = self.feature_columns.copy()
        if need_target:
            subset.append(self.target_col)

        data = data.dropna(subset=subset)
        data = data[data["전용면적"] > 0]
        data = data[data["거래금액"] > 0]
        data = data[data["건물연식"] >= 0]

        return data

    def _build_estimator(self):
        """선택한 분류 모델 객체를 생성합니다."""
        if self.model_type == "lightgbm":
            if LGBMClassifier is None:
                raise ModuleNotFoundError(
                    "lightgbm이 설치되어 있지 않습니다. `pip install lightgbm` 후 다시 실행하거나 "
                    "model_type='random_forest'를 사용하세요."
                )
            return LGBMClassifier(
                objective="multiclass",
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1,
            )

        if self.model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=200,
                max_depth=None,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1,
                class_weight="balanced_subsample",
            )

        raise ValueError("model_type은 'lightgbm' 또는 'random_forest'만 지원합니다.")

    def _build_pipeline(self) -> Pipeline:
        """전처리 + 분류 모델 파이프라인을 생성합니다."""
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self.numeric_cols),
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    self.categorical_cols,
                ),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )

        return Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", self._build_estimator()),
            ]
        )

    def fit(self, X_train: pd.DataFrame, y_train: Sequence[str], **kwargs) -> "BrandGradeClassifier":
        """BaseModel 표준 인터페이스에 맞춰 모델을 학습합니다."""
        self._model = self._build_pipeline()
        self._model.fit(X_train, y_train, **kwargs)
        self._is_trained = True
        self.feature_columns_ = list(X_train.columns)
        self.classes_ = list(self._model.named_steps["model"].classes_)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """브랜드 등급을 예측합니다."""
        self._check_trained()
        return self._model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray | None:
        """브랜드 등급별 예측 확률을 반환합니다."""
        self._check_trained()
        if hasattr(self._model, "predict_proba"):
            return self._model.predict_proba(X)
        return None

    def evaluate(self, X_test: pd.DataFrame, y_test: Sequence[str]) -> dict:
        """테스트 데이터로 분류 성능을 평가합니다."""
        self._check_trained()
        y_pred = self.predict(X_test)

        self.confusion_matrix_ = confusion_matrix(y_test, y_pred, labels=self.classes_)

        metrics = {
            "Accuracy": float(accuracy_score(y_test, y_pred)),
            "Precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
            "Recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
            "F1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        }
        return metrics

    def fit_from_dataframe(self, df: pd.DataFrame) -> dict[str, float]:
        """원본 데이터프레임으로 학습/평가를 한 번에 수행합니다."""
        data = self.prepare_dataframe(df, need_target=True)
        X = data[self.feature_columns]
        y = data[self.target_col]

        if self.sample_size is not None and len(X) > self.sample_size:
            sample_idx = X.sample(n=self.sample_size, random_state=self.random_state).index
            X = X.loc[sample_idx]
            y = y.loc[sample_idx]

        # 클래스 수가 충분하면 stratify 적용
        value_counts = y.value_counts()
        stratify = y if len(value_counts) > 1 and value_counts.min() >= 2 else None

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify,
        )

        self.fit(X_train, y_train)
        self.metrics_ = self.evaluate(X_test, y_test)
        self.metrics_["train_rows"] = int(len(X_train))
        self.metrics_["test_rows"] = int(len(X_test))
        self.metrics_["n_classes"] = int(y.nunique())

        return self.metrics_

    def predict_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """원본 데이터프레임에 예측 브랜드 등급과 확률을 붙여 반환합니다."""
        self._check_trained()
        data = self.prepare_dataframe(df, need_target=False)
        X = data[self.feature_columns]

        pred = self.predict(X)
        result = df.loc[X.index].copy()
        result["예측_brand_grade"] = pred

        prob = self.predict_proba(X)
        if prob is not None and self.classes_ is not None:
            max_prob = prob.max(axis=1)
            result["예측확률"] = max_prob
            for i, cls in enumerate(self.classes_):
                result[f"확률_{cls}"] = prob[:, i]

        return result

    def classification_report_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """원본 데이터프레임 기준 classification report를 DataFrame으로 반환합니다."""
        self._check_trained()
        data = self.prepare_dataframe(df, need_target=True)
        X = data[self.feature_columns]
        y = data[self.target_col]
        y_pred = self.predict(X)

        report = classification_report(
            y,
            y_pred,
            labels=self.classes_,
            output_dict=True,
            zero_division=0,
        )
        return pd.DataFrame(report).T

    def get_feature_names(self) -> list[str]:
        """전처리 후 실제 모델에 들어간 피처명을 반환합니다."""
        self._check_trained()
        preprocessor = self._model.named_steps["preprocessor"]
        return list(preprocessor.get_feature_names_out())

    def get_feature_importance_df(self, top_n: Optional[int] = 30) -> pd.DataFrame:
        """피처 중요도를 DataFrame으로 반환합니다."""
        self._check_trained()
        estimator = self._model.named_steps["model"]

        if not hasattr(estimator, "feature_importances_"):
            raise AttributeError("현재 모델은 feature_importances_를 지원하지 않습니다.")

        importance_df = pd.DataFrame(
            {
                "feature": self.get_feature_names(),
                "importance": estimator.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

        if top_n is not None:
            importance_df = importance_df.head(top_n)

        return importance_df.reset_index(drop=True)

    def save(self, path: str) -> None:
        """학습된 모델 객체를 저장합니다."""
        self._check_trained()
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "BrandGradeClassifier":
        """저장된 모델 객체를 불러옵니다."""
        loaded = joblib.load(path)
        if not isinstance(loaded, cls):
            raise TypeError("불러온 객체가 BrandGradeClassifier가 아닙니다.")
        return loaded


def compare_brand_grade_classifiers(
    df: pd.DataFrame,
    model_types: Sequence[str] = ("lightgbm", "random_forest"),
    sample_size: Optional[int] = 100_000,
    random_state: int = 42,
) -> pd.DataFrame:
    """여러 브랜드 등급 분류 모델을 한 번에 비교합니다."""
    rows = []

    for model_type in model_types:
        model = BrandGradeClassifier(
            model_type=model_type,
            sample_size=sample_size,
            random_state=random_state,
        )
        try:
            metrics = model.fit_from_dataframe(df)
            rows.append({"model": model.name, **metrics})
        except ModuleNotFoundError as exc:
            rows.append({"model": f"BrandGradeClassifier({model_type})", "error": str(exc)})

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = pd.read_csv("Apart Deal_6.csv", encoding="cp949")

    model = BrandGradeClassifier(model_type="lightgbm", sample_size=100_000)
    metrics = model.fit_from_dataframe(df)
    print(metrics)

    report_df = model.classification_report_dataframe(df.sample(10_000, random_state=42))
    print(report_df)

    importance_df = model.get_feature_importance_df(top_n=20)
    print(importance_df)

    model.save("brand_grade_classifier.joblib")
