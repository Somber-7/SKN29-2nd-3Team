"""
models/classification/brand_grade_classifier.py — 브랜드 등급 다중 분류 모델

아파트 거래 데이터에서 브랜드명을 4개 등급으로 변환한 뒤,
입지/가격/단지 특성으로 브랜드 등급을 예측하는 XGBoost 분류 모델입니다.

등급: 프리미엄 / 일반브랜드 / 공공(LH) / 기타

주의
----
- `브랜드`, `브랜드여부`는 타깃 생성에 사용되므로 피처에서 제외합니다.
- `아파트` 컬럼에 브랜드명이 포함되어 있으므로 피처에서 제외합니다.
- 500만 건 전체 데이터 기준으로 학습합니다.
- CUDA 사용 가능 시 GPU로 학습, 없으면 CPU 자동 폴백합니다.

사용 예시
---------
    from utils.db import load_apart_deals
    from models.classification.brand_grade_classifier import BrandGradeClassifier

    df = load_apart_deals()
    model = BrandGradeClassifier()
    metrics = model.fit_from_dataframe(df)
    print(metrics)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import joblib
import numpy as np
import pandas as pd
import torch
from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler

from models.base import BaseModel


PREMIUM_KEYWORDS = [
    "래미안", "자이", "힐스테이트", "아이파크", "푸르지오",
    "e편한세상", "이편한세상", "더샵", "롯데캐슬", "아크로",
    "SK뷰", "SK VIEW",
]
PUBLIC_KEYWORDS = ["LH", "주공"]
GENERAL_BRAND_KEYWORDS = [
    "위브", "하늘채", "베르디움", "린", "해링턴", "스위첸",
    "데시앙", "노블랜드", "스타힐스", "파밀리에", "어울림",
    "블루밍", "S-클래스", "한신", "부영", "한라", "풍림",
    "쌍용", "벽산", "동아", "두산", "금호", "코오롱",
    "호반", "우미", "효성", "대방", "서희", "중흥", "반도", "KCC",
    "대림산업", "삼익주택", "경남기업", "동원개발", "신동아건설", "동부건설",
    "삼부토건", "현진건설", "금강주택", "성지건설", "광양건설", "라온건설", "아남건설",
    "현대건설", "삼성물산", "포스코이앤씨", "GS건설", "HDC현대산업개발",
    "DL이앤씨", "롯데건설",
]
GRADE_ORDER = ["기타", "공공(LH)", "일반브랜드", "프리미엄"]


@dataclass
class BrandGradeClassifier(BaseModel):
    """아파트 브랜드 등급 다중 분류 모델 (XGBoost GPU).

    500만 건 전체 데이터를 직접 학습합니다.
    CUDA 사용 가능 시 GPU로 학습, 없으면 CPU 자동 폴백합니다.

    BaseModel 표준 인터페이스:
        fit(X_train, y_train) / predict(X) / evaluate(X_test, y_test)

    편의 메서드:
        fit_from_dataframe(df) / predict_dataframe(df)
        classification_report_dataframe(df) / save(path) / load(path)
    """

    target_col: str = "brand_grade"
    current_year: int = 2026
    test_size: float = 0.2
    random_state: int = 42

    numeric_cols: list[str] = field(default_factory=lambda: [
        "거래금액", "전용면적", "층", "건물연식", "기준금리",
        "위도", "경도", "인근학교수", "인근역수", "세대수",
        "거래연도", "거래월", "지역코드",
    ])
    categorical_cols: list[str] = field(default_factory=lambda: ["시군구"])

    metrics_: Optional[dict] = field(default=None, init=False)
    classes_: Optional[list[str]] = field(default=None, init=False)
    feature_columns_: Optional[list[str]] = field(default=None, init=False)
    confusion_matrix_: Optional[np.ndarray] = field(default=None, init=False)
    _label_encoder: LabelEncoder = field(default_factory=LabelEncoder, init=False)

    def __post_init__(self) -> None:
        super().__init__(name="BrandGradeClassifier(XGBoost)")

    @property
    def feature_columns(self) -> list[str]:
        return self.numeric_cols + self.categorical_cols

    @staticmethod
    def _clean_numeric(series: pd.Series) -> pd.Series:
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
        if not text or text == "기타":
            return "기타"
        upper = text.upper()
        if any(k.upper() in upper for k in PUBLIC_KEYWORDS):
            return "공공(LH)"
        if any(k.upper() in upper for k in PREMIUM_KEYWORDS):
            return "프리미엄"
        if any(k.upper() in upper for k in GENERAL_BRAND_KEYWORDS):
            return "일반브랜드"
        return "기타"

    def prepare_dataframe(self, df: pd.DataFrame, need_target: bool = True) -> pd.DataFrame:
        """원본 DataFrame을 모델 입력 형태로 변환합니다."""
        required = [
            "지역코드", "시군구", "거래일", "건축년도", "층",
            "전용면적", "거래금액", "기준금리", "위도", "경도",
            "인근학교수", "인근역수", "세대수",
        ]
        if need_target and self.target_col not in df.columns:
            required.append("브랜드")

        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"필수 컬럼 없음: {missing}")

        use_cols = required.copy()
        if need_target and self.target_col in df.columns:
            use_cols.append(self.target_col)

        data = df[use_cols].copy()

        for col in ["지역코드", "건축년도", "층", "전용면적", "거래금액", "기준금리",
                    "위도", "경도", "인근학교수", "인근역수", "세대수"]:
            data[col] = self._clean_numeric(data[col])

        data["시군구"] = data["시군구"].astype(str).str.strip()

        dt = pd.to_datetime(data["거래일"], errors="coerce")
        data["거래연도"] = dt.dt.year
        data["거래월"]   = dt.dt.month
        data["건물연식"] = self.current_year - data["건축년도"]

        if need_target:
            if self.target_col not in data.columns:
                data[self.target_col] = data["브랜드"].apply(self.make_brand_grade)
            data[self.target_col] = pd.Categorical(
                data[self.target_col].astype(str),
                categories=GRADE_ORDER, ordered=True,
            ).astype(str)

        drop_cols = ["거래일", "건축년도"]
        if "브랜드" in data.columns:
            drop_cols.append("브랜드")
        data = data.drop(columns=drop_cols)

        subset = self.feature_columns + ([self.target_col] if need_target else [])
        data = data.dropna(subset=subset)
        data = data[(data["전용면적"] > 0) & (data["거래금액"] > 0) & (data["건물연식"] >= 0)]

        return data

    def _build_pipeline(self) -> Pipeline:
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self.numeric_cols),
                ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), self.categorical_cols),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        estimator = XGBClassifier(
            objective="multi:softprob",
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            device=device,
            n_jobs=-1,
            verbosity=0,
            eval_metric="mlogloss",
            early_stopping_rounds=30,
        )
        return Pipeline([("preprocessor", preprocessor), ("model", estimator)])

    def fit(self, X_train: pd.DataFrame, y_train: Sequence[str], sample_weight: np.ndarray | None = None, **kwargs) -> "BrandGradeClassifier":
        # XGBoost는 문자열 타깃 미지원 — LabelEncoder로 정수 변환
        y_encoded = self._label_encoder.fit_transform(y_train)
        self.classes_ = list(self._label_encoder.classes_)

        self._model = self._build_pipeline()
        self._model.named_steps["model"].set_params(num_class=len(self.classes_))

        fit_params = dict(kwargs)
        if sample_weight is not None:
            fit_params["model__sample_weight"] = sample_weight
        self._model.fit(X_train, y_encoded, **fit_params)

        self._is_trained = True
        self.feature_columns_ = list(X_train.columns)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._check_trained()
        y_encoded = self._model.predict(X)
        return self._label_encoder.inverse_transform(y_encoded.astype(int))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray | None:
        self._check_trained()
        return self._model.predict_proba(X)

    def evaluate(self, X_test: pd.DataFrame, y_test: Sequence[str]) -> dict:
        self._check_trained()
        y_pred = self.predict(X_test)
        self.confusion_matrix_ = confusion_matrix(y_test, y_pred, labels=self.classes_)
        return {
            "Accuracy":  float(accuracy_score(y_test, y_pred)),
            "Precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
            "Recall":    float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
            "F1":        float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        }

    def fit_from_dataframe(self, df: pd.DataFrame) -> dict:
        """원본 DataFrame 전체로 학습/평가를 한 번에 수행합니다."""
        data = self.prepare_dataframe(df, need_target=True)
        X = data[self.feature_columns]
        y = data[self.target_col]

        value_counts = y.value_counts()
        stratify = y if (len(value_counts) > 1 and value_counts.min() >= 2) else None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify,
        )

        # 클래스 역비율 가중치 sqrt 완화 — 완전 역비율 대비 절반 강도
        class_counts = y_train.value_counts()
        class_weight = np.sqrt(len(y_train) / (len(class_counts) * class_counts))
        sample_weight = y_train.map(class_weight).to_numpy()

        # Early Stopping용 validation set 분리 (train의 10%)
        X_tr, X_val, y_tr, y_val, sw_tr, _ = train_test_split(
            X_train, y_train, sample_weight,
            test_size=0.1,
            random_state=self.random_state,
            stratify=y_train,
        )
        y_tr_enc  = self._label_encoder.fit_transform(y_tr)
        y_val_enc = self._label_encoder.transform(y_val)

        self._model = self._build_pipeline()
        self._model.named_steps["model"].set_params(num_class=len(set(y_tr_enc)))

        X_tr_prep  = self._model.named_steps["preprocessor"].fit_transform(X_tr)
        X_val_prep = self._model.named_steps["preprocessor"].transform(X_val)

        self._model.named_steps["model"].fit(
            X_tr_prep, y_tr_enc,
            sample_weight=sw_tr,
            eval_set=[(X_val_prep, y_val_enc)],
            verbose=False,
        )
        self._is_trained = True
        self.classes_ = list(self._label_encoder.classes_)
        self.feature_columns_ = list(X_train.columns)
        self.metrics_ = self.evaluate(X_test, y_test)
        self.metrics_["train_rows"]      = int(len(X_train))
        self.metrics_["test_rows"]       = int(len(X_test))
        self.metrics_["n_classes"]       = int(y.nunique())
        self.metrics_["best_iteration"]  = int(self._model.named_steps["model"].best_iteration)

        return self.metrics_

    def predict_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """원본 DataFrame에 예측 등급과 확률 컬럼을 추가해 반환합니다."""
        self._check_trained()
        data = self.prepare_dataframe(df, need_target=False)
        X = data[self.feature_columns]

        result = df.loc[X.index].copy()
        result["예측_brand_grade"] = self.predict(X)

        prob = self.predict_proba(X)
        if prob is not None and self.classes_:
            result["예측확률"] = prob.max(axis=1)
            for i, cls in enumerate(self.classes_):
                result[f"확률_{cls}"] = prob[:, i]

        return result

    def classification_report_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """classification report를 DataFrame으로 반환합니다."""
        self._check_trained()
        data = self.prepare_dataframe(df, need_target=True)
        X, y = data[self.feature_columns], data[self.target_col]
        report = classification_report(
            y, self.predict(X), labels=self.classes_,
            output_dict=True, zero_division=0,
        )
        return pd.DataFrame(report).T

    def get_feature_names(self) -> list[str]:
        self._check_trained()
        return list(self._model.named_steps["preprocessor"].get_feature_names_out())

    def get_feature_importance_df(self, top_n: Optional[int] = 30) -> pd.DataFrame:
        self._check_trained()
        estimator = self._model.named_steps["model"]
        df_imp = pd.DataFrame({
            "feature":    self.get_feature_names(),
            "importance": estimator.feature_importances_,
        }).sort_values("importance", ascending=False)
        return df_imp.head(top_n).reset_index(drop=True) if top_n else df_imp.reset_index(drop=True)

    def save(self, path: str) -> None:
        self._check_trained()
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "BrandGradeClassifier":
        loaded = joblib.load(path)
        if not isinstance(loaded, cls):
            raise TypeError("불러온 객체가 BrandGradeClassifier가 아닙니다.")
        return loaded
