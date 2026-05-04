"""
아파트 이상 거래 탐지 모델.

거래금액, 평당가, 면적, 층, 건물연식, 위치, 주변 인프라, 세대수 등을 바탕으로
일반적인 거래 패턴과 다른 "특이 거래"를 탐지합니다.

주의
----
이 모델이 찾는 결과는 법적/실제 이상 거래가 아니라,
머신러닝 모델 기준에서 통계적으로 특이한 거래입니다.
Streamlit이나 보고서에서는 "이상 거래"보다는 "특이 거래" 또는
"모델 기준 이상치"라고 표현하는 것을 권장합니다.

사용 예시
---------
import pandas as pd
from models.anomaly.anomaly_transaction_model import AnomalyTransactionModel

df = pd.read_csv("Apart Deal_6.csv", encoding="cp949")

model = AnomalyTransactionModel(
    contamination=0.03,
    sample_size=200_000,
    random_state=42,
)

model.fit_from_dataframe(df)

result_df = model.detect_from_dataframe(df)
print(result_df[["아파트", "거래금액", "평당가", "anomaly_label", "anomaly_score"]].head())

summary = model.summarize_anomalies(df)
print(summary)

model.save("anomaly_transaction_model.joblib")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class AnomalyTransactionModel:
    """아파트 실거래 특이 거래 탐지 모델.

    IsolationForest를 사용하여 거래 패턴에서 벗어난 데이터를 탐지합니다.

    Parameters
    ----------
    contamination:
        전체 데이터 중 이상치로 볼 비율입니다.
        예: 0.03이면 약 3%를 특이 거래로 분류합니다.
    current_year:
        건물연식 계산 기준 연도입니다.
    sample_size:
        학습에 사용할 최대 샘플 수입니다.
        전체 데이터가 크면 일부 샘플로 학습한 뒤 전체 데이터에 적용할 수 있습니다.
    random_state:
        재현성을 위한 난수 시드입니다.
    """

    contamination: float = 0.03
    current_year: int = 2026
    sample_size: Optional[int] = 200_000
    random_state: int = 42

    numeric_cols: list[str] = field(
        default_factory=lambda: [
            "거래금액",
            "전용면적",
            "평당가",
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
    model: Optional[Pipeline] = field(default=None, init=False)
    feature_columns_: Optional[list[str]] = field(default=None, init=False)
    train_rows_: int = field(default=0, init=False)

    @property
    def feature_columns(self) -> list[str]:
        """모델 입력 피처 목록을 반환합니다."""
        return self.numeric_cols

    @staticmethod
    def _clean_numeric(series: pd.Series) -> pd.Series:
        """쉼표가 포함된 문자열 숫자를 안전하게 숫자형으로 변환합니다."""
        return pd.to_numeric(
            series.astype(str).str.strip().str.replace(",", "", regex=False),
            errors="coerce",
        )

    def _required_input_cols(self) -> list[str]:
        """원본 데이터프레임에 필요한 컬럼 목록을 반환합니다."""
        return [
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

    def _validate_columns(self, df: pd.DataFrame) -> None:
        missing = [col for col in self._required_input_cols() if col not in df.columns]
        if missing:
            raise ValueError(f"필수 컬럼이 없습니다: {missing}")

    def prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """이상 거래 탐지에 사용할 피처 데이터프레임을 생성합니다.

        생성되는 파생 피처
        ----------------
        - 평당가 = 거래금액 / (전용면적 / 3.3)
        - 건물연식 = current_year - 건축년도
        - 거래연도, 거래월 = 거래일에서 추출
        """
        self._validate_columns(df)

        data = df[self._required_input_cols()].copy()

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

        거래일 = pd.to_datetime(data["거래일"], errors="coerce")
        data["거래연도"] = 거래일.dt.year
        data["거래월"] = 거래일.dt.month

        data["건물연식"] = self.current_year - data["건축년도"]

        # 전용면적이 0이거나 음수이면 평당가 계산이 불가능하므로 NaN 처리
        data.loc[data["전용면적"] <= 0, "전용면적"] = np.nan
        data["평당가"] = data["거래금액"] / (data["전용면적"] / 3.3)

        # 모델에는 건축년도/거래일 원본 대신 파생 피처 사용
        data = data.drop(columns=["건축년도", "거래일"])

        data = data.replace([np.inf, -np.inf], np.nan)
        data = data.dropna(subset=self.feature_columns)

        return data

    def _build_pipeline(self) -> Pipeline:
        """전처리 + IsolationForest 파이프라인을 생성합니다."""
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self.numeric_cols),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )

        detector = IsolationForest(
            n_estimators=200,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1,
        )

        return Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", detector),
            ]
        )

    def fit(self, X: pd.DataFrame, **kwargs) -> "AnomalyTransactionModel":
        """전처리된 피처 데이터로 IsolationForest를 학습합니다.

        일반적으로는 원본 df를 넣는 fit_from_dataframe() 사용을 권장합니다.
        """
        if kwargs:
            # IsolationForest 파라미터를 런타임에 일부 변경할 수 있게 허용
            for key, value in kwargs.items():
                if hasattr(self, key):
                    setattr(self, key, value)

        self.model = self._build_pipeline()
        self.model.fit(X[self.feature_columns])
        self.feature_columns_ = list(X[self.feature_columns].columns)
        self.train_rows_ = int(len(X))
        return self

    def fit_from_dataframe(self, df: pd.DataFrame) -> "AnomalyTransactionModel":
        """원본 데이터프레임으로 이상 거래 탐지 모델을 학습합니다."""
        data = self.prepare_dataframe(df)
        X = data[self.feature_columns]

        if self.sample_size is not None and len(X) > self.sample_size:
            X = X.sample(n=self.sample_size, random_state=self.random_state)

        return self.fit(X)

    def _check_trained(self) -> None:
        if self.model is None:
            raise RuntimeError("모델이 아직 학습되지 않았습니다. fit_from_dataframe()을 먼저 호출하세요.")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """전처리된 피처 데이터에 대해 정상/이상 라벨을 예측합니다.

        Returns
        -------
        np.ndarray
            1이면 정상 거래, -1이면 특이 거래입니다.
        """
        self._check_trained()
        return self.model.predict(X[self.feature_columns])

    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        """IsolationForest의 anomaly score를 반환합니다.

        값이 작을수록 더 특이한 거래입니다.
        """
        self._check_trained()
        return self.model.decision_function(X[self.feature_columns])

    def detect_from_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """원본 데이터프레임에 이상 거래 탐지 결과 컬럼을 추가해서 반환합니다.

        추가 컬럼
        --------
        - anomaly_raw_label: IsolationForest 원본 라벨. 정상=1, 이상=-1
        - anomaly_label: 한글 라벨. 정상거래/특이거래
        - anomaly_score: 값이 작을수록 더 특이함
        - anomaly_rank: 특이한 정도의 순위. 1이 가장 특이함
        """
        self._check_trained()

        prepared = self.prepare_dataframe(df)
        labels = self.predict(prepared)
        scores = self.decision_function(prepared)

        result = df.loc[prepared.index].copy()
        result["평당가"] = prepared["평당가"]
        result["건물연식"] = prepared["건물연식"]
        result["거래연도"] = prepared["거래연도"]
        result["거래월"] = prepared["거래월"]
        result["anomaly_raw_label"] = labels
        result["anomaly_label"] = np.where(labels == -1, "특이거래", "정상거래")
        result["anomaly_score"] = scores
        result["anomaly_rank"] = result["anomaly_score"].rank(method="first", ascending=True).astype(int)

        return result.sort_values("anomaly_score", ascending=True)

    def summarize_anomalies(self, df: pd.DataFrame) -> dict[str, float | int]:
        """이상 거래 탐지 결과 요약 지표를 반환합니다."""
        result = self.detect_from_dataframe(df)
        anomaly_mask = result["anomaly_raw_label"] == -1

        total_count = int(len(result))
        anomaly_count = int(anomaly_mask.sum())
        normal_count = total_count - anomaly_count

        return {
            "total_count": total_count,
            "normal_count": normal_count,
            "anomaly_count": anomaly_count,
            "anomaly_ratio": float(anomaly_count / total_count) if total_count else 0.0,
            "avg_price_normal": float(result.loc[~anomaly_mask, "거래금액"].mean()) if normal_count else np.nan,
            "avg_price_anomaly": float(result.loc[anomaly_mask, "거래금액"].mean()) if anomaly_count else np.nan,
            "avg_ppy_normal": float(result.loc[~anomaly_mask, "평당가"].mean()) if normal_count else np.nan,
            "avg_ppy_anomaly": float(result.loc[anomaly_mask, "평당가"].mean()) if anomaly_count else np.nan,
            "train_rows": self.train_rows_,
        }

    def top_anomalies(self, df: pd.DataFrame, n: int = 30) -> pd.DataFrame:
        """가장 특이한 거래 n개를 반환합니다."""
        result = self.detect_from_dataframe(df)
        return result.head(n).reset_index(drop=True)

    def save(self, path: str) -> None:
        """모델 객체를 저장합니다."""
        self._check_trained()
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "AnomalyTransactionModel":
        """저장된 모델 객체를 불러옵니다."""
        loaded = joblib.load(path)
        if not isinstance(loaded, cls):
            raise TypeError("불러온 객체가 AnomalyTransactionModel이 아닙니다.")
        return loaded


if __name__ == "__main__":
    import time
    import pandas as pd

    print("=== 1. 데이터 불러오기 및 전처리 ===")
    start_time = time.time()
    
    # 데이터 로드
    df = pd.read_csv("../data/raw/Apart Deal_6.csv", encoding="utf-8", low_memory=False)
    
    # 평균 계산 에러 방지를 위해 거래금액을 순수 숫자로 변환
    df["거래금액"] = df["거래금액"].astype(str).str.replace(",", "", regex=False).astype(float)
    
    # [핵심 추가] 중복 데이터 완벽 제거
    before_count = len(df)
    df = df.drop_duplicates(keep="first").reset_index(drop=True)
    after_count = len(df)
    
    print(f"중복 제거 완료: {before_count:,}건 -> {after_count:,}건 (총 {before_count - after_count:,}건 중복 삭제됨)")
    print(f"전처리 완료! - 소요시간: {time.time() - start_time:.2f}초")

    print("\n=== 2. 모델 학습 ===")
    model = AnomalyTransactionModel()
    model.fit_from_dataframe(df)
    print("모델 학습 완료!")

    print("\n=== 3. 전체 데이터 이상 거래 탐지 (Batch 처리) ===")
    batch_size = 100000
    result_list = []

    for i in range(0, len(df), batch_size):
        df_chunk = df.iloc[i : i + batch_size].copy()
        print(f"  -> 배치 처리 중: [{i:,} ~ {i + len(df_chunk):,}] / {len(df):,}")
        
        # 조각난 데이터 탐지
        chunk_result = model.detect_from_dataframe(df_chunk)
        result_list.append(chunk_result)

    print("\n=== 4. 결과 병합 및 최종 처리 ===")
    final_result_df = pd.concat(result_list, ignore_index=True)

    # 전체 데이터를 기준으로 진짜 순위(rank) 다시 계산
    final_result_df["anomaly_rank"] = final_result_df["anomaly_score"].rank(method="first", ascending=True).astype(int)
    print("최종 병합 및 순위 계산 완료!")

    print("\n=== 5. 중복 제거된 진짜 특이 거래 TOP 10 ===")
    top_anomalies = final_result_df[final_result_df["anomaly_raw_label"] == -1].sort_values("anomaly_score", ascending=True).head(10)
     
    print(top_anomalies[['시군구', '아파트', '거래금액', '전용면적', '층', '건축년도', 'anomaly_score', 'anomaly_rank']]) 