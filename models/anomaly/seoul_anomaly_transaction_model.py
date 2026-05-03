"""
서울특별시 시군구별 이상 거래 탐지 모델.

각 구(시군구)의 거래 패턴을 독립적으로 학습한 뒤,
구 내부 기준에서 특이한 거래를 탐지합니다.

원본 AnomalyTransactionModel과의 차이점
--------------------------------------
- 서울 데이터만 사용 (지역코드 11 또는 시군구에 "서울" 포함)
- 전체 모델 1개 대신, 시군구별로 IsolationForest를 각각 학습
- 따라서 "강남구 기준 특이 거래", "노원구 기준 특이 거래" 처럼
  구마다 독립된 기준으로 이상치를 판단합니다.

주의
----
이 모델의 결과는 실제 불법/허위 거래를 의미하지 않습니다.
발표나 서비스에서는 "모델 기준 특이 거래", "구 내 일반 패턴과 다른 거래"로 표현하세요.

사용 예시
---------
import pandas as pd
from models.anomaly.anomaly_transaction_model_seoul import SeoulDistrictAnomalyModel

df = pd.read_csv("Apart Deal_6.csv", encoding="cp949")

model = SeoulDistrictAnomalyModel(contamination=0.03, random_state=42)
model.fit_from_dataframe(df)

result_df = model.detect_from_dataframe(df)
print(result_df[["시군구", "아파트", "거래금액", "평당가", "anomaly_label", "anomaly_score"]].head(20))

summary = model.summarize_by_district(df)
print(summary)
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

_SEOUL_REGION_CODE_PREFIX = "11"


@dataclass
class SeoulDistrictAnomalyModel:
    """서울특별시 시군구별 이상 거래 탐지 모델.

    시군구마다 IsolationForest를 독립적으로 학습하여,
    구 내부의 일반 거래 패턴과 다른 특이 거래를 탐지합니다.

    Parameters
    ----------
    contamination:
        각 구 데이터 중 특이 거래로 볼 비율 (0 < contamination < 0.5).
    current_year:
        건물연식 계산 기준 연도.
    min_samples:
        구별 최소 학습 샘플 수. 이보다 적은 구는 학습에서 제외됩니다.
    random_state:
        재현성을 위한 난수 시드.
    """

    contamination: float = 0.03
    current_year: int = 2026
    min_samples: int = 30
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

    # 구별 모델 저장소: {"강남구": Pipeline, ...}
    district_models_: dict[str, Pipeline] = field(default_factory=dict, init=False)
    fitted_districts_: list[str] = field(default_factory=list, init=False)
    skipped_districts_: list[str] = field(default_factory=list, init=False)

    @staticmethod
    def _clean_numeric(series: pd.Series) -> pd.Series:
        """쉼표가 포함된 문자열 숫자를 안전하게 숫자형으로 변환합니다."""
        return pd.to_numeric(
            series.astype(str).str.strip().str.replace(",", "", regex=False),
            errors="coerce",
        )

    def _required_input_cols(self) -> list[str]:
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

    def _filter_seoul(self, df: pd.DataFrame) -> pd.DataFrame:
        """서울특별시 데이터만 추출합니다.

        지역코드가 "11"로 시작하거나 시군구에 "서울"이 포함된 행을 반환합니다.
        """
        code_mask = df["지역코드"].astype(str).str.strip().str.startswith(_SEOUL_REGION_CODE_PREFIX)
        name_mask = df["시군구"].astype(str).str.contains("서울", na=False)
        return df[code_mask | name_mask].copy()

    def _extract_district_name(self, sigungu: pd.Series) -> pd.Series:
        """시군구 컬럼에서 구 이름만 추출합니다.

        예: "서울 강남구" -> "강남구", "강남구" -> "강남구"
        """
        return sigungu.astype(str).str.strip().str.replace(r"^서울\s*", "", regex=True)

    def prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """서울 데이터 필터링 및 피처 생성을 수행합니다.

        생성 파생 피처
        -------------
        - 평당가 = 거래금액 / (전용면적 / 3.3)
        - 건물연식 = current_year - 건축년도
        - 거래연도, 거래월
        - 구명 = 시군구에서 "서울 " 제거한 구 이름
        """
        self._validate_columns(df)
        data = self._filter_seoul(df)

        if data.empty:
            raise ValueError("서울 데이터가 없습니다. 지역코드(11) 또는 시군구('서울' 포함) 컬럼을 확인하세요.")

        data = data[self._required_input_cols()].copy()

        numeric_source_cols = [
            "건축년도", "층", "전용면적", "거래금액",
            "기준금리", "위도", "경도", "인근학교수", "인근역수", "세대수",
        ]
        for col in numeric_source_cols:
            data[col] = self._clean_numeric(data[col])

        data["지역코드"] = data["지역코드"].astype(str).str.strip()
        data["시군구"] = data["시군구"].astype(str).str.strip()
        data["구명"] = self._extract_district_name(data["시군구"])

        거래일 = pd.to_datetime(data["거래일"], errors="coerce")
        data["거래연도"] = 거래일.dt.year
        data["거래월"] = 거래일.dt.month

        data["건물연식"] = self.current_year - data["건축년도"]

        data.loc[data["전용면적"] <= 0, "전용면적"] = np.nan
        data["평당가"] = data["거래금액"] / (data["전용면적"] / 3.3)

        data = data.drop(columns=["건축년도", "거래일"])
        data = data.replace([np.inf, -np.inf], np.nan)
        data = data.dropna(subset=self.numeric_cols)

        return data

    def _build_district_pipeline(self) -> Pipeline:
        """구별 IsolationForest 파이프라인을 생성합니다.

        범주형 피처(지역코드/시군구)는 구별 모델에서는 상수이므로 제외하고
        수치형 피처만 사용합니다.
        """
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self.numeric_cols),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )
        detector = IsolationForest(
            n_estimators=100,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1,
        )
        return Pipeline(steps=[("preprocessor", preprocessor), ("model", detector)])

    def fit_from_dataframe(self, df: pd.DataFrame) -> "SeoulDistrictAnomalyModel":
        """서울 데이터를 구별로 분리하여 IsolationForest를 학습합니다."""
        data = self.prepare_dataframe(df)

        self.district_models_ = {}
        self.fitted_districts_ = []
        self.skipped_districts_ = []

        for district, group in data.groupby("구명"):
            if len(group) < self.min_samples:
                self.skipped_districts_.append(district)
                continue

            pipe = self._build_district_pipeline()
            pipe.fit(group[self.numeric_cols])
            self.district_models_[district] = pipe
            self.fitted_districts_.append(district)

        if not self.fitted_districts_:
            raise RuntimeError("학습된 구가 없습니다. min_samples를 낮추거나 데이터를 확인하세요.")

        print(f"[SeoulDistrictAnomalyModel] 학습 완료: {len(self.fitted_districts_)}개 구")
        if self.skipped_districts_:
            print(f"  샘플 부족으로 제외된 구 ({self.min_samples}건 미만): {self.skipped_districts_}")

        return self

    def _check_trained(self) -> None:
        if not self.district_models_:
            raise RuntimeError("모델이 학습되지 않았습니다. fit_from_dataframe()을 먼저 호출하세요.")

    def detect_from_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """원본 데이터프레임에 구별 이상 거래 탐지 결과를 추가합니다.

        추가 컬럼
        --------
        - 구명: 시군구에서 추출한 구 이름
        - 평당가, 건물연식, 거래연도, 거래월: 파생 피처
        - anomaly_raw_label: 1=정상, -1=특이
        - anomaly_label: "정상거래" / "특이거래"
        - anomaly_score: 값이 작을수록 더 특이함 (구 내 상대 점수)
        - anomaly_rank: 구 내 특이 순위. 1이 가장 특이함
        - district_rank: 전체 서울 데이터 기준 순위
        """
        self._check_trained()

        prepared = self.prepare_dataframe(df)
        result_parts = []

        for district, group in prepared.groupby("구명"):
            if district not in self.district_models_:
                # 학습되지 않은 구는 NaN으로 처리
                group = group.copy()
                group["anomaly_raw_label"] = np.nan
                group["anomaly_label"] = "미탐지(학습데이터부족)"
                group["anomaly_score"] = np.nan
                group["anomaly_rank"] = np.nan
                result_parts.append(group)
                continue

            pipe = self.district_models_[district]
            labels = pipe.predict(group[self.numeric_cols])
            scores = pipe.decision_function(group[self.numeric_cols])

            group = group.copy()
            group["anomaly_raw_label"] = labels
            group["anomaly_label"] = np.where(labels == -1, "특이거래", "정상거래")
            group["anomaly_score"] = scores

            # 구 내 순위 (작은 score = 더 특이)
            group["anomaly_rank"] = (
                group["anomaly_score"].rank(method="first", ascending=True).astype(int)
            )
            result_parts.append(group)

        if not result_parts:
            raise RuntimeError("탐지된 데이터가 없습니다.")

        result = pd.concat(result_parts)

        # 원본 df의 나머지 컬럼 병합
        base = df.loc[result.index].copy()
        for col in ["평당가", "건물연식", "거래연도", "거래월", "구명",
                    "anomaly_raw_label", "anomaly_label", "anomaly_score", "anomaly_rank"]:
            base[col] = result[col]

        # 전체 서울 기준 순위
        base["district_rank"] = (
            base["anomaly_score"].rank(method="first", ascending=True)
        )

        return base.sort_values("anomaly_score", ascending=True)

    def summarize_by_district(self, df: pd.DataFrame) -> pd.DataFrame:
        """구별 이상 거래 탐지 결과 요약 테이블을 반환합니다.

        Returns
        -------
        pd.DataFrame
            인덱스: 구명, 컬럼: 총거래수, 특이거래수, 특이거래비율,
                   정상거래_평균평당가, 특이거래_평균평당가, 평당가차이
        """
        result = self.detect_from_dataframe(df)
        result = result[result["anomaly_raw_label"].notna()]

        rows = []
        for district, group in result.groupby("구명"):
            anomaly = group[group["anomaly_raw_label"] == -1]
            normal = group[group["anomaly_raw_label"] == 1]

            total = len(group)
            anomaly_count = len(anomaly)

            rows.append({
                "구명": district,
                "총거래수": total,
                "특이거래수": anomaly_count,
                "특이거래비율(%)": round(anomaly_count / total * 100, 2) if total else 0.0,
                "정상거래_평균평당가": round(normal["평당가"].mean(), 0) if len(normal) else np.nan,
                "특이거래_평균평당가": round(anomaly["평당가"].mean(), 0) if anomaly_count else np.nan,
                "평균평당가_차이": round(
                    anomaly["평당가"].mean() - normal["평당가"].mean(), 0
                ) if (anomaly_count and len(normal)) else np.nan,
                "특이거래_평균거래금액": round(anomaly["거래금액"].mean(), 0) if anomaly_count else np.nan,
                "특이거래_최저score": round(anomaly["anomaly_score"].min(), 4) if anomaly_count else np.nan,
            })

        summary = pd.DataFrame(rows).set_index("구명")
        return summary.sort_values("특이거래비율(%)", ascending=False)

    def top_anomalies_by_district(self, df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
        """구별로 가장 특이한 거래 상위 n개씩 반환합니다."""
        result = self.detect_from_dataframe(df)
        anomalies = result[result["anomaly_raw_label"] == -1]

        top_list = (
            anomalies
            .sort_values("anomaly_score", ascending=True)
            .groupby("구명", group_keys=False)
            .head(n)
        )
        return top_list.sort_values(["구명", "anomaly_score"]).reset_index(drop=True)

    def top_anomalies(self, df: pd.DataFrame, n: int = 50) -> pd.DataFrame:
        """서울 전체 기준 가장 특이한 거래 n개를 반환합니다."""
        result = self.detect_from_dataframe(df)
        anomalies = result[result["anomaly_raw_label"] == -1]
        return anomalies.head(n).reset_index(drop=True)

    def save(self, path: str) -> None:
        """모델 객체를 저장합니다."""
        self._check_trained()
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "SeoulDistrictAnomalyModel":
        """저장된 모델 객체를 불러옵니다."""
        loaded = joblib.load(path)
        if not isinstance(loaded, cls):
            raise TypeError("불러온 객체가 SeoulDistrictAnomalyModel이 아닙니다.")
        return loaded


if __name__ == "__main__":
    import time

    print("=== 1. 데이터 로드 ===")
    start = time.time()
    df = pd.read_csv("data/raw/Apart Deal_6.csv", encoding="utf-8", low_memory=False)
    df["거래금액"] = df["거래금액"].astype(str).str.replace(",", "", regex=False).astype(float)
    df = df.drop_duplicates(keep="first").reset_index(drop=True)
    print(f"로드 완료: {len(df):,}건 ({time.time() - start:.2f}초)")

    print("\n=== 2. 서울 구별 모델 학습 ===")
    model = SeoulDistrictAnomalyModel(contamination=0.03, random_state=42)
    model.fit_from_dataframe(df)

    print("\n=== 3. 구별 요약 ===")
    summary = model.summarize_by_district(df)
    print(summary.to_string())

    print("\n=== 4. 구별 TOP 10 특이 거래 ===")
    top = model.top_anomalies_by_district(df, n=10)
    print(top[["구명", "아파트", "거래금액", "평당가", "anomaly_score", "anomaly_rank"]].to_string())

    model.save("seoul_district_anomaly_model.joblib")
    print("\n모델 저장 완료: seoul_district_anomaly_model.joblib")
