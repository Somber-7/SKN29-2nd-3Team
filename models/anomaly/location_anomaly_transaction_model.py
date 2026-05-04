"""
전국 시군구별 이상 거래 탐지 모델.

246개 시군구 단위로 IsolationForest를 독립 학습하여
각 시군구 내부 거래 패턴 기준의 특이 거래를 탐지합니다.

사용 예시
--------- 
import pandas as pd
from models.anomaly.location_anomaly_transaction_model import LocationAnomalyModel

df = pd.read_csv("data/raw/Apart Deal_6.csv", encoding="utf-8", low_memory=False)
df["거래금액"] = df["거래금액"].astype(str).str.replace(",", "", regex=False).astype(float)

model = LocationAnomalyModel(contamination=0.03)
model.fit_from_dataframe(df)

top10 = model.top_anomalies_top1_per_region(df, n=10)
print(top10[["시군구", "아파트", "거래금액", "평당가", "anomaly_score", "anomaly_rank"]])
"""

from __future__ import annotations

from dataclasses import dataclass, field

import joblib 
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class LocationAnomalyModel:
    """전국 시군구별 이상 거래 탐지 모델.

    246개 시군구마다 IsolationForest를 독립 학습하여
    시군구 내부 패턴 기준으로 특이 거래를 탐지합니다.

    Parameters
    ----------
    contamination:
        각 시군구 데이터 중 특이 거래로 볼 비율 (0 < contamination < 0.5).
    current_year:
        건물연식 계산 기준 연도.
    min_samples:
        시군구별 최소 학습 샘플 수. 미만이면 학습에서 제외.
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

    location_models_: dict[str, Pipeline] = field(default_factory=dict, init=False)
    fitted_locations_: list[str] = field(default_factory=list, init=False)
    skipped_locations_: list[str] = field(default_factory=list, init=False)

    # ------------------------------------------------------------------ #
    # 내부 유틸                                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _clean_numeric(series: pd.Series) -> pd.Series:
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
        missing = [c for c in self._required_input_cols() if c not in df.columns]
        if missing:
            raise ValueError(f"필수 컬럼이 없습니다: {missing}")

    # ------------------------------------------------------------------ #
    # 전처리                                                               #
    # ------------------------------------------------------------------ #

    def prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """피처 생성 및 시군구 컬럼을 추가합니다."""
        self._validate_columns(df)
        data = df[self._required_input_cols()].copy()

        for col in ["건축년도", "층", "전용면적", "거래금액",
                    "기준금리", "위도", "경도", "인근학교수", "인근역수", "세대수"]:
            data[col] = self._clean_numeric(data[col])

        data["시군구"] = data["시군구"].astype(str).str.strip()

        거래일 = pd.to_datetime(data["거래일"], errors="coerce")
        data["거래연도"] = 거래일.dt.year
        data["거래월"] = 거래일.dt.month
        data["건물연식"] = self.current_year - data["건축년도"]

        data.loc[data["전용면적"] <= 0, "전용면적"] = np.nan
        data["평당가"] = data["거래금액"] / (data["전용면적"] / 3.3)

        data = data.drop(columns=["건축년도", "거래일", "지역코드"])
        data = data.replace([np.inf, -np.inf], np.nan)
        data = data.dropna(subset=self.numeric_cols)

        return data

    # ------------------------------------------------------------------ #
    # 학습                                                                 #
    # ------------------------------------------------------------------ #

    def _build_pipeline(self) -> Pipeline:
        preprocessor = ColumnTransformer(
            transformers=[("num", StandardScaler(), self.numeric_cols)],
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

    def fit_from_dataframe(self, df: pd.DataFrame) -> "LocationAnomalyModel":
        """시군구별로 IsolationForest를 학습합니다."""
        data = self.prepare_dataframe(df)

        self.location_models_ = {}
        self.fitted_locations_ = []
        self.skipped_locations_ = []

        for sigungu, group in data.groupby("시군구"):
            if len(group) < self.min_samples:
                self.skipped_locations_.append(sigungu)
                continue
            pipe = self._build_pipeline()
            pipe.fit(group[self.numeric_cols])
            self.location_models_[sigungu] = pipe
            self.fitted_locations_.append(sigungu)

        if not self.fitted_locations_:
            raise RuntimeError("학습된 시군구가 없습니다. min_samples를 낮추거나 데이터를 확인하세요.")

        print(f"[LocationAnomalyModel] 학습 완료: {len(self.fitted_locations_)}개 시군구")
        if self.skipped_locations_:
            print(f"  제외된 시군구({self.min_samples}건 미만): {self.skipped_locations_}")
        return self

    def _check_trained(self) -> None:
        if not self.location_models_:
            raise RuntimeError("모델이 학습되지 않았습니다. fit_from_dataframe()을 먼저 호출하세요.")

    # ------------------------------------------------------------------ #
    # 탐지                                                                 #
    # ------------------------------------------------------------------ #

    def detect_from_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """원본 데이터프레임에 시군구별 이상 탐지 결과 컬럼을 추가합니다.

        추가 컬럼
        --------
        - 평당가 / 건물연식 / 거래연도 / 거래월 : 파생 피처
        - anomaly_raw_label : 1=정상, -1=특이
        - anomaly_label     : "정상거래" / "특이거래" / "미탐지(학습데이터부족)"
        - anomaly_score     : 값이 작을수록 더 특이 (시군구 내 상대 점수)
        - anomaly_rank      : 시군구 내 특이 순위 (1 = 가장 특이)
        - total_rank        : 전국 기준 순위
        """
        self._check_trained()

        prepared = self.prepare_dataframe(df)
        parts = []

        for sigungu, group in prepared.groupby("시군구"):
            g = group.copy()
            if sigungu not in self.location_models_:
                g["anomaly_raw_label"] = np.nan
                g["anomaly_label"] = "미탐지(학습데이터부족)"
                g["anomaly_score"] = np.nan
                g["anomaly_rank"] = np.nan
                parts.append(g)
                continue

            pipe = self.location_models_[sigungu]
            g["anomaly_raw_label"] = pipe.predict(g[self.numeric_cols])
            g["anomaly_score"] = pipe.decision_function(g[self.numeric_cols])
            g["anomaly_label"] = np.where(g["anomaly_raw_label"] == -1, "특이거래", "정상거래")
            g["anomaly_rank"] = (
                g["anomaly_score"].rank(method="first", ascending=True).astype(int)
            )
            parts.append(g)

        result = pd.concat(parts)

        base = df.loc[result.index].copy()
        for col in ["평당가", "건물연식", "거래연도", "거래월",
                    "anomaly_raw_label", "anomaly_label", "anomaly_score", "anomaly_rank"]:
            base[col] = result[col]

        base["total_rank"] = base["anomaly_score"].rank(method="first", ascending=True)

        return base.sort_values("anomaly_score", ascending=True)

    # ------------------------------------------------------------------ #
    # 요약 / TOP N                                                         #
    # ------------------------------------------------------------------ #

    def top_anomalies_top1_per_region(self, df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
        """시군구별 top 1 이상치를 도출한 뒤, 전체 중 top n을 반환합니다.

        Steps
        -----
        1. 246개 시군구별로 가장 특이한 거래 1건씩 추출 (최대 246행)
        2. 해당 246건 중 anomaly_score 오름차순으로 상위 n건 반환
        """
        result = self.detect_from_dataframe(df)
        anomalies = result[result["anomaly_raw_label"] == -1]

        top1_per_region = (
            anomalies
            .sort_values("anomaly_score", ascending=True)
            .groupby("시군구", group_keys=False)
            .head(1)
        )

        return top1_per_region.sort_values("anomaly_score", ascending=True).head(n).reset_index(drop=True)

    def summarize_by_location(self, df: pd.DataFrame) -> pd.DataFrame:
        """시군구별 이상 거래 요약 테이블을 반환합니다.

        컬럼
        ----
        총거래수, 특이거래수, 특이거래비율(%), 정상거래_평균평당가,
        특이거래_평균평당가, 평균평당가_차이, 특이거래_평균거래금액,
        특이거래_최저score
        """
        result = self.detect_from_dataframe(df)
        result = result[result["anomaly_raw_label"].notna()]

        rows = []
        for sigungu, group in result.groupby("시군구"):
            anomaly = group[group["anomaly_raw_label"] == -1]
            normal = group[group["anomaly_raw_label"] == 1]
            total = len(group)
            ac = len(anomaly)

            rows.append({
                "시군구": sigungu,
                "총거래수": total,
                "특이거래수": ac,
                "특이거래비율(%)": round(ac / total * 100, 2) if total else 0.0,
                "정상거래_평균평당가": round(normal["평당가"].mean(), 0) if len(normal) else np.nan,
                "특이거래_평균평당가": round(anomaly["평당가"].mean(), 0) if ac else np.nan,
                "평균평당가_차이": round(
                    anomaly["평당가"].mean() - normal["평당가"].mean(), 0
                ) if (ac and len(normal)) else np.nan,
                "특이거래_평균거래금액": round(anomaly["거래금액"].mean(), 0) if ac else np.nan,
                "특이거래_최저score": round(anomaly["anomaly_score"].min(), 4) if ac else np.nan,
            })

        summary = pd.DataFrame(rows).set_index("시군구")
        return summary.sort_values("특이거래비율(%)", ascending=False)

    # ------------------------------------------------------------------ #
    # 저장 / 불러오기                                                      #
    # ------------------------------------------------------------------ #

    def save(self, path: str) -> None:
        self._check_trained()
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "LocationAnomalyModel":
        loaded = joblib.load(path)
        if not isinstance(loaded, cls):
            raise TypeError("불러온 객체가 LocationAnomalyModel이 아닙니다.")
        return loaded


# ------------------------------------------------------------------ #
# 직접 실행                                                            #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    import time

    t = time.time()
    df = pd.read_csv("data/raw/Apart Deal_6.csv", encoding="utf-8", low_memory=False)
    df["거래금액"] = df["거래금액"].astype(str).str.replace(",", "", regex=False).astype(float)
    df = df.drop_duplicates(keep="first").reset_index(drop=True)
    print(f"로드 완료: {len(df):,}건 ({time.time() - t:.2f}초)")

    model = LocationAnomalyModel(contamination=0.03, random_state=42)
    model.fit_from_dataframe(df)

    print("\n--- 시군구별 TOP 1 이상치 → 전체 TOP 10 특이 거래 ---")
    top = model.top_anomalies_top1_per_region(df, n=10)
    cols = ["시군구", "아파트", "거래금액", "평당가", "anomaly_score", "anomaly_rank"]
    available = [c for c in cols if c in top.columns]
    print(top[available].to_string(index=False))
