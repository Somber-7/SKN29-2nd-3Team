"""
가격 프리미엄 분석 모델.

회귀 모델이 예측한 기준 가격과 실제 거래금액의 차이를 이용해
브랜드, 역세권, 학세권, 대단지 여부에 따른 가격 프리미엄을 분석합니다.

권장 위치
---------
models/analysis/price_premium_analyzer.py

사용 예시
---------
import pandas as pd
from models.regression.price_regression_models import XGBoostPriceModel
from models.analysis.price_premium_analyzer import PricePremiumAnalyzer

# 1. 데이터 로드
df = pd.read_csv("Apart Deal_6.csv", encoding="cp949")

# 2. 기준 가격 예측 모델 학습
price_model = XGBoostPriceModel(sample_size=100_000, random_state=42)
price_model.fit_from_dataframe(df)

# 3. 프리미엄 분석
analyzer = PricePremiumAnalyzer(price_model=price_model)
premium_df = analyzer.analyze(df)

# 4. 요약 결과 확인
print(analyzer.summarize_by_group(premium_df, "역세권여부"))
print(analyzer.summarize_by_group(premium_df, "학세권여부"))
print(analyzer.summarize_by_group(premium_df, "대단지여부"))
print(analyzer.summarize_by_group(premium_df, "브랜드여부"))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@dataclass
class PricePremiumAnalyzer:
    """가격 프리미엄 분석 클래스.

    이 클래스는 별도의 예측 모델이라기보다는, 이미 학습된 거래금액 회귀 모델을 이용해
    실제 거래금액과 예측 거래금액의 차이를 계산하는 분석 도구입니다.

    계산되는 핵심 컬럼
    ----------------
    - 예측거래금액: 회귀 모델이 예측한 기준 가격
    - 프리미엄금액: 실제 거래금액 - 예측거래금액
    - 프리미엄률: 프리미엄금액 / 예측거래금액
    - 프리미엄등급: 큰 할인 / 할인 / 보통 / 프리미엄 / 고프리미엄

    해석 예시
    --------
    프리미엄률이 0.10이면, 모델이 예측한 기준 가격보다 실제 거래금액이 약 10% 높다는 뜻입니다.
    프리미엄률이 -0.08이면, 모델이 예측한 기준 가격보다 실제 거래금액이 약 8% 낮다는 뜻입니다.
    """

    price_model: object
    target_col: str = "거래금액"
    area_col: str = "전용면적"
    station_count_col: str = "인근역수"
    school_count_col: str = "인근학교수"
    household_col: str = "세대수"
    brand_col: str = "브랜드여부"
    station_threshold: int = 1
    school_threshold: int = 1
    large_complex_threshold: int = 1000
    premium_bins: list[float] = field(
        default_factory=lambda: [-np.inf, -0.15, -0.05, 0.05, 0.15, np.inf]
    )
    premium_labels: list[str] = field(
        default_factory=lambda: ["큰 할인", "할인", "보통", "프리미엄", "고프리미엄"]
    )

    @staticmethod
    def _clean_numeric(series: pd.Series) -> pd.Series:
        """쉼표가 포함된 문자열 숫자를 안전하게 숫자형으로 변환합니다."""
        return pd.to_numeric(
            series.astype(str).str.strip().str.replace(",", "", regex=False),
            errors="coerce",
        )

    def _validate_columns(self, df: pd.DataFrame) -> None:
        required = [self.target_col, self.area_col]
        optional = [
            self.station_count_col,
            self.school_count_col,
            self.household_col,
            self.brand_col,
        ]
        missing_required = [col for col in required if col not in df.columns]
        if missing_required:
            raise ValueError(f"필수 컬럼이 없습니다: {missing_required}")

        # 옵션 컬럼은 없으면 자동으로 분석에서 제외합니다.
        self.available_group_cols_ = [col for col in optional if col in df.columns]

    def _predict_price(self, df: pd.DataFrame) -> pd.Series:
        """학습된 회귀 모델로 예측 거래금액을 계산합니다."""
        if hasattr(self.price_model, "predict_series"):
            pred = self.price_model.predict_series(df)
        elif hasattr(self.price_model, "predict_dataframe"):
            pred = self.price_model.predict_dataframe(df)
        elif hasattr(self.price_model, "predict"):
            pred = self.price_model.predict(df)
        else:
            raise TypeError("price_model은 predict(), predict_series(), predict_dataframe() 중 하나를 지원해야 합니다.")

        if isinstance(pred, pd.DataFrame):
            if "예측거래금액" in pred.columns:
                pred = pred["예측거래금액"]
            else:
                pred = pred.iloc[:, 0]

        pred = pd.Series(pred, index=df.index, name="예측거래금액")
        return self._clean_numeric(pred)

    def _add_basic_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """프리미엄 분석에 필요한 파생 컬럼을 추가합니다."""
        data[self.target_col] = self._clean_numeric(data[self.target_col])
        data[self.area_col] = self._clean_numeric(data[self.area_col])

        data["평당가"] = data[self.target_col] / (data[self.area_col] / 3.3)

        if self.station_count_col in data.columns:
            data[self.station_count_col] = self._clean_numeric(data[self.station_count_col])
            data["역세권여부"] = np.where(
                data[self.station_count_col] >= self.station_threshold,
                "역세권",
                "비역세권",
            )

        if self.school_count_col in data.columns:
            data[self.school_count_col] = self._clean_numeric(data[self.school_count_col])
            data["학세권여부"] = np.where(
                data[self.school_count_col] >= self.school_threshold,
                "학세권",
                "비학세권",
            )

        if self.household_col in data.columns:
            data[self.household_col] = self._clean_numeric(data[self.household_col])
            data["대단지여부"] = np.where(
                data[self.household_col] >= self.large_complex_threshold,
                "대단지",
                "일반단지",
            )

        if self.brand_col in data.columns:
            data[self.brand_col] = self._clean_numeric(data[self.brand_col]).fillna(0).astype(int)
            data["브랜드구분"] = np.where(
                data[self.brand_col] == 1,
                "브랜드",
                "비브랜드",
            )

        return data

    def analyze(self, df: pd.DataFrame, drop_invalid: bool = True) -> pd.DataFrame:
        """실제 거래금액과 예측 거래금액의 차이를 계산합니다.

        Parameters
        ----------
        df:
            원본 거래 데이터프레임
        drop_invalid:
            True이면 거래금액, 전용면적, 예측거래금액이 결측이거나 0 이하인 행을 제거합니다.

        Returns
        -------
        pd.DataFrame
            원본 데이터에 프리미엄 관련 컬럼이 추가된 데이터프레임
        """
        self._validate_columns(df)

        data = df.copy()
        data = self._add_basic_features(data)

        predicted_price = self._predict_price(data)
        data["예측거래금액"] = predicted_price

        if drop_invalid:
            data = data.dropna(subset=[self.target_col, self.area_col, "예측거래금액"])
            data = data[
                (data[self.target_col] > 0)
                & (data[self.area_col] > 0)
                & (data["예측거래금액"] > 0)
            ]

        data["프리미엄금액"] = data[self.target_col] - data["예측거래금액"]
        data["프리미엄률"] = data["프리미엄금액"] / data["예측거래금액"]
        data["절대오차"] = (data[self.target_col] - data["예측거래금액"]).abs()
        data["오차율"] = data["절대오차"] / data[self.target_col]

        data["프리미엄등급"] = pd.cut(
            data["프리미엄률"],
            bins=self.premium_bins,
            labels=self.premium_labels,
        )

        return data

    def evaluate_price_model(self, premium_df: pd.DataFrame) -> dict[str, float]:
        """프리미엄 분석에 사용한 회귀 모델의 예측 성능을 계산합니다."""
        required = [self.target_col, "예측거래금액"]
        missing = [col for col in required if col not in premium_df.columns]
        if missing:
            raise ValueError(f"필수 컬럼이 없습니다: {missing}")

        y_true = premium_df[self.target_col]
        y_pred = premium_df["예측거래금액"]

        mse = mean_squared_error(y_true, y_pred)
        return {
            "MAE": float(mean_absolute_error(y_true, y_pred)),
            "MSE": float(mse),
            "RMSE": float(np.sqrt(mse)),
            "MAPE": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
            "R2": float(r2_score(y_true, y_pred)),
            "rows": int(len(premium_df)),
        }

    def summarize_by_group(
        self,
        premium_df: pd.DataFrame,
        group_col: str,
        min_count: int = 30,
        sort_by: str = "평균프리미엄률",
        ascending: bool = False,
    ) -> pd.DataFrame:
        """그룹별 평균 프리미엄을 요약합니다.

        Parameters
        ----------
        premium_df:
            analyze() 결과 데이터프레임
        group_col:
            요약 기준 컬럼. 예: "역세권여부", "학세권여부", "대단지여부", "브랜드구분", "시군구"
        min_count:
            거래 건수가 너무 적은 그룹을 제외하기 위한 최소 건수
        sort_by:
            정렬 기준 컬럼
        ascending:
            오름차순 정렬 여부
        """
        required = [group_col, self.target_col, "예측거래금액", "프리미엄금액", "프리미엄률"]
        missing = [col for col in required if col not in premium_df.columns]
        if missing:
            raise ValueError(f"필수 컬럼이 없습니다: {missing}")

        summary = (
            premium_df.groupby(group_col, dropna=False)
            .agg(
                거래건수=(self.target_col, "size"),
                평균거래금액=(self.target_col, "mean"),
                평균예측거래금액=("예측거래금액", "mean"),
                평균프리미엄금액=("프리미엄금액", "mean"),
                중앙값프리미엄금액=("프리미엄금액", "median"),
                평균프리미엄률=("프리미엄률", "mean"),
                중앙값프리미엄률=("프리미엄률", "median"),
                평균평당가=("평당가", "mean"),
            )
            .reset_index()
        )

        summary = summary[summary["거래건수"] >= min_count]
        if sort_by in summary.columns:
            summary = summary.sort_values(sort_by, ascending=ascending)

        return summary.reset_index(drop=True)

    def compare_binary_groups(
        self,
        premium_df: pd.DataFrame,
        group_col: str,
        positive_label: str,
        negative_label: str,
    ) -> dict[str, float | str | int]:
        """두 그룹의 평균 프리미엄 차이를 비교합니다.

        예시
        ----
        compare_binary_groups(premium_df, "역세권여부", "역세권", "비역세권")
        """
        if group_col not in premium_df.columns:
            raise ValueError(f"'{group_col}' 컬럼이 없습니다.")

        pos = premium_df[premium_df[group_col] == positive_label]
        neg = premium_df[premium_df[group_col] == negative_label]

        if len(pos) == 0 or len(neg) == 0:
            raise ValueError("비교할 두 그룹 중 하나가 비어 있습니다.")

        pos_mean = pos["프리미엄률"].mean()
        neg_mean = neg["프리미엄률"].mean()

        return {
            "group_col": group_col,
            "positive_label": positive_label,
            "negative_label": negative_label,
            "positive_count": int(len(pos)),
            "negative_count": int(len(neg)),
            "positive_mean_premium_rate": float(pos_mean),
            "negative_mean_premium_rate": float(neg_mean),
            "premium_rate_gap": float(pos_mean - neg_mean),
            "positive_mean_premium_amount": float(pos["프리미엄금액"].mean()),
            "negative_mean_premium_amount": float(neg["프리미엄금액"].mean()),
            "premium_amount_gap": float(pos["프리미엄금액"].mean() - neg["프리미엄금액"].mean()),
        }

    def top_premium_transactions(
        self,
        premium_df: pd.DataFrame,
        n: int = 30,
        positive: bool = True,
    ) -> pd.DataFrame:
        """프리미엄률 기준 상위 또는 하위 거래를 반환합니다."""
        sort_col = "프리미엄률"
        result = premium_df.sort_values(sort_col, ascending=not positive).head(n)
        return result.reset_index(drop=True)

    def premium_pivot_by_region(
        self,
        premium_df: pd.DataFrame,
        region_col: str = "시군구",
        group_col: str = "브랜드구분",
        min_count: int = 30,
    ) -> pd.DataFrame:
        """지역 x 그룹 기준으로 평균 프리미엄률 피벗 테이블을 만듭니다."""
        required = [region_col, group_col, "프리미엄률", self.target_col]
        missing = [col for col in required if col not in premium_df.columns]
        if missing:
            raise ValueError(f"필수 컬럼이 없습니다: {missing}")

        grouped = (
            premium_df.groupby([region_col, group_col])
            .agg(
                거래건수=(self.target_col, "size"),
                평균프리미엄률=("프리미엄률", "mean"),
                평균프리미엄금액=("프리미엄금액", "mean"),
            )
            .reset_index()
        )
        grouped = grouped[grouped["거래건수"] >= min_count]

        pivot = grouped.pivot(index=region_col, columns=group_col, values="평균프리미엄률")
        return pivot.reset_index()

    def save(self, path: str) -> None:
        """분석기 객체를 저장합니다."""
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "PricePremiumAnalyzer":
        """저장된 분석기 객체를 불러옵니다."""
        loaded = joblib.load(path)
        if not isinstance(loaded, cls):
            raise TypeError("불러온 객체가 PricePremiumAnalyzer가 아닙니다.")
        return loaded


if __name__ == "__main__":
    # 실행 예시는 프로젝트 환경에 맞게 수정해서 사용하세요.
    import pandas as pd

    try:
        from models.regression.price_regression_models import XGBoostPriceModel
    except ModuleNotFoundError:
        from price_regression_models import XGBoostPriceModel

    df = pd.read_csv("Apart Deal_6.csv", encoding="cp949")

    price_model = XGBoostPriceModel(sample_size=100_000, random_state=42)
    price_model.fit_from_dataframe(df)

    analyzer = PricePremiumAnalyzer(price_model=price_model)
    premium_df = analyzer.analyze(df)

    print(analyzer.evaluate_price_model(premium_df))

    for col in ["역세권여부", "학세권여부", "대단지여부", "브랜드구분", "시군구"]:
        if col in premium_df.columns:
            print(f"\n[{col}별 프리미엄 요약]")
            print(analyzer.summarize_by_group(premium_df, col).head(20))
