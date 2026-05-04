"""
scripts/save_models.py — 학습된 모델을 pkl 파일로 저장

실행 방법:
    python scripts/save_models.py

저장 위치:
    data/models/brand_grade_classifier.pkl
    data/models/torch_kmeans_clustering.pkl
    data/models/premium_analysis_results.pkl
"""

import sys
import time
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.db import load_apart_deals

MODELS_DIR = Path(__file__).parent.parent / "data" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def save_classification_model(df):
    from models.classification.brand_grade_classifier import BrandGradeClassifier

    print("\n[1/2] 브랜드 등급 분류 모델 학습 중...")
    t0 = time.time()
    model = BrandGradeClassifier()
    metrics = model.fit_from_dataframe(df)
    elapsed = time.time() - t0

    print(f"  학습 시간 : {elapsed:.1f}초")
    for k, v in metrics.items():
        print(f"  {k:<10}: {v}")

    path = MODELS_DIR / "brand_grade_classifier.pkl"
    model.save(str(path))
    print(f"  저장 완료 : {path}")
    return model


def save_clustering_model(df):
    from models.clustering.torch_kmeans_models import TorchKMeansLocationClusterModel

    print("\n[2/2] 지역 군집화 모델 학습 중...")
    t0 = time.time()
    model = TorchKMeansLocationClusterModel(n_clusters=7)
    model.fit_from_dataframe(df)
    elapsed = time.time() - t0

    print(f"  학습 시간 : {elapsed:.1f}초")
    for k, v in (model.metrics_ or {}).items():
        val = f"{v:,.0f}" if isinstance(v, int) else f"{v:.4f}" if isinstance(v, float) else str(v)
        print(f"  {k:<25}: {val}")

    path = MODELS_DIR / "torch_kmeans_clustering.pkl"
    model.save(str(path))
    print(f"  저장 완료 : {path}")
    return model


def save_premium_analysis(df):
    from models.regression.price_regression_models import XGBoostPriceModel
    from models.regression.price_premium_analyzer import PricePremiumAnalyzer

    PREMIUM_NUMERIC_COLS = [
        "전용면적", "층", "건물연식", "기준금리", "세대수", "거래연도", "거래월",
    ]
    GRADE_ORDER = ["큰 할인", "할인", "보통", "프리미엄", "고프리미엄"]

    print("\n[3/3] 저·고평가 프리미엄 분석 학습 및 결과 저장 중...")
    t0 = time.time()

    price_model = XGBoostPriceModel(
        sample_size=200_000,
        random_state=42,
        numeric_cols=PREMIUM_NUMERIC_COLS,
    )
    price_model.fit_from_dataframe(df)

    analyzer = PricePremiumAnalyzer(price_model=price_model)
    premium_df = analyzer.analyze(df)

    metrics = analyzer.evaluate_price_model(premium_df)
    sigungu_df = analyzer.summarize_by_group(premium_df, "시군구", min_count=500)

    group_summaries = {}
    for col in ["역세권여부", "학세권여부", "브랜드구분"]:
        if col in premium_df.columns:
            s = analyzer.summarize_by_group(premium_df, col, min_count=100)
            if len(s) >= 2:
                group_summaries[col] = s

    grade_counts = {
        g: int((premium_df["프리미엄등급"] == g).sum()) for g in GRADE_ORDER
    }

    scatter_cols = ["거래금액", "예측거래금액", "프리미엄률"]
    if "시군구" in premium_df.columns:
        scatter_cols.append("시군구")
    scatter_sample = (
        premium_df[scatter_cols]
        .sample(min(4000, len(premium_df)), random_state=42)
        .reset_index(drop=True)
    )

    results = {
        "metrics":        metrics,
        "sigungu_df":     sigungu_df,
        "group_summaries": group_summaries,
        "grade_counts":   grade_counts,
        "scatter_sample": scatter_sample,
    }

    path = MODELS_DIR / "premium_analysis_results.pkl"
    joblib.dump(results, path)
    print(f"  학습+분석 시간 : {time.time() - t0:.1f}초")
    print(f"  저장 완료      : {path}")
    return results


def main():
    print("=" * 55)
    print("  모델 저장 스크립트")
    print(f"  저장 위치: {MODELS_DIR}")
    print("=" * 55)

    print("\n데이터 로딩 중...")
    t0 = time.time()
    df = load_apart_deals()
    print(f"  로딩 완료 : {len(df):,}건 ({time.time() - t0:.1f}초)")

    save_classification_model(df)
    save_clustering_model(df)
    save_premium_analysis(df)

    print("\n" + "=" * 55)
    print("  모델 저장 완료")
    print("=" * 55)
    print("\n저장된 파일:")
    for f in sorted(MODELS_DIR.glob("*.pkl")):
        size_mb = f.stat().st_size / (1024 ** 2)
        print(f"  {f.name:<40} {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
