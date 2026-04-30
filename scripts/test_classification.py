"""
scripts/test_classification.py — 분류 모델 학습 및 평가 스크립트

실행 방법:
    python scripts/test_classification.py

결과는 logs/YYYYMMDD_HHMMSS_classification_results.md 로 저장됩니다.
"""

import time
import torch
from _common import Logger, section, load_data, log_header, LOG_DIR


def test_classification(logger, df):
    section(logger, "분류 모델 — BrandGradeClassifier (XGBoost GPU)")

    from models.classification.brand_grade_classifier import BrandGradeClassifier

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.log(f"  device      : {device}")
    logger.log(f"  학습 데이터 : {len(df):,}건")

    model = BrandGradeClassifier()
    t0 = time.time()
    metrics = model.fit_from_dataframe(df)
    elapsed = time.time() - t0

    logger.log(f"  학습 시간   : {elapsed:.1f}초")
    logger.log()
    logger.log("  [ 평가 지표 ]")
    for k, v in metrics.items():
        val = f"{v:,.0f}" if isinstance(v, int) else f"{v:.4f}"
        logger.log(f"    {k:<15}: {val}")

    logger.log()
    logger.log("  [ 피처 중요도 Top 15 ]")
    imp_df = model.get_feature_importance_df(top_n=15)
    for _, row in imp_df.iterrows():
        logger.log(f"    {row['feature']:<35} {row['importance']:.4f}")

    logger.log()
    logger.log("  [ 클래스별 분류 리포트 ]")
    report_df = model.classification_report_dataframe(df.sample(min(50_000, len(df)), random_state=42))
    logger.log(report_df.to_string())

    return model


def main():
    logger = Logger()
    run_at = log_header(logger)
    log_path = LOG_DIR / f"{run_at.strftime('%Y%m%d_%H%M%S')}_classification_results.md"

    df = load_data(logger)
    test_classification(logger, df)

    logger.log()
    logger.log("=" * 55)
    logger.log("  분류 모델 학습 완료")
    logger.log("=" * 55)

    logger.save(log_path)


if __name__ == "__main__":
    main()
