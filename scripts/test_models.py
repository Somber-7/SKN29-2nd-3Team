"""
scripts/test_models.py — 분류 / 군집화 모델 동작 확인 스크립트

실행 방법:
    python scripts/test_models.py

결과는 logs/YYYYMMDD_HHMMSS_model_results.md 로 저장됩니다.
"""

import sys
import time
from datetime import datetime
from pathlib import Path
from io import StringIO

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from utils.db import load_apart_deals

SAMPLE = None  # 전체 500만 건 사용

LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)


class Logger:
    """터미널 출력과 파일 기록을 동시에 처리합니다."""

    def __init__(self):
        self._buf = StringIO()

    def log(self, msg: str = ""):
        print(msg)
        self._buf.write(msg + "\n")

    def save(self, path: Path):
        path.write_text(self._buf.getvalue(), encoding="utf-8")
        print(f"\n결과 저장 완료: {path}")


logger = Logger()


def section(title: str):
    logger.log()
    logger.log("=" * 55)
    logger.log(f"  {title}")
    logger.log("=" * 55)


def test_classification(df):
    section("분류 모델 — BrandGradeClassifier (XGBoost GPU)")

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


def test_clustering(df):
    section("군집화 모델 — KMeansLocationClusterModel (MiniBatchKMeans)")

    from models.clustering.location_cluster_models import KMeansLocationClusterModel

    logger.log(f"  학습 데이터 : {len(df):,}건")

    model = KMeansLocationClusterModel(n_clusters=5)
    t0 = time.time()
    model.fit_from_dataframe(df)
    elapsed = time.time() - t0

    logger.log(f"  학습 시간   : {elapsed:.1f}초")
    logger.log()
    logger.log("  [ 평가 지표 ]")
    for k, v in (model.metrics_ or {}).items():
        val = f"{v:,.0f}" if isinstance(v, int) else f"{v:.4f}" if isinstance(v, float) else str(v)
        logger.log(f"    {k:<25}: {val}")

    logger.log()
    logger.log("  [ 군집별 요약 ]")
    summary = model.summarize_clusters(df)
    logger.log(summary.to_string(index=False))

    return model


def main():
    run_at = datetime.now()
    log_path = LOG_DIR / f"{run_at.strftime('%Y%m%d_%H%M%S')}_model_results.md"

    logger.log(f"# 모델 학습 결과")
    logger.log(f"- 실행 시각 : {run_at.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.log(f"- GPU       : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A (CPU)'}")
    logger.log(f"- 데이터    : {'전체' if SAMPLE is None else f'{SAMPLE:,}건 샘플'}")

    section("데이터 로딩")
    t0 = time.time()
    df = load_apart_deals(limit=SAMPLE)
    logger.log(f"  로딩 완료 : {len(df):,}건 ({time.time() - t0:.1f}초)")

    test_classification(df)
    test_clustering(df)

    logger.log()
    logger.log("=" * 55)
    logger.log("  모든 모델 학습 완료")
    logger.log("=" * 55)

    logger.save(log_path)


if __name__ == "__main__":
    main()
