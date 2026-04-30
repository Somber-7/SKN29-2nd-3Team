"""
scripts/test_clustering.py — 군집화 모델 학습 및 평가 스크립트

실행 방법:
    python scripts/test_clustering.py

결과는 logs/YYYYMMDD_HHMMSS_clustering_results.md 로 저장됩니다.
"""

import time
from _common import Logger, section, load_data, log_header, LOG_DIR


def test_clustering(logger, df, n_clusters=5):
    section(logger, f"군집화 모델 — KMeansLocationClusterModel (MiniBatchKMeans, k={n_clusters})")

    from models.clustering.location_cluster_models import KMeansLocationClusterModel

    logger.log(f"  학습 데이터 : {len(df):,}건")

    model = KMeansLocationClusterModel(n_clusters=n_clusters)
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


def test_find_best_k(logger, df, k_values=range(2, 11)):
    section(logger, "최적 k 탐색 (Elbow)")

    from models.clustering.location_cluster_models import find_best_k

    logger.log(f"  탐색 범위 : k={min(k_values)}~{max(k_values)}")
    result_df = find_best_k(df, k_values=k_values)

    logger.log()
    logger.log("  [ k별 평가 지표 ]")
    logger.log(result_df.to_string(index=False))

    return result_df


def main():
    logger = Logger()
    run_at = log_header(logger)
    log_path = LOG_DIR / f"{run_at.strftime('%Y%m%d_%H%M%S')}_clustering_results.md"

    df = load_data(logger)
    test_clustering(logger, df, n_clusters=7)

    logger.log()
    logger.log("=" * 55)
    logger.log("  군집화 모델 학습 완료")
    logger.log("=" * 55)

    logger.save(log_path)


if __name__ == "__main__":
    main()
