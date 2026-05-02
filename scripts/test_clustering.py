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


def test_clustering_pca(logger, df, n_clusters=7, n_components=3):
    section(logger, f"군집화 모델 — KMeansLocationClusterModel (PCA={n_components}, k={n_clusters})")

    from models.clustering.location_cluster_models import KMeansLocationClusterModel

    logger.log(f"  학습 데이터 : {len(df):,}건")

    model = KMeansLocationClusterModel(n_clusters=n_clusters, n_components=n_components)
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


def test_clustering_by_year(logger, df, n_clusters=7, years=range(2015, 2024)):
    section(logger, f"군집화 모델 — 연도별 KMeans (k={n_clusters}, {min(years)}~{max(years)})")

    from models.clustering.location_cluster_models import KMeansLocationClusterModel
    import pandas as pd

    # 거래일 컬럼에서 연도 파싱
    df = df.copy()
    df["거래년도"] = pd.to_datetime(df["거래일"], errors="coerce").dt.year

    rows = []
    for year in years:
        df_year = df[df["거래년도"] == year]
        if len(df_year) < 1000:
            continue

        model = KMeansLocationClusterModel(n_clusters=n_clusters)
        model.fit_from_dataframe(df_year)
        m = model.metrics_ or {}
        rows.append({
            "연도": year,
            "거래건수": len(df_year),
            "Silhouette": round(m.get("Silhouette", float("nan")), 4),
            "Davies-Bouldin": round(m.get("Davies-Bouldin", float("nan")), 4),
            "Calinski-Harabasz": round(m.get("Calinski-Harabasz", float("nan")), 1),
        })
        logger.log(f"  {year}년 — {len(df_year):,}건 | Silhouette={m.get('Silhouette', float('nan')):.4f} | DB={m.get('Davies-Bouldin', float('nan')):.4f}")

    logger.log()
    logger.log("  [ 연도별 요약 ]")
    logger.log(pd.DataFrame(rows).to_string(index=False))

    return rows


def test_torch_kmeans(logger, df, n_clusters=7, feature_weights=None, n_init=5, label=""):
    tag = f" [{label}]" if label else ""
    weights = feature_weights or {"위도": 3.0, "경도": 3.0}
    section(logger, f"TorchKMeans GPU — k={n_clusters}, weights={weights}, n_init={n_init}{tag}")

    from models.clustering.torch_kmeans_models import TorchKMeansLocationClusterModel

    logger.log(f"  학습 데이터 : {len(df):,}건")

    model = TorchKMeansLocationClusterModel(
        n_clusters=n_clusters,
        feature_weights=weights,
        n_init=n_init,
    )
    logger.log(f"  device      : {model.device}")

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

    # 거래활성도 가중치 탐색 — k=7, weights×5, n_init=10 고정
    for w_act in [1.5, 2.0, 3.0, 5.0]:
        test_torch_kmeans(logger, df, n_clusters=7,
                          feature_weights={"위도": 5.0, "경도": 5.0, "거래활성도": float(w_act)}, n_init=10,
                          label=f"활성도×{w_act}")

    logger.log()
    logger.log("=" * 55)
    logger.log("  군집화 모델 학습 완료")
    logger.log("=" * 55)

    logger.save(log_path)


if __name__ == "__main__":
    main()
