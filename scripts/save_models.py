"""
scripts/save_models.py — 학습된 모델을 pkl 파일로 저장

실행 방법:
    python scripts/save_models.py

저장 위치:
    data/models/brand_grade_classifier.pkl
    data/models/torch_kmeans_clustering.pkl
"""

import sys
import time
from pathlib import Path

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

    print("\n" + "=" * 55)
    print("  모델 저장 완료")
    print("=" * 55)
    print("\n저장된 파일:")
    for f in sorted(MODELS_DIR.glob("*.pkl")):
        size_mb = f.stat().st_size / (1024 ** 2)
        print(f"  {f.name:<40} {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
