"""
models/clustering/torch_kmeans_models.py — PyTorch GPU KMeans 군집화 모델

sklearn MiniBatchKMeans와 달리 전체 데이터를 GPU에 올려 완전한 수렴을 보장합니다.
K-Means++ 초기화 대신 랜덤 초기화를 사용하여 이전 hanging 문제를 회피합니다.
VRAM 초과 방지를 위해 거리 계산은 배치 단위로 처리합니다.

사용 예시
---------
    from utils.db import load_apart_deals
    from models.clustering.torch_kmeans_models import TorchKMeansLocationClusterModel

    df = load_apart_deals()
    model = TorchKMeansLocationClusterModel(n_clusters=7)
    model.fit_from_dataframe(df)

    print(model.metrics_)
    print(model.summarize_clusters(df))
"""

from __future__ import annotations

from typing import Optional, Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from models.base import BaseClusterModel


def _clustering_metrics(X: np.ndarray, labels: np.ndarray) -> dict:
    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

    n_clusters = int(len(set(labels)))
    metrics = {
        "n_samples":         int(len(labels)),
        "n_clusters":        n_clusters,
        "Silhouette":        np.nan,
        "Davies-Bouldin":    np.nan,
        "Calinski-Harabasz": np.nan,
    }
    if n_clusters >= 2 and len(X) > n_clusters:
        sample_size = min(50_000, len(X))
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X), sample_size, replace=False)
        metrics["Silhouette"]        = float(silhouette_score(X[idx], labels[idx]))
        metrics["Davies-Bouldin"]    = float(davies_bouldin_score(X[idx], labels[idx]))
        metrics["Calinski-Harabasz"] = float(calinski_harabasz_score(X[idx], labels[idx]))
    return metrics


def _assign_labels(X: torch.Tensor, centroids: torch.Tensor, batch_size: int = 1_000_000) -> torch.Tensor:
    """거리 계산 후 레이블 할당.

    VRAM이 충분하면 전체를 한 번에 처리하고, OOM 발생 시 배치로 자동 폴백합니다.
    500만×7 거리 행렬 ≈ 140MB — RTX 5060 8GB에서는 전체 처리 가능.
    """
    try:
        # (n, 1, d) - (1, k, d) → (n, k) — 전체 한 번에
        diff = X.unsqueeze(1) - centroids.unsqueeze(0)
        dists = (diff ** 2).sum(dim=2)
        return dists.argmin(dim=1)
    except torch.cuda.OutOfMemoryError:
        # VRAM 부족 시 배치 폴백
        labels = torch.empty(len(X), dtype=torch.long, device=X.device)
        for start in range(0, len(X), batch_size):
            end = min(start + batch_size, len(X))
            diff = X[start:end].unsqueeze(1) - centroids.unsqueeze(0)
            dists = (diff ** 2).sum(dim=2)
            labels[start:end] = dists.argmin(dim=1)
        return labels


class TorchKMeansLocationClusterModel(BaseClusterModel):
    """PyTorch GPU 기반 전체 배치 KMeans 군집화 모델.

    전체 데이터를 GPU에 올려 완전한 수렴을 보장합니다.
    랜덤 초기화 + n_init 반복으로 초기화 운을 줄입니다.

    파생 피처:
        평당가     = log1p(거래금액 / (전용면적 / 3.3))
        건물연식   = current_year - 건축년도
        거래활성도 = 아파트별 거래건수 / 전체 거래건수
    """

    def __init__(
        self,
        n_clusters: int = 7,
        max_iter: int = 300,
        n_init: int = 10,
        tol: float = 1e-4,
        current_year: int = 2026,
        random_state: int = 42,
        feature_cols: Optional[list[str]] = None,
        feature_weights: Optional[dict[str, float]] = None,
        device: Optional[str] = None,
    ):
        super().__init__(name="TorchKMeans(GPU)")
        self.n_clusters    = n_clusters
        self.max_iter      = max_iter
        self.n_init        = n_init
        self.tol           = tol
        self.current_year  = current_year
        self.random_state  = random_state
        self.feature_cols  = feature_cols or [
            "위도", "경도", "평당가", "건물연식", "거래활성도",
        ]
        self.feature_weights = feature_weights or {"위도": 5.0, "경도": 5.0}
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._scaler: Optional[StandardScaler] = None
        self.centroids_: Optional[np.ndarray] = None
        self.metrics_: Optional[dict] = None
        self.apt_activity_map_: Optional[pd.Series] = None

    @staticmethod
    def _clean_numeric(series: pd.Series) -> pd.Series:
        return pd.to_numeric(
            series.astype(str).str.strip().str.replace(",", "", regex=False),
            errors="coerce",
        )

    def _validate_columns(self, df: pd.DataFrame, required: Iterable[str]) -> None:
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"필수 컬럼 없음: {missing}")

    def prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        required = ["위도", "경도", "거래금액", "전용면적", "건축년도", "아파트"]
        self._validate_columns(df, required)

        extra = ["인근학교수", "인근역수", "세대수", "시군구", "법정동", "지역코드"]
        use_cols = list(dict.fromkeys(required + [c for c in extra if c in df.columns]))
        data = df[use_cols].copy()

        for col in ["위도", "경도", "거래금액", "전용면적", "건축년도",
                    "인근학교수", "인근역수", "세대수"]:
            if col not in data.columns:
                data[col] = 0
            data[col] = self._clean_numeric(data[col])

        data["평당가"]   = np.log1p(data["거래금액"] / (data["전용면적"] / 3.3))
        data["건물연식"] = self.current_year - data["건축년도"]

        if self.apt_activity_map_ is not None:
            data["거래활성도"] = data["아파트"].map(self.apt_activity_map_).fillna(
                self.apt_activity_map_.median()
            )
        else:
            apt_counts = data["아파트"].map(data["아파트"].value_counts())
            data["거래활성도"] = apt_counts / len(data)

        data = data.replace([np.inf, -np.inf], np.nan)
        data = data.dropna(subset=self.feature_cols)
        data = data[(data["전용면적"] > 0) & (data["거래금액"] > 0)]
        return data

    def _apply_weights(self, X_scaled: np.ndarray) -> np.ndarray:
        result = X_scaled.copy()
        for col, w in self.feature_weights.items():
            if col in self.feature_cols:
                result[:, self.feature_cols.index(col)] *= w
        return result

    def _kmeans_once(self, X: torch.Tensor, seed: int) -> tuple[torch.Tensor, torch.Tensor, float]:
        """랜덤 초기화로 KMeans 1회 실행. (centroids, labels, inertia) 반환."""
        torch.manual_seed(seed)
        perm = torch.randperm(len(X), device=X.device)
        centroids = X[perm[:self.n_clusters]].clone()

        labels = torch.zeros(len(X), dtype=torch.long, device=X.device)
        for i in range(self.max_iter):
            new_labels = _assign_labels(X, centroids)

            # centroid 업데이트
            new_centroids = torch.zeros_like(centroids)
            counts = torch.zeros(self.n_clusters, device=X.device)
            for k in range(self.n_clusters):
                mask = new_labels == k
                if mask.sum() > 0:
                    new_centroids[k] = X[mask].mean(dim=0)
                    counts[k] = mask.sum().float()
                else:
                    # 빈 군집 — 랜덤 재초기화
                    new_centroids[k] = X[torch.randint(len(X), (1,), device=X.device)].squeeze()

            shift = ((new_centroids - centroids) ** 2).sum().sqrt().item()
            centroids = new_centroids
            labels = new_labels

            if (i + 1) % 10 == 0:
                print(f"    iter {i+1:>3} — centroid shift: {shift:.6f}")

            if shift < self.tol:
                print(f"    수렴 (iter={i+1}, shift={shift:.6f})")
                break

        # inertia 계산
        diff = X - centroids[labels]
        inertia = (diff ** 2).sum().item()
        return centroids, labels, inertia

    def fit(self, _X) -> "TorchKMeansLocationClusterModel":
        raise NotImplementedError("fit_from_dataframe()을 사용하세요.")

    def evaluate(self, _X) -> dict:
        if self.centroids_ is None:
            raise RuntimeError(f"[{self.name}] fit_from_dataframe()을 먼저 호출하세요.")
        return self.metrics_ or {}

    def fit_from_dataframe(self, df: pd.DataFrame) -> "TorchKMeansLocationClusterModel":
        data = self.prepare_dataframe(df)

        if "거래활성도" in self.feature_cols:
            self.apt_activity_map_ = (
                data["아파트"].value_counts() / len(data)
            ).rename("거래활성도")

        X_np = data[self.feature_cols].values.astype(np.float32)
        self._scaler = StandardScaler()
        X_scaled = self._apply_weights(self._scaler.fit_transform(X_np)).astype(np.float32)

        print(f"[{self.name}] device={self.device}, n={len(X_scaled):,}, k={self.n_clusters}, n_init={self.n_init}")
        X_gpu = torch.tensor(X_scaled, device=self.device)

        best_inertia = float("inf")
        best_centroids = None
        best_labels = None

        for i in range(self.n_init):
            print(f"  [init {i+1}/{self.n_init}]")
            centroids, labels, inertia = self._kmeans_once(X_gpu, seed=self.random_state + i)
            print(f"  → inertia={inertia:.2f}")
            if inertia < best_inertia:
                best_inertia = inertia
                best_centroids = centroids
                best_labels = labels

        self.centroids_ = best_centroids.cpu().numpy()
        self._labels = best_labels.cpu().numpy()
        self.metrics_ = _clustering_metrics(X_scaled, self._labels)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.centroids_ is None:
            raise RuntimeError(f"[{self.name}] fit_from_dataframe()을 먼저 호출하세요.")
        data = self.prepare_dataframe(df)
        X_np = data[self.feature_cols].values.astype(np.float32)
        X_scaled = self._apply_weights(self._scaler.transform(X_np)).astype(np.float32)
        X_gpu = torch.tensor(X_scaled, device=self.device)
        centroids_gpu = torch.tensor(self.centroids_, device=self.device)
        labels = _assign_labels(X_gpu, centroids_gpu)
        return labels.cpu().numpy()

    def summarize_clusters(self, df: pd.DataFrame, label_col: str = "cluster") -> pd.DataFrame:
        if self.centroids_ is None:
            raise RuntimeError(f"[{self.name}] fit_from_dataframe()을 먼저 호출하세요.")
        data = self.prepare_dataframe(df)
        labels = self.predict(df)
        data = data.iloc[:len(labels)].copy()
        data[label_col] = labels
        data["평당가_원단위"] = np.expm1(data["평당가"])

        summary = data.groupby(label_col).agg(
            거래건수      =("거래금액",      "count"),
            평균거래금액  =("거래금액",      "mean"),
            중앙거래금액  =("거래금액",      "median"),
            평균평당가    =("평당가_원단위", "mean"),
            중앙평당가    =("평당가_원단위", "median"),
            평균전용면적  =("전용면적",      "mean"),
            평균건물연식  =("건물연식",      "mean"),
            평균인근학교수=("인근학교수",    "mean"),
            평균인근역수  =("인근역수",      "mean"),
            평균세대수    =("세대수",        "mean"),
            중심위도      =("위도",          "mean"),
            중심경도      =("경도",          "mean"),
        )
        return summary.reset_index().sort_values("거래건수", ascending=False)

    def get_params(self) -> dict:
        return {
            "n_clusters":      self.n_clusters,
            "max_iter":        self.max_iter,
            "n_init":          self.n_init,
            "tol":             self.tol,
            "feature_cols":    self.feature_cols,
            "feature_weights": self.feature_weights,
            "device":          self.device,
        }
