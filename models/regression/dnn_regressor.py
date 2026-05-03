
# models/regression/dnn_regressor.py
# models/regression/dnn_regressor.py
from __future__ import annotations

import time
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from models.regression.price_regression_models import ApartmentPriceRegressionBase


class _DNNNet(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: int, neurons: int,
                 dropout: float, use_bn: bool):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(in_dim, neurons))
            if use_bn:
                layers.append(nn.BatchNorm1d(neurons))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = neurons
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class DNNRegressorModel(ApartmentPriceRegressionBase):
    """PyTorch DNN 회귀 모델 — 거래금액 예측."""

    def __init__(
        self,
        hidden_layers: int = 3,
        neurons: int = 256,
        dropout: float = 0.3,
        lr: float = 0.001,
        batch_size: int = 256,
        epochs: int = 50,
        use_bn: bool = True,
        early_stopping: bool = True,
        patience: int = 5,
        sample_size: Optional[int] = 100_000,
    ):
        super().__init__(
            name="DNN",
            sample_size=sample_size,
            scale_numeric=True,  # DNN은 스케일링 필수
        )
        self.hidden_layers  = hidden_layers
        self.neurons        = neurons
        self.dropout        = dropout
        self.lr             = lr
        self.batch_size     = batch_size
        self.epochs         = epochs
        self.use_bn         = use_bn
        self.early_stopping = early_stopping
        self.patience       = patience

        self.preprocessor_: Optional[ColumnTransformer] = None
        self.net_: Optional[_DNNNet] = None
        self.train_losses_: list[float] = []
        self.val_losses_:   list[float] = []
        self.best_epoch_:   int = 0
        self.elapsed_:      float = 0.0

    def _create_estimator(self):
        return None  # 부모 클래스 Pipeline 방식 대신 직접 학습하므로 사용 안 함

    def _build_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        return ColumnTransformer([
            ("num", StandardScaler(), self.numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), self.categorical_cols),
        ])

    def fit_from_dataframe(self, df: pd.DataFrame,
                           progress_callback=None) -> "DNNRegressorModel":
        data = self.prepare_dataframe(df)
        if self.sample_size and len(data) > self.sample_size:
            data = data.sample(self.sample_size, random_state=42)

        X = data[self.feature_columns]
        y = data[self.target_col].values.astype(np.float32)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 전처리
        self.preprocessor_ = self._build_preprocessor(X_train)
        X_train_np = self.preprocessor_.fit_transform(X_train).astype(np.float32)
        X_val_np   = self.preprocessor_.transform(X_val).astype(np.float32)

        # 타깃 스케일링 (만원 단위 → 정규화)
        self._y_mean = float(y_train.mean())
        self._y_std  = float(y_train.std()) or 1.0
        y_train_s = (y_train - self._y_mean) / self._y_std
        y_val_s   = (y_val   - self._y_mean) / self._y_std

        # 텐서
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        Xt = torch.tensor(X_train_np, device=device)
        yt = torch.tensor(y_train_s,  device=device)
        Xv = torch.tensor(X_val_np,   device=device)
        yv = torch.tensor(y_val_s,    device=device)

        # 모델
        input_dim = X_train_np.shape[1]
        self.net_ = _DNNNet(input_dim, self.hidden_layers, self.neurons,
                            self.dropout, self.use_bn).to(device)
        optimizer = torch.optim.Adam(self.net_.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        dataset   = torch.utils.data.TensorDataset(Xt, yt)
        loader    = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        best_val  = float("inf")
        no_improve = 0
        self.train_losses_ = []
        self.val_losses_   = []

        t0 = time.time()
        for epoch in range(1, self.epochs + 1):
            self.net_.train()
            epoch_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(self.net_(xb), yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(xb)
            train_loss = epoch_loss / len(Xt)

            self.net_.eval()
            with torch.no_grad():
                val_loss = criterion(self.net_(Xv), yv).item()

            self.train_losses_.append(round(train_loss, 6))
            self.val_losses_.append(round(val_loss, 6))

            if progress_callback:
                progress_callback(epoch, self.epochs, train_loss, val_loss)

            if self.early_stopping:
                if val_loss < best_val - 1e-5:
                    best_val   = val_loss
                    no_improve = 0
                    self.best_epoch_ = epoch
                    self._best_state = {k: v.cpu().clone()
                                        for k, v in self.net_.state_dict().items()}
                else:
                    no_improve += 1
                    if no_improve >= self.patience:
                        break
            else:
                self.best_epoch_ = epoch

        self.elapsed_ = time.time() - t0

        # 최적 가중치 복원
        if self.early_stopping and hasattr(self, "_best_state"):
            self.net_.load_state_dict(
                {k: v.to(device) for k, v in self._best_state.items()}
            )

        # 테스트셋 평가
        self.net_.eval()
        with torch.no_grad():
            y_pred_s = self.net_(Xv).cpu().numpy()
        y_pred = y_pred_s * self._y_std + self._y_mean
        y_true = y_val

        self.metrics_ = {
            "MAE":  float(mean_absolute_error(y_true, y_pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "R2":   float(r2_score(y_true, y_pred)),
        }
        self.net_.to("cpu")
        return self

    @classmethod
    def load(cls, path: str) -> "DNNRegressorModel":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg  = ckpt["config"]
        obj  = cls(
            hidden_layers=cfg["hidden_layers"],
            neurons=cfg["neurons"],
            dropout=cfg["dropout"],
            use_bn=cfg["use_bn"],
        )
        obj.preprocessor_ = ckpt["preprocessor"]
        obj._y_mean        = ckpt["y_mean"]
        obj._y_std         = ckpt["y_std"]
        obj.net_ = _DNNNet(cfg["input_dim"], cfg["hidden_layers"],
                           cfg["neurons"], cfg["dropout"], cfg["use_bn"])
        obj.net_.load_state_dict(ckpt["net_state"])
        obj.net_.eval()
        return obj

    def predict_single(self, X: pd.DataFrame) -> float:
        assert self.net_ is not None and self.preprocessor_ is not None
        X_np = self.preprocessor_.transform(X).astype(np.float32)
        xt   = torch.tensor(X_np)
        self.net_.eval()
        with torch.no_grad():
            out = self.net_(xt).item()
        return out * self._y_std + self._y_mean
