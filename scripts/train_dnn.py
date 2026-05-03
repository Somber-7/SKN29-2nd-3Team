# scripts/train_dnn.py
# 사용법: python scripts/train_dnn.py

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import torch
import pandas as pd
from models.regression.dnn_regressor import DNNRegressorModel

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "models")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PATH  = os.path.join(OUTPUT_DIR, "dnn_regressor.pt")
META_PATH   = os.path.join(OUTPUT_DIR, "dnn_regressor_meta.json")

print("데이터 로딩 중...")
df = pd.read_parquet("data/cache/apart_deals.parquet")
print(f"전체 데이터: {len(df):,}건")

model = DNNRegressorModel(
    hidden_layers=4,
    neurons=256,
    dropout=0.2,
    lr=0.001,
    batch_size=1024,
    epochs=30,
    use_bn=True,
    early_stopping=True,
    patience=5,
    sample_size=None,  # 전체 사용
)

def progress(epoch, total, tl, vl):
    print(f"  Epoch {epoch:3d}/{total}  train={tl:.4f}  val={vl:.4f}")

print("학습 시작...")
model.fit_from_dataframe(df, progress_callback=progress)

print(f"\n학습 완료: {model.elapsed_:.1f}초")
print(f"Best epoch: {model.best_epoch_}")
print(f"MAE : {model.metrics_['MAE']:,.0f}만원")
print(f"RMSE: {model.metrics_['RMSE']:,.0f}만원")
print(f"R²  : {model.metrics_['R2']:.4f}")

# 모델 저장
torch.save({
    "net_state":    model.net_.state_dict(),
    "preprocessor": model.preprocessor_,
    "y_mean":       model._y_mean,
    "y_std":        model._y_std,
    "config": {
        "hidden_layers": model.hidden_layers,
        "neurons":       model.neurons,
        "dropout":       model.dropout,
        "use_bn":        model.use_bn,
        "input_dim":     next(iter(model.net_.parameters())).shape[1],
    },
}, MODEL_PATH)

# 메타 저장 (성능 지표 + 학습 곡선)
meta = {
    "sample_size":   len(df),
    "elapsed":       round(model.elapsed_, 1),
    "best_epoch":    model.best_epoch_,
    "metrics":       model.metrics_,
    "train_losses":  model.train_losses_,
    "val_losses":    model.val_losses_,
    "config": {
        "hidden_layers": model.hidden_layers,
        "neurons":       model.neurons,
        "dropout":       model.dropout,
        "lr":            model.lr,
        "batch_size":    model.batch_size,
        "use_bn":        model.use_bn,
    },
}
with open(META_PATH, "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print(f"\n저장 완료:")
print(f"  모델: {MODEL_PATH}")
print(f"  메타: {META_PATH}")
