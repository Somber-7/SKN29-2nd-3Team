# 📦 학습된 모델 메타데이터

> **프로젝트명**: 아파트 실거래가 분석 및 예측  
> **팀**: SKN29-2nd-3Team  
> **작성일**: 2026-05-04  
> **모델 저장 경로**: `data/models/`

---

## 학습 환경

| 항목 | 버전/사양 |
|------|---------|
| Python | 3.x |
| pandas | ≥ 2.2.0 |
| numpy | ≥ 1.26.0 |
| scikit-learn | ≥ 1.4.0 |
| lightgbm | ≥ 4.3.0 |
| xgboost | ≥ 2.0.0 |
| torch | ≥ 2.3.0 |
| joblib | ≥ 1.4.0 |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU (CUDA) |
| OS | Windows 11 |

---

## 모델 1: Linear Regression (가격 예측)

### 기본 정보

| 항목 | 내용 |
|------|------|
| 모델명 | LinearRegressionPriceModel |
| 버전 | v1.0.0 |
| 파일명 | `data/models/LinearRegression_model.pkl` |
| 파일 크기 | 7.8 KB |
| 라이브러리 | scikit-learn |
| 역할 | 베이스라인 가격 예측 |

### 하이퍼파라미터

| 파라미터 | 값 |
|---------|-----|
| fit_intercept | True |
| 정규화 | 없음 |

### 입력 데이터 스펙

- 타겟: 거래금액 (만원, float)
- 필수 전처리: StandardScaler 적용 필요
- 피처: 전용면적, 층, 건물연식, 기준금리, 위도, 경도, 인근학교수, 인근역수, 세대수, 거래연도, 거래월, 지역코드

### 모델 로드 예시

```python
import joblib

model = joblib.load('data/models/LinearRegression_model.pkl')
prediction = model.predict(X_scaled)  # X는 StandardScaler 적용 필요
```

---

## 모델 2: Random Forest (가격 예측)

### 기본 정보

| 항목 | 내용 |
|------|------|
| 모델명 | RandomForestPriceModel |
| 버전 | v1.0.0 |
| 파일명 | `data/models/RandomForest_model.pkl` |
| 파일 크기 | 84 MB |
| 라이브러리 | scikit-learn |
| 역할 | 앙상블 기반 가격 예측 |

### 하이퍼파라미터

| 파라미터 | 값 |
|---------|-----|
| n_estimators | 100 |
| max_depth | None |
| min_samples_split | 2 |
| random_state | 42 |

### 입력 데이터 스펙

- 타겟: 거래금액 (만원, float)
- 필수 전처리: 없음 (트리 기반, 스케일 불변)
- 피처: 전용면적, 층, 건물연식, 기준금리, 위도, 경도, 인근학교수, 인근역수, 세대수, 거래연도, 거래월, 지역코드

### 모델 로드 예시

```python
import joblib

model = joblib.load('data/models/RandomForest_model.pkl')
prediction = model.predict(X)  # 스케일링 불필요
```

---

## 모델 3: LightGBM (가격 예측)

### 기본 정보

| 항목 | 내용 |
|------|------|
| 모델명 | LightGBMPriceModel |
| 버전 | v1.0.0 |
| 파일명 | `data/models/LightGBM_model.pkl` |
| 파일 크기 | 2.9 MB |
| 라이브러리 | lightgbm ≥ 4.3.0 |
| 역할 | 그래디언트 부스팅 가격 예측 |

### 입력 데이터 스펙

- 타겟: 거래금액 (만원, float)
- 필수 전처리: 없음 (트리 기반)

### 모델 로드 예시

```python
import joblib

model = joblib.load('data/models/LightGBM_model.pkl')
prediction = model.predict(X)
```

---

## 모델 4: XGBoost (가격 예측)

### 기본 정보

| 항목 | 내용 |
|------|------|
| 모델명 | XGBoostPriceModel |
| 버전 | v1.0.0 |
| 파일명 | `data/models/XGBoost_model.pkl` |
| 파일 크기 | 6.6 MB |
| 라이브러리 | xgboost ≥ 2.0.0 |
| 역할 | GPU 부스팅 가격 예측 |

### 입력 데이터 스펙

- 타겟: 거래금액 (만원, float)
- 필수 전처리: 없음 (트리 기반)

### 모델 로드 예시

```python
import joblib

model = joblib.load('data/models/XGBoost_model.pkl')
prediction = model.predict(X)
```

---

## 모델 5: DNN Regressor (가격 예측)

### 기본 정보

| 항목 | 내용 |
|------|------|
| 모델명 | DNNRegressorModel |
| 버전 | v1.0.0 |
| 저장일 | 2026-05-04 |
| 파일명 | `data/models/dnn_regressor.pt` |
| 메타데이터 | `data/models/dnn_regressor_meta.json` |
| 파일 크기 | 1.04 MB |
| 라이브러리 | PyTorch ≥ 2.3.0 |
| 역할 | 딥러닝 가격 예측 (최고 성능) |

### 모델 구조

```
입력층 (수치형 특성)
    │
    ├─ Linear → BatchNorm → ReLU → Dropout(0.2)  [256 units]
    ├─ Linear → BatchNorm → ReLU → Dropout(0.2)  [256 units]
    ├─ Linear → BatchNorm → ReLU → Dropout(0.2)  [256 units]
    ├─ Linear → BatchNorm → ReLU → Dropout(0.2)  [256 units]
    └─ Linear → 출력 (거래금액, 만원)
```

### 하이퍼파라미터

| 파라미터 | 값 |
|---------|-----|
| hidden_layers | 4 |
| neurons | 256 |
| dropout | 0.2 |
| use_bn | True (Batch Normalization) |
| lr | 0.001 |
| batch_size | 1,024 |
| optimizer | Adam |
| loss | MSELoss |
| best_epoch | **30** |
| 학습 시간 | 677.9초 |
| 학습 샘플 | 5,002,839건 |

### 최종 성능

| 지표 | 값 |
|------|-----|
| MAE | **3,055만원** |
| RMSE | **5,515만원** |
| R² | **0.9632** |

### 입력 데이터 스펙

- 타겟: 거래금액 (만원, float)
- **필수 전처리**: StandardScaler 적용 필수 (신경망은 스케일 민감)
- 피처: 전용면적, 층, 건물연식, 기준금리, 위도, 경도, 인근학교수, 인근역수, 세대수, 거래연도, 거래월, 지역코드

### 모델 로드 및 예측 예시

```python
import torch
import json
from models.regression.dnn_regressor import DNNRegressorModel

# 메타데이터 로드
with open('data/models/dnn_regressor_meta.json', 'r') as f:
    meta = json.load(f)

# 모델 로드
model = DNNRegressorModel()
model.load('data/models/dnn_regressor.pt')

# 예측 (X는 StandardScaler 적용 후 입력)
predictions = model.predict(X_scaled)
print(f"예측 거래금액: {predictions[0]:.0f} 만원")
```

### 알려진 한계점

- 학습 데이터 기간: 2015~2023년 — 이후 시장 변화 반영 안 됨
- 위도/경도 결측 아파트(76,960건)는 예측 불가
- 초대형 면적(>200㎡) 예측 정확도 상대적으로 낮을 수 있음

---

## 모델 6: BrandGradeClassifier (브랜드 등급 분류)

### 기본 정보

| 항목 | 내용 |
|------|------|
| 모델명 | BrandGradeClassifier |
| 버전 | v1.0.0 |
| 저장일 | 2026-04-30 |
| 파일명 | `data/models/brand_grade_classifier.pkl` |
| 파일 크기 | 45 MB |
| 라이브러리 | xgboost ≥ 2.0.0 |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU |
| 역할 | 아파트 브랜드 등급 다중 분류 |

### 클래스 정의

| 클래스 ID | 클래스명 | 대표 브랜드 |
|----------|---------|-----------|
| 0 | 공공(LH) | LH, 주공 |
| 1 | 기타 | 미분류 단지 |
| 2 | 일반브랜드 | 위브, 하늘채, 호반 등 |
| 3 | 프리미엄 | 래미안, 자이, 힐스테이트, 아이파크 등 |

### 하이퍼파라미터

| 파라미터 | 값 |
|---------|-----|
| objective | multi:softprob |
| n_estimators | 1,000 |
| learning_rate | 0.05 |
| max_depth | 8 |
| subsample | 0.8 |
| colsample_bytree | 0.8 |
| early_stopping_rounds | 30 |
| eval_metric | mlogloss |
| device | cuda |
| best_iteration | 999 |
| 클래스 가중치 | sqrt(n / k×count) |
| 학습 시간 | 116.2초 |

### 최종 성능 (Test 기준)

| 지표 | 값 |
|------|-----|
| Accuracy | **0.9602** |
| Precision (weighted) | **0.9603** |
| Recall (weighted) | **0.9602** |
| F1 (weighted) | **0.9602** |

**클래스별**

| 클래스 | Precision | Recall | F1 | Support |
|--------|-----------|--------|-----|---------|
| 공공(LH) | 0.995 | 0.999 | 0.997 | 3,967 |
| 기타 | 0.973 | 0.961 | 0.967 | 27,701 |
| 일반브랜드 | 0.928 | 0.933 | 0.930 | 12,018 |
| 프리미엄 | 0.946 | 0.991 | 0.968 | 5,358 |

### 입력 데이터 스펙

- 타겟: 브랜드등급 (4개 클래스)
- **제외 필수**: 아파트명, 브랜드명, 브랜드여부 (타깃 누출 방지)
- 필수 피처: 거래금액, 전용면적, 층, 건물연식, 기준금리, 위도, 경도, 인근학교수, 인근역수, 세대수, 거래연도, 거래월, 지역코드, 시군구

### 예측값 해석

| 출력값 | 의미 |
|--------|------|
| 0 | 공공(LH) |
| 1 | 기타 |
| 2 | 일반브랜드 |
| 3 | 프리미엄 |
| predict_proba[:,i] | 각 클래스 확률 (0~1) |

### 모델 로드 및 예측 예시

```python
import joblib
from models.classification.brand_grade_classifier import BrandGradeClassifier

# 방법 1: 래퍼 클래스 사용
model = BrandGradeClassifier()
model.load('data/models/brand_grade_classifier.pkl')
labels = model.predict_dataframe(df)

# 방법 2: joblib 직접 로드
model = joblib.load('data/models/brand_grade_classifier.pkl')
predictions = model.predict(X)
proba = model.predict_proba(X)
```

---

## 모델 7: TorchKMeans (지역 군집화)

### 기본 정보

| 항목 | 내용 |
|------|------|
| 모델명 | TorchKMeansLocationClusterModel |
| 버전 | v1.0.0 |
| 저장일 | 2026-05-03 |
| 파일명 | `data/models/torch_kmeans_clustering.pkl` |
| 파일 크기 | 40 MB |
| 라이브러리 | PyTorch ≥ 2.3.0 |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU |
| 역할 | 전국 아파트 지역 군집화 |

### 최종 설정

| 파라미터 | 값 |
|---------|-----|
| k (군집 수) | **7** |
| 피처 | 위도, 경도, 평당가(log1p), 건물연식, 거래활성도 |
| 위도/경도 가중치 | **×5** |
| n_init | 10 |
| max_iter | 300 |
| tol | 1e-4 |
| 학습 샘플 | 4,925,803건 |
| 학습 시간 | 약 45초 |

### 최종 성능

| 지표 | 값 |
|------|-----|
| Silhouette | **0.4694** |
| Davies-Bouldin | **0.8479** |
| Calinski-Harabasz | **61,227** |

### 군집 레이블 해석

| cluster | 권역 해석 | 평균 평당가(만) | 거래 비율 |
|---------|---------|-------------|---------|
| 2 | 서울·수도권 핵심 | 1,829 | 47% |
| 1 | 부산·경남권 | 1,015 | 23% |
| 4 | 충청·전북권 | 900 | 13% |
| 5 | 창원·김해 역세권 | 771 | 9% |
| 3 | 전남·광주권 | 667 | 5% |
| 0 | 강원·경북권 | 776 | 4% |
| 6 | 서울 초고가 특수 | 1,038 | 1% |

### 입력 데이터 스펙

- **필수**: 위도, 경도 (결측 시 예측 불가)
- 내부 파생: 평당가 = log1p(거래금액 / (전용면적 / 3.3))
- 내부 파생: 건물연식 = 2026 - 건축년도
- 내부 파생: 거래활성도 = `apt_activity_map_`에 저장된 맵으로 매핑 (미등록 단지는 중앙값 대체)

### 모델 로드 및 예측 예시

```python
from models.clustering.torch_kmeans_models import TorchKMeansLocationClusterModel

model = TorchKMeansLocationClusterModel(n_clusters=7)
model.load('data/models/torch_kmeans_clustering.pkl')

# 새 데이터 군집 예측
labels = model.predict(new_df)

# 군집 요약
summary = model.summarize_clusters(df)
print(summary)
```

### 알려진 한계점

- 위도/경도 결측 아파트는 군집 배정 불가
- 거래활성도 미등록 신규 단지는 중앙값으로 대체 → 정확도 저하 가능

---

## 모델 8: Premium Analysis Results (프리미엄 분석)

### 기본 정보

| 항목 | 내용 |
|------|------|
| 파일명 | `data/models/premium_analysis_results.pkl` |
| 파일 크기 | 214 KB |
| 역할 | 지역별 가격 프리미엄 분석 결과 저장 |
| 유형 | 분석 결과 (모델 아님) |

### 데이터 구조

- 지역별 평당가 평균 및 프리미엄 지수
- 브랜드별 프리미엄 비교 데이터
- Streamlit 프리미엄 분석 페이지에서 직접 로드하여 사용

---

## 재현 방법

### 1. 환경 설정

```bash
pip install -r requirements.txt
```

### 2. 데이터베이스 준비

```bash
# MySQL DB에 데이터 삽입
python scripts/insert_data.py

# 시군구 통계 테이블 생성
python scripts/build_sigungu_stats.py
```

### 3. 모델 학습 및 저장

```bash
# 회귀·분류·군집화 모델 저장
python scripts/save_models.py

# DNN 회귀 모델 학습
python scripts/train_dnn.py

# 이상 탐지 사전 계산
python scripts/precompute_anomaly.py

# 홈 페이지 캐시 생성
python scripts/save_page_data.py
```

### 4. Streamlit 앱 실행

```bash
streamlit run app/Home.py
```

---

## 전체 저장 파일 체크리스트

| 파일명 | 크기 | 용도 | 확인 |
|--------|------|------|------|
| `data/models/LinearRegression_model.pkl` | 7.8 KB | 선형 회귀 | ✅ |
| `data/models/RandomForest_model.pkl` | 84 MB | 랜덤포레스트 회귀 | ✅ |
| `data/models/LightGBM_model.pkl` | 2.9 MB | LightGBM 회귀 | ✅ |
| `data/models/XGBoost_model.pkl` | 6.6 MB | XGBoost 회귀 | ✅ |
| `data/models/dnn_regressor.pt` | 1.04 MB | DNN 회귀 (PyTorch) | ✅ |
| `data/models/dnn_regressor_meta.json` | 1.3 KB | DNN 메타데이터 | ✅ |
| `data/models/brand_grade_classifier.pkl` | 45 MB | 브랜드 등급 분류 | ✅ |
| `data/models/torch_kmeans_clustering.pkl` | 40 MB | 지역 군집화 | ✅ |
| `data/models/premium_analysis_results.pkl` | 214 KB | 프리미엄 분석 결과 | ✅ |
