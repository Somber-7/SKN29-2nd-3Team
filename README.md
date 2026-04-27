# SKN29 2nd Project - 3Team

머신러닝 모델을 직접 실행해보는 Streamlit 기반 인터랙티브 웹 애플리케이션

---

## 디렉토리 구조

```
SKN29-2nd-3Team/
│
├── app/                        # Streamlit 애플리케이션
│   ├── pages/                  # 영역별 페이지 (회귀, 분류, 군집화 등)
│   └── components/             # 재사용 UI 컴포넌트 (차트, 입력폼 등)
│
├── models/                     # 모델 로직 (영역별 분리)
│   ├── regression/             # 선형회귀, RandomForest, LightGBM
│   ├── classification/         # SVM, DecisionTree, Ensemble
│   ├── clustering/             # K-Means, DBSCAN, Agglomerative
│   ├── dimensionality_reduction/  # PCA
│   └── neural_network/         # MLP (sklearn), DNN (PyTorch)
│
├── data/                       # 데이터 저장소
│   ├── raw/                    # 원본 데이터
│   └── processed/              # 전처리 완료 데이터
│
├── sql/                        # MySQL 쿼리 파일 (테이블당 1개 파일)
│   ├── ddl/                    # 테이블 생성/수정 (CREATE, ALTER)
│   └── dml/                    # 초기 데이터 INSERT
│
├── conf/                       # 설정 파일
│                               # - DB 접속 정보 (host, port, db, 계정)
│                               # - 앱 전역 설정
│
├── utils/                      # 공통 유틸리티
│                               # - 전처리 헬퍼
│                               # - 평가지표 계산
│                               # - 시각화 함수
│
└── assets/                     # 정적 파일
    ├── images/                 # 이미지
    └── css/                    # 스타일시트
```

---

## 기술 스택

| 구분 | 사용 기술 |
|------|----------|
| 프론트엔드 | Streamlit |
| 백엔드 / 모델 | Python, scikit-learn, PyTorch, LightGBM |
| 데이터베이스 | MySQL |
