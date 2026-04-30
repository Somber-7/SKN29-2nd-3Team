"""
utils/preprocessor.py — 데이터 전처리 유틸리티

스케일링, 인코딩, 결측치 처리, 이상치 제거, sklearn 파이프라인 빌더를 제공합니다.

일반적인 전처리 순서:
    1. impute()          — 결측치 처리
    2. remove_outliers_iqr() / log_transform()  — 이상치 제거 또는 분포 정규화
    3. onehot_encode()   — 범주형 인코딩
    4. scale_features()  — 수치형 스케일링

또는 build_pipeline()으로 2~4단계를 sklearn Pipeline 하나로 묶을 수 있습니다.

── Apart Deal 전용 함수 ──────────────────────────────────────
parse_price()       — 거래금액 str → int 변환
parse_date()        — 거래일 str → datetime, 연/월/분기 파생 피처 추출
fix_floor()         — 층 결측(NaN) 처리 (음수는 지하층으로 보존)
map_brand_grade()   — 브랜드 문자열 → 등급(프리미엄/일반/공공/기타) 변환
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


# ── 스케일링 ──────────────────────────────────────────

def get_scaler(method: str = "standard"):
    """스케일러 객체를 반환합니다.

    Args:
        method: "standard" — 평균 0, 표준편차 1로 변환 (StandardScaler, 기본값)
                "minmax"   — 0~1 범위로 변환 (MinMaxScaler, 이상치에 민감)

    Returns:
        sklearn Scaler 객체 (fit_transform() 미호출 상태)
    """
    if method == "minmax":
        return MinMaxScaler()
    return StandardScaler()


def scale_features(X_train, X_test=None, method: str = "standard"):
    """학습 데이터로 스케일러를 학습하고, 학습/테스트 데이터를 변환합니다.

    스케일러는 X_train에만 fit하고 X_test에는 transform만 적용합니다.
    (데이터 누수 방지)

    Args:
        X_train: 학습용 피처 데이터 (array-like)
        X_test:  테스트용 피처 데이터 (array-like). None이면 X_train만 변환.
        method:  "standard" | "minmax"

    Returns:
        X_test가 None인 경우:  (X_train_scaled, scaler)
        X_test가 있는 경우:   (X_train_scaled, X_test_scaled, scaler)

    예시:
        X_tr, X_te, scaler = scale_features(X_train, X_test, method="standard")
    """
    scaler = get_scaler(method)
    X_train_scaled = scaler.fit_transform(X_train)
    if X_test is not None:
        return X_train_scaled, scaler.transform(X_test), scaler
    return X_train_scaled, scaler


# ── 인코딩 ────────────────────────────────────────────

def label_encode(series: pd.Series):
    """범주형 Series를 정수로 레이블 인코딩합니다.

    순서가 있는 범주형 (예: 학력 수준) 또는 이진 분류 타깃에 적합합니다.
    순서 없는 명목형 데이터에는 onehot_encode()를 사용하세요.

    Args:
        series: 인코딩할 pandas Series (문자열 또는 범주형)

    Returns:
        (encoded_array, encoder) 튜플
        - encoded_array: 정수로 변환된 numpy 배열
        - encoder:       LabelEncoder 객체 (역변환 시 encoder.inverse_transform() 사용)

    예시:
        encoded, enc = label_encode(df["gender"])
        original = enc.inverse_transform(encoded)  # 역변환
    """
    enc = LabelEncoder()
    encoded = enc.fit_transform(series)
    return encoded, enc


def onehot_encode(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """지정한 컬럼을 One-Hot 인코딩으로 변환합니다.

    pandas get_dummies를 사용하여 각 범주값이 별도 컬럼(0/1)이 됩니다.
    원본 컬럼은 제거되고 새 컬럼이 추가된 DataFrame을 반환합니다.

    Args:
        df:      인코딩할 DataFrame
        columns: One-Hot 인코딩할 컬럼명 리스트

    Returns:
        인코딩된 새 DataFrame (원본 불변)

    예시:
        df_encoded = onehot_encode(df, columns=["gender", "region"])
        # "gender" → "gender_Male", "gender_Female" 컬럼 생성
    """
    return pd.get_dummies(df, columns=columns, drop_first=False)


# ── 결측치 처리 ───────────────────────────────────────

def impute(df: pd.DataFrame, strategy: str = "mean") -> pd.DataFrame:
    """결측치(NaN)를 지정한 전략으로 채웁니다.

    수치형 컬럼과 범주형 컬럼을 자동으로 구분하여 처리합니다.
    - 수치형: strategy 파라미터 적용
    - 범주형: 항상 최빈값(most_frequent)으로 채움

    Args:
        df:       처리할 DataFrame
        strategy: 수치형 결측치 대체 전략
                  "mean"           — 평균값 (기본값, 정규분포에 적합)
                  "median"         — 중앙값 (이상치 있을 때 권장)
                  "most_frequent"  — 최빈값

    Returns:
        결측치가 채워진 새 DataFrame (원본 불변)
    """
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()

    result = df.copy()
    if num_cols:
        imp = SimpleImputer(strategy=strategy if strategy != "most_frequent" else "mean")
        result[num_cols] = imp.fit_transform(result[num_cols])
    if cat_cols:
        imp_cat = SimpleImputer(strategy="most_frequent")
        result[cat_cols] = imp_cat.fit_transform(result[cat_cols])
    return result


# ── 전처리 파이프라인 빌더 ─────────────────────────────

def build_pipeline(model, num_cols: list, cat_cols: list, scaler: str = "standard") -> Pipeline:
    """수치형 + 범주형 전처리와 모델을 하나의 sklearn Pipeline으로 묶습니다.

    내부 구성:
        수치형 컬럼 → [결측치 평균 대체] → [StandardScaler or MinMaxScaler]
        범주형 컬럼 → [결측치 최빈값 대체] → [OneHotEncoder]
        → ColumnTransformer → model

    Args:
        model:    학습시킬 sklearn 호환 모델 (예: RandomForestClassifier())
        num_cols: 수치형 컬럼명 리스트
        cat_cols: 범주형 컬럼명 리스트
        scaler:   수치형 스케일링 방식 "standard" | "minmax"

    Returns:
        sklearn Pipeline 객체. pipe.fit(X, y) / pipe.predict(X) 로 바로 사용.

    예시:
        pipe = build_pipeline(
            model=LGBMClassifier(),
            num_cols=["age", "income"],
            cat_cols=["gender"],
        )
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
    """
    num_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", get_scaler(scaler)),
    ])
    cat_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer([
        ("num", num_transformer, num_cols),
        ("cat", cat_transformer, cat_cols),
    ])
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


# ── 이상치 탐지 ───────────────────────────────────────

def remove_outliers_iqr(df: pd.DataFrame, columns: list, factor: float = 1.5) -> pd.DataFrame:
    """IQR(사분위수 범위) 기반으로 이상치 행을 제거합니다.

    제거 기준: Q1 - factor * IQR  <  값  < Q3 + factor * IQR
    factor=1.5 (기본값)이면 일반적인 이상치, factor=3.0이면 극단적 이상치만 제거합니다.

    Args:
        df:      처리할 DataFrame
        columns: 이상치를 검사할 수치형 컬럼명 리스트
        factor:  IQR 배수. 클수록 더 적은 행이 제거됩니다. (기본 1.5)

    Returns:
        이상치 행이 제거된 새 DataFrame. 인덱스가 0부터 재설정됩니다.
    """
    mask = pd.Series([True] * len(df))
    for col in columns:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        mask &= df[col].between(q1 - factor * iqr, q3 + factor * iqr)
    return df[mask].reset_index(drop=True)


def log_transform(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """지정한 컬럼에 log1p 변환을 적용합니다.

    오른쪽으로 치우친(양의 왜도) 분포를 정규분포에 가깝게 만듭니다.
    log1p(x) = log(1 + x) 이므로 x=0인 값도 안전하게 처리됩니다.
    음수 값이 있는 컬럼에는 적용하지 마세요.

    Args:
        df:      처리할 DataFrame
        columns: 변환할 수치형 컬럼명 리스트 (양수 값만 존재해야 함)

    Returns:
        변환된 새 DataFrame (원본 불변)
    """
    result = df.copy()
    for col in columns:
        result[col] = np.log1p(result[col])
    return result


# ── Apart Deal 전용 ───────────────────────────────────

# 브랜드 → 등급 매핑 테이블 (BRAND_GUIDE.md 참고)
# 프리미엄 외부 데이터 확보 후 기준 조정 예정
_PREMIUM_BRANDS = {
    "삼성물산(래미안)", "GS건설(자이)", "현대건설(힐스테이트)",
    "HDC현대산업개발(아이파크)", "대우건설(푸르지오)", "DL이앤씨(e편한세상)",
    "포스코이앤씨(더샵)", "롯데건설(롯데캐슬)", "DL이앤씨(아크로)",
    "SK에코플랜트(SK뷰)",
}
_GENERAL_BRANDS = {
    "두산건설(위브)", "코오롱글로벌(하늘채)", "호반건설(베르디움)",
    "우미건설(린)", "효성중공업(해링턴)", "KCC건설(스위첸)",
    "태영건설(데시앙)", "대방건설(노블랜드)", "서희건설(스타힐스)",
    "신동아건설(파밀리에)", "금호건설(어울림)", "벽산건설(블루밍)",
    "중흥토건(S-클래스)", "현대건설", "삼성물산", "대우건설",
    "롯데건설", "금호건설", "두산건설", "벽산건설", "코오롱글로벌",
    "호반건설", "우미건설", "효성중공업", "신동아건설", "서희건설",
    "중흥토건", "대방건설", "포스코이앤씨", "한신공영", "쌍용건설",
    "한양", "풍림산업", "대림산업", "동아건설", "삼익주택",
    "한라", "경남기업", "동원개발", "반도건설", "동부건설",
    "금강주택", "현진건설", "우미건설", "삼부토건", "성지건설",
    "광양건설", "라온건설", "아남건설", "부영",
}
_PUBLIC_BRANDS = {"LH(주공)"}


def parse_price(df: pd.DataFrame, col: str = "거래금액") -> pd.DataFrame:
    """거래금액 컬럼을 str에서 int로 변환합니다.

    원본 데이터의 거래금액은 문자열이며 쉼표가 포함될 수 있습니다.
    변환 실패 행은 NaN으로 처리됩니다.

    Args:
        df:  처리할 DataFrame
        col: 변환할 컬럼명 (기본값: "거래금액")

    Returns:
        거래금액이 int64로 변환된 새 DataFrame (원본 불변)

    예시:
        df = parse_price(df)
        # "77,000" → 77000
    """
    result = df.copy()
    result[col] = (
        result[col]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
        .pipe(pd.to_numeric, errors="coerce")
        .astype("Int64")
    )
    return result


def parse_date(df: pd.DataFrame, col: str = "거래일") -> pd.DataFrame:
    """거래일 컬럼을 datetime으로 변환하고 연/월/분기 파생 피처를 추가합니다.

    추가되는 컬럼:
        거래년도 (int)  — 연도
        거래월   (int)  — 월 (1~12)
        거래분기 (int)  — 분기 (1~4)

    원본 거래일 컬럼은 유지됩니다. 필요 없으면 직접 drop하세요.

    Args:
        df:  처리할 DataFrame
        col: 날짜 컬럼명 (기본값: "거래일", 형식: "YYYY-MM-DD")

    Returns:
        파생 피처 3개가 추가된 새 DataFrame (원본 불변)

    예시:
        df = parse_date(df)
        # "2015-02-04" → 거래년도=2015, 거래월=2, 거래분기=1
    """
    result = df.copy()
    dt = pd.to_datetime(result[col], errors="coerce")
    result["거래년도"] = dt.dt.year
    result["거래월"]   = dt.dt.month
    result["거래분기"] = dt.dt.quarter
    return result


def fix_floor(df: pd.DataFrame, col: str = "층") -> pd.DataFrame:
    """층 컬럼의 결측치(NaN)를 처리합니다.

    음수 층(-1, -2 등)은 지하층을 의미하므로 그대로 보존합니다.
    현재는 NaN 행만 식별 가능하도록 유지하며,
    모델 학습 시 필요에 따라 impute() 또는 해당 행 제거를 선택하세요.

    Args:
        df:  처리할 DataFrame
        col: 층 컬럼명 (기본값: "층")

    Returns:
        원본과 동일한 새 DataFrame (원본 불변)
    """
    return df.copy()


def map_brand_grade(df: pd.DataFrame, col: str = "브랜드") -> pd.DataFrame:
    """브랜드 문자열을 4개 등급으로 변환한 brand_grade 컬럼을 추가합니다.

    등급 기준 (BRAND_GUIDE.md 참고):
        "프리미엄" — 래미안, 자이, 힐스테이트, 아이파크, 푸르지오 등 상위 10개
        "일반브랜드" — 중소형 민간 브랜드
        "공공(LH)"  — LH(주공)
        "기타"       — 비브랜드 및 미분류

    브랜드 프리미엄 외부 데이터 확보 후 _PREMIUM_BRANDS / _GENERAL_BRANDS 집합을
    수정하여 등급 기준을 조정하세요.

    분류 모델 사용 시: brand_grade를 타깃으로, 브랜드/브랜드여부 컬럼은 제거할 것.
    회귀 모델 사용 시: brand_grade를 피처로 사용 가능 (OneHot 인코딩 후).

    Args:
        df:  처리할 DataFrame
        col: 브랜드 컬럼명 (기본값: "브랜드")

    Returns:
        "brand_grade" 컬럼이 추가된 새 DataFrame (원본 불변)

    예시:
        df = map_brand_grade(df)
        df["brand_grade"].value_counts()
        # 기타 2738432 / 공공(LH) 394914 / 일반브랜드 ... / 프리미엄 ...
    """
    def _grade(brand: str) -> str:
        if brand in _PREMIUM_BRANDS:
            return "프리미엄"
        if brand in _PUBLIC_BRANDS:
            return "공공(LH)"
        if brand in _GENERAL_BRANDS:
            return "일반브랜드"
        return "기타"

    result = df.copy()
    result["brand_grade"] = result[col].map(_grade)
    return result
