"""
scripts/insert_data.py — CSV → MySQL 배치 INSERT

실행 방법:
    python scripts/insert_data.py

주의:
    - 실행 전 scripts/create_db.sql 로 DB/테이블을 먼저 생성해야 합니다.
    - conf/.env 에 DB 접속 정보가 올바르게 설정되어 있어야 합니다.
    - 5,002,839건 기준 약 10~20분 소요 예상 (BATCH_SIZE, PC 사양에 따라 다름).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
from utils.db import get_connection

# ── 설정 ──────────────────────────────────────────────────────
CSV_PATH   = ROOT / "data" / "raw" / "Apart Deal_6.csv"
BATCH_SIZE = 5_000
ENCODING   = "utf-8-sig"

# CSV 컬럼명 → DB 컬럼명 매핑
COL_MAP = {
    "지역코드": "region_code",
    "시군구":   "sigungu",
    "법정동":   "dong",
    "지번":     "jibun",
    "아파트":   "apt_name",
    "브랜드":   "brand",
    "브랜드여부": "is_brand",
    "건축년도": "build_year",
    "세대수":   "household_cnt",
    "거래일":   "deal_date",
    "층":       "floor",
    "전용면적": "area",
    "거래금액": "deal_amount",
    "기준금리": "base_rate",
    "위도":     "latitude",
    "경도":     "longitude",
    "인근학교수": "school_cnt",
    "인근역수": "station_cnt",
}

INSERT_SQL = """
INSERT INTO tbl_apart_deals (
    region_code, sigungu, dong, jibun,
    apt_name, brand, is_brand, build_year, household_cnt,
    deal_date, floor, area, deal_amount,
    base_rate, latitude, longitude, school_cnt, station_cnt
) VALUES (
    %s, %s, %s, %s,
    %s, %s, %s, %s, %s,
    %s, %s, %s, %s,
    %s, %s, %s, %s, %s
)
"""


def _none(val):
    """NaN / 공백 문자열 → None (MySQL NULL)"""
    if val is None:
        return None
    if isinstance(val, float) and np.isnan(val):
        return None
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(val, str) and val.strip() == "":
        return None
    return val


def row_to_tuple(row: pd.Series) -> tuple:
    return (
        _none(row["region_code"]),
        _none(row["sigungu"]),
        _none(row["dong"]),
        _none(row["jibun"]),
        _none(row["apt_name"]),
        row["brand"] if _none(row["brand"]) is not None else "기타",
        int(row["is_brand"]) if _none(row["is_brand"]) is not None else 0,
        _none(row["build_year"]),
        _none(row["household_cnt"]),
        _none(row["deal_date"]),
        _none(row["floor"]),
        _none(row["area"]),
        _none(row["deal_amount"]),
        _none(row["base_rate"]),
        _none(row["latitude"]),
        _none(row["longitude"]),
        _none(row["school_cnt"]),
        _none(row["station_cnt"]),
    )


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # 컬럼명 영문으로 변환
    df = df.rename(columns=COL_MAP)

    # 거래금액: 쉼표 제거 후 정수 변환
    df["deal_amount"] = (
        df["deal_amount"].astype(str).str.replace(",", "", regex=False).str.strip()
        .pipe(pd.to_numeric, errors="coerce")
        .astype("Int64")
    )

    # 층: 공백/float 문자열 → 정수 (지하층 음수 보존)
    df["floor"] = (
        df["floor"].astype(str).str.strip()
        .replace("", None)
        .pipe(pd.to_numeric, errors="coerce")
        .astype("Int64")
    )

    # 거래일: DATE 형식 통일
    df["deal_date"] = pd.to_datetime(df["deal_date"], errors="coerce").dt.date

    # 브랜드 결측치 처리
    df["brand"] = df["brand"].fillna("기타")
    df["is_brand"] = df["is_brand"].fillna(0).astype(int)

    return df


def insert_batch(cursor, batch: pd.DataFrame) -> int:
    params = [row_to_tuple(row) for _, row in batch.iterrows()]
    cursor.executemany(INSERT_SQL, params)
    return cursor.rowcount


def main():
    print(f"[START] CSV: {CSV_PATH}")
    print(f"        BATCH_SIZE: {BATCH_SIZE:,}")

    conn = get_connection()
    cursor = conn.cursor()

    total_inserted = 0
    chunk_no = 0

    try:
        reader = pd.read_csv(
            CSV_PATH,
            encoding=ENCODING,
            chunksize=BATCH_SIZE,
            dtype=str,          # 모든 컬럼을 문자열로 읽어 전처리에서 변환
            low_memory=False,
        )

        for chunk in reader:
            chunk_no += 1
            chunk = preprocess(chunk)

            inserted = insert_batch(cursor, chunk)
            conn.commit()
            total_inserted += inserted

            print(f"  chunk {chunk_no:4d} | +{inserted:,} rows | 누계 {total_inserted:,}")

    except Exception as e:
        conn.rollback()
        print(f"\n[ERROR] chunk {chunk_no}: {e}")
        raise
    finally:
        cursor.close()
        conn.close()

    print(f"\n[DONE] 총 {total_inserted:,}건 삽입 완료")


if __name__ == "__main__":
    main()
