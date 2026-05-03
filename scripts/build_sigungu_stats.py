"""
scripts/build_sigungu_stats.py — 시군구별 통계 테이블 생성

실행:
    python scripts/build_sigungu_stats.py

생성 테이블: tbl_sigungu_stats
    sigungu          VARCHAR(100) PK
    sido             VARCHAR(50)        -- 시/도 (지역코드 앞 2자리로 매핑)
    region_code      INT
    lat_median       DOUBLE
    lon_median       DOUBLE
    household_total  BIGINT             -- 시군구 전체 세대수
"""

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.db import fetch_all, db_cursor

CACHE_PATH = Path(__file__).parent.parent / "data" / "cache" / "apart_deals.parquet"

# 지역코드 앞 2자리 → 시/도명
SIDO_MAP = {
    "11": "서울특별시",
    "26": "부산광역시",
    "27": "대구광역시",
    "28": "인천광역시",
    "29": "광주광역시",
    "30": "대전광역시",
    "31": "울산광역시",
    "36": "세종특별자치시",
    "41": "경기도",
    "42": "강원특별자치도",
    "43": "충청북도",
    "44": "충청남도",
    "45": "전북특별자치도",
    "46": "전라남도",
    "47": "경상북도",
    "48": "경상남도",
    "50": "제주특별자치도",
}


def load_raw() -> pd.DataFrame:
    if CACHE_PATH.exists():
        print(f"[cache] {CACHE_PATH}")
        return pd.read_parquet(CACHE_PATH, columns=["시군구", "지역코드", "위도", "경도", "세대수"])
    print("[DB] 전체 조회 중...")
    rows = fetch_all(
        "SELECT sigungu AS 시군구, region_code AS 지역코드, "
        "latitude AS 위도, longitude AS 경도, household_cnt AS 세대수 "
        "FROM tbl_apart_deals"
    )
    return pd.DataFrame(rows)


def build_stats(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["지역코드", "위도", "경도", "세대수"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["시군구", "지역코드", "위도", "경도"])
    df = df[df["시군구"].str.strip() != ""]

    stats = (
        df.groupby("시군구", as_index=False)
        .agg(
            지역코드=("지역코드", "first"),
            lat_median=("위도", "median"),
            lon_median=("경도", "median"),
            household_total=("세대수", "max"),
        )
        .sort_values("지역코드")
        .reset_index(drop=True)
    )
    stats["lat_median"]      = stats["lat_median"].round(6)
    stats["lon_median"]      = stats["lon_median"].round(6)
    stats["household_total"] = stats["household_total"].fillna(0).astype(int)
    stats["sido"] = stats["지역코드"].astype(str).str[:2].map(SIDO_MAP).fillna("기타")
    return stats


def create_table():
    with db_cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS tbl_sigungu_stats")
        cur.execute("""
            CREATE TABLE tbl_sigungu_stats (
                sigungu         VARCHAR(100) NOT NULL PRIMARY KEY,
                sido            VARCHAR(50)  NOT NULL,
                region_code     INT          NOT NULL,
                lat_median      DOUBLE       NOT NULL,
                lon_median      DOUBLE       NOT NULL,
                household_total BIGINT       NOT NULL
            ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
        """)
    print("  테이블 생성 완료: tbl_sigungu_stats")


def insert_stats(stats: pd.DataFrame):
    rows = [
        (row["시군구"], row["sido"], int(row["지역코드"]),
         float(row["lat_median"]), float(row["lon_median"]), int(row["household_total"]))
        for _, row in stats.iterrows()
    ]
    with db_cursor() as cur:
        cur.executemany(
            "INSERT INTO tbl_sigungu_stats (sigungu, sido, region_code, lat_median, lon_median, household_total) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            rows,
        )
    print(f"  {len(rows)}개 시군구 저장 완료")


def main():
    print("=" * 50)
    print("  tbl_sigungu_stats 빌드")
    print("=" * 50)

    print("\n[1] 데이터 로딩...")
    df = load_raw()
    print(f"    {len(df):,}건 로딩 완료")

    print("\n[2] 시군구별 집계...")
    stats = build_stats(df)
    print(f"    {len(stats)}개 시군구 집계 완료")
    print(stats.head(5).to_string(index=False))

    print("\n[3] DB 저장...")
    create_table()
    insert_stats(stats)

    print("\n완료.")


if __name__ == "__main__":
    main()
