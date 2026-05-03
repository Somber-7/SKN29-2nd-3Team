"""
scripts/build_sigungu_stats.py — 시군구별 통계 테이블 생성

실행:
    python scripts/build_sigungu_stats.py

생성 테이블: tbl_sigungu_stats
    sigungu              VARCHAR(100) PK
    sido                 VARCHAR(50)        -- 시/도 (지역코드 앞 2자리로 매핑)
    region_code          INT
    lat_median           DOUBLE
    lon_median           DOUBLE
    household_total      BIGINT             -- 시군구 전체 세대수
    geo_code             VARCHAR(10)        -- GeoJSON 통계청 경계 코드 (지도 매핑용)
    avg_price_per_pyeong INT                -- 평균 평당가 만원/3.3㎡ (지도 시각화용)
"""

import sys
from pathlib import Path
from collections import defaultdict
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

# DB region_code 앞 2자리 → GeoJSON code 앞 2자리 (코드 체계가 다름)
DB_TO_GEO_PREFIX = {
    "11": "11", "26": "21", "27": "22", "28": "23",
    "29": "24", "30": "25", "31": "26", "36": "29",
    "41": "31", "42": "32", "43": "33", "44": "34",
    "45": "35", "46": "36", "47": "37", "48": "38", "50": "39",
}


def load_raw() -> pd.DataFrame:
    if CACHE_PATH.exists():
        print(f"[cache] {CACHE_PATH}")
        return pd.read_parquet(
            CACHE_PATH,
            columns=["시군구", "지역코드", "위도", "경도", "세대수", "거래금액", "전용면적"],
        )
    print("[DB] 전체 조회 중...")
    rows = fetch_all(
        "SELECT sigungu AS 시군구, region_code AS 지역코드, "
        "latitude AS 위도, longitude AS 경도, household_cnt AS 세대수, "
        "deal_amount AS 거래금액, area AS 전용면적 "
        "FROM tbl_apart_deals"
    )
    return pd.DataFrame(rows)


def build_geo_code_map() -> dict:
    """DB region_code → GeoJSON code 매핑 딕셔너리 반환.

    같은 시/도 prefix 내에서 region_code 오름차순 ↔ GeoJSON code 오름차순으로 1:1 매핑.
    GeoJSON을 직접 로드하지 않도록 오프라인 하드코딩 매핑을 사용.
    """
    # GeoJSON 통계청 코드 전체 목록 (prefix별 오름차순, 2018년 기준 250개)
    GEO_CODES_BY_PREFIX = {
        "11": ["11010","11020","11030","11040","11050","11060","11070","11080","11090","11100",
               "11110","11120","11130","11140","11150","11160","11170","11180","11190","11200",
               "11210","11220","11230","11240","11250"],
        "21": ["21010","21020","21030","21040","21050","21060","21070","21080","21090","21100",
               "21110","21120","21130","21140","21150","21310"],
        "22": ["22010","22020","22030","22040","22050","22060","22070","22310"],
        "23": ["23010","23020","23030","23040","23050","23060","23070","23080","23310","23320"],
        "24": ["24010","24020","24030","24040","24050"],
        "25": ["25010","25020","25030","25040","25050"],
        "26": ["26010","26020","26030","26040","26310"],
        "29": ["29010"],
        "31": ["31011","31012","31013","31014","31021","31022","31023","31030","31041","31042",
               "31050","31060","31070","31080","31091","31092","31101","31103","31104","31110",
               "31120","31130","31140","31150","31160","31170","31180","31191","31192","31193",
               "31200","31210","31220","31230","31240","31250","31260","31270","31280","31350",
               "31370","31380"],
        "32": ["32010","32020","32030","32040","32050","32060","32070","32310","32320","32330",
               "32340","32350","32360","32370","32380","32390","32400","32410"],
        "33": ["33020","33030","33041","33042","33043","33044","33320","33330","33340","33350",
               "33360","33370","33380","33390"],
        "34": ["34011","34012","34020","34030","34040","34050","34060","34070","34080","34310",
               "34330","34340","34350","34360","34370","34380"],
        "35": ["35011","35012","35020","35030","35040","35050","35060","35310","35320","35330",
               "35340","35350","35360","35370","35380"],
        "36": ["36010","36020","36030","36040","36060","36310","36320","36330","36350","36360",
               "36370","36380","36390","36400","36410","36420","36430","36440","36450","36460",
               "36470","36480"],
        "37": ["37011","37012","37020","37030","37040","37050","37060","37070","37080","37090",
               "37100","37310","37320","37330","37340","37350","37360","37370","37380","37390",
               "37400","37410","37420","37430"],
        "38": ["38030","38050","38060","38070","38080","38090","38100","38111","38112","38113",
               "38114","38115","38310","38320","38330","38340","38350","38360","38370","38380",
               "38390","38400"],
        "39": ["39010","39020"],
    }

    # DB region_code 목록 (DB에서 조회)
    rows = fetch_all(
        "SELECT DISTINCT region_code FROM tbl_apart_deals ORDER BY region_code"
    )
    db_codes = [str(r["region_code"]) for r in rows]

    db_by_prefix = defaultdict(list)
    for rc in db_codes:
        db_by_prefix[rc[:2]].append(rc)

    mapping = {}
    for db_prefix, geo_prefix in DB_TO_GEO_PREFIX.items():
        db_list = db_by_prefix.get(db_prefix, [])
        geo_list = GEO_CODES_BY_PREFIX.get(geo_prefix, [])
        for db_code, geo_code in zip(db_list, geo_list):
            mapping[db_code] = geo_code

    return mapping


def build_stats(df: pd.DataFrame, geo_code_map: dict) -> pd.DataFrame:
    for col in ["지역코드", "위도", "경도", "세대수", "거래금액", "전용면적"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["시군구", "지역코드", "위도", "경도"])
    df = df[df["시군구"].str.strip() != ""]

    # 평당가 컬럼 생성 (면적 0 제외)
    df = df[df["전용면적"] > 0].copy()
    df["평당가"] = df["거래금액"] / (df["전용면적"] / 3.3)

    stats = (
        df.groupby("시군구", as_index=False)
        .agg(
            지역코드=("지역코드", "first"),
            lat_median=("위도", "median"),
            lon_median=("경도", "median"),
            household_total=("세대수", "max"),
            avg_price_per_pyeong=("평당가", "mean"),
        )
        .sort_values("지역코드")
        .reset_index(drop=True)
    )

    stats["lat_median"]           = stats["lat_median"].round(6)
    stats["lon_median"]           = stats["lon_median"].round(6)
    stats["household_total"]      = stats["household_total"].fillna(0).astype(int)
    stats["avg_price_per_pyeong"] = stats["avg_price_per_pyeong"].round(0).fillna(0).astype(int)
    stats["sido"]                 = stats["지역코드"].astype(str).str[:2].map(SIDO_MAP).fillna("기타")
    stats["geo_code"]             = stats["지역코드"].astype(str).map(geo_code_map)

    return stats


def create_table():
    with db_cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS tbl_sigungu_stats")
        cur.execute("""
            CREATE TABLE tbl_sigungu_stats (
                sigungu              VARCHAR(100) NOT NULL PRIMARY KEY,
                sido                 VARCHAR(50)  NOT NULL,
                region_code          INT          NOT NULL,
                lat_median           DOUBLE       NOT NULL,
                lon_median           DOUBLE       NOT NULL,
                household_total      BIGINT       NOT NULL,
                geo_code             VARCHAR(10)  NULL,
                avg_price_per_pyeong INT          NULL,
                INDEX idx_sido (sido),
                INDEX idx_region_code (region_code),
                INDEX idx_geo_code (geo_code)
            ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
        """)
    print("  테이블 생성 완료: tbl_sigungu_stats")


def insert_stats(stats: pd.DataFrame):
    rows = [
        (
            row["시군구"], row["sido"], int(row["지역코드"]),
            float(row["lat_median"]), float(row["lon_median"]),
            int(row["household_total"]),
            row["geo_code"] if pd.notna(row["geo_code"]) else None,
            int(row["avg_price_per_pyeong"]) if pd.notna(row["avg_price_per_pyeong"]) else None,
        )
        for _, row in stats.iterrows()
    ]
    with db_cursor() as cur:
        cur.executemany(
            "INSERT INTO tbl_sigungu_stats "
            "(sigungu, sido, region_code, lat_median, lon_median, household_total, geo_code, avg_price_per_pyeong) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
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

    print("\n[2] geo_code 매핑 테이블 생성...")
    geo_code_map = build_geo_code_map()
    print(f"    {len(geo_code_map)}개 region_code 매핑 완료")

    print("\n[3] 시군구별 집계...")
    stats = build_stats(df, geo_code_map)
    print(f"    {len(stats)}개 시군구 집계 완료")
    print(stats[["시군구", "sido", "geo_code", "avg_price_per_pyeong"]].head(5).to_string(index=False))

    print("\n[4] DB 저장...")
    create_table()
    insert_stats(stats)

    print("\n완료.")


if __name__ == "__main__":
    main()
