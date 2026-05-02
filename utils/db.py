"""
utils/db.py — MySQL 데이터베이스 연결 유틸리티

conf/.env 파일의 접속 정보를 자동으로 로드합니다.
모든 쿼리 함수는 연결 열기/닫기, commit/rollback을 자동 처리합니다.

사용 예시:
    from utils.db import fetch_all, fetch_one, execute, load_apart_deals

    # 여러 행 조회
    rows = fetch_all("SELECT * FROM users WHERE age > %s", (20,))

    # 단일 행 조회
    user = fetch_one("SELECT * FROM users WHERE id = %s", (1,))

    # INSERT / UPDATE / DELETE
    affected = execute("DELETE FROM logs WHERE id = %s", (42,))

    # 아파트 거래 데이터 — 한글 컬럼명 DataFrame으로 반환
    import pandas as pd
    df = load_apart_deals(limit=100_000)
    df = load_apart_deals(sigungu="강남구", limit=50_000)
"""

import os
from pathlib import Path
from contextlib import contextmanager
from typing import Optional

import pandas as pd
import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv

_CACHE_PATH = Path(__file__).parent.parent / "data" / "cache" / "apart_deals.parquet"

load_dotenv(Path(__file__).parent.parent / "conf" / ".env")


def get_connection():
    """conf/.env 값을 읽어 MySQL 연결 객체를 반환합니다.

    직접 사용하기보다 db_cursor() context manager를 사용하는 것을 권장합니다.
    연결 후 반드시 .close()를 호출해야 합니다.
    """
    return mysql.connector.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", 3306)),
        user=os.getenv("DB_USER", "root"),
        password=os.getenv("DB_PASSWORD", ""),
        database=os.getenv("DB_NAME", ""),
    )


@contextmanager
def db_cursor(dictionary=True):
    """커서를 context manager로 제공합니다. 예외 발생 시 자동 rollback합니다.

    Args:
        dictionary: True면 결과를 dict로 반환, False면 tuple로 반환

    사용 예시:
        with db_cursor() as cursor:
            cursor.execute("SELECT * FROM users")
            rows = cursor.fetchall()
    """
    conn = get_connection()
    cursor = conn.cursor(dictionary=dictionary)
    try:
        yield cursor
        conn.commit()
    except Error as e:
        conn.rollback()
        raise e
    finally:
        cursor.close()
        conn.close()


def fetch_all(query: str, params: tuple = ()) -> list[dict]:
    """SELECT 결과 전체를 dict 리스트로 반환합니다.

    Args:
        query:  실행할 SQL 문자열. 파라미터는 %s 플레이스홀더 사용
        params: 플레이스홀더에 바인딩할 값 튜플 (SQL 인젝션 방지)

    Returns:
        행(row)마다 컬럼명을 키로 갖는 dict 리스트. 결과 없으면 빈 리스트.

    예시:
        rows = fetch_all("SELECT * FROM users WHERE age > %s", (20,))
        # [{"id": 1, "name": "홍길동", "age": 25}, ...]
    """
    with db_cursor() as cursor:
        cursor.execute(query, params)
        return cursor.fetchall()


def fetch_one(query: str, params: tuple = ()) -> dict | None:
    """SELECT 결과 첫 번째 행을 dict로 반환합니다. 결과 없으면 None.

    Args:
        query:  실행할 SQL 문자열
        params: 플레이스홀더에 바인딩할 값 튜플

    Returns:
        첫 번째 행의 dict, 결과 없으면 None

    예시:
        user = fetch_one("SELECT * FROM users WHERE id = %s", (1,))
        if user:
            print(user["name"])
    """
    with db_cursor() as cursor:
        cursor.execute(query, params)
        return cursor.fetchone()


def execute(query: str, params: tuple = ()) -> int:
    """INSERT / UPDATE / DELETE 쿼리를 실행하고 영향받은 행 수를 반환합니다.

    Args:
        query:  실행할 SQL 문자열
        params: 플레이스홀더에 바인딩할 값 튜플

    Returns:
        영향받은 행(row) 수 (int)

    예시:
        count = execute("UPDATE users SET active = 0 WHERE id = %s", (5,))
        print(f"{count}개 행 변경됨")
    """
    with db_cursor() as cursor:
        cursor.execute(query, params)
        return cursor.rowcount


# ── Apart Deal 전용 ───────────────────────────────────

# DB 영문 컬럼명 → 원본 한글 컬럼명 매핑
# 모델 파일은 한글 컬럼명 기준으로 작성되어 있으므로 쿼리에서 alias 처리합니다.
_ALIAS_SQL = """
    region_code   AS 지역코드,
    sigungu       AS 시군구,
    dong          AS 법정동,
    jibun         AS 지번,
    apt_name      AS 아파트,
    brand         AS 브랜드,
    is_brand      AS 브랜드여부,
    build_year    AS 건축년도,
    household_cnt AS 세대수,
    deal_date     AS 거래일,
    floor         AS 층,
    area          AS 전용면적,
    deal_amount   AS 거래금액,
    base_rate     AS 기준금리,
    latitude      AS 위도,
    longitude     AS 경도,
    school_cnt    AS 인근학교수,
    station_cnt   AS 인근역수
"""


def load_apart_deals(
    sigungu: Optional[str] = None,
    apt_name: Optional[str] = None,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    limit: Optional[int] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """tbl_apart_deals를 한글 컬럼명 DataFrame으로 반환합니다.

    모델 파일(classification, clustering 등)이 기대하는 한글 컬럼명을
    DB 조회 시 alias로 자동 변환하여 반환합니다.

    Args:
        sigungu:   시군구 필터 (예: "강남구"). None이면 전체.
        apt_name:  아파트명 필터 (부분 일치). None이면 전체.
        year_from: 거래 시작 연도 (포함). None이면 제한 없음.
        year_to:   거래 종료 연도 (포함). None이면 제한 없음.
        limit:     최대 반환 행 수. None이면 전체 (500만 건 주의).
        use_cache: True면 Parquet 캐시 사용 (필터/limit 없는 전체 조회 시만 적용).
                   캐시 없으면 DB에서 로딩 후 자동 저장. False면 항상 DB 조회.

    Returns:
        한글 컬럼명을 가진 pandas DataFrame.
        컬럼: 지역코드, 시군구, 법정동, 지번, 아파트, 브랜드, 브랜드여부,
              건축년도, 세대수, 거래일, 층, 전용면적, 거래금액,
              기준금리, 위도, 경도, 인근학교수, 인근역수

    예시:
        df = load_apart_deals()                                  # 캐시 사용 (빠름)
        df = load_apart_deals(use_cache=False)                   # DB 직접 조회 + 캐시 갱신
        df = load_apart_deals(sigungu="강남구", limit=50_000)    # 필터 시 항상 DB 조회
    """
    is_full_load = not any([sigungu, apt_name, year_from, year_to, limit])

    if use_cache and is_full_load:
        if _CACHE_PATH.exists():
            print(f"[cache] Parquet 캐시 로딩 — {_CACHE_PATH}")
            return pd.read_parquet(_CACHE_PATH)
        print("[cache] 캐시 없음 — DB에서 전체 로딩 후 저장합니다.")

    conditions = []
    params = []

    if sigungu:
        conditions.append("sigungu = %s")
        params.append(sigungu)
    if apt_name:
        conditions.append("apt_name LIKE %s")
        params.append(f"%{apt_name}%")
    if year_from:
        conditions.append("YEAR(deal_date) >= %s")
        params.append(year_from)
    if year_to:
        conditions.append("YEAR(deal_date) <= %s")
        params.append(year_to)

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    limit_clause = f"LIMIT {int(limit)}" if limit else ""

    query = f"SELECT {_ALIAS_SQL} FROM tbl_apart_deals {where} {limit_clause}"

    rows = fetch_all(query, tuple(params))
    df = pd.DataFrame(rows)

    if use_cache and is_full_load:
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(_CACHE_PATH, index=False)
        print(f"[cache] Parquet 캐시 저장 완료 — {_CACHE_PATH}")

    return df
