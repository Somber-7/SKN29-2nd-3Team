"""
utils/db.py — MySQL 데이터베이스 연결 유틸리티

conf/.env 파일의 접속 정보를 자동으로 로드합니다.
모든 쿼리 함수는 연결 열기/닫기, commit/rollback을 자동 처리합니다.

사용 예시:
    from utils.db import fetch_all, fetch_one, execute

    # 여러 행 조회
    rows = fetch_all("SELECT * FROM users WHERE age > %s", (20,))

    # 단일 행 조회
    user = fetch_one("SELECT * FROM users WHERE id = %s", (1,))

    # INSERT / UPDATE / DELETE
    affected = execute("DELETE FROM logs WHERE id = %s", (42,))
"""

import os
from pathlib import Path
from contextlib import contextmanager

import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv

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
