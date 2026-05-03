-- =============================================================
-- sql/ddl/create_db.sql
-- Apart Deal 프로젝트 DB 및 테이블 생성
--
-- 실행 방법:
--   mysql -u root -p < sql/ddl/create_db.sql
-- =============================================================

CREATE DATABASE IF NOT EXISTS apart_deal
    DEFAULT CHARACTER SET utf8mb4
    DEFAULT COLLATE utf8mb4_unicode_ci;

USE apart_deal;

-- -------------------------------------------------------------
-- 아파트 실거래가 테이블
-- 거래금액은 원본이 문자열(쉼표 포함)이므로 INT로 변환 후 저장합니다.
-- 층은 음수(-1, -2 등)가 지하층을 의미하므로 그대로 보존합니다.
-- -------------------------------------------------------------
CREATE TABLE IF NOT EXISTS tbl_apart_deals (
    id              BIGINT          NOT NULL AUTO_INCREMENT,

    -- 지역 정보
    region_code     INT             NOT NULL COMMENT '지역코드 — 행정구역 코드 (5자리)',
    sigungu         VARCHAR(50)     COMMENT '시군구 — 시/군/구 명칭',
    dong            VARCHAR(50)     NOT NULL COMMENT '법정동 — 법정동 명칭',
    jibun           VARCHAR(20)     COMMENT '지번 — 부번 포함 (예: 95-15)',

    -- 단지 정보
    apt_name        VARCHAR(100)    COMMENT '아파트 — 단지명',
    brand           VARCHAR(50)     NOT NULL DEFAULT '기타' COMMENT '브랜드 — 브랜드명 (기타 포함)',
    is_brand        TINYINT         NOT NULL DEFAULT 0 COMMENT '브랜드여부 — 0/1',
    build_year      SMALLINT        COMMENT '건축년도 — 준공 연도',
    household_cnt   INT             COMMENT '세대수 — 단지 전체 세대수',

    -- 거래 정보
    deal_date       DATE            NOT NULL COMMENT '거래일 — YYYY-MM-DD',
    floor           SMALLINT        COMMENT '층 — 음수 = 지하층 (예: -1 = B1)',
    area            DECIMAL(8,2)    COMMENT '전용면적 — ㎡',
    deal_amount     INT             COMMENT '거래금액 — 만원, 쉼표 제거 후 저장',

    -- 거시 지표
    base_rate       DECIMAL(5,2)    COMMENT '기준금리 — 거래 시점 기준금리 (%)',

    -- 위치 정보
    latitude        DECIMAL(12,9)   COMMENT '위도 — WGS84',
    longitude       DECIMAL(12,9)   COMMENT '경도 — WGS84',

    -- 입지 정보
    school_cnt      SMALLINT        COMMENT '인근학교수 — 반경 내 학교 수',
    station_cnt     SMALLINT        COMMENT '인근역수 — 반경 내 지하철역 수',

    PRIMARY KEY (id),
    INDEX idx_region_code   (region_code),
    INDEX idx_sigungu       (sigungu),
    INDEX idx_apt_name      (apt_name),
    INDEX idx_deal_date     (deal_date),
    INDEX idx_brand         (brand),
    INDEX idx_deal_amount   (deal_amount)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  COMMENT='아파트 실거래가 원본 데이터 (5,002,839건)';


-- -------------------------------------------------------------
-- 시군구별 통계 테이블
-- tbl_apart_deals에서 집계한 시군구 단위 요약 테이블.
-- scripts/build_sigungu_stats.py 실행으로 생성/갱신합니다.
-- 분류 모델 샘플 예측 UI에서 위도·경도·세대수 기본값으로 사용합니다.
-- -------------------------------------------------------------
CREATE TABLE IF NOT EXISTS tbl_sigungu_stats (
    sigungu          VARCHAR(100)    NOT NULL COMMENT '시군구 — 시/군/구 명칭 (원본 전체명)',
    sido             VARCHAR(50)     NOT NULL COMMENT '시/도 — 지역코드 앞 2자리로 매핑한 광역시/도명',
    region_code      INT             NOT NULL COMMENT '지역코드 — 행정구역 코드 (5자리), 오름차순 정렬 기준',
    lat_median       DOUBLE          NOT NULL COMMENT '위도 중앙값 — 해당 시군구 거래 위도의 median',
    lon_median       DOUBLE          NOT NULL COMMENT '경도 중앙값 — 해당 시군구 거래 경도의 median',
    household_total  BIGINT          NOT NULL COMMENT '전체 세대수 — 시군구 합계 세대수',

    PRIMARY KEY (sigungu),
    INDEX idx_sido (sido),
    INDEX idx_region_code (region_code)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  COMMENT='시군구별 통계 요약 (248개 시군구, build_sigungu_stats.py 생성)';
