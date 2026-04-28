CREATE DATABASE IF NOT EXISTS gaming_mental_health
    DEFAULT CHARACTER SET utf8mb4
    DEFAULT COLLATE utf8mb4_unicode_ci;

USE gaming_mental_health;

CREATE TABLE IF NOT EXISTS tbl_gaming_mental_health (
    id                          INT             UNSIGNED NOT NULL AUTO_INCREMENT   COMMENT '고유 식별자',

    -- 인구통계 (Demographic)
    age                         INT             NOT NULL    COMMENT '나이',
    gender                      VARCHAR(10)     NOT NULL    COMMENT '성별 (Male / Female)',
    income                      INT             NOT NULL    COMMENT '월 소득 추정치 (달러)',

    -- 게임 행동 (Gaming Behavior)
    daily_gaming_hours          FLOAT           NOT NULL    COMMENT '하루 평균 게임 시간 (시간)',
    weekly_sessions             INT             NOT NULL    COMMENT '주간 게임 세션 수',
    years_gaming                INT             NOT NULL    COMMENT '게임 경력 (년)',
    weekend_gaming_hours        FLOAT           NOT NULL    COMMENT '주말 게임 시간 (시간)',
    multiplayer_ratio           FLOAT           NOT NULL    COMMENT '멀티플레이 비율 (0~1)',
    violent_games_ratio         FLOAT           NOT NULL    COMMENT '폭력적 게임 비율 (0~1)',
    mobile_gaming_ratio         FLOAT           NOT NULL    COMMENT '모바일 게임 비율 (0~1)',
    night_gaming_ratio          FLOAT           NOT NULL    COMMENT '야간 게임 비율 (0~1)',
    competitive_rank            INT             NOT NULL    COMMENT '경쟁 게임 랭킹 점수',
    esports_interest            INT             NOT NULL    COMMENT 'e스포츠 참여 관심도 (0~10)',
    streaming_hours             FLOAT           NOT NULL    COMMENT '주간 게임 스트리밍 시청 시간 (시간)',
    microtransactions_spending  FLOAT           NOT NULL    COMMENT '월 인앱 결제 지출액 (달러)',
    headset_usage               INT             NOT NULL    COMMENT '헤드셋 사용 여부 (0: 미사용 / 1: 사용)',

    -- 정신 건강 (Psychological Health)
    stress_level                INT             NOT NULL    COMMENT '스트레스 수준 (1~10)',
    anxiety_score               FLOAT           NOT NULL    COMMENT '불안 점수 (0~10)',
    depression_score            FLOAT           NOT NULL    COMMENT '우울 점수 (0~10)',
    addiction_level             FLOAT           NOT NULL    COMMENT '게임 중독 수준 (0~10)',
    loneliness_score            FLOAT           NOT NULL    COMMENT '외로움 점수 (0~10)',
    aggression_score            FLOAT           NOT NULL    COMMENT '공격성 점수 (0~10)',
    happiness_score             FLOAT           NOT NULL    COMMENT '행복감 점수 (0~10)',

    -- 사회 환경 (Social Environment)
    social_interaction_score    FLOAT           NOT NULL    COMMENT '사회적 활동 지수 (0~10)',
    relationship_satisfaction   FLOAT           NOT NULL    COMMENT '대인관계 만족도 (0~10)',
    friends_gaming_count        INT             NOT NULL    COMMENT '함께 게임하는 친구 수',
    online_friends              INT             NOT NULL    COMMENT '온라인 친구 수',
    toxic_exposure              FLOAT           NOT NULL    COMMENT '독성 커뮤니티 노출 비율 (0~1)',
    parental_supervision        INT             NOT NULL    COMMENT '부모 감독 점수 (0~10)',

    -- 라이프스타일 (Lifestyle)
    sleep_hours                 FLOAT           NOT NULL    COMMENT '하루 평균 수면 시간 (시간)',
    exercise_hours              FLOAT           NOT NULL    COMMENT '주간 운동 시간 (시간)',
    caffeine_intake             FLOAT           NOT NULL    COMMENT '하루 카페인 섭취량 (잔)',
    screen_time_total           FLOAT           NOT NULL    COMMENT '하루 총 스크린 타임 (시간)',
    internet_quality            INT             NOT NULL    COMMENT '인터넷 연결 품질 점수 (1~10)',

    -- 신체 건강 (Physical Health)
    bmi                         FLOAT           NOT NULL    COMMENT '체질량지수 (Body Mass Index)',
    eye_strain_score            FLOAT           NOT NULL    COMMENT '눈 피로도 점수 (0~10)',
    back_pain_score             FLOAT           NOT NULL    COMMENT '허리 통증 점수 (0~10)',

    -- 생산성 / 성과 (Productivity / Performance)
    academic_performance        FLOAT           NOT NULL    COMMENT '학업 성취도 점수 (0~100)',
    work_productivity           FLOAT           NOT NULL    COMMENT '업무 생산성 점수 (0~100)',

    PRIMARY KEY (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='게임 이용자 정신 건강 데이터';
