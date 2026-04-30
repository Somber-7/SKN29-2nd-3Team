import pandas as pd

df = pd.read_csv('station.csv')

# 원본 보존을 위해 copy
df_clean = df.copy()

# 비교용 역사명 생성
# 예: 공덕역 -> 공덕, 공덕 -> 공덕
df_clean["역사명_정규화"] = (
    df_clean["역사명"]
    .astype(str)
    .str.strip()
    .str.replace(r"역$", "", regex=True)
)

# 정규화된 역사명 기준으로 중복 제거
df_clean = df_clean.drop_duplicates(
    subset=["역사명_정규화"],
    keep="first"
).reset_index(drop=True)

# 필요 없으면 비교용 컬럼 삭제
df_clean = df_clean.drop(columns=["역사명_정규화"])

df_clean.to_csv('station_cleaned.csv')