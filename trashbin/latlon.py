import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

# threshold 설정
threshold_km = 0.75

# 지구 반지름, km
EARTH_RADIUS_KM = 6371.0088

df1 = pd.read_csv('거리.csv', encoding='utf-8')
df2 = pd.read_csv('station_cleaned.csv', encoding='cp949')

# 결측치 제거용 인덱스 보존
df1_valid = df1.dropna(subset=["위도", "경도"]).copy()
df2_valid = df2.dropna(subset=["위도", "경도"]).copy()

# BallTree는 [위도, 경도] 순서의 라디안 값을 사용
df1_coords_rad = np.radians(df1_valid[["위도", "경도"]].to_numpy())
df2_coords_rad = np.radians(df2_valid[["위도", "경도"]].to_numpy())

# df2 좌표를 기준으로 BallTree 생성
tree = BallTree(df2_coords_rad, metric="haversine")

# km 단위 threshold를 라디안 단위로 변환
threshold_rad = threshold_km / EARTH_RADIUS_KM

# df1의 각 좌표에 대해 threshold 이내 df2 좌표 개수 계산
indices = tree.query_radius(df1_coords_rad, r=threshold_rad)

# 결과 저장
df1["인근역수"] = 0
df1.loc[df1_valid.index, "인근역수"] = [len(idx) for idx in indices]

df1.to_csv("거리.csv")