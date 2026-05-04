"""
scripts/save_page_data.py — 페이지 표시용 집계 데이터 사전 계산 후 pkl 저장

실행 방법:
    python scripts/save_page_data.py

저장 위치:
    data/cache/home_data.pkl
    data/cache/price_trend_data.pkl
    data/cache/location_data.pkl
"""

import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.db import load_apart_deals, fetch_all

CACHE_DIR = Path(__file__).parent.parent / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 개요 (Home)
# =============================================================================

def save_home_data(df: pd.DataFrame):
    print("\n[1/3] 개요 데이터 계산 중...")
    t0 = time.time()

    df = df.copy()
    df["거래일"] = pd.to_datetime(df["거래일"])
    df["거래금액"] = pd.to_numeric(df["거래금액"], errors="coerce")
    df = df.dropna(subset=["거래금액"])

    sido_map_rows = fetch_all("SELECT sigungu, sido FROM tbl_sigungu_stats")
    sg_to_sido = {r["sigungu"]: r["sido"] for r in sido_map_rows}
    df["시도"] = df["시군구"].map(sg_to_sido)

    cnt      = len(df)
    avg_amt  = int(df["거래금액"].mean().round(0))
    min_date = df["거래일"].min().strftime("%Y.%m")
    max_date = df["거래일"].max().strftime("%Y.%m")

    vc = df["시군구"].value_counts()
    top_sg   = vc.idxmax()
    top_sido = sg_to_sido.get(top_sg, "")
    top_cnt  = int(vc.max())

    df["연월"] = df["거래일"].dt.to_period("M").dt.to_timestamp()
    monthly = df.groupby("연월").agg(
        거래량=("거래금액", "count"),
        평균가=("거래금액", "mean"),
    ).reset_index()
    monthly["평균가"] = monthly["평균가"].round(0).astype(int)

    df["연도"] = df["거래일"].dt.year
    yearly = df.groupby("연도").agg(
        거래량=("거래금액", "count"),
        평균가=("거래금액", "mean"),
    ).reset_index()
    yearly["평균가"] = yearly["평균가"].round(0).astype(int)

    result = {
        "cnt": cnt, "avg_amt": avg_amt,
        "min_date": min_date, "max_date": max_date,
        "top_sg": top_sg, "top_sido": top_sido, "top_cnt": top_cnt,
        "monthly": monthly, "yearly": yearly,
    }

    path = CACHE_DIR / "home_data.pkl"
    joblib.dump(result, path)
    print(f"  완료 : {time.time() - t0:.1f}초  →  {path}")


# =============================================================================
# 가격 추이 분석
# =============================================================================

def save_price_trend_data(df: pd.DataFrame):
    print("\n[2/3] 가격 추이 데이터 계산 중...")
    t0 = time.time()

    df = df.copy()
    df["거래일"] = pd.to_datetime(df["거래일"])
    df["거래금액"] = pd.to_numeric(df["거래금액"], errors="coerce")
    df = df.dropna(subset=["거래금액"])

    sido_map_rows = fetch_all("SELECT sigungu, sido FROM tbl_sigungu_stats")
    sg_to_sido = {r["sigungu"]: r["sido"] for r in sido_map_rows}
    df["시도"] = df["시군구"].map(sg_to_sido)
    df["연월"] = df["거래일"].dt.to_period("M").dt.to_timestamp()

    # 전국 월별
    national = df.groupby("연월")["거래금액"].mean().round(0).reset_index()
    national.columns = ["date", "전국"]

    # 시도별 월별
    sido_monthly = df.groupby(["시도", "연월"])["거래금액"].mean().round(0).reset_index()
    sido_monthly.columns = ["시도", "date", "avg"]

    # 시군구별 월별
    sigungu_monthly = df.groupby(["시군구", "연월"])["거래금액"].mean().round(0).reset_index()
    sigungu_monthly.columns = ["시군구", "date", "avg"]

    # 메타
    sido_list = sorted(df["시도"].dropna().unique().tolist())
    sigungu_map: dict[str, list[str]] = {}
    sg_full_map: dict[str, dict[str, str]] = {}
    for s in sido_list:
        sgs = sorted(df[df["시도"] == s]["시군구"].dropna().unique().tolist())
        sigungu_map[s] = [sg.split()[-1] for sg in sgs]
        sg_full_map[s] = {sg.split()[-1]: sg for sg in sgs}

    result = {
        "national": national,
        "sido_monthly": sido_monthly,
        "sigungu_monthly": sigungu_monthly,
        "sido_list": sido_list,
        "sigungu_map": sigungu_map,
        "sg_full_map": sg_full_map,
    }

    path = CACHE_DIR / "price_trend_data.pkl"
    joblib.dump(result, path)
    print(f"  완료 : {time.time() - t0:.1f}초  →  {path}")


# =============================================================================
# 입지 분석
# =============================================================================

def save_location_data(df: pd.DataFrame):
    print("\n[3/3] 입지 분석 데이터 계산 중...")
    t0 = time.time()

    cols = ["기준금리", "인근역수", "인근학교수", "세대수", "건축년도", "거래금액", "브랜드여부"]
    data = df[cols].copy()
    for c in cols:
        data[c] = pd.to_numeric(data[c], errors="coerce")
    data = data.dropna()

    # 상관계수
    corr = data.corr()["거래금액"].to_dict()

    # 역세권 여부별
    data["역세권"] = data["인근역수"].apply(lambda x: "역세권 (1개 이상)" if x >= 1 else "비역세권")
    subway_df = data.groupby("역세권")["거래금액"].mean().round(0).reset_index()

    # 인근 역 수 구간별
    data["역수_구간"] = data["인근역수"].clip(upper=5).astype(int).astype(str)
    data.loc[data["인근역수"] >= 5, "역수_구간"] = "5+"
    station_grp_df = (
        data.groupby("역수_구간")["거래금액"].mean().round(0)
        .reindex(["0", "1", "2", "3", "4", "5+"])
        .reset_index()
    )
    station_grp_df.columns = ["인근 역 수", "거래금액"]

    # 인근 학교 수 구간별
    data["학교수_구간"] = data["인근학교수"].clip(upper=6).astype(int).astype(str)
    data.loc[data["인근학교수"] >= 6, "학교수_구간"] = "6+"
    school_grp_df = (
        data.groupby("학교수_구간")["거래금액"].mean().round(0)
        .reindex(["0", "1", "2", "3", "4", "5", "6+"])
        .reset_index()
    )
    school_grp_df.columns = ["인근 학교 수", "거래금액"]

    # 브랜드 여부별
    brand_df = data.groupby("브랜드여부")["거래금액"].mean().round(0).reset_index()
    brand_df["구분"] = brand_df["브랜드여부"].map({1: "브랜드", 0: "비브랜드"})
    brand_df = brand_df[["구분", "거래금액"]]

    # 건축연식 구간별
    bins   = [1970, 1980, 1990, 2000, 2005, 2010, 2015, 2020, 2025]
    labels = ["~1980", "80~90", "90~00", "00~05", "05~10", "10~15", "15~20", "20~"]
    data["연식_구간"] = pd.cut(data["건축년도"], bins=bins, labels=labels, right=True)
    era_df = (
        data.groupby("연식_구간", observed=True)["거래금액"].mean().round(0)
        .reset_index()
    )
    era_df.columns = ["건축연식", "거래금액"]

    result = {
        "corr": corr,
        "subway_df": subway_df,
        "station_grp_df": station_grp_df,
        "school_grp_df": school_grp_df,
        "brand_df": brand_df,
        "era_df": era_df,
    }

    path = CACHE_DIR / "location_data.pkl"
    joblib.dump(result, path)
    print(f"  완료 : {time.time() - t0:.1f}초  →  {path}")


# =============================================================================
# main
# =============================================================================

def main():
    print("=" * 55)
    print("  페이지 데이터 사전 계산 스크립트")
    print(f"  저장 위치: {CACHE_DIR}")
    print("=" * 55)

    print("\n데이터 로딩 중...")
    t0 = time.time()
    df = load_apart_deals()
    print(f"  로딩 완료 : {len(df):,}건 ({time.time() - t0:.1f}초)")

    save_home_data(df)
    save_price_trend_data(df)
    save_location_data(df)

    print("\n" + "=" * 55)
    print("  저장 완료")
    print("=" * 55)
    print("\n저장된 파일:")
    for f in ["home_data.pkl", "price_trend_data.pkl", "location_data.pkl"]:
        p = CACHE_DIR / f
        if p.exists():
            size_mb = p.stat().st_size / (1024 ** 2)
            print(f"  {f:<35} {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
