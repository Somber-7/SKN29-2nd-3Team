"""scripts/_common.py — 테스트 스크립트 공통 유틸"""

import sys
import time
from datetime import datetime
from pathlib import Path
from io import StringIO

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from utils.db import load_apart_deals

SAMPLE = None  # 전체 500만 건 사용

LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)


class Logger:
    """터미널 출력과 파일 기록을 동시에 처리합니다."""

    def __init__(self):
        self._buf = StringIO()

    def log(self, msg: str = ""):
        print(msg)
        self._buf.write(msg + "\n")

    def save(self, path: Path):
        path.write_text(self._buf.getvalue(), encoding="utf-8")
        print(f"\n결과 저장 완료: {path}")


def section(logger: Logger, title: str):
    logger.log()
    logger.log("=" * 55)
    logger.log(f"  {title}")
    logger.log("=" * 55)


def load_data(logger: Logger) -> "pd.DataFrame":
    import pandas as pd
    section(logger, "데이터 로딩")
    t0 = time.time()
    df = load_apart_deals(limit=SAMPLE)
    logger.log(f"  로딩 완료 : {len(df):,}건 ({time.time() - t0:.1f}초)")
    return df


def log_header(logger: Logger):
    run_at = datetime.now()
    logger.log(f"# 모델 학습 결과")
    logger.log(f"- 실행 시각 : {run_at.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.log(f"- GPU       : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A (CPU)'}")
    logger.log(f"- 데이터    : {'전체' if SAMPLE is None else f'{SAMPLE:,}건 샘플'}")
    return run_at
