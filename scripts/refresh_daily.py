"""오늘 날짜 기준으로 샘플 데이터를 재생성하고 예측까지 한 번에 갱신합니다.

순서: make_sample_data(날씨/포트홀/보수 이력 재생성, roads.csv는 보존) -> predict(오늘 기준 예측)
-> 기존 predictions_v2.csv에 있던 address 컬럼을 grid_id 기준으로 재활용(카카오 API 재호출 없음).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.common import load_config, resolve_path
from src.predict import main as run_predict


def main() -> None:
    config = load_config(BASE_DIR / "config.yaml")
    prediction_path = resolve_path(config, "predictions")

    address_backup: pd.DataFrame | None = None
    if prediction_path.exists():
        previous = pd.read_csv(prediction_path)
        if "address" in previous.columns:
            address_backup = previous[["grid_id", "address"]]

    print("1/2 샘플 데이터 재생성 중 (오늘 기준)...")
    subprocess.run(
        [sys.executable, str(BASE_DIR / "scripts" / "make_sample_data.py")],
        check=True,
        cwd=BASE_DIR,
    )

    print("2/2 예측 재실행 중...")
    run_predict(str(BASE_DIR / "config.yaml"))

    if address_backup is not None:
        pred = pd.read_csv(prediction_path)
        merged = pred.merge(address_backup, on="grid_id", how="left")
        merged.to_csv(prediction_path, index=False)
        missing = int(merged["address"].isna().sum())
        print(f"주소 컬럼 재활용 완료 (매칭 실패 {missing}개, 필요 시 scripts/geocode_grids.py로 채우십시오)")

    print("완료: Streamlit 앱을 새로고침하면 오늘 날짜 기준 예측이 반영됩니다.")


if __name__ == "__main__":
    main()
