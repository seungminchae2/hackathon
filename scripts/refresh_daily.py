"""
기상청 단기예보를 갱신하고 오늘 날짜 기준
도로 위험도 예측을 자동으로 갱신합니다.

순서:
1. 기존 predictions_v2.csv의 address 컬럼 백업
2. 기상청 단기예보 API 호출
3. weather_forecast.csv 갱신
4. 오늘 날짜 기준 위험도 예측
5. 기존 address 컬럼 재활용
"""

from __future__ import annotations

import subprocess
import sys
from datetime import date
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent

sys.path.insert(
    0,
    str(BASE_DIR),
)

from src.common import (
    load_config,
    resolve_path,
)

from src.predict import (
    main as run_predict,
)


CONFIG_PATH = (
    BASE_DIR
    / "config.yaml"
)

WEATHER_UPDATE_SCRIPT = (
    BASE_DIR
    / "scripts"
    / "update_weather_forecast.py"
)


def backup_addresses(
    prediction_path: Path,
) -> pd.DataFrame | None:

    if not prediction_path.exists():
        return None

    previous = pd.read_csv(
        prediction_path,
        dtype={
            "grid_id": str,
        },
    )

    if (
        "grid_id" not in previous.columns
        or "address" not in previous.columns
    ):
        return None

    address_backup = (
        previous[
            [
                "grid_id",
                "address",
            ]
        ]
        .drop_duplicates(
            subset=[
                "grid_id",
            ]
        )
        .copy()
    )

    address_backup[
        "grid_id"
    ] = (
        address_backup[
            "grid_id"
        ]
        .astype(str)
    )

    return address_backup


def restore_addresses(
    prediction_path: Path,
    address_backup: pd.DataFrame | None,
) -> None:

    if address_backup is None:
        print(
            "기존 address 컬럼이 없어 "
            "주소 재활용을 건너뜁니다."
        )
        return

    pred = pd.read_csv(
        prediction_path,
        dtype={
            "grid_id": str,
        },
    )

    if "address" in pred.columns:

        pred = pred.drop(
            columns=[
                "address",
            ]
        )

    pred["grid_id"] = (
        pred[
            "grid_id"
        ]
        .astype(str)
    )

    merged = pred.merge(
        address_backup,
        on="grid_id",
        how="left",
    )

    merged.to_csv(
        prediction_path,
        index=False,
    )

    missing = int(
        merged[
            "address"
        ]
        .isna()
        .sum()
    )

    print(
        f"주소 컬럼 재활용 완료 "
        f"(매칭 실패 {missing:,}개)"
    )


def main() -> None:

    config = load_config(
        CONFIG_PATH
    )

    prediction_path = resolve_path(
        config,
        "predictions",
    )

    # -----------------------------------------
    # 1. 기존 주소 백업
    # -----------------------------------------

    address_backup = (
        backup_addresses(
            prediction_path
        )
    )

    # -----------------------------------------
    # 2. 기상청 단기예보 API 갱신
    # -----------------------------------------

    print()
    print(
        "1/2 기상청 단기예보 갱신 중..."
    )

    subprocess.run(
        [
            sys.executable,
            str(
                WEATHER_UPDATE_SCRIPT
            ),
        ],
        check=True,
        cwd=BASE_DIR,
    )

    # -----------------------------------------
    # 3. 오늘 날짜 기준 위험도 예측
    # -----------------------------------------

    today = (
        date.today()
        .isoformat()
    )

    print()
    print(
        f"2/2 {today} 기준 "
        f"도로 위험도 예측 중..."
    )

    run_predict(
        str(
            CONFIG_PATH
        ),
        today,
    )

    # -----------------------------------------
    # 4. 주소 복원
    # -----------------------------------------

    restore_addresses(
        prediction_path,
        address_backup,
    )

    print()
    print(
        "완료: Streamlit 앱을 새로고침하면 "
        "기상청 최신 단기예보가 반영된 "
        "오늘 기준 위험도가 표시됩니다."
    )


if __name__ == "__main__":
    main()