from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import unquote

import numpy as np
import pandas as pd
import requests


BASE_DIR = Path(__file__).resolve().parents[1]

OUTPUT_PATH = (
    BASE_DIR
    / "data"
    / "weather_forecast.csv"
)

ENV_PATH = (
    BASE_DIR
    / ".env"
)

API_URL = (
    "https://apis.data.go.kr/"
    "1360000/"
    "VilageFcstInfoService_2.0/"
    "getVilageFcst"
)


# ==========================================================
# Road Doctor에서 사용하는 5개 도시
#
# nx / ny는 기상청 단기예보 격자 좌표
# ==========================================================

STATIONS = [
    {
        "station_id": "JB01",
        "station_name": "전주",
        "lat": 35.8242,
        "lon": 127.1480,
        "nx": 63,
        "ny": 89,
    },
    {
        "station_id": "JB02",
        "station_name": "군산",
        "lat": 35.9677,
        "lon": 126.7366,
        "nx": 56,
        "ny": 92,
    },
    {
        "station_id": "JB03",
        "station_name": "익산",
        "lat": 35.9483,
        "lon": 126.9576,
        "nx": 60,
        "ny": 91,
    },
    {
        "station_id": "JB04",
        "station_name": "남원",
        "lat": 35.4164,
        "lon": 127.3904,
        "nx": 68,
        "ny": 80,
    },
    {
        "station_id": "JB05",
        "station_name": "정읍",
        "lat": 35.5699,
        "lon": 126.8560,
        "nx": 58,
        "ny": 83,
    },
]


# ==========================================================
# .env 읽기
# ==========================================================

def load_env(
    path: Path,
) -> None:

    if not path.exists():
        return

    for raw_line in path.read_text(
        encoding="utf-8"
    ).splitlines():

        line = raw_line.strip()

        if (
            not line
            or line.startswith("#")
            or "=" not in line
        ):
            continue

        key, value = line.split(
            "=",
            1,
        )

        os.environ[key.strip()] = (
            value
            .strip()
            .strip('"')
            .strip("'")
        )


# ==========================================================
# API 발표시각 결정
#
# 단기예보 발표시각:
# 02, 05, 08, 11, 14, 17, 20, 23시
#
# API 반영 지연을 고려하여
# 발표 후 약 15분이 지난 시각만 사용
# ==========================================================

def get_latest_base_datetime() -> datetime:

    now = datetime.now()

    base_hours = [
        2,
        5,
        8,
        11,
        14,
        17,
        20,
        23,
    ]

    candidates = []

    for day_offset in [
        0,
        -1,
    ]:

        target_date = (
            now
            + timedelta(
                days=day_offset
            )
        )

        for hour in base_hours:

            candidate = (
                target_date.replace(
                    hour=hour,
                    minute=0,
                    second=0,
                    microsecond=0,
                )
            )

            # API 데이터 생성 지연 고려
            available_time = (
                candidate
                + timedelta(
                    minutes=15
                )
            )

            if available_time <= now:

                candidates.append(
                    candidate
                )

    if not candidates:

        raise RuntimeError(
            "사용 가능한 단기예보 발표시각을 "
            "찾지 못했습니다."
        )

    return max(
        candidates
    )


# ==========================================================
# 강수량 문자열 파싱
# ==========================================================

def parse_precipitation(
    value: object,
) -> float:

    if value is None:
        return 0.0

    text = str(
        value
    ).strip()

    if not text:
        return 0.0

    zero_values = {
        "강수없음",
        "없음",
        "0",
        "0.0",
    }

    if text in zero_values:
        return 0.0

    if "미만" in text:

        # 예: "1.0mm 미만"
        numbers = (
            text
            .replace(
                "mm",
                ""
            )
            .replace(
                "미만",
                ""
            )
            .strip()
        )

        try:
            return float(
                numbers
            ) / 2.0

        except ValueError:
            return 0.0

    if "~" in text:

        # 예: 30~50mm
        cleaned = (
            text
            .replace(
                "mm",
                ""
            )
            .replace(
                "이상",
                ""
            )
        )

        pieces = (
            cleaned
            .split("~")
        )

        try:

            numbers = [
                float(piece.strip())
                for piece in pieces
                if piece.strip()
            ]

            if numbers:
                return float(
                    np.mean(
                        numbers
                    )
                )

        except ValueError:
            return 0.0

    cleaned = (
        text
        .replace(
            "mm",
            ""
        )
        .replace(
            "이상",
            ""
        )
        .strip()
    )

    try:
        return float(
            cleaned
        )

    except ValueError:
        return 0.0


# ==========================================================
# 적설 파싱
# ==========================================================

def parse_snowfall(
    value: object,
) -> float:

    if value is None:
        return 0.0

    text = str(
        value
    ).strip()

    if not text:
        return 0.0

    if text in {
        "적설없음",
        "없음",
        "0",
        "0.0",
    }:
        return 0.0

    if "미만" in text:

        cleaned = (
            text
            .replace(
                "cm",
                ""
            )
            .replace(
                "미만",
                ""
            )
            .strip()
        )

        try:
            return float(
                cleaned
            ) / 2.0

        except ValueError:
            return 0.0

    if "~" in text:

        cleaned = (
            text
            .replace(
                "cm",
                ""
            )
            .replace(
                "이상",
                ""
            )
        )

        pieces = (
            cleaned
            .split("~")
        )

        try:

            numbers = [
                float(piece.strip())
                for piece in pieces
                if piece.strip()
            ]

            if numbers:
                return float(
                    np.mean(
                        numbers
                    )
                )

        except ValueError:
            return 0.0

    cleaned = (
        text
        .replace(
            "cm",
            ""
        )
        .replace(
            "이상",
            ""
        )
        .strip()
    )

    try:
        return float(
            cleaned
        )

    except ValueError:
        return 0.0


# ==========================================================
# 한 지역 API 호출
# ==========================================================

def request_forecast(
    service_key: str,
    nx: int,
    ny: int,
    base_datetime: datetime,
) -> list[dict]:

    # 공공데이터포털에서
    # Encoding Key를 복사했다면
    # requests가 다시 encode하지 않도록
    # 한 번 decode 후 params로 전달
    decoded_key = unquote(
        service_key
    )

    params = {
        "serviceKey": decoded_key,
        "pageNo": 1,
        "numOfRows": 1000,
        "dataType": "JSON",

        "base_date":
            base_datetime.strftime(
                "%Y%m%d"
            ),

        "base_time":
            base_datetime.strftime(
                "%H00"
            ),

        "nx": int(nx),
        "ny": int(ny),
    }

    response = requests.get(
        API_URL,
        params=params,
        timeout=30,
    )

    response.raise_for_status()

    try:

        payload = (
            response.json()
        )

    except Exception as exc:

        raise RuntimeError(
            "기상청 API 응답을 JSON으로 "
            "해석하지 못했습니다.\n"
            f"응답 앞부분: "
            f"{response.text[:500]}"
        ) from exc

    header = (
        payload
        .get(
            "response",
            {},
        )
        .get(
            "header",
            {},
        )
    )

    result_code = str(
        header.get(
            "resultCode",
            ""
        )
    )

    result_msg = str(
        header.get(
            "resultMsg",
            ""
        )
    )

    if result_code != "00":

        raise RuntimeError(
            "기상청 API 오류: "
            f"{result_code} / "
            f"{result_msg}"
        )

    body = (
        payload
        .get(
            "response",
            {},
        )
        .get(
            "body",
            {},
        )
    )

    items = (
        body
        .get(
            "items",
            {},
        )
        .get(
            "item",
            [],
        )
    )

    if not items:

        raise RuntimeError(
            f"예보 데이터가 없습니다. "
            f"nx={nx}, ny={ny}, "
            f"base={base_datetime}"
        )

    return items


# ==========================================================
# 한 지역을 일별 데이터로 변환
# ==========================================================

def aggregate_station(
    station: dict,
    items: list[dict],
) -> pd.DataFrame:

    raw = pd.DataFrame(
        items
    )

    required = [
        "category",
        "fcstDate",
        "fcstTime",
        "fcstValue",
    ]

    missing = [
        column
        for column in required
        if column not in raw.columns
    ]

    if missing:

        raise RuntimeError(
            "API 응답 필수 컬럼이 없습니다: "
            f"{missing}"
        )

    raw["date"] = pd.to_datetime(
        raw[
            "fcstDate"
        ],
        format="%Y%m%d",
        errors="coerce",
    )

    raw = raw.dropna(
        subset=[
            "date"
        ]
    )

    rows = []

    for forecast_date, day in (
        raw.groupby(
            "date"
        )
    ):

        # -----------------------------------------
        # TMP
        # 시간별 기온
        # -----------------------------------------

        tmp = pd.to_numeric(
            day.loc[
                day[
                    "category"
                ].eq(
                    "TMP"
                ),
                "fcstValue",
            ],
            errors="coerce",
        ).dropna()

        # TMP가 없는 날짜는 사용 불가
        if tmp.empty:
            continue

        avg_temp = float(
            tmp.mean()
        )

        min_temp = float(
            tmp.min()
        )

        max_temp = float(
            tmp.max()
        )

        # -----------------------------------------
        # REH
        # 습도
        # -----------------------------------------

        reh = pd.to_numeric(
            day.loc[
                day[
                    "category"
                ].eq(
                    "REH"
                ),
                "fcstValue",
            ],
            errors="coerce",
        ).dropna()

        humidity = (
            float(
                reh.mean()
            )
            if not reh.empty
            else np.nan
        )

        # -----------------------------------------
        # PCP
        # 1시간 강수량
        # -----------------------------------------

        pcp = (
            day.loc[
                day[
                    "category"
                ].eq(
                    "PCP"
                ),
                "fcstValue",
            ]
            .map(
                parse_precipitation
            )
        )

        precipitation = (
            float(
                pcp.sum()
            )
            if len(pcp)
            else 0.0
        )

        # -----------------------------------------
        # SNO
        # 1시간 신적설
        # -----------------------------------------

        sno = (
            day.loc[
                day[
                    "category"
                ].eq(
                    "SNO"
                ),
                "fcstValue",
            ]
            .map(
                parse_snowfall
            )
        )

        snowfall = (
            float(
                sno.sum()
            )
            if len(sno)
            else 0.0
        )

        rows.append(
            {
                "date":
                    forecast_date.date(),

                "station_id":
                    station[
                        "station_id"
                    ],

                "station_name":
                    station[
                        "station_name"
                    ],

                "lat":
                    station[
                        "lat"
                    ],

                "lon":
                    station[
                        "lon"
                    ],

                "avg_temp":
                    round(
                        avg_temp,
                        2,
                    ),

                "min_temp":
                    round(
                        min_temp,
                        2,
                    ),

                "max_temp":
                    round(
                        max_temp,
                        2,
                    ),

                "precipitation":
                    round(
                        precipitation,
                        2,
                    ),

                "snowfall":
                    round(
                        snowfall,
                        2,
                    ),

                "humidity":
                    round(
                        humidity,
                        2,
                    )
                    if not pd.isna(
                        humidity
                    )
                    else np.nan,
            }
        )

    result = pd.DataFrame(
        rows
    )

    return result


# ==========================================================
# 메인
# ==========================================================

def main() -> None:

    load_env(
        ENV_PATH
    )

    service_key = (
        os.getenv(
            "KMA_SERVICE_KEY",
            ""
        )
        .strip()
    )

    if not service_key:

        raise RuntimeError(
            ".env에 "
            "KMA_SERVICE_KEY가 없습니다."
        )

    base_datetime = (
        get_latest_base_datetime()
    )

    print(
        "기상청 단기예보 기준:"
    )

    print(
        base_datetime.strftime(
            "%Y-%m-%d %H:%M"
        )
    )

    frames = []

    for station in STATIONS:

        print()
        print(
            f"[{station['station_name']}] "
            f"nx={station['nx']} "
            f"ny={station['ny']}"
        )

        items = request_forecast(
            service_key=service_key,
            nx=station["nx"],
            ny=station["ny"],
            base_datetime=base_datetime,
        )

        result = aggregate_station(
            station,
            items,
        )

        if result.empty:

            print(
                "  예보 데이터 없음"
            )

            continue

        print(
            result.to_string(
                index=False
            )
        )

        frames.append(
            result
        )

    if not frames:

        raise RuntimeError(
            "수집된 예보 데이터가 없습니다."
        )

    forecast = pd.concat(
        frames,
        ignore_index=True,
    )

    # -----------------------------------------
    # 모델에서 사용하려는
    # 오늘 이후 날짜만 유지
    # -----------------------------------------

    today = (
        pd.Timestamp.today()
        .normalize()
    )

    forecast["date"] = pd.to_datetime(
        forecast[
            "date"
        ]
    )

    forecast = forecast.loc[
        forecast[
            "date"
        ]
        >= today
    ].copy()

    # -----------------------------------------
    # 최대 3일 사용
    # -----------------------------------------

    available_dates = sorted(
        forecast[
            "date"
        ].unique()
    )

    selected_dates = (
        available_dates[:3]
    )

    forecast = forecast.loc[
        forecast[
            "date"
        ].isin(
            selected_dates
        )
    ].copy()

    # -----------------------------------------
    # 최종 컬럼 순서
    # 기존 weather_forecast.csv와 동일
    # -----------------------------------------

    columns = [
        "date",
        "station_id",
        "station_name",
        "lat",
        "lon",
        "avg_temp",
        "min_temp",
        "max_temp",
        "precipitation",
        "snowfall",
        "humidity",
    ]

    forecast = (
        forecast[
            columns
        ]
        .sort_values(
            [
                "station_id",
                "date",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    forecast[
        "date"
    ] = (
        forecast[
            "date"
        ]
        .dt.strftime(
            "%Y-%m-%d"
        )
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    forecast.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    print()
    print("=" * 70)

    print(
        f"저장 완료: "
        f"{OUTPUT_PATH}"
    )

    print(
        f"행 수: "
        f"{len(forecast)}"
    )

    print()

    print(
        forecast.to_string(
            index=False
        )
    )


if __name__ == "__main__":

    try:

        main()

    except Exception as exc:

        print(
            f"[ERROR] {exc}",
            file=sys.stderr,
        )

        raise