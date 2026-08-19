from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from .common import load_config, resolve_path


OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

STATIONS = {
    "JB01": {"station_name": "전주", "lat": 35.8242, "lon": 127.1480},
    "JB02": {"station_name": "군산", "lat": 35.9677, "lon": 126.7366},
    "JB03": {"station_name": "익산", "lat": 35.9483, "lon": 126.9576},
    "JB04": {"station_name": "남원", "lat": 35.4164, "lon": 127.3904},
    "JB05": {"station_name": "정읍", "lat": 35.5699, "lon": 126.8560},
}


def _fetch_station(station_id: str, info: dict, forecast_days: int = 3) -> pd.DataFrame:
    params = {
        "latitude": info["lat"],
        "longitude": info["lon"],
        "hourly": "temperature_2m,relative_humidity_2m,precipitation,snowfall",
        "timezone": "Asia/Seoul",
        "forecast_days": forecast_days,
    }
    url = f"{OPEN_METEO_URL}?{urlencode(params)}"
    request = Request(
        url,
        headers={
            "User-Agent": "RoadDoctor-Hackathon/1.0",
            "Accept": "application/json",
        },
    )

    with urlopen(request, timeout=20) as response:
        payload = json.loads(response.read().decode("utf-8"))

    hourly = payload.get("hourly", {})
    required = [
        "time",
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "snowfall",
    ]
    missing = [key for key in required if key not in hourly]
    if missing:
        raise ValueError(f"{info['station_name']} 예보 응답에 필요한 항목이 없습니다: {missing}")

    frame = pd.DataFrame(
        {
            "datetime": pd.to_datetime(hourly["time"], errors="coerce"),
            "temperature": pd.to_numeric(hourly["temperature_2m"], errors="coerce"),
            "humidity": pd.to_numeric(hourly["relative_humidity_2m"], errors="coerce"),
            "precipitation": pd.to_numeric(hourly["precipitation"], errors="coerce"),
            "snowfall": pd.to_numeric(hourly["snowfall"], errors="coerce"),
        }
    ).dropna(subset=["datetime"])

    frame["date"] = frame["datetime"].dt.normalize()

    daily = (
        frame.groupby("date", as_index=False)
        .agg(
            avg_temp=("temperature", "mean"),
            min_temp=("temperature", "min"),
            max_temp=("temperature", "max"),
            precipitation=("precipitation", "sum"),
            snowfall=("snowfall", "sum"),
            humidity=("humidity", "mean"),
        )
    )

    daily.insert(1, "station_id", station_id)
    daily.insert(2, "station_name", info["station_name"])
    daily.insert(3, "lat", info["lat"])
    daily.insert(4, "lon", info["lon"])
    return daily


def refresh_weather_forecast(
    config_path: str = "config.yaml",
    forecast_days: int = 3,
    force: bool = False,
    max_age_minutes: int = 30,
) -> bool:
    config = load_config(config_path)
    output_path = resolve_path(config, "weather_forecast")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not force:
        age_seconds = time.time() - output_path.stat().st_mtime
        if age_seconds < max_age_minutes * 60:
            print(
                f"예보 파일이 최근 {max_age_minutes}분 이내 갱신되어 재사용합니다: "
                f"{output_path}"
            )
            return False

    rows = []
    for station_id, info in STATIONS.items():
        rows.append(_fetch_station(station_id, info, forecast_days=forecast_days))

    forecast = pd.concat(rows, ignore_index=True)
    forecast["date"] = pd.to_datetime(forecast["date"]).dt.strftime("%Y-%m-%d")

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
    forecast = forecast[columns].sort_values(["station_id", "date"])

    forecast.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"최신 {forecast_days}일 예보 저장 완료: {output_path}")
    print(
        forecast[
            ["date", "station_id", "station_name", "avg_temp",
             "min_temp", "max_temp", "precipitation", "humidity"]
        ].to_string(index=False)
    )
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--days", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-age-minutes", type=int, default=30)
    args = parser.parse_args()

    refresh_weather_forecast(
        config_path=args.config,
        forecast_days=args.days,
        force=args.force,
        max_age_minutes=args.max_age_minutes,
    )


if __name__ == "__main__":
    main()
