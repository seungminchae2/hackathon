from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(42)

# 전북 내 데모용 관측소 좌표: 전주, 군산, 익산, 남원, 정읍 근처
stations = pd.DataFrame(
    [
        ("JB01", "전주", 35.8242, 127.1480),
        ("JB02", "군산", 35.9677, 126.7366),
        ("JB03", "익산", 35.9483, 126.9576),
        ("JB04", "남원", 35.4164, 127.3904),
        ("JB05", "정읍", 35.5699, 126.8560),
    ],
    columns=["station_id", "station_name", "lat", "lon"],
)

today = pd.Timestamp.now().normalize()
history_dates = pd.date_range(today - pd.Timedelta(days=730), today - pd.Timedelta(days=3), freq="D")
weather_rows = []
for station in stations.itertuples(index=False):
    phase = rng.uniform(-0.25, 0.25)
    for date in history_dates:
        seasonal = 12 + 13 * np.sin(2 * np.pi * (date.dayofyear / 365.25 - 0.24 + phase * 0.05))
        avg = seasonal + rng.normal(0, 2.4)
        temp_range = max(4, rng.normal(9, 2.2))
        min_temp = avg - temp_range / 2
        max_temp = avg + temp_range / 2
        rain_event = rng.random() < (0.18 if date.month not in [6, 7, 8] else 0.31)
        precipitation = rng.gamma(1.4, 7.0) if rain_event else 0.0
        snowfall = rng.gamma(1.2, 1.1) if date.month in [1, 2, 12] and min_temp < 1 and rng.random() < 0.15 else 0.0
        humidity = np.clip(58 + precipitation * 0.8 + rng.normal(0, 10), 25, 100)
        weather_rows.append(
            {
                "date": date.date().isoformat(),
                "station_id": station.station_id,
                "station_name": station.station_name,
                "lat": station.lat,
                "lon": station.lon,
                "avg_temp": round(avg, 2),
                "min_temp": round(min_temp, 2),
                "max_temp": round(max_temp, 2),
                "precipitation": round(precipitation, 2),
                "snowfall": round(snowfall, 2),
                "humidity": round(humidity, 2),
            }
        )
weather = pd.DataFrame(weather_rows)
weather.to_csv(DATA_DIR / "weather_history.csv", index=False)

forecast_dates = pd.date_range(today - pd.Timedelta(days=2), today, freq="D")
forecast_rows = []
for station in stations.itertuples(index=False):
    for i, date in enumerate(forecast_dates):
        avg = -1 + i * 2 + rng.normal(0, 1)
        min_temp = avg - 5
        max_temp = avg + 5
        precipitation = [18, 4, 0][i] + rng.uniform(0, 3)
        forecast_rows.append(
            {
                "date": date.date().isoformat(),
                "station_id": station.station_id,
                "station_name": station.station_name,
                "lat": station.lat,
                "lon": station.lon,
                "avg_temp": round(avg, 2),
                "min_temp": round(min_temp, 2),
                "max_temp": round(max_temp, 2),
                "precipitation": round(precipitation, 2),
                "snowfall": round(max(0, 1.5 - i * 0.7), 2),
                "humidity": round(82 - i * 5, 2),
            }
        )
pd.DataFrame(forecast_rows).to_csv(DATA_DIR / "weather_forecast.csv", index=False)

# 도로 점: 실제 도로망(data/roads.csv)이 이미 있으면 그대로 재사용하고,
# 없을 때만 관측소 주변 도로 축을 단순 생성해 대체합니다.
roads_path = DATA_DIR / "roads.csv"
if roads_path.exists():
    roads = pd.read_csv(roads_path)
else:
    road_rows = []
    road_id = 1
    for station in stations.itertuples(index=False):
        for angle in [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]:
            for step in np.linspace(-0.08, 0.08, 34):
                lat = station.lat + step * np.sin(angle) + rng.normal(0, 0.0007)
                lon = station.lon + step * np.cos(angle) + rng.normal(0, 0.0007)
                road_rows.append(
                    {
                        "road_point_id": f"R{road_id:05d}",
                        "lat": round(lat, 6),
                        "lon": round(lon, 6),
                        "road_name": f"{station.station_name} 데모도로",
                    }
                )
                road_id += 1
    roads = pd.DataFrame(road_rows)
    roads.to_csv(roads_path, index=False)

# 포트홀: 겨울철, 강수 직후, 반복 위치에 더 자주 생기도록 합성
road_sample = roads.sample(65, random_state=42).reset_index(drop=True)
pothole_rows = []
for idx, road in road_sample.iterrows():
    n_events = rng.integers(1, 5 if idx < 12 else 3)
    for _ in range(n_events):
        winter_bias = rng.random() < 0.67
        candidates = (
            history_dates[history_dates.month.isin([1, 2, 3, 11, 12])]
            if winter_bias
            else history_dates
        )
        date = pd.Timestamp(rng.choice(candidates))
        pothole_rows.append(
            {
                "event_id": f"P{len(pothole_rows)+1:04d}",
                "event_date": date.date().isoformat(),
                "lat": round(road.lat + rng.normal(0, 0.0005), 6),
                "lon": round(road.lon + rng.normal(0, 0.0005), 6),
                "severity": int(rng.integers(1, 4)),
                "road_name": road.road_name,
            }
        )
potholes = pd.DataFrame(pothole_rows).sort_values("event_date")
potholes.to_csv(DATA_DIR / "potholes.csv", index=False)

repair_rows = []
for idx, event in potholes.sample(frac=0.72, random_state=7).iterrows():
    repair_date = pd.Timestamp(event.event_date) + pd.Timedelta(days=int(rng.integers(1, 18)))
    repair_rows.append(
        {
            "repair_id": f"F{len(repair_rows)+1:04d}",
            "repair_date": repair_date.date().isoformat(),
            "lat": event.lat,
            "lon": event.lon,
            "repair_type": rng.choice(["임시보수", "상온아스콘", "덧씌우기"]),
        }
    )
pd.DataFrame(repair_rows).to_csv(DATA_DIR / "repairs.csv", index=False)

print(f"샘플 데이터 생성 완료: {DATA_DIR}")
