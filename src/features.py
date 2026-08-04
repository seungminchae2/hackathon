from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .common import (
    add_grid_columns,
    haversine_distance_matrix,
    parse_date_column,
    require_columns,
    safe_numeric,
)


WEATHER_COLUMNS = [
    "avg_temp",
    "min_temp",
    "max_temp",
    "precipitation",
    "snowfall",
    "humidity",
]

FEATURE_COLUMNS = [
    "avg_temp",
    "min_temp",
    "max_temp",
    "temp_range",
    "precipitation",
    "precip_3d",
    "precip_7d",
    "snowfall",
    "humidity",
    "freeze_thaw",
    "freeze_thaw_7d",
    "past_potholes_30d",
    "past_potholes_90d",
    "past_potholes_total",
    "days_since_last_repair",
    "month",
    "day_of_year_sin",
    "day_of_year_cos",
]


@dataclass
class PreparedData:
    panel: pd.DataFrame
    grid_catalog: pd.DataFrame
    station_map: pd.DataFrame


def load_potholes(path: Path, grid_size_m: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    require_columns(df, ["event_date", "lat", "lon"], "potholes.csv")
    df = parse_date_column(df, "event_date")
    df = add_grid_columns(df, grid_size_m=grid_size_m)
    if "severity" not in df.columns:
        df["severity"] = 1
    return df


def load_repairs(path: Path, grid_size_m: int) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["repair_date", "grid_id", "lat", "lon"])
    df = pd.read_csv(path)
    require_columns(df, ["repair_date", "lat", "lon"], "repairs.csv")
    df = parse_date_column(df, "repair_date")
    return add_grid_columns(df, grid_size_m=grid_size_m)


def load_roads(path: Path, grid_size_m: int, potholes: pd.DataFrame) -> pd.DataFrame:
    if path.exists():
        roads = pd.read_csv(path)
        require_columns(roads, ["lat", "lon"], "roads.csv")
        roads = add_grid_columns(roads, grid_size_m=grid_size_m)
        catalog = roads[["grid_id", "grid_x", "grid_y", "grid_lat", "grid_lon"]].drop_duplicates()
    else:
        # 도로 데이터가 없으면 양성 격자 주변 8개 격자를 후보로 만들어 데모가 동작하게 합니다.
        base = potholes[["grid_x", "grid_y"]].drop_duplicates()
        rows: list[dict[str, int]] = []
        for row in base.itertuples(index=False):
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    rows.append({"grid_x": row.grid_x + dx, "grid_y": row.grid_y + dy})
        catalog = pd.DataFrame(rows).drop_duplicates()
        lat_step = grid_size_m / 111_320.0
        lon_step = grid_size_m / (111_320.0 * np.cos(np.radians(35.8)))
        catalog["grid_id"] = catalog["grid_x"].astype(str) + "_" + catalog["grid_y"].astype(str)
        catalog["grid_lat"] = (catalog["grid_y"] + 0.5) * lat_step
        catalog["grid_lon"] = (catalog["grid_x"] + 0.5) * lon_step

    return catalog.reset_index(drop=True)


def load_weather(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    require_columns(df, ["date", "station_id", "lat", "lon"], str(path.name))
    df = parse_date_column(df, "date")
    df = safe_numeric(df, WEATHER_COLUMNS)
    df["precipitation"] = df["precipitation"].fillna(0).clip(lower=0)
    df["snowfall"] = df["snowfall"].fillna(0).clip(lower=0)
    df["humidity"] = df["humidity"].fillna(df.groupby("station_id")["humidity"].transform("median"))
    df["humidity"] = df["humidity"].fillna(df["humidity"].median())
    for col in ["avg_temp", "min_temp", "max_temp"]:
        df[col] = df[col].fillna(df.groupby("station_id")[col].transform("median"))
        df[col] = df[col].fillna(df[col].median())
    return df.sort_values(["station_id", "date"]).reset_index(drop=True)


def map_grids_to_stations(grid_catalog: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    stations = weather[["station_id", "lat", "lon"]].drop_duplicates("station_id")
    distances = haversine_distance_matrix(
        grid_catalog["grid_lat"].to_numpy(),
        grid_catalog["grid_lon"].to_numpy(),
        stations["lat"].to_numpy(),
        stations["lon"].to_numpy(),
    )
    nearest_idx = distances.argmin(axis=1)
    station_ids = stations.iloc[nearest_idx]["station_id"].to_numpy()
    nearest_km = distances[np.arange(len(grid_catalog)), nearest_idx]
    return pd.DataFrame(
        {
            "grid_id": grid_catalog["grid_id"].to_numpy(),
            "station_id": station_ids,
            "weather_station_distance_km": nearest_km,
        }
    )


def engineer_weather_features(weather: pd.DataFrame) -> pd.DataFrame:
    df = weather.copy().sort_values(["station_id", "date"])
    df["temp_range"] = df["max_temp"] - df["min_temp"]
    df["freeze_thaw"] = ((df["min_temp"] < 0) & (df["max_temp"] > 0)).astype(int)

    grouped = df.groupby("station_id", group_keys=False)
    df["precip_3d"] = grouped["precipitation"].transform(
        lambda s: s.rolling(3, min_periods=1).sum()
    )
    df["precip_7d"] = grouped["precipitation"].transform(
        lambda s: s.rolling(7, min_periods=1).sum()
    )
    df["freeze_thaw_7d"] = grouped["freeze_thaw"].transform(
        lambda s: s.rolling(7, min_periods=1).sum()
    )
    return df


def build_daily_panel(
    potholes: pd.DataFrame,
    repairs: pd.DataFrame,
    grid_catalog: pd.DataFrame,
    weather: pd.DataFrame,
    station_map: pd.DataFrame,
    start_date: str | None = None,
    end_date: str | None = None,
    include_target: bool = True,
) -> pd.DataFrame:
    weather = engineer_weather_features(weather)
    min_date = pd.Timestamp(start_date) if start_date else weather["date"].min()
    max_date = pd.Timestamp(end_date) if end_date else weather["date"].max()
    dates = pd.date_range(min_date, max_date, freq="D")

    panel = (
        grid_catalog.assign(_key=1)
        .merge(pd.DataFrame({"date": dates, "_key": 1}), on="_key")
        .drop(columns="_key")
        .merge(station_map, on="grid_id", how="left")
        .merge(weather, on=["station_id", "date"], how="left", suffixes=("", "_weather"))
    )

    event_daily = (
        potholes.groupby(["grid_id", "event_date"])
        .size()
        .rename("pothole_count")
        .reset_index()
        .rename(columns={"event_date": "date"})
    )
    panel = panel.merge(event_daily, on=["grid_id", "date"], how="left")
    panel["pothole_count"] = panel["pothole_count"].fillna(0).astype(int)
    if include_target:
        panel["target"] = (panel["pothole_count"] > 0).astype(int)

    # 과거 발생 이력은 당일 정답 누수를 막기 위해 shift(1) 후 계산합니다.
    panel = panel.sort_values(["grid_id", "date"])
    shifted = panel.groupby("grid_id")["pothole_count"].shift(1).fillna(0)
    panel["past_potholes_30d"] = shifted.groupby(panel["grid_id"]).transform(
        lambda s: s.rolling(30, min_periods=1).sum()
    )
    panel["past_potholes_90d"] = shifted.groupby(panel["grid_id"]).transform(
        lambda s: s.rolling(90, min_periods=1).sum()
    )
    panel["past_potholes_total"] = shifted.groupby(panel["grid_id"]).cumsum()

    panel = add_days_since_last_repair(panel, repairs)
    panel["month"] = panel["date"].dt.month
    day_of_year = panel["date"].dt.dayofyear
    panel["day_of_year_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    panel["day_of_year_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)

    panel = safe_numeric(panel, FEATURE_COLUMNS)
    for col in FEATURE_COLUMNS:
        if panel[col].isna().all():
            panel[col] = 0
        else:
            panel[col] = panel[col].fillna(panel[col].median())

    return panel.reset_index(drop=True)


def add_days_since_last_repair(panel: pd.DataFrame, repairs: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    if repairs.empty:
        out["days_since_last_repair"] = 3650
        return out

    repair_daily = repairs[["grid_id", "repair_date"]].drop_duplicates().sort_values(
        ["grid_id", "repair_date"]
    )
    chunks: list[pd.DataFrame] = []
    for grid_id, chunk in out.groupby("grid_id", sort=False):
        chunk = chunk.sort_values("date").copy()
        r = repair_daily[repair_daily["grid_id"] == grid_id][["repair_date"]].sort_values(
            "repair_date"
        )
        if r.empty:
            chunk["last_repair_date"] = pd.NaT
        else:
            chunk = pd.merge_asof(
                chunk,
                r,
                left_on="date",
                right_on="repair_date",
                direction="backward",
                allow_exact_matches=True,
            )
            chunk = chunk.rename(columns={"repair_date": "last_repair_date"})
        chunks.append(chunk)

    out = pd.concat(chunks, ignore_index=True)
    out["days_since_last_repair"] = (out["date"] - out["last_repair_date"]).dt.days
    out["days_since_last_repair"] = out["days_since_last_repair"].fillna(3650).clip(0, 3650)
    return out


def prepare_dataset(
    pothole_path: Path,
    repair_path: Path,
    road_path: Path,
    weather_path: Path,
    grid_size_m: int,
    start_date: str | None = None,
    end_date: str | None = None,
    include_target: bool = True,
) -> PreparedData:
    potholes = load_potholes(pothole_path, grid_size_m)
    repairs = load_repairs(repair_path, grid_size_m)
    grid_catalog = load_roads(road_path, grid_size_m, potholes)
    weather = load_weather(weather_path)
    station_map = map_grids_to_stations(grid_catalog, weather)
    panel = build_daily_panel(
        potholes=potholes,
        repairs=repairs,
        grid_catalog=grid_catalog,
        weather=weather,
        station_map=station_map,
        start_date=start_date,
        end_date=end_date,
        include_target=include_target,
    )
    return PreparedData(panel=panel, grid_catalog=grid_catalog, station_map=station_map)


def downsample_negatives(
    df: pd.DataFrame,
    ratio: int,
    random_seed: int,
) -> pd.DataFrame:
    positives = df[df["target"] == 1]
    negatives = df[df["target"] == 0]
    if positives.empty:
        raise ValueError("양성 포트홀 데이터가 없습니다.")
    max_negatives = min(len(negatives), len(positives) * ratio)
    sampled_negatives = negatives.sample(n=max_negatives, random_state=random_seed)
    return (
        pd.concat([positives, sampled_negatives], ignore_index=True)
        .sort_values("date")
        .reset_index(drop=True)
    )


def risk_level(score: float) -> str:
    if score >= 0.75:
        return "매우 높음"
    if score >= 0.50:
        return "높음"
    if score >= 0.25:
        return "보통"
    return "낮음"


def explain_risk(row: pd.Series) -> str:
    reasons: list[str] = []
    if row.get("precip_3d", 0) >= 30:
        reasons.append("최근 3일 누적강수")
    elif row.get("precip_7d", 0) >= 50:
        reasons.append("최근 7일 누적강수")
    if row.get("freeze_thaw_7d", 0) >= 2:
        reasons.append("동결·융해 반복")
    if row.get("temp_range", 0) >= 12:
        reasons.append("큰 일교차")
    if row.get("past_potholes_90d", 0) >= 1:
        reasons.append("최근 반복 발생")
    if row.get("past_potholes_total", 0) >= 2:
        reasons.append("상습 발생 구간")
    if row.get("days_since_last_repair", 0) >= 730:
        reasons.append("보수 후 장기 경과")
    return ", ".join(reasons[:3]) if reasons else "기상·과거 이력 종합 위험"
