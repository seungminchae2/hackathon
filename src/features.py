from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

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

BASE_FEATURE_COLUMNS = [
    "grid_lat",
    "grid_lon",
    "precip_3d",
    "precip_7d",
    "freeze_thaw_7d",
    "past_potholes_90d",
    "past_potholes_total",
    "has_repair_history",
    "days_since_last_repair",
]
OPTIONAL_FEATURE_COLUMNS = ["traffic_volume", "road_importance"]

# 기존 코드에서 import하던 이름을 유지하되, 실제 학습 컬럼은 select_feature_columns로 정합니다.
FEATURE_COLUMNS = BASE_FEATURE_COLUMNS

FEATURE_GROUPS = {
    "spatial": ["grid_lat", "grid_lon"],
    "precipitation": ["precip_3d", "precip_7d"],
    "freeze_thaw": ["freeze_thaw_7d"],
    "recent_recurrence": ["past_potholes_90d"],
    "long_term_history": ["past_potholes_total"],
    "repair": ["has_repair_history", "days_since_last_repair"],
    "traffic": ["traffic_volume"],
    "road_importance": ["road_importance"],
}


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

        optional = []
        for column in OPTIONAL_FEATURE_COLUMNS:
            if column in roads.columns:
                roads[column] = pd.to_numeric(roads[column], errors="coerce")
                if roads[column].notna().any():
                    optional.append(column)

        aggregations: dict[str, str] = {
            "grid_x": "first",
            "grid_y": "first",
            "grid_lat": "first",
            "grid_lon": "first",
        }
        aggregations.update({column: "median" for column in optional})
        catalog = roads.groupby("grid_id", as_index=False).agg(aggregations)
    else:
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
    for column in ["avg_temp", "min_temp", "max_temp"]:
        df[column] = df[column].fillna(df.groupby("station_id")[column].transform("median"))
        df[column] = df[column].fillna(df[column].median())
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
    return pd.DataFrame(
        {
            "grid_id": grid_catalog["grid_id"].to_numpy(),
            "station_id": stations.iloc[nearest_idx]["station_id"].to_numpy(),
            "weather_station_distance_km": distances[np.arange(len(grid_catalog)), nearest_idx],
        }
    )


def engineer_weather_features(weather: pd.DataFrame) -> pd.DataFrame:
    df = weather.copy().sort_values(["station_id", "date"])
    df["freeze_thaw"] = ((df["min_temp"] < 0) & (df["max_temp"] > 0)).astype(int)
    grouped = df.groupby("station_id", group_keys=False)
    df["precip_3d"] = grouped["precipitation"].transform(
        lambda series: series.rolling(3, min_periods=1).sum()
    )
    df["precip_7d"] = grouped["precipitation"].transform(
        lambda series: series.rolling(7, min_periods=1).sum()
    )
    df["freeze_thaw_7d"] = grouped["freeze_thaw"].transform(
        lambda series: series.rolling(7, min_periods=1).sum()
    )
    return df


def _forward_window_count(series: pd.Series, horizon_days: int) -> pd.Series:
    shifted = series.shift(-1)
    return (
        shifted.iloc[::-1]
        .rolling(horizon_days, min_periods=horizon_days)
        .sum()
        .iloc[::-1]
    )


def build_daily_panel(
    potholes: pd.DataFrame,
    repairs: pd.DataFrame,
    grid_catalog: pd.DataFrame,
    weather: pd.DataFrame,
    station_map: pd.DataFrame,
    start_date: str | None = None,
    end_date: str | None = None,
    include_target: bool = True,
    target_horizon_days: int = 30,
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

    panel = panel.sort_values(["grid_id", "date"])
    shifted = panel.groupby("grid_id")["pothole_count"].shift(1).fillna(0)
    panel["past_potholes_90d"] = shifted.groupby(panel["grid_id"]).transform(
        lambda series: series.rolling(90, min_periods=1).sum()
    )
    panel["past_potholes_total"] = shifted.groupby(panel["grid_id"]).cumsum()
    panel = add_days_since_last_repair(panel, repairs)

    if include_target:
        future_count = panel.groupby("grid_id")["pothole_count"].transform(
            lambda series: _forward_window_count(series, target_horizon_days)
        )
        panel = panel.loc[future_count.notna()].copy()
        panel["target_next_30d"] = (future_count.loc[panel.index] > 0).astype(int)
        panel["target"] = panel["target_next_30d"]

    available_features = BASE_FEATURE_COLUMNS + [
        column for column in OPTIONAL_FEATURE_COLUMNS if column in panel.columns
    ]
    panel = safe_numeric(panel, available_features)
    panel.loc[panel["has_repair_history"].eq(0), "days_since_last_repair"] = 0.0
    for column in available_features:
        if panel[column].isna().all():
            if column in BASE_FEATURE_COLUMNS:
                raise ValueError(f"필수 feature 전체가 결측입니다: {column}")
            continue
        panel[column] = panel[column].fillna(panel[column].median())

    return panel.reset_index(drop=True)


def add_days_since_last_repair(panel: pd.DataFrame, repairs: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    if repairs.empty:
        out["has_repair_history"] = 0
        out["days_since_last_repair"] = 0.0
        return out

    repair_daily = repairs[["grid_id", "repair_date"]].drop_duplicates().sort_values(
        ["grid_id", "repair_date"]
    )
    chunks: list[pd.DataFrame] = []
    for grid_id, chunk in out.groupby("grid_id", sort=False):
        chunk = chunk.sort_values("date").copy()
        repair_dates = repair_daily.loc[
            repair_daily["grid_id"].eq(grid_id), ["repair_date"]
        ].sort_values("repair_date")
        if repair_dates.empty:
            chunk["last_repair_date"] = pd.NaT
        else:
            chunk = pd.merge_asof(
                chunk,
                repair_dates,
                left_on="date",
                right_on="repair_date",
                direction="backward",
                allow_exact_matches=True,
            ).rename(columns={"repair_date": "last_repair_date"})
        chunks.append(chunk)

    out = pd.concat(chunks, ignore_index=True)
    out["has_repair_history"] = out["last_repair_date"].notna().astype(int)
    out["days_since_last_repair"] = (
        (out["date"] - out["last_repair_date"]).dt.days
        .where(out["has_repair_history"].eq(1), 0)
        .fillna(0)
        .clip(lower=0)
    )
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
    target_horizon_days: int = 30,
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
        target_horizon_days=target_horizon_days,
    )
    return PreparedData(panel=panel, grid_catalog=grid_catalog, station_map=station_map)


def select_feature_columns(frame: pd.DataFrame) -> list[str]:
    missing = [column for column in BASE_FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"필수 feature가 없습니다: {missing}")
    return BASE_FEATURE_COLUMNS + [
        column
        for column in OPTIONAL_FEATURE_COLUMNS
        if column in frame.columns and frame[column].notna().any()
    ]


def prepare_feature_matrix(
    frame: pd.DataFrame,
    feature_columns: list[str],
    medians: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    output_medians = dict(medians or {})
    matrix = pd.DataFrame(index=frame.index)
    for column in feature_columns:
        if column not in frame.columns:
            values = pd.Series(np.nan, index=frame.index, dtype=float)
        else:
            values = pd.to_numeric(frame[column], errors="coerce")
        if medians is None:
            if column == "days_since_last_repair":
                repaired = values.loc[frame["has_repair_history"].eq(1)]
                median = repaired.median() if repaired.notna().any() else 0.0
            else:
                median = values.median()
            if pd.isna(median):
                raise ValueError(f"학습 구간에서 {column}의 유효값이 없습니다.")
            output_medians[column] = float(median)
        matrix[column] = values.fillna(output_medians[column]).astype(float)

    if "days_since_last_repair" in matrix.columns:
        matrix.loc[frame["has_repair_history"].eq(0), "days_since_last_repair"] = 0.0
    return matrix, output_medians


def robust_scale(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna().clip(lower=0)
    return max(float(values.quantile(0.95)) if not values.empty else 1.0, 1.0)


def build_priority_scales(frame: pd.DataFrame) -> tuple[dict[str, float], dict[str, float]]:
    recurrence = {
        "past_potholes_90d": robust_scale(frame["past_potholes_90d"]),
        "past_potholes_total": robust_scale(frame["past_potholes_total"]),
    }
    importance = {
        column: robust_scale(frame[column])
        for column in OPTIONAL_FEATURE_COLUMNS
        if column in frame.columns and frame[column].notna().any()
    }
    return recurrence, importance


def calculate_priority_components(
    frame: pd.DataFrame,
    risk_score: pd.Series,
    recurrence_scales: dict[str, float],
    importance_scales: dict[str, float],
) -> pd.DataFrame:
    recent = (
        pd.to_numeric(frame["past_potholes_90d"], errors="coerce")
        .fillna(0)
        .clip(lower=0)
        .div(recurrence_scales["past_potholes_90d"])
        .clip(upper=1)
    )
    total = (
        pd.to_numeric(frame["past_potholes_total"], errors="coerce")
        .fillna(0)
        .clip(lower=0)
        .div(recurrence_scales["past_potholes_total"])
        .clip(upper=1)
    )
    recurrence_score = 0.7 * recent + 0.3 * total

    importance_parts: list[pd.Series] = []
    for column, scale in importance_scales.items():
        if column in frame.columns:
            importance_parts.append(
                pd.to_numeric(frame[column], errors="coerce").clip(lower=0).div(scale).clip(upper=1)
            )
    if importance_parts:
        importance_score = pd.concat(importance_parts, axis=1).mean(axis=1, skipna=True)
    else:
        importance_score = pd.Series(np.nan, index=frame.index, dtype=float)

    has_importance = importance_score.notna()
    denominator = 0.90 + 0.10 * has_importance.astype(float)
    return pd.DataFrame(
        {
            "recurrence_score": recurrence_score,
            "importance_score": importance_score,
            "priority_weight_risk": 0.75 / denominator,
            "priority_weight_recurrence": 0.15 / denominator,
            "priority_weight_importance": np.where(has_importance, 0.10 / denominator, 0.0),
            "priority_score": (
                0.75 * risk_score
                + 0.15 * recurrence_score
                + 0.10 * importance_score.fillna(0)
            )
            / denominator,
        },
        index=frame.index,
    )


def risk_level_from_percentile(percentile: float) -> str:
    if percentile > 0.95:
        return "매우 높음"
    if percentile > 0.80:
        return "높음"
    if percentile > 0.50:
        return "보통"
    return "낮음"


def _reason_text(group: str, row: pd.Series) -> str:
    if group == "spatial":
        return "격자 위치의 공간 위험 패턴"
    if group == "precipitation":
        return f"누적강수 영향(3일 {row['precip_3d']:.1f}mm, 7일 {row['precip_7d']:.1f}mm)"
    if group == "freeze_thaw":
        return f"최근 7일 동결·융해 {row['freeze_thaw_7d']:.0f}회"
    if group == "recent_recurrence":
        return f"최근 90일 포트홀 {row['past_potholes_90d']:.0f}건"
    if group == "long_term_history":
        return f"과거 누적 포트홀 {row['past_potholes_total']:.0f}건"
    if group == "repair":
        if int(row["has_repair_history"]) == 0:
            return "보수이력 없음의 모델 기여"
        return f"보수 후 {row['days_since_last_repair']:.0f}일 경과 영향"
    if group == "traffic":
        return f"교통량 영향({row['traffic_volume']:.0f})"
    if group == "road_importance":
        return f"도로중요도 영향({row['road_importance']:.2f})"
    return group


def explain_with_contributions(
    model: Any,
    matrix: pd.DataFrame,
    source: pd.DataFrame,
    max_reasons: int = 3,
) -> list[str]:
    import xgboost as xgb

    contributions = model.get_booster().predict(
        xgb.DMatrix(matrix, feature_names=list(matrix.columns)),
        pred_contribs=True,
    )
    contributions = np.asarray(contributions)
    if contributions.ndim == 3:
        contributions = contributions[:, 1, :]
    contributions = contributions[:, :-1]
    positions = {feature: index for index, feature in enumerate(matrix.columns)}
    groups = {
        group: [positions[name] for name in names if name in positions]
        for group, names in FEATURE_GROUPS.items()
    }
    groups = {group: indices for group, indices in groups.items() if indices}

    reasons: list[str] = []
    for row_position, (_, row) in enumerate(source.iterrows()):
        group_values = {
            group: float(contributions[row_position, indices].sum())
            for group, indices in groups.items()
        }
        positive = [
            group
            for group, value in sorted(group_values.items(), key=lambda item: item[1], reverse=True)
            if value > 1e-9
        ][:max_reasons]
        reasons.append(
            "; ".join(_reason_text(group, row) for group in positive)
            if positive
            else "모델상 두드러진 양(+) 기여 요인 없음"
        )
    return reasons
