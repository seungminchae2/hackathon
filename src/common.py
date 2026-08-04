from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["_base_dir"] = str(path.resolve().parent)
    return config


def resolve_path(config: dict[str, Any], key: str) -> Path:
    base_dir = Path(config["_base_dir"])
    raw = Path(config["paths"][key])
    return raw if raw.is_absolute() else base_dir / raw


def require_columns(df: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{name}에 필요한 컬럼이 없습니다: {missing}")


def parse_date_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    out = df.copy()
    out[column] = pd.to_datetime(out[column], errors="coerce").dt.normalize()
    out = out.dropna(subset=[column])
    return out


def meters_to_lat_step(meters: float) -> float:
    return meters / 111_320.0


def meters_to_lon_step(meters: float, reference_lat: float = 35.8) -> float:
    return meters / (111_320.0 * np.cos(np.radians(reference_lat)))


def add_grid_columns(
    df: pd.DataFrame,
    grid_size_m: int = 500,
    lat_col: str = "lat",
    lon_col: str = "lon",
    reference_lat: float = 35.8,
) -> pd.DataFrame:
    """해커톤용 근사 격자. 전북 범위에서는 500m 수준 시각화에 충분합니다."""
    require_columns(df, [lat_col, lon_col], "좌표 데이터")
    out = df.copy()
    out[lat_col] = pd.to_numeric(out[lat_col], errors="coerce")
    out[lon_col] = pd.to_numeric(out[lon_col], errors="coerce")
    out = out.dropna(subset=[lat_col, lon_col])

    lat_step = meters_to_lat_step(grid_size_m)
    lon_step = meters_to_lon_step(grid_size_m, reference_lat)

    out["grid_y"] = np.floor(out[lat_col] / lat_step).astype(int)
    out["grid_x"] = np.floor(out[lon_col] / lon_step).astype(int)
    out["grid_id"] = out["grid_x"].astype(str) + "_" + out["grid_y"].astype(str)
    out["grid_lat"] = (out["grid_y"] + 0.5) * lat_step
    out["grid_lon"] = (out["grid_x"] + 0.5) * lon_step
    return out


def haversine_distance_matrix(
    source_lat: np.ndarray,
    source_lon: np.ndarray,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
) -> np.ndarray:
    """source N개와 target M개 사이 거리(km) 행렬을 반환합니다."""
    r = 6371.0088
    s_lat = np.radians(source_lat)[:, None]
    s_lon = np.radians(source_lon)[:, None]
    t_lat = np.radians(target_lat)[None, :]
    t_lon = np.radians(target_lon)[None, :]

    dlat = t_lat - s_lat
    dlon = t_lon - s_lon
    a = np.sin(dlat / 2) ** 2 + np.cos(s_lat) * np.cos(t_lat) * np.sin(dlon / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def safe_numeric(df: pd.DataFrame, columns: list[str], fill: float | None = None) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")
        if fill is not None:
            out[col] = out[col].fillna(fill)
    return out
