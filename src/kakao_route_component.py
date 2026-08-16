from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit.components.v1 as components


MAP_COLUMNS = [
    "grid_id",
    "grid_lat",
    "grid_lon",
    "risk_score",
    "risk_level",
    "risk_reason",
    "priority_score",
    "priority_rank",
    "recurrence_score",
    "importance_score",
    "precip_3d",
    "precip_7d",
    "freeze_thaw_7d",
    "past_potholes_90d",
    "past_potholes_total",
    "has_repair_history",
    "days_since_last_repair",
]

_COMPONENT_DIR = Path(__file__).resolve().parent / "kakao_component"
_KAKAO_MAP_COMPONENT = components.declare_component(
    "road_doctor_kakao_route_map_v3",
    path=str(_COMPONENT_DIR),
)


def _records_for_map(predictions: pd.DataFrame) -> list[dict]:
    frame = predictions.copy()
    for column in MAP_COLUMNS:
        if column not in frame.columns:
            frame[column] = None
    frame = frame[MAP_COLUMNS].replace({np.nan: None})
    return frame.to_dict(orient="records")


def show_kakao_map(
    predictions: pd.DataFrame,
    app_key: str,
    height: int = 730,
    route_plan: dict[str, Any] | None = None,
) -> None:
    _KAKAO_MAP_COMPONENT(
        rows=_records_for_map(predictions),
        appKey=app_key.strip(),
        componentHeight=height,
        routePlan=route_plan,
        default=None,
    )
