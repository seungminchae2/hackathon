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
    "risk_percentile",
    "is_top_95",
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
    "road_doctor_kakao_map_v2",
    path=str(_COMPONENT_DIR),
)


def _records_for_map(predictions: pd.DataFrame) -> list[dict]:
    frame = predictions.copy()
    for column in MAP_COLUMNS:
        if column not in frame.columns:
            frame[column] = None
    frame = frame[MAP_COLUMNS].replace({np.nan: None})
    
    # 넘파이/판다스 타입을 순수 파이썬 기본형(bool, float, int)으로 강제 변환하여 
    # Streamlit 컴포넌트 간 JSON 직렬화 오류를 원천 차단합니다.
    records = []
    for row in frame.to_dict(orient="records"):
        clean_row = {}
        for k, v in row.items():
            if v is None:
                clean_row[k] = None
            elif isinstance(v, (np.bool_, bool)):
                clean_row[k] = bool(v)
            elif isinstance(v, (np.integer, int)):
                clean_row[k] = int(v)
            elif isinstance(v, (np.floating, float)):
                clean_row[k] = float(v)
            else:
                clean_row[k] = v
        records.append(clean_row)
    return records


def show_kakao_map(
    predictions: pd.DataFrame,
    app_key: str,
    height: int = 730,
    route_plan: dict[str, Any] | None = None,
) -> None:
    """동일 출처의 Streamlit 컴포넌트로 카카오 지도와 보수 목록을 표시합니다."""
    _KAKAO_MAP_COMPONENT(
        rows=_records_for_map(predictions),
        appKey=app_key.strip(),
        componentHeight=height,
        routePlan=route_plan,
        default=None,
    )