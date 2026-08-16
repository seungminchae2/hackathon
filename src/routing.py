from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
import requests

from .common import haversine_distance_matrix


LOCAL_ADDRESS_URL = "https://dapi.kakao.com/v2/local/search/address.json"
LOCAL_KEYWORD_URL = "https://dapi.kakao.com/v2/local/search/keyword.json"
DIRECTIONS_URL = "https://apis-navi.kakaomobility.com/v1/directions"
HIGH_RISK_LEVELS = {"매우 높음", "높음"}


class KakaoApiError(RuntimeError):
    pass


def _headers(rest_api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"KakaoAK {rest_api_key}",
        "Content-Type": "application/json",
    }


def _get_json(
    url: str,
    rest_api_key: str,
    params: dict[str, Any],
    timeout: int = 12,
) -> dict[str, Any]:
    try:
        response = requests.get(
            url,
            headers=_headers(rest_api_key),
            params=params,
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise KakaoApiError("카카오 API에 연결하지 못했습니다.") from exc
    if not response.ok:
        try:
            detail = response.json().get("message") or response.json().get("msg")
        except (ValueError, AttributeError):
            detail = None
        suffix = f" — {detail}" if detail else ""
        raise KakaoApiError(f"카카오 API 오류({response.status_code}){suffix}")
    try:
        return response.json()
    except ValueError as exc:
        raise KakaoApiError("카카오 API 응답이 JSON 형식이 아닙니다.") from exc


def _coordinate_query(query: str) -> dict[str, Any] | None:
    match = re.fullmatch(
        r"\s*(12[4-9]|13[0-2])(?:\.\d+)?\s*[, ]\s*(3[3-9])(?:\.\d+)?\s*",
        query,
    )
    if not match:
        return None
    numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", query)
    lon, lat = map(float, numbers[:2])
    return {"name": query.strip(), "address": "직접 입력 좌표", "lon": lon, "lat": lat}


def resolve_place(query: str, rest_api_key: str) -> dict[str, Any]:
    """주소·장소명 또는 '경도,위도'를 하나의 좌표로 변환합니다."""
    query = query.strip()
    if not query:
        raise KakaoApiError("출발지와 도착지를 모두 입력하십시오.")

    coordinates = _coordinate_query(query)
    if coordinates:
        return coordinates

    address_payload = _get_json(
        LOCAL_ADDRESS_URL,
        rest_api_key,
        {"query": query, "size": 5},
    )
    documents = address_payload.get("documents") or []
    if documents:
        document = documents[0]
        road_address = document.get("road_address") or {}
        return {
            "name": road_address.get("building_name") or query,
            "address": road_address.get("address_name") or document.get("address_name") or query,
            "lon": float(document["x"]),
            "lat": float(document["y"]),
        }

    keyword_payload = _get_json(
        LOCAL_KEYWORD_URL,
        rest_api_key,
        {"query": query, "size": 15, "sort": "accuracy"},
    )
    documents = keyword_payload.get("documents") or []
    if not documents:
        raise KakaoApiError(f"장소를 찾지 못했습니다: {query}")

    def jeonbuk_first(document: dict[str, Any]) -> tuple[int, int]:
        address = f"{document.get('address_name', '')} {document.get('road_address_name', '')}"
        is_jeonbuk = any(name in address for name in ["전북특별자치도", "전라북도", "전북"])
        return (0 if is_jeonbuk else 1, documents.index(document))

    document = sorted(documents, key=jeonbuk_first)[0]
    return {
        "name": document.get("place_name") or query,
        "address": document.get("road_address_name") or document.get("address_name") or query,
        "lon": float(document["x"]),
        "lat": float(document["y"]),
    }


def _extract_route(route: dict[str, Any], route_index: int) -> dict[str, Any]:
    summary = route.get("summary") or {}
    path: list[list[float]] = []
    road_names: Counter[str] = Counter()
    for section in route.get("sections") or []:
        for road in section.get("roads") or []:
            vertices = road.get("vertexes") or []
            road_name = str(road.get("name") or "").strip()
            if road_name:
                road_names[road_name] += max(int(road.get("distance") or 0), 1)
            for position in range(0, len(vertices) - 1, 2):
                point = [float(vertices[position]), float(vertices[position + 1])]
                if not path or point != path[-1]:
                    path.append(point)

    if len(path) < 2:
        origin = summary.get("origin") or {}
        destination = summary.get("destination") or {}
        path = [
            [float(origin.get("x", 0)), float(origin.get("y", 0))],
            [float(destination.get("x", 0)), float(destination.get("y", 0))],
        ]

    return {
        "id": f"route-{route_index + 1}",
        "label": "카카오 추천 경로" if route_index == 0 else f"대안 경로 {route_index}",
        "path": path,
        "distance_m": int(summary.get("distance") or 0),
        "duration_s": int(summary.get("duration") or 0),
        "fare_won": int((summary.get("fare") or {}).get("taxi") or 0),
        "road_names": [name for name, _ in road_names.most_common(3)],
    }


def fetch_directions(
    origin: dict[str, Any],
    destination: dict[str, Any],
    rest_api_key: str,
) -> list[dict[str, Any]]:
    """추천·최단시간·최단거리 요청을 합치고 중복 경로를 제거합니다."""
    collected: list[dict[str, Any]] = []
    signatures: set[tuple[int, int, int, int]] = set()
    for priority in ["RECOMMEND", "TIME", "DISTANCE"]:
        payload = _get_json(
            DIRECTIONS_URL,
            rest_api_key,
            {
                "origin": f"{origin['lon']},{origin['lat']},name={origin['name']}",
                "destination": f"{destination['lon']},{destination['lat']},name={destination['name']}",
                "priority": priority,
                "alternatives": "true",
                "road_details": "true",
                "summary": "false",
                "roadevent": 0,
            },
        )
        for raw_route in payload.get("routes") or []:
            if int(raw_route.get("result_code", 0)) != 0:
                continue
            route = _extract_route(raw_route, len(collected))
            middle = route["path"][len(route["path"]) // 2]
            signature = (
                round(route["distance_m"] / 100),
                round(route["duration_s"] / 30),
                round(middle[0] * 1000),
                round(middle[1] * 1000),
            )
            if signature not in signatures:
                signatures.add(signature)
                collected.append(route)
        if len(collected) >= 3:
            break

    if not collected:
        raise KakaoApiError("출발지와 도착지 사이의 자동차 경로를 찾지 못했습니다.")
    return collected[:3]


def _haversine_km(point_a: list[float], point_b: list[float]) -> float:
    lon1, lat1 = map(math.radians, point_a)
    lon2, lat2 = map(math.radians, point_b)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    value = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6371.0088 * 2 * math.asin(math.sqrt(value))


def _sample_path(path: list[list[float]], interval_km: float = 0.15) -> np.ndarray:
    samples: list[list[float]] = [path[0]]
    for start, end in zip(path[:-1], path[1:]):
        distance = _haversine_km(start, end)
        count = max(1, int(math.ceil(distance / interval_km)))
        for step in range(1, count + 1):
            ratio = step / count
            samples.append(
                [
                    start[0] + (end[0] - start[0]) * ratio,
                    start[1] + (end[1] - start[1]) * ratio,
                ]
            )
    return np.asarray(samples, dtype=float)


def analyze_route_risk(
    route: dict[str, Any],
    predictions: pd.DataFrame,
    match_radius_km: float = 0.36,
) -> dict[str, Any]:
    samples = _sample_path(route["path"])
    grid = predictions.reset_index(drop=True)
    distances = haversine_distance_matrix(
        samples[:, 1],
        samples[:, 0],
        pd.to_numeric(grid["grid_lat"], errors="coerce").to_numpy(),
        pd.to_numeric(grid["grid_lon"], errors="coerce").to_numpy(),
    )
    nearest_indices = distances.argmin(axis=1)
    nearest_distances = distances[np.arange(len(samples)), nearest_indices]
    within = nearest_distances <= match_radius_km
    risk_scores = pd.to_numeric(grid["risk_score"], errors="coerce").fillna(0).to_numpy()
    nearest_scores = risk_scores[nearest_indices]
    exposure_values = np.where(within, nearest_scores, 0.0)
    nearest_levels = grid["risk_level"].astype(str).to_numpy()[nearest_indices]
    high_samples = within & np.isin(nearest_levels, list(HIGH_RISK_LEVELS))
    high_indices = sorted(
        set(nearest_indices[high_samples]),
        key=lambda index: risk_scores[index],
        reverse=True,
    )
    matched_indices = set(nearest_indices[within])

    output = dict(route)
    output.update(
        {
            "risk_exposure": float(exposure_values.mean()),
            "max_risk": float(max((risk_scores[index] for index in matched_indices), default=0.0)),
            "high_risk_count": len(high_indices),
            "high_risk_km": float(route["distance_m"] / 1000 * high_samples.mean()),
            "danger_grid_ids": [str(grid.iloc[index]["grid_id"]) for index in high_indices],
            "danger_points": [
                {
                    "grid_id": str(grid.iloc[index]["grid_id"]),
                    "lat": float(grid.iloc[index]["grid_lat"]),
                    "lon": float(grid.iloc[index]["grid_lon"]),
                    "risk_score": float(risk_scores[index]),
                    "risk_level": str(grid.iloc[index]["risk_level"]),
                }
                for index in high_indices[:30]
            ],
        }
    )
    return output


def choose_safe_route(routes: list[dict[str, Any]]) -> tuple[int, str]:
    baseline = routes[0]
    if baseline["high_risk_count"] == 0:
        return 0, "기본 경로에서 고위험 격자가 발견되지 않았습니다."

    duration_limit = max(baseline["duration_s"] * 1.30, baseline["duration_s"] + 300)
    eligible = [
        (index, route)
        for index, route in enumerate(routes)
        if route["duration_s"] <= duration_limit
    ]
    safest_index, safest = min(
        eligible,
        key=lambda item: (
            item[1]["risk_exposure"],
            item[1]["high_risk_count"],
            item[1]["duration_s"],
        ),
    )
    improved = (
        safest["risk_exposure"] < baseline["risk_exposure"] * 0.95
        or safest["high_risk_count"] < baseline["high_risk_count"]
    )
    if safest_index == 0 or not improved:
        return 0, "30% 이내 시간 증가로 위험을 줄이는 대안 경로가 없어 기본 경로를 유지합니다."

    reduction = (
        1 - safest["risk_exposure"] / baseline["risk_exposure"]
        if baseline["risk_exposure"] > 0
        else 0
    )
    extra_minutes = max(0, round((safest["duration_s"] - baseline["duration_s"]) / 60))
    return safest_index, f"기본 경로보다 위험 노출을 {reduction:.0%} 줄이는 우회 경로입니다. 예상 시간은 {extra_minutes}분 늘어납니다."


def build_safe_route_plan(
    origin_query: str,
    destination_query: str,
    rest_api_key: str,
    predictions: pd.DataFrame,
) -> dict[str, Any]:
    origin = resolve_place(origin_query, rest_api_key)
    destination = resolve_place(destination_query, rest_api_key)
    raw_routes = fetch_directions(origin, destination, rest_api_key)
    analyzed = [analyze_route_risk(route, predictions) for route in raw_routes]
    recommended_index, message = choose_safe_route(analyzed)

    client_routes: list[dict[str, Any]] = []
    for index, route in enumerate(analyzed):
        step = max(1, len(route["path"]) // 1200)
        path = route["path"][::step]
        if path[-1] != route["path"][-1]:
            path.append(route["path"][-1])
        selected = index == recommended_index
        client_routes.append(
            {
                **route,
                "path": path,
                "selected": selected,
                "label": "안전 우회 추천" if selected and index != 0 else route["label"],
                "color": "#13795b" if selected else ("#e76f51" if index == 0 else "#64748b"),
            }
        )

    recommended = client_routes[recommended_index]
    return {
        "origin": origin,
        "destination": destination,
        "routes": client_routes,
        "recommended_id": recommended["id"],
        "message": message,
        "is_detour": recommended_index != 0,
        "recommended": {
            "label": recommended["label"],
            "distance_km": recommended["distance_m"] / 1000,
            "duration_min": recommended["duration_s"] / 60,
            "risk_exposure": recommended["risk_exposure"],
            "high_risk_count": recommended["high_risk_count"],
            "high_risk_km": recommended["high_risk_km"],
            "road_names": recommended["road_names"],
        },
    }
