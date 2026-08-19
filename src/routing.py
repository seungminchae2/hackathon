from __future__ import annotations

from typing import Any

import requests


COORD2ADDRESS_URL = "https://dapi.kakao.com/v2/local/geo/coord2address.json"


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


def reverse_geocode(lat: float, lon: float, rest_api_key: str) -> str:
    """위경도 좌표를 도로명 주소(없으면 지번 주소) 문자열로 변환합니다."""
    payload = _get_json(
        COORD2ADDRESS_URL,
        rest_api_key,
        {"x": lon, "y": lat, "input_coord": "WGS84"},
    )
    documents = payload.get("documents") or []
    if not documents:
        return ""
    document = documents[0]
    road_address = document.get("road_address") or {}
    address = document.get("address") or {}
    return road_address.get("address_name") or address.get("address_name") or ""
