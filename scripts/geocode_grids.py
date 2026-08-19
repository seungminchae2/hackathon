"""예측 CSV의 grid_lat/grid_lon을 카카오 좌표->주소 변환으로 채워 address 컬럼을 추가합니다."""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.common import load_config, resolve_path
from src.routing import KakaoApiError, reverse_geocode


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ[key.strip()] = value.strip().strip('"').strip("'")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--sleep", type=float, default=0.05, help="요청 사이 대기 시간(초)")
    parser.add_argument("--workers", type=int, default=8, help="동시 요청 개수")
    parser.add_argument(
        "--only-missing",
        action="store_true",
        help="이미 address가 채워진 좌표는 건너뛰고, 비어있는 것만 새로 변환합니다.",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent
    load_env(base_dir / ".env")
    config = load_config(base_dir / args.config)

    rest_key_env = config.get("kakao", {}).get("rest_api_key_env", "KAKAO_REST_API_KEY")
    rest_api_key = os.getenv(rest_key_env, "").strip()
    if not rest_api_key:
        raise SystemExit(f".env에 {rest_key_env}가 설정되어 있어야 합니다.")

    prediction_path = resolve_path(config, "predictions")
    frame = pd.read_csv(prediction_path)
    if "address" not in frame.columns:
        frame["address"] = ""

    if args.only_missing:
        missing_mask = frame["address"].isna() | frame["address"].astype(str).str.strip().eq("")
        target_rows = frame.loc[missing_mask]
    else:
        target_rows = frame

    unique_coords = list(
        target_rows[["grid_lat", "grid_lon"]].drop_duplicates().itertuples(index=False, name=None)
    )
    total = len(unique_coords)
    address_by_coord: dict[tuple[float, float], str] = {}
    done = 0
    lock = threading.Lock()

    def fetch(coord: tuple[float, float]) -> tuple[tuple[float, float], str]:
        lat, lon = float(coord[0]), float(coord[1])
        time.sleep(args.sleep)
        try:
            return (lat, lon), reverse_geocode(lat, lon, rest_api_key)
        except KakaoApiError as exc:
            print(f"실패 ({lat}, {lon}): {exc}")
            return (lat, lon), ""

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fetch, coord) for coord in unique_coords]
        for future in as_completed(futures):
            key, address = future.result()
            address_by_coord[key] = address
            with lock:
                done += 1
                if done % 100 == 0 or done == total:
                    print(f"[{done}/{total}] 변환 중...")

    def resolved_address(lat: float, lon: float, existing: str) -> str:
        key = (float(lat), float(lon))
        if key in address_by_coord:
            return address_by_coord[key]
        return existing

    frame["address"] = [
        resolved_address(lat, lon, existing)
        for lat, lon, existing in zip(frame["grid_lat"], frame["grid_lon"], frame["address"])
    ]
    missing = int((frame["address"].isna() | frame["address"].astype(str).str.strip().eq("")).sum())
    frame.to_csv(prediction_path, index=False)
    print(f"완료: {prediction_path} 에 address 컬럼 저장 (주소 못 찾은 격자 {missing}개)")


if __name__ == "__main__":
    main()
