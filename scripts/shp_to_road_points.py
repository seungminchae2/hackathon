from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


def main(
    input_path: str,
    output_path: str,
    spacing_m: float,
    bbox: tuple[float, float, float, float] | None,
    max_road_rank: int | None,
) -> None:
    gdf = gpd.read_file(input_path)
    if gdf.empty:
        raise ValueError("SHP에 도로 객체가 없습니다.")
    if gdf.crs is None:
        raise ValueError("SHP 좌표계 정보가 없습니다.")

    if max_road_rank is not None and "ROAD_RANK" in gdf.columns:
        ranks = pd.to_numeric(gdf["ROAD_RANK"], errors="coerce")
        gdf = gdf[ranks <= max_road_rank]
        if gdf.empty:
            raise ValueError("max_road_rank 조건을 만족하는 도로 객체가 없습니다.")
        print(f"도로 등급(ROAD_RANK<={max_road_rank}) 필터 적용: {len(gdf):,}개 객체 남음")

    if bbox is not None:
        min_lon, min_lat, max_lon, max_lat = bbox
        wgs84 = gdf.to_crs(epsg=4326)
        gdf = gdf.loc[wgs84.cx[min_lon:max_lon, min_lat:max_lat].index]
        if gdf.empty:
            raise ValueError("bbox 범위 안에 도로 객체가 없습니다. 좌표를 확인하십시오.")
        print(f"bbox 필터 적용: {len(gdf):,}개 객체 남음")

    metric = gdf.to_crs(epsg=5186).explode(index_parts=False)
    rows = []
    point_id = 1
    for idx, row in metric.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty or geom.length == 0:
            continue
        distances = np.arange(0, geom.length + spacing_m, spacing_m)
        for distance in distances:
            point = geom.interpolate(min(distance, geom.length))
            rows.append(
                {
                    "road_point_id": f"R{point_id:08d}",
                    "x": point.x,
                    "y": point.y,
                    "road_name": row.get("road_name", row.get("ROAD_NAME", "")),
                }
            )
            point_id += 1

    points = gpd.GeoDataFrame(
        rows,
        geometry=gpd.points_from_xy([r["x"] for r in rows], [r["y"] for r in rows]),
        crs="EPSG:5186",
    ).to_crs(epsg=4326)
    out = pd.DataFrame(
        {
            "road_point_id": points["road_point_id"],
            "lat": points.geometry.y,
            "lon": points.geometry.x,
            "road_name": points["road_name"],
        }
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)
    print(f"변환 완료: {output} ({len(out):,}개 점)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="data/roads.csv")
    parser.add_argument("--spacing-m", type=float, default=200)
    parser.add_argument(
        "--bbox",
        default=None,
        help="min_lon,min_lat,max_lon,max_lat (WGS84). 전국 데이터에서 특정 지역만 골라낼 때 사용",
    )
    parser.add_argument(
        "--max-road-rank",
        type=int,
        default=None,
        help="표준노드링크 ROAD_RANK 이하만 포함 (예: 106=시군도 이상만, 107 제외 시 일반시도/골목 제거)",
    )
    args = parser.parse_args()
    bbox = tuple(float(v) for v in args.bbox.split(",")) if args.bbox else None
    if bbox is not None and len(bbox) != 4:
        raise SystemExit("--bbox는 min_lon,min_lat,max_lon,max_lat 4개 값이어야 합니다.")
    main(args.input, args.output, args.spacing_m, bbox, args.max_road_rank)
