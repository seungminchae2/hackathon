from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


def main(input_path: str, output_path: str, spacing_m: float) -> None:
    gdf = gpd.read_file(input_path)
    if gdf.empty:
        raise ValueError("SHP에 도로 객체가 없습니다.")
    if gdf.crs is None:
        raise ValueError("SHP 좌표계 정보가 없습니다.")

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
    args = parser.parse_args()
    main(args.input, args.output, args.spacing_m)
