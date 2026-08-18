import pandas as pd
import numpy as np
from pathlib import Path


ROADS_FILE = Path("data/roads_with_sewer_age.csv")
REPAIRS_FILE = Path("data/repairs.csv")
OUTPUT_FILE = Path("data/roads_with_sewer_repair.csv")

MATCH_RADIUS_M = 300


def haversine_distance(
    lat1,
    lon1,
    lat2,
    lon2
):

    earth_radius_m = 6371000

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        +
        np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    c = 2 * np.arctan2(
        np.sqrt(a),
        np.sqrt(1 - a)
    )

    return earth_radius_m * c


roads = pd.read_csv(
    ROADS_FILE,
    encoding="utf-8-sig"
)

repairs = pd.read_csv(
    REPAIRS_FILE,
    encoding="utf-8-sig"
)


required_road_columns = [
    "lat",
    "lon",
]

required_repair_columns = [
    "repair_id",
    "repair_date",
    "lat",
    "lon",
    "repair_type",
]


missing_roads = [
    col
    for col in required_road_columns
    if col not in roads.columns
]

missing_repairs = [
    col
    for col in required_repair_columns
    if col not in repairs.columns
]


if missing_roads:
    raise ValueError(
        f"도로 파일에 필요한 컬럼이 없습니다: {missing_roads}"
    )

if missing_repairs:
    raise ValueError(
        f"repairs.csv에 필요한 컬럼이 없습니다: {missing_repairs}"
    )


roads["lat"] = pd.to_numeric(
    roads["lat"],
    errors="coerce"
)

roads["lon"] = pd.to_numeric(
    roads["lon"],
    errors="coerce"
)

repairs["lat"] = pd.to_numeric(
    repairs["lat"],
    errors="coerce"
)

repairs["lon"] = pd.to_numeric(
    repairs["lon"],
    errors="coerce"
)

repairs["repair_date"] = pd.to_datetime(
    repairs["repair_date"],
    errors="coerce"
)


repairs = repairs.dropna(
    subset=[
        "lat",
        "lon",
        "repair_date",
    ]
).copy()


results = []

repair_lats = repairs["lat"].to_numpy()
repair_lons = repairs["lon"].to_numpy()


for _, road in roads.iterrows():

    road_lat = road["lat"]
    road_lon = road["lon"]

    if pd.isna(road_lat) or pd.isna(road_lon):

        results.append(
            {
                "has_nearby_repair": 0,
                "repair_id": np.nan,
                "last_repair_date": pd.NaT,
                "repair_type": np.nan,
                "distance_to_last_repair_m": np.nan,
                "nearby_repair_count": 0,
            }
        )

        continue


    distances = haversine_distance(
        road_lat,
        road_lon,
        repair_lats,
        repair_lons
    )


    nearby_mask = (
        distances <= MATCH_RADIUS_M
    )


    nearby_count = int(
        nearby_mask.sum()
    )


    if nearby_count == 0:

        results.append(
            {
                "has_nearby_repair": 0,
                "repair_id": np.nan,
                "last_repair_date": pd.NaT,
                "repair_type": np.nan,
                "distance_to_last_repair_m": np.nan,
                "nearby_repair_count": 0,
            }
        )

        continue


    nearby_repairs = (
        repairs.loc[
            nearby_mask
        ]
        .copy()
    )


    nearby_repairs[
        "distance_m"
    ] = distances[
        nearby_mask
    ]


    latest_idx = (
        nearby_repairs[
            "repair_date"
        ]
        .idxmax()
    )


    latest_repair = (
        nearby_repairs.loc[
            latest_idx
        ]
    )


    results.append(
        {
            "has_nearby_repair": 1,
            "repair_id": latest_repair[
                "repair_id"
            ],
            "last_repair_date": latest_repair[
                "repair_date"
            ],
            "repair_type": latest_repair[
                "repair_type"
            ],
            "distance_to_last_repair_m": latest_repair[
                "distance_m"
            ],
            "nearby_repair_count": nearby_count,
        }
    )


repair_features = pd.DataFrame(
    results
)


roads = pd.concat(
    [
        roads.reset_index(drop=True),
        repair_features.reset_index(drop=True),
    ],
    axis=1
)


OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


roads.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


matched_count = int(
    roads[
        "has_nearby_repair"
    ].sum()
)

total_count = len(roads)

match_rate = (
    matched_count
    / total_count
    * 100
)


print(
    "도로 포인트 + 보수이력 결합 완료"
)

print(
    f"저장 위치: {OUTPUT_FILE}"
)

print(
    f"전체 도로 포인트: {total_count}"
)

print(
    f"{MATCH_RADIUS_M}m 이내 보수이력 매칭: {matched_count}"
)

print(
    f"매칭률: {match_rate:.2f}%"
)


if matched_count > 0:

    matched = roads[
        roads[
            "has_nearby_repair"
        ]
        == 1
    ]

    print(
        "\n매칭 거리 통계(m)"
    )

    print(
        matched[
            "distance_to_last_repair_m"
        ]
        .describe()
    )


print(
    "\n매칭 결과 예시"
)


display_columns = [
    col
    for col in [
        "point_id",
        "road_name",
        "city",
        "lat",
        "lon",
        "repair_id",
        "last_repair_date",
        "repair_type",
        "distance_to_last_repair_m",
        "nearby_repair_count",
        "has_nearby_repair",
    ]
    if col in roads.columns
]


print(
    roads[
        display_columns
    ]
    .head(30)
    .to_string(index=False)
)