import pandas as pd
import numpy as np
import re
from pathlib import Path


ROADS_FILE = Path(
    "data/roads_with_sewer_repair.csv"
)

TRAFFIC_FILE = Path(
    "data/전북국도교통량(상시조사교통량).csv"
)

TRAFFIC_COORD_FILE = Path(
    "data/2025년 도로종류별 교통량 및 X,Y좌표.xlsx"
)

OUTPUT_FILE = Path(
    "data/roads_with_sewer_repair_traffic.csv"
)


MAX_TRAFFIC_DISTANCE_M = 3000


ESAL_FACTORS = {
    "승용차": 0.0002,
    "버스": 0.852,
    "소형화물": 0.004,
    "중형화물": 1.735,
    "대형화물": 3.169,
}


TARGET_CITIES = [
    "전주시",
    "군산시",
    "익산시",
    "남원시",
    "정읍시",
]


def read_csv_auto_encoding(path, **kwargs):

    encodings = [
        "utf-8-sig",
        "utf-8",
        "cp949",
        "euc-kr",
    ]

    last_error = None

    for encoding in encodings:

        try:
            return pd.read_csv(
                path,
                encoding=encoding,
                **kwargs
            )

        except UnicodeDecodeError as e:
            last_error = e

    raise last_error


def clean_number(series):

    return pd.to_numeric(
        series
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.strip(),
        errors="coerce"
    ).fillna(0)


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


def extract_station_id(section_name):

    if pd.isna(section_name):
        return None

    match = re.search(
        r"\((\d{4}-\d{2})\)",
        str(section_name)
    )

    if match:
        return match.group(1)

    return None


MANUAL_CITY_MAP = {
    "대강면-동계면(1317-00)": "남원시",
    "식정동-장수읍(1909-05)": "남원시",
    "정읍IC-흥덕면(2202-01)": "정읍시",
    "김제시-대야면(2912-01)": "군산시",
    "개정면-성산면(2914-00)": "군산시",
}


def get_traffic_city(section):

    if pd.isna(section):
        return None

    text = str(section)

    for city in TARGET_CITIES:

        if city in text:
            return city

    return MANUAL_CITY_MAP.get(text)


def minmax_from_reference(
    values,
    reference_values
):

    values = pd.to_numeric(
        values,
        errors="coerce"
    )

    reference_values = pd.to_numeric(
        reference_values,
        errors="coerce"
    ).dropna()

    if len(reference_values) == 0:

        return pd.Series(
            np.nan,
            index=values.index
        )

    minimum = reference_values.min()
    maximum = reference_values.max()

    if maximum == minimum:

        return pd.Series(
            0.0,
            index=values.index
        )

    return (
        (values - minimum)
        /
        (maximum - minimum)
        * 100
    ).clip(
        0,
        100
    )


# ============================================================
# 1. 도로 포인트 데이터
#
# 하수관로 + 300m 보수이력까지 결합된
# 680개 포인트를 기준으로 사용한다.
# ============================================================

roads = read_csv_auto_encoding(
    ROADS_FILE
)


roads["lat"] = pd.to_numeric(
    roads["lat"],
    errors="coerce"
)

roads["lon"] = pd.to_numeric(
    roads["lon"],
    errors="coerce"
)


if "point_id" not in roads.columns:

    roads = roads.reset_index(
        drop=True
    )

    roads["point_id"] = np.arange(
        len(roads)
    )


print(
    "도로 포인트 수:",
    len(roads)
)


# ============================================================
# 2. 기존 전북 상시교통량 CSV 읽기
# ============================================================

raw = read_csv_auto_encoding(
    TRAFFIC_FILE,
    header=None
)


traffic = raw.iloc[
    2:,
    :17
].copy()


traffic.columns = [
    "노선명",
    "구간명",
    "AADT",
    "승용차",
    "버스",
    "소형화물",
    "중형화물",
    "대형화물",
    "평균교통량_계",
    "평균교통량_비율",
    "주말_승용차",
    "주말_버스",
    "주말_소형화물",
    "주말_중형화물",
    "주말_대형화물",
    "주말교통량_계",
    "주말교통량_비율",
]


traffic = traffic[
    traffic["구간명"].notna()
].copy()


traffic["노선명"] = (
    traffic["노선명"]
    .replace(
        r"^\s*$",
        np.nan,
        regex=True
    )
    .ffill()
)


for col in [
    "AADT",
    "승용차",
    "버스",
    "소형화물",
    "중형화물",
    "대형화물",
]:

    traffic[col] = clean_number(
        traffic[col]
    )


traffic["station_id"] = (
    traffic["구간명"]
    .apply(
        extract_station_id
    )
)


traffic["city"] = (
    traffic["구간명"]
    .apply(
        get_traffic_city
    )
)


print(
    "교통량 조사구간 수:",
    len(traffic)
)

print(
    "지점번호 추출 성공:",
    traffic[
        "station_id"
    ]
    .notna()
    .sum()
)


# ============================================================
# 3. ESAL 계산
# ============================================================

for vehicle, factor in ESAL_FACTORS.items():

    traffic[
        f"{vehicle}_ESAL"
    ] = (
        traffic[vehicle]
        * factor
    )


traffic["traffic_esal"] = sum(
    traffic[
        f"{vehicle}_ESAL"
    ]
    for vehicle in ESAL_FACTORS
)


traffic["total_traffic"] = (
    traffic[
        list(
            ESAL_FACTORS.keys()
        )
    ]
    .sum(
        axis=1
    )
)


traffic["heavy_traffic"] = (
    traffic["중형화물"]
    + traffic["대형화물"]
)


traffic["heavy_vehicle_ratio"] = (
    traffic["heavy_traffic"]
    /
    traffic[
        "total_traffic"
    ].replace(
        0,
        np.nan
    )
)


# ============================================================
# 4. 공식 X,Y 좌표 파일 읽기
# ============================================================

coords = pd.read_excel(
    TRAFFIC_COORD_FILE,
    sheet_name="X,Y좌표"
)


coords = coords[
    (
        coords["도로종류"]
        == "일반국도"
    )
    &
    (
        coords["구분"]
        == "상시"
    )
    &
    (
        coords["도/시"]
        == "전북"
    )
].copy()


coords["station_id"] = (
    coords["지점번호"]
    .astype(str)
    .str.strip()
)


coords["traffic_lat"] = pd.to_numeric(
    coords["XCODE"],
    errors="coerce"
)


coords["traffic_lon"] = pd.to_numeric(
    coords["YCODE"],
    errors="coerce"
)


coords = coords.dropna(
    subset=[
        "traffic_lat",
        "traffic_lon",
    ]
)


print(
    "공식 전북 일반국도 상시조사 좌표 수:",
    len(coords)
)


# ============================================================
# 5. ESAL + 공식 좌표 결합
# ============================================================

traffic_locations = traffic.merge(
    coords[
        [
            "station_id",
            "호선",
            "구간",
            "시/군",
            "traffic_lat",
            "traffic_lon",
        ]
    ],
    on="station_id",
    how="left"
)


matched_coords = (
    traffic_locations[
        "traffic_lat"
    ]
    .notna()
    .sum()
)


print(
    "교통량 + 공식좌표 결합 성공:",
    matched_coords,
    "/",
    len(traffic_locations)
)


unmatched_coords = traffic_locations[
    traffic_locations[
        "traffic_lat"
    ]
    .isna()
]


if len(unmatched_coords) > 0:

    print(
        "\n좌표 매칭 실패 구간"
    )

    print(
        unmatched_coords[
            [
                "노선명",
                "구간명",
                "station_id",
            ]
        ]
        .to_string(
            index=False
        )
    )


traffic_locations = traffic_locations[
    traffic_locations[
        "traffic_lat"
    ]
    .notna()
].copy()


# ============================================================
# 6. 도시별 ESAL 중앙값 생성
#
# 최근접 조사점이 3km보다 먼 경우 사용할 fallback.
#
# 조사점이 지나치게 먼 상황에서
# 무조건 가장 가까운 국도의 교통량을 적용하는 것보다
# 같은 도시의 관측 교통량 중앙값을 사용하는 방식이다.
#
# 3km는 국가 공식 기준이 아니라
# 프로젝트 공간매칭 임계값이다.
# ============================================================

city_medians = (
    traffic_locations[
        traffic_locations[
            "city"
        ].notna()
    ]
    .groupby(
        "city",
        as_index=False
    )
    .agg(
        city_median_esal=(
            "traffic_esal",
            "median"
        ),
        city_median_total_traffic=(
            "total_traffic",
            "median"
        ),
        city_median_heavy_ratio=(
            "heavy_vehicle_ratio",
            "median"
        )
    )
)


city_median_dict = (
    city_medians
    .set_index(
        "city"
    )
    .to_dict(
        orient="index"
    )
)


print(
    "\n도시별 ESAL 중앙값"
)


print(
    city_medians
    .to_string(
        index=False
    )
)


# ============================================================
# 7. 포인트별 교통량 매칭
#
# ① 도로 포인트와 같은 city의 교통량 조사점만 후보로 사용
#
# ② 그중 가장 가까운 조사점 탐색
#
# ③ 거리가 3km 이하
#    → 실제 조사점 ESAL 사용
#
# ④ 3km 초과
#    → 해당 도시 ESAL 중앙값 사용
#
# 이를 통해 5~13km 떨어진 조사점의 값이
# 무조건 연결되는 문제를 방지한다.
# ============================================================

results = []


for _, road in roads.iterrows():

    road_lat = road["lat"]
    road_lon = road["lon"]
    road_city = road["city"]


    if (
        pd.isna(road_lat)
        or pd.isna(road_lon)
        or pd.isna(road_city)
    ):

        results.append(
            {
                "traffic_station_id": np.nan,
                "traffic_route": np.nan,
                "traffic_section": np.nan,
                "traffic_distance_m": np.nan,
                "traffic_esal": np.nan,
                "total_traffic": np.nan,
                "heavy_vehicle_ratio": np.nan,
                "traffic_match_type": "unmatched",
            }
        )

        continue


    # 같은 도시의 조사점만 후보
    candidates = traffic_locations[
        traffic_locations[
            "city"
        ]
        == road_city
    ].copy()


    # 같은 도시 조사점 자체가 없는 경우
    if len(candidates) == 0:

        median_info = city_median_dict.get(
            road_city
        )

        if median_info is None:

            results.append(
                {
                    "traffic_station_id": np.nan,
                    "traffic_route": np.nan,
                    "traffic_section": np.nan,
                    "traffic_distance_m": np.nan,
                    "traffic_esal": np.nan,
                    "total_traffic": np.nan,
                    "heavy_vehicle_ratio": np.nan,
                    "traffic_match_type": "unmatched",
                }
            )

        else:

            results.append(
                {
                    "traffic_station_id": np.nan,
                    "traffic_route": np.nan,
                    "traffic_section": np.nan,
                    "traffic_distance_m": np.nan,
                    "traffic_esal":
                        median_info[
                            "city_median_esal"
                        ],
                    "total_traffic":
                        median_info[
                            "city_median_total_traffic"
                        ],
                    "heavy_vehicle_ratio":
                        median_info[
                            "city_median_heavy_ratio"
                        ],
                    "traffic_match_type":
                        "city_median",
                }
            )

        continue


    candidate_lats = (
        candidates[
            "traffic_lat"
        ]
        .to_numpy()
    )

    candidate_lons = (
        candidates[
            "traffic_lon"
        ]
        .to_numpy()
    )


    distances = haversine_distance(
        road_lat,
        road_lon,
        candidate_lats,
        candidate_lons
    )


    nearest_position = int(
        np.argmin(
            distances
        )
    )


    nearest_distance = float(
        distances[
            nearest_position
        ]
    )


    nearest = candidates.iloc[
        nearest_position
    ]


    # ========================================================
    # 3km 이내
    # → 실제 조사점 사용
    # ========================================================

    if (
        nearest_distance
        <= MAX_TRAFFIC_DISTANCE_M
    ):

        results.append(
            {
                "traffic_station_id":
                    nearest[
                        "station_id"
                    ],

                "traffic_route":
                    nearest[
                        "노선명"
                    ],

                "traffic_section":
                    nearest[
                        "구간명"
                    ],

                "traffic_distance_m":
                    nearest_distance,

                "traffic_esal":
                    nearest[
                        "traffic_esal"
                    ],

                "total_traffic":
                    nearest[
                        "total_traffic"
                    ],

                "heavy_vehicle_ratio":
                    nearest[
                        "heavy_vehicle_ratio"
                    ],

                "traffic_match_type":
                    "nearest_station",
            }
        )


    # ========================================================
    # 3km 초과
    # → 도시 중앙값 사용
    #
    # 실제 최근접 조사점 정보와 거리는 남겨두지만
    # ESAL은 도시 중앙값으로 대체한다.
    # ========================================================

    else:

        median_info = city_median_dict.get(
            road_city
        )


        if median_info is None:

            results.append(
                {
                    "traffic_station_id":
                        nearest[
                            "station_id"
                        ],

                    "traffic_route":
                        nearest[
                            "노선명"
                        ],

                    "traffic_section":
                        nearest[
                            "구간명"
                        ],

                    "traffic_distance_m":
                        nearest_distance,

                    "traffic_esal": np.nan,
                    "total_traffic": np.nan,
                    "heavy_vehicle_ratio": np.nan,

                    "traffic_match_type":
                        "unmatched",
                }
            )

        else:

            results.append(
                {
                    "traffic_station_id":
                        nearest[
                            "station_id"
                        ],

                    "traffic_route":
                        nearest[
                            "노선명"
                        ],

                    "traffic_section":
                        nearest[
                            "구간명"
                        ],

                    "traffic_distance_m":
                        nearest_distance,

                    "traffic_esal":
                        median_info[
                            "city_median_esal"
                        ],

                    "total_traffic":
                        median_info[
                            "city_median_total_traffic"
                        ],

                    "heavy_vehicle_ratio":
                        median_info[
                            "city_median_heavy_ratio"
                        ],

                    "traffic_match_type":
                        "city_median",
                }
            )


traffic_features = pd.DataFrame(
    results
)


roads = pd.concat(
    [
        roads.reset_index(
            drop=True
        ),
        traffic_features.reset_index(
            drop=True
        ),
    ],
    axis=1
)


# ============================================================
# 8. traffic_score 계산
#
# ESAL은 편차가 크므로 log1p를 적용한다.
#
# 점수 범위는 실제 48개 교통량 조사구간의
# ESAL 최소~최대를 기준으로 고정한다.
#
# 이렇게 해야 어떤 포인트가 선택됐는지에 따라
# Min-Max 기준 자체가 변하지 않는다.
# ============================================================

reference_log_esal = np.log1p(
    traffic_locations[
        "traffic_esal"
    ]
)


roads[
    "traffic_score"
] = (
    minmax_from_reference(
        np.log1p(
            roads[
                "traffic_esal"
            ]
        ),
        reference_log_esal
    )
)


# ============================================================
# 9. 결과 저장
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


roads.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 10. 결과 검증
# ============================================================

nearest_count = int(
    (
        roads[
            "traffic_match_type"
        ]
        == "nearest_station"
    )
    .sum()
)


median_count = int(
    (
        roads[
            "traffic_match_type"
        ]
        == "city_median"
    )
    .sum()
)


unmatched_count = int(
    (
        roads[
            "traffic_match_type"
        ]
        == "unmatched"
    )
    .sum()
)


print(
    "\n도로 포인트 + ESAL 공간매칭 완료"
)

print(
    f"저장 위치: {OUTPUT_FILE}"
)

print(
    f"전체 포인트: {len(roads)}"
)


print(
    f"\n{MAX_TRAFFIC_DISTANCE_M / 1000:.0f}km "
    f"이내 실제 조사점 매칭: {nearest_count}"
)

print(
    f"도시 중앙값 대체: {median_count}"
)

print(
    f"매칭 실패: {unmatched_count}"
)


# ============================================================
# 실제 조사점을 사용한 포인트의 거리만 통계 확인
# ============================================================

nearest_matched = roads[
    roads[
        "traffic_match_type"
    ]
    == "nearest_station"
]


if len(
    nearest_matched
) > 0:

    print(
        "\n실제 조사점 매칭 거리 통계(m)"
    )

    print(
        nearest_matched[
            "traffic_distance_m"
        ]
        .describe()
    )


print(
    "\ntraffic_match_type 분포"
)

print(
    roads[
        "traffic_match_type"
    ]
    .value_counts()
)


print(
    "\n서로 다른 실제/대체 traffic_esal 개수:",
    roads[
        "traffic_esal"
    ]
    .nunique()
)


print(
    "서로 다른 traffic_score 개수:",
    roads[
        "traffic_score"
    ]
    .nunique()
)


print(
    "\n시별 traffic_score 개수"
)

print(
    roads
    .groupby(
        "city"
    )[
        "traffic_score"
    ]
    .nunique()
)


print(
    "\n가장 많이 사용된 조사점"
)

print(
    roads[
        roads[
            "traffic_match_type"
        ]
        == "nearest_station"
    ][
        [
            "traffic_station_id",
            "traffic_section",
        ]
    ]
    .value_counts()
    .head(20)
)


print(
    "\n매칭 결과 예시"
)


display_columns = [
    "point_id",
    "road_name",
    "city",
    "lat",
    "lon",
    "traffic_station_id",
    "traffic_route",
    "traffic_section",
    "traffic_distance_m",
    "traffic_match_type",
    "traffic_esal",
    "traffic_score",
]


display_columns = [
    col
    for col in display_columns
    if col in roads.columns
]


print(
    roads[
        display_columns
    ]
    .head(30)
    .to_string(
        index=False
    )
)