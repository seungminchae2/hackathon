import pandas as pd
import numpy as np
from pathlib import Path


RISK_FILE = Path(
    "data/road_risk_all_dates.csv"
)

POTHOLE_FILE = Path(
    "data/potholes.csv"
)

REPAIRS_FILE = Path(
    "data/repairs.csv"
)

OUTPUT_FILE = Path(
    "data/pothole_training_data.csv"
)


# ============================================================
# 포트홀 라벨링 기준
#
# 공간 기준:
# 도로 포인트 반경 200m
#
# 시간 기준:
# 기준일 다음 날부터 향후 7일 이내
#
# 예:
#
# 기준일 = 2025-01-01
#
# 2025-01-02 ~ 2025-01-08 사이에
# 해당 도로 포인트 200m 이내에서 포트홀이 발생하면
#
# pothole_label = 1
#
# 아니면
#
# pothole_label = 0
#
# 이 기준은 프로젝트의 예측 단위를
# "향후 7일 이내 주변 포트홀 발생 여부"로 정의한 것이다.
# ============================================================

POTHOLE_MATCH_RADIUS_M = 200
PREDICTION_DAYS = 7


# ============================================================
# 보수이력 매칭 기준
#
# 도로 포인트 반경 300m 이내 보수기록을 사용한다.
#
# 단, 머신러닝 학습에서는 반드시
#
# repair_date <= 평가 날짜
#
# 를 만족하는 과거 보수만 사용한다.
#
# 이를 통해 미래 보수정보가 과거 데이터에 들어가는
# data leakage를 방지한다.
# ============================================================

REPAIR_MATCH_RADIUS_M = 300


# ============================================================
# Haversine 거리
#
# 위도 / 경도를 이용해 두 위치 사이의
# 대권거리를 meter 단위로 계산한다.
# ============================================================

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
        np.sin(
            dlat / 2
        ) ** 2
        +
        np.cos(lat1)
        * np.cos(lat2)
        * np.sin(
            dlon / 2
        ) ** 2
    )

    c = (
        2
        * np.arctan2(
            np.sqrt(a),
            np.sqrt(1 - a)
        )
    )

    return (
        earth_radius_m
        * c
    )


# ============================================================
# 보수 감쇠값
#
# 기존 규칙 기반 위험도에서 사용하던 기준.
#
# 최근 90일 이내
# → -10점
#
# 91~180일
# → -7점
#
# 181~365일
# → -4점
#
# 366~730일
# → -2점
#
# 730일 초과
# → 0점
#
# ML에서는 repair_adjustment보다는
#
# has_past_repair
# days_since_last_repair
#
# 원변수를 사용하는 것이 더 중요하다.
# ============================================================

def calculate_repair_adjustment(
    days_since_repair
):

    if pd.isna(
        days_since_repair
    ):
        return 0.0

    if days_since_repair < 0:
        return 0.0

    if days_since_repair <= 90:
        return -10.0

    if days_since_repair <= 180:
        return -7.0

    if days_since_repair <= 365:
        return -4.0

    if days_since_repair <= 730:
        return -2.0

    return 0.0


# ============================================================
# 1. 날짜별 도로 위험요인 데이터 불러오기
# ============================================================

print(
    "도로 위험도 데이터 불러오는 중..."
)


risk = pd.read_csv(
    RISK_FILE,
    encoding="utf-8-sig",
    low_memory=False
)


risk[
    "date"
] = pd.to_datetime(
    risk[
        "date"
    ],
    errors="coerce"
)


risk[
    "lat"
] = pd.to_numeric(
    risk[
        "lat"
    ],
    errors="coerce"
)


risk[
    "lon"
] = pd.to_numeric(
    risk[
        "lon"
    ],
    errors="coerce"
)


risk[
    "point_id"
] = pd.to_numeric(
    risk[
        "point_id"
    ],
    errors="coerce"
)


risk = risk.dropna(
    subset=[
        "point_id",
        "date",
        "lat",
        "lon",
    ]
).copy()


risk[
    "point_id"
] = (
    risk[
        "point_id"
    ]
    .astype(int)
)


print(
    "전체 포인트-날짜 행 수:",
    len(risk)
)


print(
    "도로 포인트 수:",
    risk[
        "point_id"
    ]
    .nunique()
)


print(
    "날짜 범위:",
    risk[
        "date"
    ]
    .min()
    .date(),
    "~",
    risk[
        "date"
    ]
    .max()
    .date()
)


# ============================================================
# 2. 고유 도로 포인트 추출
#
# 같은 point_id는 날짜가 달라도 위치가 동일하므로
# 680개 고유 위치만 추출해서
# 공간거리 계산에 사용한다.
# ============================================================

points = (
    risk[
        [
            "point_id",
            "lat",
            "lon",
        ]
    ]
    .drop_duplicates(
        subset=[
            "point_id"
        ]
    )
    .sort_values(
        "point_id"
    )
    .reset_index(
        drop=True
    )
)


point_lats = (
    points[
        "lat"
    ]
    .to_numpy()
)


point_lons = (
    points[
        "lon"
    ]
    .to_numpy()
)


print(
    "\n공간 매칭 대상 도로 포인트:",
    len(points)
)


# ============================================================
# 3. 보수이력 데이터 읽기
# ============================================================

repairs = pd.read_csv(
    REPAIRS_FILE,
    encoding="utf-8-sig"
)


required_repair_columns = [
    "repair_id",
    "repair_date",
    "lat",
    "lon",
]


missing_repairs = [
    col
    for col in required_repair_columns
    if col not in repairs.columns
]


if missing_repairs:

    raise ValueError(
        "repairs.csv에 필요한 컬럼이 없습니다: "
        f"{missing_repairs}"
    )


repairs[
    "repair_date"
] = pd.to_datetime(
    repairs[
        "repair_date"
    ],
    errors="coerce"
)


repairs[
    "lat"
] = pd.to_numeric(
    repairs[
        "lat"
    ],
    errors="coerce"
)


repairs[
    "lon"
] = pd.to_numeric(
    repairs[
        "lon"
    ],
    errors="coerce"
)


repairs = repairs.dropna(
    subset=[
        "repair_date",
        "lat",
        "lon",
    ]
).copy()


print(
    "\n전체 보수기록 수:",
    len(repairs)
)


print(
    "보수 날짜 범위:",
    repairs[
        "repair_date"
    ]
    .min()
    .date(),
    "~",
    repairs[
        "repair_date"
    ]
    .max()
    .date()
)


# ============================================================
# 4. 보수지점 ↔ 도로포인트 공간매칭
#
# 각 repair 위치에서
# 300m 이내 도로 포인트를 모두 찾는다.
#
# 이 단계에서는 날짜를 고려하지 않고
# 공간 관계만 생성한다.
# ============================================================

repair_matches = []


for _, repair in repairs.iterrows():

    distances = haversine_distance(
        repair[
            "lat"
        ],
        repair[
            "lon"
        ],
        point_lats,
        point_lons
    )


    nearby_positions = np.where(
        distances
        <= REPAIR_MATCH_RADIUS_M
    )[0]


    for position in nearby_positions:

        point = points.iloc[
            position
        ]


        repair_matches.append(
            {
                "point_id":
                    int(
                        point[
                            "point_id"
                        ]
                    ),

                "repair_id":
                    repair[
                        "repair_id"
                    ],

                "repair_date":
                    repair[
                        "repair_date"
                    ],

                "repair_distance_m":
                    float(
                        distances[
                            position
                        ]
                    ),
            }
        )


repair_matches = pd.DataFrame(
    repair_matches
)


if len(
    repair_matches
) == 0:

    print(
        "\n경고:"
        " 300m 이내에서 도로 포인트와"
        " 매칭된 보수기록이 없습니다."
    )


else:

    print(
        "\n300m 이내 보수-도로 공간매칭 건수:",
        len(
            repair_matches
        )
    )


    print(
        "보수기록이 연결된 도로 포인트 수:",
        repair_matches[
            "point_id"
        ]
        .nunique()
    )


    print(
        "\n보수-도로 거리 통계(m)"
    )


    print(
        repair_matches[
            "repair_distance_m"
        ]
        .describe()
    )


# ============================================================
# 5. 같은 포인트 + 같은 보수일 중복 제거
#
# 동일한 날짜에 같은 포인트 근처에
# 복수의 보수기록이 존재한다면
#
# 가장 가까운 보수기록 하나만 대표로 사용한다.
# ============================================================

if len(
    repair_matches
) > 0:

    repair_history = (
        repair_matches
        .sort_values(
            [
                "point_id",
                "repair_date",
                "repair_distance_m",
            ]
        )
        .drop_duplicates(
            subset=[
                "point_id",
                "repair_date",
            ],
            keep="first"
        )
        .copy()
    )

else:

    repair_history = pd.DataFrame(
        columns=[
            "point_id",
            "repair_id",
            "repair_date",
            "repair_distance_m",
        ]
    )


# ============================================================
# 6. 날짜별 최근 과거 보수 찾기
#
# 이전 코드에서는 전체 데이터에 merge_asof를 한 번
# 적용하면서 pandas의 정렬 조건 때문에
#
# ValueError:
# left keys must be sorted
#
# 오류가 발생했다.
#
# 이번 버전에서는 point_id 하나씩 분리해서
# merge_asof를 실행한다.
#
#
# 예:
#
# point 10
#
# 보수:
# 2024-03-01
# 2025-05-01
#
#
# 평가일:
# 2024-08-10
#
# → 2024-03-01 사용
#
#
# 평가일:
# 2025-07-01
#
# → 2025-05-01 사용
#
#
# 평가일보다 미래의 보수는 절대 사용하지 않는다.
# ============================================================

merged_groups = []


for point_id, point_risk in risk.groupby(
    "point_id",
    sort=False
):

    point_risk = (
        point_risk
        .sort_values(
            "date"
        )
        .copy()
    )


    point_repairs = (
        repair_history[
            repair_history[
                "point_id"
            ]
            == point_id
        ]
        .sort_values(
            "repair_date"
        )
        .copy()
    )


    # 해당 포인트 주변에 보수기록이 없는 경우
    if len(
        point_repairs
    ) == 0:

        point_risk[
            "repair_id"
        ] = np.nan

        point_risk[
            "repair_date"
        ] = pd.NaT

        point_risk[
            "repair_distance_m"
        ] = np.nan


        merged_groups.append(
            point_risk
        )

        continue


    point_repairs = point_repairs[
        [
            "repair_id",
            "repair_date",
            "repair_distance_m",
        ]
    ].copy()


    merged = pd.merge_asof(
        point_risk,
        point_repairs,
        left_on="date",
        right_on="repair_date",
        direction="backward",
        allow_exact_matches=True
    )


    merged_groups.append(
        merged
    )


risk = pd.concat(
    merged_groups,
    ignore_index=True
)


risk = risk.sort_values(
    [
        "point_id",
        "date",
    ]
).reset_index(
    drop=True
)


# ============================================================
# 7. 시점별 보수 피처 생성
#
# has_past_repair:
#
# 해당 날짜 기준 과거 또는 당일에
# 300m 이내 보수이력이 존재했는지 여부
#
#
# days_since_last_repair:
#
# 현재 평가날짜 - 가장 최근 과거 보수일
#
#
# 기존 has_nearby_repair는 사용하지 않는다.
# 미래 보수 존재 여부까지 반영될 수 있기 때문이다.
# ============================================================

risk[
    "has_past_repair"
] = (
    risk[
        "repair_date"
    ]
    .notna()
    .astype(int)
)


risk[
    "days_since_last_repair"
] = (
    risk[
        "date"
    ]
    -
    risk[
        "repair_date"
    ]
).dt.days


risk[
    "repair_adjustment"
] = (
    risk[
        "days_since_last_repair"
    ]
    .apply(
        calculate_repair_adjustment
    )
)


# ============================================================
# 8. 미래 보수정보 누출 검사
#
# repair_date가 date보다 큰 행은
# 단 하나도 존재하면 안 된다.
# ============================================================

future_repair_count = int(
    (
        risk[
            "repair_date"
        ]
        >
        risk[
            "date"
        ]
    )
    .sum()
)


print(
    "\n미래 보수정보 누출 검사"
)


print(
    "평가일보다 미래 보수기록:",
    future_repair_count
)


if future_repair_count > 0:

    raise ValueError(
        "미래 보수정보가 학습 데이터에 포함됐습니다."
    )


print(
    "과거 보수이력이 존재하는 학습 후보 행:",
    int(
        risk[
            "has_past_repair"
        ]
        .sum()
    )
)


# ============================================================
# 9. 포트홀 데이터 읽기
# ============================================================

potholes = pd.read_csv(
    POTHOLE_FILE,
    encoding="utf-8-sig"
)


required_pothole_columns = [
    "event_id",
    "event_date",
    "lat",
    "lon",
]


missing_potholes = [
    col
    for col in required_pothole_columns
    if col not in potholes.columns
]


if missing_potholes:

    raise ValueError(
        "potholes.csv에 필요한 컬럼이 없습니다: "
        f"{missing_potholes}"
    )


potholes[
    "event_date"
] = pd.to_datetime(
    potholes[
        "event_date"
    ],
    errors="coerce"
)


potholes[
    "lat"
] = pd.to_numeric(
    potholes[
        "lat"
    ],
    errors="coerce"
)


potholes[
    "lon"
] = pd.to_numeric(
    potholes[
        "lon"
    ],
    errors="coerce"
)


potholes = potholes.dropna(
    subset=[
        "event_date",
        "lat",
        "lon",
    ]
).copy()


print(
    "\n포트홀 발생 건수:",
    len(potholes)
)


print(
    "포트홀 발생 날짜 범위:",
    potholes[
        "event_date"
    ]
    .min()
    .date(),
    "~",
    potholes[
        "event_date"
    ]
    .max()
    .date()
)


# ============================================================
# 10. 포트홀 ↔ 도로 포인트 공간매칭
#
# 포트홀 위치 반경 200m 안에 있는
# 도로 포인트를 찾는다.
# ============================================================

pothole_matches = []


for _, pothole in potholes.iterrows():

    distances = haversine_distance(
        pothole[
            "lat"
        ],
        pothole[
            "lon"
        ],
        point_lats,
        point_lons
    )


    nearby_positions = np.where(
        distances
        <= POTHOLE_MATCH_RADIUS_M
    )[0]


    for position in nearby_positions:

        point = points.iloc[
            position
        ]


        pothole_matches.append(
            {
                "event_id":
                    pothole[
                        "event_id"
                    ],

                "event_date":
                    pothole[
                        "event_date"
                    ],

                "point_id":
                    int(
                        point[
                            "point_id"
                        ]
                    ),

                "pothole_distance_m":
                    float(
                        distances[
                            position
                        ]
                    ),
            }
        )


pothole_matches = pd.DataFrame(
    pothole_matches
)


if len(
    pothole_matches
) == 0:

    raise ValueError(
        "200m 이내에서 도로 포인트와 "
        "매칭된 포트홀 기록이 하나도 없습니다."
    )


print(
    "\n200m 이내 포트홀 공간 매칭 건수:",
    len(
        pothole_matches
    )
)


print(
    "도로 포인트와 연결된 포트홀 수:",
    pothole_matches[
        "event_id"
    ]
    .nunique()
)


print(
    "포트홀이 연결된 도로 포인트 수:",
    pothole_matches[
        "point_id"
    ]
    .nunique()
)


print(
    "\n포트홀-도로포인트 거리 통계(m)"
)


print(
    pothole_matches[
        "pothole_distance_m"
    ]
    .describe()
)


# ============================================================
# 11. 향후 7일 포트홀 라벨 생성
#
# 포트홀 발생일이 8월 10일이면:
#
# 8월 3일
# 8월 4일
# 8월 5일
# 8월 6일
# 8월 7일
# 8월 8일
# 8월 9일
#
# 이 7개 날짜의 동일 point_id를
# pothole_label = 1로 만든다.
#
# 포트홀 발생 당일은 미래 예측이 아니므로
# 양성 예측 기간에 포함하지 않는다.
# ============================================================

label_counts = {}


for _, match in pothole_matches.iterrows():

    point_id = int(
        match[
            "point_id"
        ]
    )


    event_date = match[
        "event_date"
    ]


    for days_before in range(
        1,
        PREDICTION_DAYS + 1
    ):

        prediction_date = (
            event_date
            -
            pd.Timedelta(
                days=days_before
            )
        )


        key = (
            point_id,
            prediction_date
        )


        label_counts[
            key
        ] = (
            label_counts.get(
                key,
                0
            )
            + 1
        )


risk_keys = list(
    zip(
        risk[
            "point_id"
        ],
        risk[
            "date"
        ]
    )
)


risk[
    "pothole_count_7d"
] = [
    label_counts.get(
        key,
        0
    )
    for key in risk_keys
]


risk[
    "pothole_label"
] = (
    risk[
        "pothole_count_7d"
    ]
    > 0
).astype(int)


# ============================================================
# 12. 라벨 관측기간 제한
#
# 포트홀 데이터 마지막 날짜가 2025-12-27인 경우,
#
# 2025-12-21 이후 날짜에서는
# 향후 7일 전체를 관찰할 수 없다.
#
# 따라서 마지막 포트홀 관측일 - 7일까지 사용한다.
# ============================================================

last_pothole_date = (
    potholes[
        "event_date"
    ]
    .max()
)


valid_label_end_date = (
    last_pothole_date
    -
    pd.Timedelta(
        days=PREDICTION_DAYS
    )
)


risk = risk[
    risk[
        "date"
    ]
    <= valid_label_end_date
].copy()


# ============================================================
# 13. 학습 피처 선택
#
# 규칙 기반 risk_score 자체는 ML 입력에서 제외한다.
#
# 우리가 직접 정한 가중치를 다시 학습하는 것이 아니라
# 원변수와 실제 포트홀 발생 사이의 관계를
# 모델이 직접 학습하도록 한다.
#
#
# 주요 입력:
#
# 동결융해 반복
# 누적강수
# 노후 하수관 비율
# ESAL
# 과거 보수 여부
# 마지막 보수 후 경과일
#
#
# 정답:
#
# pothole_label
# ============================================================

training_columns = [
    "point_id",
    "date",
    "road_name",
    "city",
    "lat",
    "lon",

    "freeze_thaw_14d",

    "rain_7d",
    "rain_14d",

    "sewer_old30_ratio",

    "traffic_esal",
    "traffic_match_type",
    "traffic_distance_m",

    "has_past_repair",
    "repair_id",
    "repair_date",
    "repair_distance_m",
    "days_since_last_repair",

    "pothole_count_7d",
    "pothole_label",
]


training_columns = [
    col
    for col in training_columns
    if col in risk.columns
]


training = risk[
    training_columns
].copy()


# ============================================================
# 14. 결과 저장
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


training.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 15. 결과 검증
# ============================================================

positive_count = int(
    training[
        "pothole_label"
    ]
    .sum()
)


negative_count = int(
    (
        training[
            "pothole_label"
        ]
        == 0
    )
    .sum()
)


total_count = len(
    training
)


positive_rate = (
    positive_count
    /
    total_count
    * 100

    if total_count > 0

    else 0
)


print(
    "\n포트홀 머신러닝 학습 데이터 생성 완료"
)


print(
    f"저장 위치: {OUTPUT_FILE}"
)


print(
    f"전체 학습 행 수: {total_count}"
)


print(
    f"양성(label=1): {positive_count}"
)


print(
    f"음성(label=0): {negative_count}"
)


print(
    f"양성 비율: {positive_rate:.4f}%"
)


print(
    "학습 데이터 마지막 날짜:",
    training[
        "date"
    ]
    .max()
    .date()
)


# ============================================================
# 과거 보수이력 상태
# ============================================================

print(
    "\n과거 보수이력 여부"
)


print(
    training[
        "has_past_repair"
    ]
    .value_counts()
)


past_repair_rows = training[
    training[
        "has_past_repair"
    ]
    == 1
]


print(
    "\n과거 보수이력이 있는 행의 경과일수 통계"
)


if len(
    past_repair_rows
) > 0:

    print(
        past_repair_rows[
            "days_since_last_repair"
        ]
        .describe()
    )

else:

    print(
        "과거 보수이력이 있는 행 없음"
    )


# ============================================================
# 미래 보수정보 누출 최종 검사
# ============================================================

future_leakage = int(
    (
        training[
            "repair_date"
        ]
        >
        training[
            "date"
        ]
    )
    .sum()
)


print(
    "\n미래 보수정보 포함 여부"
)


print(
    "미래 보수 행:",
    future_leakage
)


if future_leakage > 0:

    raise ValueError(
        "학습 데이터에 미래 보수정보가 포함되어 있습니다."
    )


# ============================================================
# 양성 포인트 확인
# ============================================================

print(
    "\n포인트별 양성 발생 횟수 상위 20개"
)


print(
    training[
        training[
            "pothole_label"
        ]
        == 1
    ]
    .groupby(
        "point_id"
    )
    .size()
    .sort_values(
        ascending=False
    )
    .head(20)
)


# ============================================================
# 양성 데이터 예시
# ============================================================

print(
    "\n양성 학습 데이터 예시"
)


print(
    training[
        training[
            "pothole_label"
        ]
        == 1
    ]
    .head(20)
    .to_string(
        index=False
    )
)