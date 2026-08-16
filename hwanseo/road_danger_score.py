import pandas as pd
import numpy as np
from pathlib import Path


ROADS_FILE = Path(
    "data/roads_with_sewer_repair_traffic.csv"
)

WEATHER_FILE = Path(
    "data/weather_history.csv"
)

OUTPUT_ALL = Path(
    "data/road_risk_all_dates.csv"
)

OUTPUT_LATEST = Path(
    "data/road_risk_latest.csv"
)


WEIGHT_FREEZE = 0.35
WEIGHT_RAIN = 0.25
WEIGHT_SEWER = 0.20
WEIGHT_TRAFFIC = 0.20


CITY_MAP = {
    "전주": "전주시",
    "군산": "군산시",
    "익산": "익산시",
    "남원": "남원시",
    "정읍": "정읍시",
}


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


def find_column(
    df,
    candidates
):

    for col in candidates:

        if col in df.columns:
            return col

    return None


def calculate_freeze_thaw(
    weather
):

    min_temp_col = find_column(
        weather,
        [
            "min_temp",
            "최저기온",
            "최저기온(°C)",
            "최저기온(℃)",
        ]
    )

    max_temp_col = find_column(
        weather,
        [
            "max_temp",
            "최고기온",
            "최고기온(°C)",
            "최고기온(℃)",
        ]
    )

    if min_temp_col is None:

        raise ValueError(
            "weather_history.csv에서 "
            "최저기온 컬럼을 찾을 수 없습니다."
        )

    if max_temp_col is None:

        raise ValueError(
            "weather_history.csv에서 "
            "최고기온 컬럼을 찾을 수 없습니다."
        )

    weather[min_temp_col] = pd.to_numeric(
        weather[min_temp_col],
        errors="coerce"
    )

    weather[max_temp_col] = pd.to_numeric(
        weather[max_temp_col],
        errors="coerce"
    )

    weather["freeze_thaw_day"] = (
        (
            weather[min_temp_col]
            <= 0
        )
        &
        (
            weather[max_temp_col]
            > 0
        )
    ).astype(int)

    weather["freeze_thaw_14d"] = (
        weather
        .groupby(
            "city"
        )[
            "freeze_thaw_day"
        ]
        .transform(
            lambda x:
                x.rolling(
                    14,
                    min_periods=1
                ).sum()
        )
    )

    return weather


def freeze_score(
    count
):

    if pd.isna(count):
        return 0.0

    if count == 0:
        return 0.0

    if count <= 2:
        return 20.0

    if count <= 4:
        return 40.0

    if count <= 6:
        return 60.0

    if count <= 9:
        return 80.0

    return 100.0


def calculate_rain(
    weather
):

    rain_col = find_column(
        weather,
        [
            "rain",
            "precipitation",
            "강수량",
            "일강수량",
            "일강수량(mm)",
        ]
    )

    if rain_col is None:

        raise ValueError(
            "weather_history.csv에서 "
            "강수량 컬럼을 찾을 수 없습니다."
        )

    weather[rain_col] = pd.to_numeric(
        weather[rain_col],
        errors="coerce"
    ).fillna(0)

    weather["rain_7d"] = (
        weather
        .groupby(
            "city"
        )[rain_col]
        .transform(
            lambda x:
                x.rolling(
                    7,
                    min_periods=1
                ).sum()
        )
    )

    weather["rain_14d"] = (
        weather
        .groupby(
            "city"
        )[rain_col]
        .transform(
            lambda x:
                x.rolling(
                    14,
                    min_periods=1
                ).sum()
        )
    )

    return weather


def calculate_rain_score(
    rain_7d,
    rain_14d
):

    score_7d = (
        rain_7d
        / 50
        * 100
    )

    score_14d = (
        rain_14d
        / 100
        * 100
    )

    score_7d = np.clip(
        score_7d,
        0,
        100
    )

    score_14d = np.clip(
        score_14d,
        0,
        100
    )

    return (
        score_7d * 0.6
        +
        score_14d * 0.4
    )


def calculate_repair_adjustment(
    days_since_repair,
    has_nearby_repair
):

    if (
        pd.isna(has_nearby_repair)
        or has_nearby_repair == 0
        or pd.isna(days_since_repair)
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


def get_risk_level(
    score
):

    if score >= 70:
        return "매우 위험"

    if score >= 55:
        return "위험"

    if score >= 40:
        return "주의"

    if score >= 25:
        return "관심"

    return "낮음"


def get_main_risk_factor(
    row
):

    contributions = {
        "동결융해":
            row["freeze_score"]
            * WEIGHT_FREEZE,

        "누적강수":
            row["rain_score"]
            * WEIGHT_RAIN,

        "하수관로 노후도":
            row["sewer_score"]
            * WEIGHT_SEWER,

        "교통하중":
            row["traffic_score"]
            * WEIGHT_TRAFFIC,
    }

    return max(
        contributions,
        key=contributions.get
    )


def traffic_match_description(
    row
):

    match_type = row.get(
        "traffic_match_type"
    )

    if (
        match_type
        == "nearest_station"
    ):

        distance = row.get(
            "traffic_distance_m"
        )

        if pd.notna(distance):

            return (
                f"{distance / 1000:.1f}km 이내 "
                f"실제 조사점 적용"
            )

        return "실제 조사점 적용"

    if (
        match_type
        == "city_median"
    ):

        return "도시 중앙값 적용"

    return "교통량 매칭정보 없음"


def make_risk_reason(
    row
):

    traffic_description = (
        traffic_match_description(
            row
        )
    )

    reason = (
        f"주요원인={row['main_risk_factor']}"
        f" | 동결융해={row['freeze_score']:.1f}점"
        f" (14일 {row['freeze_thaw_14d']:.0f}회)"
        f" | 누적강수={row['rain_score']:.1f}점"
        f" (7일 {row['rain_7d']:.1f}mm,"
        f" 14일 {row['rain_14d']:.1f}mm)"
        f" | 하수관로 노후도={row['sewer_score']:.1f}점"
        f" (30년 이상 비율 "
        f"{row['sewer_old30_ratio'] * 100:.1f}%)"
        f" | 교통하중={row['traffic_score']:.1f}점"
        f" (ESAL {row['traffic_esal']:.1f}, "
        f"{traffic_description})"
    )

    if (
        row.get(
            "has_nearby_repair",
            0
        )
        == 1
        and pd.notna(
            row.get(
                "days_since_last_repair"
            )
        )
    ):

        distance = row.get(
            "distance_to_last_repair_m"
        )

        if pd.notna(distance):

            reason += (
                f" | 최근보수="
                f"{row['days_since_last_repair']:.0f}일 전"
                f" ({distance:.0f}m,"
                f" 보정 {row['repair_adjustment']:.0f}점)"
            )

        else:

            reason += (
                f" | 최근보수="
                f"{row['days_since_last_repair']:.0f}일 전"
                f" (보정 "
                f"{row['repair_adjustment']:.0f}점)"
            )

    else:

        reason += (
            " | 300m 이내 확인된 보수이력 없음"
        )

    return reason


# ============================================================
# 도로 포인트 데이터
#
# 이 파일에는 이미:
#
# 하수관로 노후도
# 300m 보수이력
# 포인트별 ESAL
# traffic_score
# traffic_match_type
#
# 가 결합되어 있다.
# ============================================================

roads = read_csv_auto_encoding(
    ROADS_FILE
)


required_road_columns = [
    "point_id",
    "road_name",
    "city",
    "lat",
    "lon",
    "sewer_old30_ratio",
    "traffic_esal",
    "traffic_score",
]


missing = [
    col
    for col in required_road_columns
    if col not in roads.columns
]


if missing:

    raise ValueError(
        "도로 데이터에 필요한 컬럼이 없습니다: "
        f"{missing}"
    )


print(
    "도로 포인트 수:",
    len(roads)
)


print(
    "하수관로 매칭 성공:",
    roads[
        "sewer_old30_ratio"
    ]
    .notna()
    .sum()
)


print(
    "교통량 ESAL 매칭 성공:",
    roads[
        "traffic_esal"
    ]
    .notna()
    .sum()
)


# ============================================================
# 하수관로 점수
#
# sewer_old30_ratio는 0~1 비율이다.
#
# 예:
# 0.762 → 76.2점
#
# 별도 Min-Max 없이 비율 자체를 점수화한다.
# ============================================================

roads[
    "sewer_old30_ratio"
] = pd.to_numeric(
    roads[
        "sewer_old30_ratio"
    ],
    errors="coerce"
)


roads[
    "sewer_score"
] = (
    roads[
        "sewer_old30_ratio"
    ]
    * 100
).clip(
    0,
    100
)


# ============================================================
# 교통량
#
# merge_traffic_esal.py에서 계산된
# 포인트별 ESAL과 traffic_score를 그대로 사용한다.
#
# 더 이상 road_danger_score.py 내부에서
# 교통량 CSV를 다시 읽거나
# 도시별 중앙값을 다시 계산하지 않는다.
# ============================================================

roads[
    "traffic_esal"
] = pd.to_numeric(
    roads[
        "traffic_esal"
    ],
    errors="coerce"
)


roads[
    "traffic_score"
] = pd.to_numeric(
    roads[
        "traffic_score"
    ],
    errors="coerce"
)


# ============================================================
# 보수이력
# ============================================================

if (
    "has_nearby_repair"
    not in roads.columns
):

    roads[
        "has_nearby_repair"
    ] = 0


roads[
    "has_nearby_repair"
] = pd.to_numeric(
    roads[
        "has_nearby_repair"
    ],
    errors="coerce"
).fillna(0)


if (
    "last_repair_date"
    in roads.columns
):

    roads[
        "last_repair_date"
    ] = pd.to_datetime(
        roads[
            "last_repair_date"
        ],
        errors="coerce"
    )


# ============================================================
# 날씨 데이터
# ============================================================

weather = read_csv_auto_encoding(
    WEATHER_FILE
)


date_col = find_column(
    weather,
    [
        "date",
        "날짜",
        "일시",
    ]
)


if date_col is None:

    raise ValueError(
        "weather_history.csv에서 "
        "날짜 컬럼을 찾을 수 없습니다."
    )


if date_col != "date":

    weather = weather.rename(
        columns={
            date_col: "date"
        }
    )


weather["date"] = pd.to_datetime(
    weather["date"],
    errors="coerce"
)


# ============================================================
# 실제 weather_history.csv에는 city가 없고
# station_name이 존재한다.
#
# 예:
#
# 전주 → 전주시
# 군산 → 군산시
# 익산 → 익산시
# 남원 → 남원시
# 정읍 → 정읍시
#
# 로 변환한다.
# ============================================================

if "city" not in weather.columns:

    if "station_name" not in weather.columns:

        raise ValueError(
            "weather_history.csv에 "
            "city 또는 station_name 컬럼이 필요합니다."
        )

    weather["city"] = (
        weather[
            "station_name"
        ]
        .map(
            CITY_MAP
        )
    )


weather = weather[
    weather[
        "city"
    ]
    .notna()
].copy()


weather = weather.dropna(
    subset=[
        "date",
        "city",
    ]
)


weather = weather.sort_values(
    [
        "city",
        "date",
    ]
).reset_index(
    drop=True
)


print(
    "날씨 데이터 도시:",
    sorted(
        weather[
            "city"
        ]
        .dropna()
        .unique()
        .tolist()
    )
)


# ============================================================
# 동결융해 계산
# ============================================================

weather = calculate_freeze_thaw(
    weather
)


weather[
    "freeze_score"
] = (
    weather[
        "freeze_thaw_14d"
    ]
    .apply(
        freeze_score
    )
)


# ============================================================
# 누적강수 계산
# ============================================================

weather = calculate_rain(
    weather
)


weather[
    "rain_score"
] = calculate_rain_score(
    weather[
        "rain_7d"
    ],
    weather[
        "rain_14d"
    ]
)


weather_features = weather[
    [
        "city",
        "date",
        "freeze_thaw_14d",
        "freeze_score",
        "rain_7d",
        "rain_14d",
        "rain_score",
    ]
].copy()


# ============================================================
# 도로 포인트 × 날짜 날씨 결합
# ============================================================

result = roads.merge(
    weather_features,
    on="city",
    how="inner"
)


# ============================================================
# 보수 후 경과일 계산
#
# 보수이력이 없는 경우
# 임의로 3650일을 넣지 않는다.
# ============================================================

if (
    "last_repair_date"
    in result.columns
):

    result[
        "days_since_last_repair"
    ] = (
        result[
            "date"
        ]
        -
        result[
            "last_repair_date"
        ]
    ).dt.days

else:

    result[
        "days_since_last_repair"
    ] = np.nan


# 평가 날짜보다 미래에 발생한 보수기록은
# 해당 날짜 위험도에 영향을 주지 않는다.
result.loc[
    result[
        "days_since_last_repair"
    ]
    < 0,
    "days_since_last_repair"
] = np.nan


result[
    "repair_adjustment"
] = result.apply(
    lambda row:
        calculate_repair_adjustment(
            row[
                "days_since_last_repair"
            ],
            row[
                "has_nearby_repair"
            ]
        ),
    axis=1
)


# ============================================================
# 기본 위험도
#
# 동결융해 35%
# 누적강수 25%
# 하수관로 노후도 20%
# ESAL 교통하중 20%
#
# 합계 100%
# ============================================================

result[
    "freeze_contribution"
] = (
    result[
        "freeze_score"
    ]
    * WEIGHT_FREEZE
)


result[
    "rain_contribution"
] = (
    result[
        "rain_score"
    ]
    * WEIGHT_RAIN
)


result[
    "sewer_contribution"
] = (
    result[
        "sewer_score"
    ]
    * WEIGHT_SEWER
)


result[
    "traffic_contribution"
] = (
    result[
        "traffic_score"
    ]
    * WEIGHT_TRAFFIC
)


result[
    "base_risk_score"
] = (
    result[
        "freeze_contribution"
    ]
    +
    result[
        "rain_contribution"
    ]
    +
    result[
        "sewer_contribution"
    ]
    +
    result[
        "traffic_contribution"
    ]
)


# ============================================================
# 최종 위험도
#
# 기본 위험도 + 최근 보수 감쇠
# ============================================================

result[
    "risk_score"
] = (
    result[
        "base_risk_score"
    ]
    +
    result[
        "repair_adjustment"
    ]
).clip(
    0,
    100
)


# ============================================================
# 위험등급
# ============================================================

result[
    "risk_level"
] = (
    result[
        "risk_score"
    ]
    .apply(
        get_risk_level
    )
)


# ============================================================
# 주요 위험원인
# ============================================================

result[
    "main_risk_factor"
] = result.apply(
    get_main_risk_factor,
    axis=1
)


# ============================================================
# 위험 이유
# ============================================================

result[
    "risk_reason"
] = result.apply(
    make_risk_reason,
    axis=1
)


# ============================================================
# 전체 결과 저장
# ============================================================

OUTPUT_ALL.parent.mkdir(
    parents=True,
    exist_ok=True
)


result.to_csv(
    OUTPUT_ALL,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 최신 날짜 결과
# ============================================================

latest_date = (
    result[
        "date"
    ]
    .max()
)


latest = result[
    result[
        "date"
    ]
    == latest_date
].copy()


latest = latest.sort_values(
    "risk_score",
    ascending=False
)


latest.to_csv(
    OUTPUT_LATEST,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 결과 검증
# ============================================================

print(
    "\n도로 위험도 계산 완료"
)


print(
    "도로 포인트 수:",
    roads[
        "point_id"
    ]
    .nunique()
)


print(
    "전체 날짜 결과:",
    OUTPUT_ALL
)


print(
    "최신 위험도 결과:",
    OUTPUT_LATEST
)


print(
    "최신 날씨 날짜:",
    latest_date.date()
)


print(
    "\n하수관로 결측치:",
    int(
        roads[
            "sewer_old30_ratio"
        ]
        .isna()
        .sum()
    )
)


print(
    "교통량 ESAL 결측치:",
    int(
        roads[
            "traffic_esal"
        ]
        .isna()
        .sum()
    )
)


# ============================================================
# 교통량 매칭방식 확인
# ============================================================

if (
    "traffic_match_type"
    in roads.columns
):

    print(
        "\n교통량 매칭 방식"
    )

    print(
        roads[
            "traffic_match_type"
        ]
        .value_counts(
            dropna=False
        )
    )


# ============================================================
# 시별 하수관로
# ============================================================

print(
    "\n시별 하수관로 노후도"
)


print(
    roads
    .groupby(
        "city"
    )[
        [
            "sewer_old30_ratio",
            "sewer_score",
        ]
    ]
    .mean()
)


# ============================================================
# 시별 교통량 다양성
# ============================================================

print(
    "\n시별 교통량 점수 종류"
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


# ============================================================
# 보수 감쇠값
# ============================================================

print(
    "\n최신 날짜 보수 감쇠값 분포"
)


print(
    latest[
        "repair_adjustment"
    ]
    .value_counts()
    .sort_index()
)


# ============================================================
# 위험도 상위 20개 포인트
# ============================================================

print(
    "\n위험도 상위 20개 포인트"
)


display_columns = [
    "point_id",
    "road_name",
    "city",
    "lat",
    "lon",
    "date",

    "freeze_thaw_14d",
    "freeze_score",

    "rain_7d",
    "rain_14d",
    "rain_score",

    "sewer_old30_ratio",
    "sewer_score",

    "traffic_station_id",
    "traffic_distance_m",
    "traffic_match_type",
    "traffic_esal",
    "traffic_score",

    "last_repair_date",
    "days_since_last_repair",
    "distance_to_last_repair_m",
    "repair_adjustment",

    "freeze_contribution",
    "rain_contribution",
    "sewer_contribution",
    "traffic_contribution",

    "base_risk_score",
    "risk_score",
    "risk_level",
    "main_risk_factor",
    "risk_reason",
]


display_columns = [
    col
    for col in display_columns
    if col in latest.columns
]


print(
    latest[
        display_columns
    ]
    .head(20)
    .to_string(
        index=False
    )
)


# ============================================================
# 시별 risk_score 다양성
# ============================================================

print(
    "\n시별 서로 다른 risk_score 개수"
)


print(
    latest
    .groupby(
        "city"
    )[
        "risk_score"
    ]
    .nunique()
)


print(
    "\n전체 서로 다른 risk_score 개수:",
    latest[
        "risk_score"
    ]
    .nunique()
)


print(
    "\n가장 많이 중복된 risk_score"
)


print(
    latest[
        "risk_score"
    ]
    .value_counts()
    .head(15)
)


# ============================================================
# 교통량 매칭방식별 결과
# ============================================================

if (
    "traffic_match_type"
    in latest.columns
):

    print(
        "\n교통량 매칭방식별 포인트 수 및 평균 위험도"
    )

    print(
        latest
        .groupby(
            "traffic_match_type"
        )
        .agg(
            point_count=(
                "point_id",
                "count"
            ),
            mean_traffic_score=(
                "traffic_score",
                "mean"
            ),
            mean_risk_score=(
                "risk_score",
                "mean"
            ),
        )
    )