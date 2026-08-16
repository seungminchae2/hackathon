import pandas as pd
import numpy as np
from pathlib import Path


ROADS_FILE = Path("data/roads_with_sewer_age.csv")
TRAFFIC_FILE = Path("data/전북국도교통량(상시조사교통량).csv")
WEATHER_FILE = Path("data/weather_history.csv")

OUTPUT_ALL = Path("data/road_risk_all_dates.csv")
OUTPUT_LATEST = Path("data/road_risk_latest.csv")


TARGET_CITIES = [
    "전주시",
    "군산시",
    "익산시",
    "남원시",
    "정읍시",
]


# ============================================================
# ESAL 계수
#
# ESAL은 서로 다른 차량 종류가 도로 포장에 미치는
# 반복하중 영향을 표준 축하중으로 환산하기 위한 지표이다.
#
# 단순 교통량보다 중형·대형 화물차가 도로에 주는
# 상대적으로 큰 하중을 반영하기 위해 사용한다.
#
# 아래 계수는 기존 프로젝트에서 사용한 값을 그대로 유지한다.
# ============================================================

ESAL_FACTORS = {
    "승용차": 0.0002,
    "버스": 0.852,
    "소형화물": 0.004,
    "중형화물": 1.735,
    "대형화물": 3.169,
}


# ============================================================
# 공통 함수
# ============================================================

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


def minmax_score(series):

    series = pd.to_numeric(
        series,
        errors="coerce"
    ).fillna(0)

    minimum = series.min()
    maximum = series.max()

    if maximum == minimum:

        return pd.Series(
            0.0,
            index=series.index
        )

    return (
        (series - minimum)
        / (maximum - minimum)
        * 100
    ).clip(
        0,
        100
    )


def piecewise_score(
    value,
    thresholds,
    scores
):

    if pd.isna(value):

        return 0.0

    return float(
        np.interp(
            value,
            thresholds,
            scores
        )
    )


# ============================================================
# 1. 도로 포인트 + 하수관로 데이터
#
# merge_sewer_age.py 결과물을 직접 사용한다.
#
# 이 파일에는 680개 도로 포인트가 있고:
#
# road_name
# city
# lat
# lon
# sewer_old30_ratio
# average_sewer_age
#
# 등의 컬럼이 들어있다.
#
# 따라서 기존처럼 하수관로 엑셀을 road_danger_score.py에서
# 다시 읽어서 merge하지 않는다.
# ============================================================

roads = read_csv_auto_encoding(
    ROADS_FILE
)


required_road_columns = [
    "road_name",
    "city",
    "lat",
    "lon",
    "sewer_old30_ratio",
]


missing = [
    col
    for col in required_road_columns
    if col not in roads.columns
]


if missing:

    raise ValueError(
        f"roads_with_sewer_age.csv에 필요한 컬럼이 없습니다: {missing}\n"
        f"현재 컬럼: {roads.columns.tolist()}"
    )


roads["lat"] = pd.to_numeric(
    roads["lat"],
    errors="coerce"
)


roads["lon"] = pd.to_numeric(
    roads["lon"],
    errors="coerce"
)


roads["sewer_old30_ratio"] = pd.to_numeric(
    roads["sewer_old30_ratio"],
    errors="coerce"
)


# ============================================================
# 하수관로 노후도 점수
#
# sewer_old30_ratio는 0~1 비율이다.
#
# 예:
#
# 0.762
# → 30년 이상 노후관 비율 약 76.2%
# → sewer_score = 76.2점
#
# 노후관 비율 자체가 이미 0~100%라는 의미를 가진 값이므로
# 다른 도시와 Min-Max 정규화하지 않고 그대로 사용한다.
#
# 즉:
#
# 0%   → 0점
# 50%  → 50점
# 100% → 100점
# ============================================================

roads["sewer_score"] = (
    roads["sewer_old30_ratio"]
    * 100
).clip(
    0,
    100
)


# 각 포인트 고유 ID
roads = roads.reset_index(
    drop=True
)


roads["point_id"] = (
    np.arange(
        len(roads)
    )
)


print(
    "도로 포인트 수:",
    len(roads)
)


print(
    "하수관로 매칭 성공:",
    roads["sewer_old30_ratio"]
    .notna()
    .sum()
)


# ============================================================
# 2. 교통량 CSV 읽기
#
# 실제 교통량 파일은 2단 헤더 구조이다.
#
# 첫 두 줄을 제외하고 실제 데이터를 사용한다.
# ============================================================

def read_traffic_file(path):

    raw = read_csv_auto_encoding(
        path,
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
        .replace("", np.nan)
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

    return traffic


traffic = read_traffic_file(
    TRAFFIC_FILE
)


# ============================================================
# 교통량 구간 → 도시 매핑
#
# 구간명에 도시 이름이 직접 들어가는 경우 우선 사용한다.
#
# 추가로 이전 확인에서 행정구역상 5개시에 포함되는 것으로
# 확인된 구간을 수동 매핑한다.
# ============================================================

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

    return MANUAL_CITY_MAP.get(
        text
    )


traffic["city"] = (
    traffic["구간명"]
    .apply(
        get_traffic_city
    )
)


traffic = traffic[
    traffic["city"].notna()
].copy()


# ============================================================
# ESAL 계산
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
    / traffic[
        "total_traffic"
    ].replace(
        0,
        np.nan
    )
)


# ============================================================
# 현재 교통량 데이터에는 각 조사구간의 실제 선형 좌표가 없다.
#
# 따라서 680개 포인트와 정확한 최근접 공간매칭은
# 현재 파일만으로 수행할 수 없다.
#
# 임시로 각 도시의 교통량 구간 ESAL 중앙값을 사용한다.
#
# 평균보다 중앙값을 사용하는 이유:
# 한 개의 초고교통량 도로가 도시 전체 포인트의
# 교통량 위험도를 과도하게 높이는 것을 줄이기 위함이다.
#
# 이 값은 향후 교통량 조사구간의 좌표/geometry 확보 시
# 포인트별 nearest spatial join으로 교체하는 것이 바람직하다.
# ============================================================

city_traffic = (
    traffic
    .groupby(
        "city",
        as_index=False
    )
    .agg(
        traffic_esal=(
            "traffic_esal",
            "median"
        ),
        total_traffic=(
            "total_traffic",
            "median"
        ),
        heavy_vehicle_ratio=(
            "heavy_vehicle_ratio",
            "median"
        )
    )
)


city_traffic["traffic_score"] = (
    minmax_score(
        np.log1p(
            city_traffic[
                "traffic_esal"
            ]
        )
    )
)


roads = roads.merge(
    city_traffic,
    on="city",
    how="left"
)


# ============================================================
# 3. 날씨 데이터
# ============================================================

weather = read_csv_auto_encoding(
    WEATHER_FILE
)


weather["date"] = pd.to_datetime(
    weather["date"]
)


for col in [
    "min_temp",
    "max_temp",
    "precipitation",
]:

    weather[col] = pd.to_numeric(
        weather[col],
        errors="coerce"
    ).fillna(0)


weather = weather.sort_values(
    [
        "station_name",
        "date",
    ]
).copy()


# ============================================================
# 4. 동결융해
#
# 하루 최저기온 <= 0℃
# AND
# 하루 최고기온 > 0℃
#
# 이면 하루 동안 0℃ 경계를 통과한 것으로 보고
# 동결융해 가능일로 판정한다.
#
# 실제 포장 내부 온도가 아닌 대기온도를 사용하는
# proxy 지표라는 한계가 있다.
# ============================================================

weather["freeze_thaw"] = (
    (
        weather["min_temp"]
        <= 0
    )
    &
    (
        weather["max_temp"]
        > 0
    )
).astype(int)


weather[
    "freeze_thaw_14d"
] = (
    weather
    .groupby(
        "station_name"
    )[
        "freeze_thaw"
    ]
    .transform(
        lambda x:
        x.rolling(
            14,
            min_periods=1
        ).sum()
    )
)


# ============================================================
# 최근 14일 동결융해 점수
#
# 0회       → 0점
# 1~2회     → 20점
# 3~4회     → 40점
# 5~6회     → 60점
# 7~9회     → 80점
# 10회 이상 → 100점
#
# 동결융해 반복이 포장 열화에 영향을 준다는 문헌적 근거를
# 바탕으로 반복 횟수가 많을수록 위험도를 높인다.
#
# 정확한 20점 간격은 정부 공식 위험등급이 아니라
# 프로젝트의 engineering heuristic이다.
# ============================================================

def calculate_freeze_score(count):

    if count <= 0:
        return 0

    if count <= 2:
        return 20

    if count <= 4:
        return 40

    if count <= 6:
        return 60

    if count <= 9:
        return 80

    return 100


weather[
    "freeze_score"
] = (
    weather[
        "freeze_thaw_14d"
    ]
    .apply(
        calculate_freeze_score
    )
)


# ============================================================
# 5. 누적강수
# ============================================================

weather["rain_7d"] = (
    weather
    .groupby(
        "station_name"
    )[
        "precipitation"
    ]
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
        "station_name"
    )[
        "precipitation"
    ]
    .transform(
        lambda x:
        x.rolling(
            14,
            min_periods=1
        ).sum()
    )
)


# ============================================================
# 강수 위험 임계값
#
# 전북 5개시 2024~2025 자료의 6~9월 누적강수 분위수 기반.
#
# 7일:
# P50 = 16.98
# P75 = 30.05
# P90 = 41.94
# P95 = 50.95
# P99 = 64.81
#
# 14일:
# P50 = 38.74
# P75 = 55.03
# P90 = 73.21
# P95 = 82.47
# P99 = 98.25
#
# P50/P75/P90/P95/P99를
# 각각 20/40/60/80/100점으로 대응한다.
# ============================================================

RAIN7_THRESHOLDS = [
    0,
    16.98,
    30.05,
    41.94,
    50.95,
    64.81,
]


RAIN14_THRESHOLDS = [
    0,
    38.74,
    55.03,
    73.21,
    82.47,
    98.25,
]


RAIN_SCORES = [
    0,
    20,
    40,
    60,
    80,
    100,
]


weather[
    "rain_7d_score"
] = (
    weather["rain_7d"]
    .apply(
        lambda x:
        piecewise_score(
            x,
            RAIN7_THRESHOLDS,
            RAIN_SCORES
        )
    )
)


weather[
    "rain_14d_score"
] = (
    weather["rain_14d"]
    .apply(
        lambda x:
        piecewise_score(
            x,
            RAIN14_THRESHOLDS,
            RAIN_SCORES
        )
    )
)


# 7일 또는 14일 중 높은 위험을 사용한다.
weather["rain_score"] = (
    weather[
        [
            "rain_7d_score",
            "rain_14d_score",
        ]
    ]
    .max(
        axis=1
    )
)


# ============================================================
# 기상관측소 → 시 연결
# ============================================================

CITY_MAP = {
    "전주": "전주시",
    "군산": "군산시",
    "익산": "익산시",
    "남원": "남원시",
    "정읍": "정읍시",
}


weather["city"] = (
    weather[
        "station_name"
    ]
    .map(
        CITY_MAP
    )
)


weather = weather[
    weather["city"].notna()
].copy()


# ============================================================
# 6. 680개 도로 포인트 + 날씨
# ============================================================

risk = roads.merge(
    weather[
        [
            "date",
            "city",
            "min_temp",
            "max_temp",
            "precipitation",
            "freeze_thaw",
            "freeze_thaw_14d",
            "freeze_score",
            "rain_7d",
            "rain_14d",
            "rain_7d_score",
            "rain_14d_score",
            "rain_score",
        ]
    ],
    on="city",
    how="left"
)


# ============================================================
# 7. 계절별 가중치
#
# 겨울/해빙기 11~4월:
#
# 동결융해 35%
# 강수     15%
# 하수관   25%
# 교통     25%
#
#
# 장마/여름철 6~9월:
#
# 동결융해 5%
# 강수     45%
# 하수관   25%
# 교통     25%
#
#
# 5월/10월:
#
# 동결융해 15%
# 강수     35%
# 하수관   25%
# 교통     25%
#
#
# 환경요인의 계절적 메커니즘을 반영한 규칙 기반 가중치이며
# 공식 정부 포트홀 가중치는 아니다.
# ============================================================

def get_weights(month):

    if month in [
        11,
        12,
        1,
        2,
        3,
        4,
    ]:

        return (
            0.35,
            0.15,
            0.25,
            0.25,
        )

    if month in [
        6,
        7,
        8,
        9,
    ]:

        return (
            0.05,
            0.45,
            0.25,
            0.25,
        )

    return (
        0.15,
        0.35,
        0.25,
        0.25,
    )


risk["month"] = (
    risk["date"]
    .dt.month
)


weights = (
    risk["month"]
    .apply(
        get_weights
    )
)


risk[
    [
        "freeze_weight",
        "rain_weight",
        "sewer_weight",
        "traffic_weight",
    ]
] = pd.DataFrame(
    weights.tolist(),
    index=risk.index
)


# ============================================================
# 8. 각 요인의 기여점수
# ============================================================

risk[
    "freeze_contribution"
] = (
    risk["freeze_score"]
    * risk["freeze_weight"]
)


risk[
    "rain_contribution"
] = (
    risk["rain_score"]
    * risk["rain_weight"]
)


risk[
    "sewer_contribution"
] = (
    risk["sewer_score"]
    * risk["sewer_weight"]
)


risk[
    "traffic_contribution"
] = (
    risk["traffic_score"]
    * risk["traffic_weight"]
)


risk[
    "base_risk_score"
] = (
    risk["freeze_contribution"]
    + risk["rain_contribution"]
    + risk["sewer_contribution"]
    + risk["traffic_contribution"]
)


# ============================================================
# 동결융해 × 강수 상호작용
# ============================================================

risk["interaction_bonus"] = np.where(
    (
        risk["freeze_score"] >= 60
    )
    &
    (
        risk["rain_score"] >= 60
    ),
    10,
    0
)


risk["risk_score"] = (
    risk["base_risk_score"]
    + risk["interaction_bonus"]
).clip(
    0,
    100
)


# ============================================================
# 위험등급
# ============================================================

risk["risk_level"] = pd.cut(
    risk["risk_score"],
    bins=[
        -1,
        30,
        50,
        70,
        100,
    ],
    labels=[
        "안전",
        "주의",
        "위험",
        "매우 위험",
    ]
)


# ============================================================
# 9. 주요 위험요인
# ============================================================

contribution_columns = {
    "freeze_contribution": "동결융해",
    "rain_contribution": "누적강수",
    "sewer_contribution": "하수관로 노후도",
    "traffic_contribution": "교통하중",
}


risk[
    "main_risk_factor"
] = (
    risk[
        list(
            contribution_columns.keys()
        )
    ]
    .idxmax(
        axis=1
    )
    .map(
        contribution_columns
    )
)


# ============================================================
# 10. risk_reason
#
# 기존 결과에서 risk_reason에 하수관로가 표시되지 않는
# 문제가 있었으므로 모든 핵심 피처를 명시한다.
#
# 따라서 하수관로가 1순위 원인이 아니더라도
# sewer_old30_ratio / sewer_score가 항상 결과에 남는다.
# ============================================================

def make_risk_reason(row):

    return (
        f"주요원인={row['main_risk_factor']} | "
        f"동결융해={row['freeze_score']:.1f}점 "
        f"(14일 {row['freeze_thaw_14d']:.0f}회) | "
        f"누적강수={row['rain_score']:.1f}점 "
        f"(7일 {row['rain_7d']:.1f}mm, "
        f"14일 {row['rain_14d']:.1f}mm) | "
        f"하수관로 노후도={row['sewer_score']:.1f}점 "
        f"(30년 이상 비율 "
        f"{row['sewer_old30_ratio'] * 100:.1f}%) | "
        f"교통하중={row['traffic_score']:.1f}점 "
        f"(ESAL {row['traffic_esal']:.1f})"
    )


risk["risk_reason"] = (
    risk.apply(
        make_risk_reason,
        axis=1
    )
)


# ============================================================
# 11. 결과 저장
# ============================================================

OUTPUT_ALL.parent.mkdir(
    parents=True,
    exist_ok=True
)


risk.to_csv(
    OUTPUT_ALL,
    index=False,
    encoding="utf-8-sig"
)


latest_date = (
    risk["date"]
    .max()
)


latest_risk = (
    risk[
        risk["date"]
        == latest_date
    ]
    .copy()
)


latest_risk = latest_risk.sort_values(
    "risk_score",
    ascending=False
)


latest_risk.to_csv(
    OUTPUT_LATEST,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 12. 결과 검증
# ============================================================

print(
    "도로 위험도 계산 완료"
)


print(
    f"도로 포인트 수: {roads['point_id'].nunique()}"
)


print(
    f"전체 날짜 결과: {OUTPUT_ALL}"
)


print(
    f"최신 위험도 결과: {OUTPUT_LATEST}"
)


print(
    f"최신 날씨 날짜: {latest_date.date()}"
)


print(
    "\n하수관로 결측치:",
    latest_risk[
        "sewer_old30_ratio"
    ]
    .isna()
    .sum()
)


print(
    "\n시별 하수관로 노후도"
)


print(
    latest_risk[
        [
            "city",
            "sewer_old30_ratio",
            "sewer_score",
        ]
    ]
    .drop_duplicates()
    .sort_values(
        "city"
    )
    .to_string(
        index=False
    )
)


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
    "traffic_esal",
    "traffic_score",
    "risk_score",
    "risk_level",
    "main_risk_factor",
    "risk_reason",
]


print(
    latest_risk[
        display_columns
    ]
    .head(
        20
    )
    .to_string(
        index=False
    )
)


# ============================================================
# 13. 동일 risk_score 개수 확인
#
# 같은 시 내 포인트들이 같은 위험도를 갖는 문제가
# 얼마나 남아 있는지 확인하기 위한 진단 출력이다.
# ============================================================

print(
    "\n시별 서로 다른 risk_score 개수"
)


print(
    latest_risk
    .groupby(
        "city"
    )[
        "risk_score"
    ]
    .nunique()
)


print(
    "\n전체 서로 다른 risk_score 개수:",
    latest_risk[
        "risk_score"
    ]
    .nunique()
)