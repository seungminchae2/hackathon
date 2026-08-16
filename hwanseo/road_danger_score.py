import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# 0. 파일 경로
# ============================================================

# merge_sewer_age.py
# + merge_repairs.py(300m 기준)
# 를 거친 최종 도로 포인트 파일
ROADS_FILE = Path(
    "data/roads_with_sewer_repair.csv"
)

TRAFFIC_FILE = Path(
    "data/전북국도교통량(상시조사교통량).csv"
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


TARGET_CITIES = [
    "전주시",
    "군산시",
    "익산시",
    "남원시",
    "정읍시",
]


# ============================================================
# 1. ESAL 계수
#
# ESAL(Equivalent Single Axle Load)은 차량 종류별로
# 서로 다른 포장 하중 영향을 표준 축하중으로 환산하기 위한
# 포장공학 지표이다.
#
# 단순 차량 수보다 화물차 및 버스의 큰 도로 하중을
# 반영하기 위해 사용한다.
#
# 아래 값은 기존 프로젝트에서 사용하던 계수를 유지한다.
# ============================================================

ESAL_FACTORS = {
    "승용차": 0.0002,
    "버스": 0.852,
    "소형화물": 0.004,
    "중형화물": 1.735,
    "대형화물": 3.169,
}


# ============================================================
# 2. 공통 함수
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
        .str.replace(
            ",",
            "",
            regex=False
        )
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
# 3. 도로 포인트 + 하수관로 + 보수이력 불러오기
#
# roads_with_sewer_repair.csv
#
# = roads.csv
# + 하수관로 노후도
# + 반경 300m 보수이력
#
# 주요 컬럼:
#
# road_name
# city
# lat
# lon
# sewer_old30_ratio
# average_sewer_age
# has_nearby_repair
# repair_id
# last_repair_date
# repair_type
# distance_to_last_repair_m
# nearby_repair_count
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
    "has_nearby_repair",
    "last_repair_date",
]


missing_columns = [
    col
    for col in required_road_columns
    if col not in roads.columns
]


if missing_columns:

    raise ValueError(
        "roads_with_sewer_repair.csv에 "
        f"필요한 컬럼이 없습니다: {missing_columns}\n"
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

roads["has_nearby_repair"] = pd.to_numeric(
    roads["has_nearby_repair"],
    errors="coerce"
).fillna(0)


roads["last_repair_date"] = pd.to_datetime(
    roads["last_repair_date"],
    errors="coerce"
)


# point_id가 아직 없다면 생성
if "point_id" not in roads.columns:

    roads = roads.reset_index(
        drop=True
    )

    roads["point_id"] = np.arange(
        len(roads)
    )


# ============================================================
# 4. 하수관로 노후도 점수
#
# sewer_old30_ratio는
# 30년 이상 노후 하수관로의 비율이다.
#
# 예:
#
# 0.762
# → 76.2%
# → sewer_score = 76.2
#
# 이미 0~1이라는 의미 있는 절대비율이므로
# Min-Max 정규화를 하지 않는다.
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


print(
    f"도로 포인트 수: {len(roads)}"
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
    "300m 이내 보수이력 포인트:",
    int(
        roads[
            "has_nearby_repair"
        ].sum()
    )
)


# ============================================================
# 5. 교통량 데이터 읽기
#
# 실제 CSV는 2단 헤더 구조이므로
# 첫 두 행 이후부터 실제 데이터로 사용한다.
# ============================================================

def read_traffic_file(path):

    raw = read_csv_auto_encoding(
        path,
        header=None
    )

    if raw.shape[1] < 17:

        raise ValueError(
            "교통량 CSV 컬럼 수가 예상보다 적습니다.\n"
            f"현재 컬럼 수: {raw.shape[1]}"
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
            "",
            np.nan
        )
        .ffill()
    )

    numeric_columns = [
        "AADT",
        "승용차",
        "버스",
        "소형화물",
        "중형화물",
        "대형화물",
    ]

    for col in numeric_columns:

        traffic[col] = clean_number(
            traffic[col]
        )

    return traffic


traffic = read_traffic_file(
    TRAFFIC_FILE
)


# ============================================================
# 6. 교통량 조사구간 → 대상 도시
#
# 구간명에 도시가 직접 들어있는 경우 자동 추출한다.
#
# 이전 확인에서 행정구역상 대상 5개시에 포함되는
# 구간 일부는 수동으로 추가한다.
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
# 7. ESAL 계산
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
# 8. 현재 ESAL 공간매칭 방법
#
# 현재 상시교통량 데이터에는 조사구간의 정확한
# geometry/좌표가 없기 때문에 680개 포인트와
# 직접 nearest spatial join을 할 수 없다.
#
# 따라서 현재 버전에서는 같은 도시 내 조사구간들의
# ESAL 중앙값을 사용한다.
#
# 평균이 아니라 중앙값을 사용하는 이유:
# 초고교통량 도로 한 개가 도시 전체 점수를
# 과도하게 지배하는 것을 방지하기 위함.
#
# 향후 교통량 조사구간 좌표 확보 시
# 포인트별 최근접 ESAL로 교체해야 한다.
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
# 9. 날씨 데이터
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
# 10. 동결융해 판정
#
# 최저기온 <= 0℃
# AND
# 최고기온 > 0℃
#
# 이면 하루 동안 0℃ 경계를 통과한 것으로 보고
# 동결융해 가능일로 판정한다.
#
# 물의 상변화가 0℃ 부근에서 발생하고,
# freeze-thaw가 포장 열화에 영향을 준다는
# 포장공학적 근거를 이용한다.
#
# 단 실제 노면 내부 온도가 아니라
# 기상관측소 대기온도를 사용하는 proxy라는 한계가 있다.
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
            window=14,
            min_periods=1
        ).sum()
    )
)


# ============================================================
# 11. 동결융해 위험점수
#
# 최근 14일:
#
# 0회       → 0점
# 1~2회     → 20점
# 3~4회     → 40점
# 5~6회     → 60점
# 7~9회     → 80점
# 10회 이상 → 100점
#
# 반복 동결융해가 포장 열화를 증가시킨다는 근거는 있으나
# 정확한 20점 단위는 국가 공식 포트홀 위험등급이 아니다.
#
# 반복횟수가 많을수록 위험을 증가시키기 위한
# engineering heuristic이다.
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
# 12. 최근 7일 / 14일 누적강수
#
# 수분은 포장 내부의 노상/기층 지지력을 감소시키고
# 동결융해 및 반복 교통하중과 결합하면
# 포장 손상 위험을 높일 수 있다.
#
# 최근 7일:
# 단기간 강수 및 최근 수분 유입
#
# 최근 14일:
# 장기간 지속된 습윤 상태
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
            window=7,
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
            window=14,
            min_periods=1
        ).sum()
    )
)


# ============================================================
# 13. 강수 위험 임계값
#
# 공식적으로
# "7일 XXmm면 포트홀 위험"
# 같은 전국 단일 기준은 없기 때문에
#
# 전북 5개시 2024~2025 날씨 데이터의
# 6~9월 누적강수 분위수를 이용한다.
#
#
# 7일 누적강수:
#
# P50 = 16.98mm
# P75 = 30.05mm
# P90 = 41.94mm
# P95 = 50.95mm
# P99 = 64.81mm
#
#
# 14일 누적강수:
#
# P50 = 38.74mm
# P75 = 55.03mm
# P90 = 73.21mm
# P95 = 82.47mm
# P99 = 98.25mm
#
#
# 점수:
#
# 0   → 0
# P50 → 20
# P75 → 40
# P90 → 60
# P95 → 80
# P99 → 100
#
# 따라서 100점은 실제 전북 데이터의
# 상위 약 1% 수준의 누적강수 상태를 의미한다.
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
    weather[
        "rain_7d"
    ]
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
    weather[
        "rain_14d"
    ]
    .apply(
        lambda x:
        piecewise_score(
            x,
            RAIN14_THRESHOLDS,
            RAIN_SCORES
        )
    )
)


# ============================================================
# 14. 최종 강수점수
#
# 7일과 14일 중 높은 값을 사용한다.
#
# 단기간 집중적인 습윤과
# 장기간 지속적인 습윤 모두 위험할 수 있으므로
# 평균으로 위험이 희석되지 않도록 max를 사용한다.
#
# max 사용은 프로젝트 설계 방식이다.
# ============================================================

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
# 15. 관측소 → 도시 연결
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
# 16. 도로 포인트 + 날씨 결합
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
# 17. 최근 보수 후 경과일수
#
# 각 위험도 평가 날짜 - 최근 보수일
#
# 를 계산한다.
#
# 기존처럼 매칭 실패를
# 3650일(10년)으로 강제로 채우지 않는다.
#
# 보수기록이 없으면 NaN으로 유지한다.
#
# 중요:
#
# 보수 기록 없음
# ≠
# 10년 동안 보수 안 함
# ============================================================

risk[
    "days_since_last_repair"
] = (
    risk["date"]
    - risk["last_repair_date"]
).dt.days


# 해당 평가날짜보다 미래에 시행된 보수는
# 그 시점의 위험도 계산에 사용하지 않는다.
risk.loc[
    risk[
        "days_since_last_repair"
    ] < 0,
    "days_since_last_repair"
] = np.nan


# ============================================================
# 18. 최근 보수에 따른 위험도 감쇠
#
# 보수이력은 새로운 독립적인 위험원인으로 10%를
# 추가하는 방식보다
#
# 이미 존재하던 위험을 최근 보수로 일부 낮추는
# 보정값으로 사용한다.
#
#
# 최근 90일 이내:
# -10점
#
# 91~180일:
# -7점
#
# 181~365일:
# -4점
#
# 366~730일:
# -2점
#
# 2년 초과:
# 0점
#
# 보수기록 없음:
# 0점
#
#
# 이 값은 국가 공식 포트홀 감소율이 아니라
# 최근 보수 효과를 반영하기 위한
# 프로젝트 heuristic이다.
#
# 보수기록이 없다고 위험도를 임의로 높이지 않는 것이 중요하다.
# ============================================================

def calculate_repair_adjustment(row):

    if (
        row["has_nearby_repair"] == 0
        or pd.isna(
            row[
                "days_since_last_repair"
            ]
        )
    ):
        return 0.0

    days = row[
        "days_since_last_repair"
    ]

    if days <= 90:
        return -10.0

    if days <= 180:
        return -7.0

    if days <= 365:
        return -4.0

    if days <= 730:
        return -2.0

    return 0.0


risk[
    "repair_adjustment"
] = (
    risk.apply(
        calculate_repair_adjustment,
        axis=1
    )
)


# ============================================================
# 19. 계절별 기본 가중치
#
# 겨울·해빙기 11~4월
#
# 동결융해 35%
# 강수     15%
# 하수관   25%
# 교통     25%
#
#
# 장마·여름 6~9월
#
# 동결융해 5%
# 강수     45%
# 하수관   25%
# 교통     25%
#
#
# 전환기 5월·10월
#
# 동결융해 15%
# 강수     35%
# 하수관   25%
# 교통     25%
#
#
# 동결융해와 수분이 계절에 따라 포장에 미치는 영향이
# 달라지는 점을 반영한다.
#
# 단 정확한 35/45/25%는 정부 공식 포트홀 가중치가 아니라
# 현재 규칙기반 모델의 engineering heuristic이다.
#
# 향후 포트홀 발생 라벨을 이용한 Logistic Regression,
# Random Forest 등으로 실제 계수를 학습하면
# 이 가중치를 데이터 기반으로 교체할 수 있다.
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
    risk[
        "date"
    ]
    .dt.month
)


weights = (
    risk[
        "month"
    ]
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
# 20. 각 위험요인 기여점수
# ============================================================

risk[
    "freeze_contribution"
] = (
    risk[
        "freeze_score"
    ]
    * risk[
        "freeze_weight"
    ]
)


risk[
    "rain_contribution"
] = (
    risk[
        "rain_score"
    ]
    * risk[
        "rain_weight"
    ]
)


risk[
    "sewer_contribution"
] = (
    risk[
        "sewer_score"
    ]
    * risk[
        "sewer_weight"
    ]
)


risk[
    "traffic_contribution"
] = (
    risk[
        "traffic_score"
    ]
    * risk[
        "traffic_weight"
    ]
)


# ============================================================
# 21. 기본 위험도
# ============================================================

risk[
    "base_risk_score"
] = (
    risk[
        "freeze_contribution"
    ]
    +
    risk[
        "rain_contribution"
    ]
    +
    risk[
        "sewer_contribution"
    ]
    +
    risk[
        "traffic_contribution"
    ]
)


# ============================================================
# 22. 동결융해 × 강수 복합효과
#
# 수분이 존재하는 상태에서 동결융해가 반복되면
# 포장 손상이 더 커질 수 있다는 물리적 메커니즘을
# 반영한다.
#
# freeze_score >= 60
# AND
# rain_score >= 60
#
# 이면 +10점.
#
# +10은 공식 정부 기준이 아니라
# 복합위험을 반영하기 위한 프로젝트 보너스값이다.
# ============================================================

risk[
    "interaction_bonus"
] = np.where(
    (
        risk[
            "freeze_score"
        ]
        >= 60
    )
    &
    (
        risk[
            "rain_score"
        ]
        >= 60
    ),
    10.0,
    0.0
)


# ============================================================
# 23. 최종 위험점수
#
# 기본 위험도
# + 기상 복합효과
# + 최근 보수 감쇠
#
# 최종 0~100점으로 제한
# ============================================================

risk[
    "risk_score"
] = (
    risk[
        "base_risk_score"
    ]
    +
    risk[
        "interaction_bonus"
    ]
    +
    risk[
        "repair_adjustment"
    ]
).clip(
    0,
    100
)


# ============================================================
# 24. 위험등급
#
# UI 표현을 위한 프로젝트 기준
#
# 0~30     안전
# 30~50    주의
# 50~70    위험
# 70~100   매우 위험
#
# 정부 공식 위험등급은 아니다.
# ============================================================

risk[
    "risk_level"
] = pd.cut(
    risk[
        "risk_score"
    ],
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
# 25. 주요 위험원인
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
# 26. risk_reason
#
# 특정 요인이 main_risk_factor가 아니더라도
# 모든 핵심 변수의 점수를 보여준다.
#
# 따라서 하수관로가 실제 위험도에 반영됐는지
# 결과 데이터에서 직접 확인할 수 있다.
# ============================================================

def make_risk_reason(row):

    reason = (
        f"주요원인={row['main_risk_factor']} | "
        f"동결융해={row['freeze_score']:.1f}점 "
        f"(14일 {row['freeze_thaw_14d']:.0f}회) | "
        f"누적강수={row['rain_score']:.1f}점 "
        f"(7일 {row['rain_7d']:.1f}mm, "
        f"14일 {row['rain_14d']:.1f}mm) | "
        f"하수관로 노후도={row['sewer_score']:.1f}점 "
        f"(30년 이상 "
        f"{row['sewer_old30_ratio'] * 100:.1f}%) | "
        f"교통하중={row['traffic_score']:.1f}점 "
        f"(ESAL {row['traffic_esal']:.1f})"
    )

    if (
        row["has_nearby_repair"] == 1
        and pd.notna(
            row[
                "days_since_last_repair"
            ]
        )
    ):

        reason += (
            f" | 최근보수={row['days_since_last_repair']:.0f}일 전"
            f" ({row['distance_to_last_repair_m']:.0f}m,"
            f" 보정 {row['repair_adjustment']:.0f}점)"
        )

    else:

        reason += (
            " | 300m 이내 확인된 보수이력 없음"
        )

    return reason


risk[
    "risk_reason"
] = (
    risk.apply(
        make_risk_reason,
        axis=1
    )
)


# ============================================================
# 27. 결과 저장
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
    risk[
        "date"
    ]
    .max()
)


latest_risk = (
    risk[
        risk[
            "date"
        ]
        == latest_date
    ]
    .copy()
)


latest_risk = (
    latest_risk
    .sort_values(
        "risk_score",
        ascending=False
    )
)


latest_risk.to_csv(
    OUTPUT_LATEST,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 28. 결과 검증
# ============================================================

print(
    "\n도로 위험도 계산 완료"
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
    "300m 보수이력 매칭 포인트:",
    int(
        latest_risk[
            "has_nearby_repair"
        ].sum()
    )
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


# ============================================================
# 29. 최근 보수 적용 결과 확인
# ============================================================

print(
    "\n보수 후 경과일수 통계"
)


matched_latest = latest_risk[
    (
        latest_risk[
            "has_nearby_repair"
        ]
        == 1
    )
    &
    (
        latest_risk[
            "days_since_last_repair"
        ]
        .notna()
    )
]


if len(
    matched_latest
) > 0:

    print(
        matched_latest[
            "days_since_last_repair"
        ]
        .describe()
    )

else:

    print(
        "현재 날짜 이전 보수기록 없음"
    )


print(
    "\n보수 감쇠값 분포"
)


print(
    latest_risk[
        "repair_adjustment"
    ]
    .value_counts()
    .sort_index()
)


# ============================================================
# 30. 위험도 상위 20개 포인트
# ============================================================

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
    "has_nearby_repair",
    "last_repair_date",
    "days_since_last_repair",
    "distance_to_last_repair_m",
    "repair_adjustment",
    "interaction_bonus",
    "risk_score",
    "risk_level",
    "main_risk_factor",
    "risk_reason",
]


display_columns = [
    col
    for col in display_columns
    if col in latest_risk.columns
]


print(
    "\n위험도 상위 20개 포인트"
)


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
# 31. 포인트별 위험도 다양성 확인
#
# 이전에는 같은 도시 136개 포인트가 모두
# 완전히 동일한 risk_score를 가졌다.
#
# 보수이력을 추가한 뒤 같은 도시 안에서도
# 몇 종류의 점수가 만들어졌는지 확인한다.
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


# ============================================================
# 32. 동일 위험점수 최대 중복 확인
# ============================================================

print(
    "\n가장 많이 중복된 risk_score"
)


print(
    latest_risk[
        "risk_score"
    ]
    .value_counts()
    .head(10)
)