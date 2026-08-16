import pandas as pd
import numpy as np
from pathlib import Path


TRAFFIC_FILE = Path("data/전북국도교통량(상시조사교통량).csv")
WEATHER_FILE = Path("data/weather_history.csv")
SEWER_FILE = Path("data/전북5개시_하수관로_노후도.xlsx")

OUTPUT_ALL = Path("data/road_risk_all_dates.csv")
OUTPUT_LATEST = Path("data/road_risk_latest.csv")


# ============================================================
# ESAL 계수
#
# ESAL(Equivalent Single Axle Load)은 서로 다른 차량의
# 도로 포장에 대한 반복하중 영향을 표준 축하중으로 환산하는 개념이다.
#
# 단순 차량 대수보다 대형차량이 포장에 미치는 영향을
# 더 크게 반영하기 위해 사용한다.
#
# 아래 값은 기존 프로젝트에서 사용한 ESAL 계수를 그대로 사용한다.
# ============================================================

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


# ============================================================
# CSV 인코딩 자동 판별
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


# ============================================================
# 숫자형 데이터 정리
#
# CSV에
#
# 23,142
#
# 같은 형태로 저장된 값을
#
# 23142
#
# 숫자로 변환한다.
# ============================================================

def clean_number(series):

    return pd.to_numeric(
        series
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.strip(),
        errors="coerce"
    ).fillna(0)


# ============================================================
# Min-Max 0~100 점수화
# ============================================================

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


# ============================================================
# 임계값 사이를 선형적으로 0~100 점수화
# ============================================================

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
# 구간명에서 대상 도시 찾기
#
# 예:
#
# 정읍시-태인면
# → 정읍시
#
# 김제IC-전주시
# → 전주시
#
# 전주시-익산시
# → 전주시 + 익산시
# ============================================================

def extract_target_cities(section_name):

    if pd.isna(section_name):

        return []

    section_name = str(section_name)

    cities = [
        city
        for city in TARGET_CITIES
        if city in section_name
    ]

    return cities


# ============================================================
# 실제 교통량 CSV 읽기
#
# 현재 파일은 2단 헤더 구조이다.
#
# 1행:
# 노선명 / 구간명 / AADT / 평균교통량 / 주말교통량
#
# 2행:
# 승용차 / 버스 / 소형화물 / 중형화물 / 대형화물 ...
#
# 따라서 처음 두 행을 제외하고 실제 데이터를 읽는다.
# ============================================================

def read_traffic_file(path):

    raw = read_csv_auto_encoding(
        path,
        header=None
    )

    if raw.shape[1] < 17:

        raise ValueError(
            "교통량 CSV의 컬럼 수가 예상보다 적습니다.\n"
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

    # GitHub 화면에서 국도1호선처럼 첫 행에만 노선명이 있고
    # 아래 행은 빈칸인 구조이므로 이전 값을 아래로 채운다.
    traffic["노선명"] = (
        traffic["노선명"]
        .replace("", np.nan)
        .ffill()
    )

    numeric_columns = [
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

    for col in numeric_columns:

        traffic[col] = clean_number(
            traffic[col]
        )

    return traffic


# ============================================================
# 교통량 데이터
# ============================================================

traffic = read_traffic_file(
    TRAFFIC_FILE
)


# 구간명에서 대상 도시 추출
traffic["cities"] = (
    traffic["구간명"]
    .apply(extract_target_cities)
)


# 전북 5개시 중 하나라도 포함된 구간만 사용
traffic = traffic[
    traffic["cities"].str.len() > 0
].copy()


# 전주시-익산시처럼 도시가 두 개인 경우
# 각각의 도시 조건을 계산할 수 있도록 행을 분리
traffic = traffic.explode(
    "cities"
)


traffic = traffic.rename(
    columns={
        "cities": "city"
    }
)


# 각 도로를 구분하기 위한 ID
traffic["road_id"] = (
    traffic["노선명"].astype(str)
    + "|"
    + traffic["구간명"].astype(str)
)


# ============================================================
# ESAL 계산
#
# 단순 교통량 대신 차량별 도로하중을 반영한다.
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
# ESAL 위험점수
#
# ESAL 값은 도로별 편차가 매우 클 수 있기 때문에
# log1p 변환 후 0~100점으로 정규화한다.
#
# log 변환 및 Min-Max 점수화 자체는 공식 정부 위험식이 아니라
# 특정 도로의 극단적인 ESAL 값이 전체 점수를 지배하지 않도록
# 하기 위한 프로젝트 데이터 처리 방식이다.
# ============================================================

traffic["traffic_score"] = minmax_score(
    np.log1p(
        traffic["traffic_esal"]
    )
)


# ============================================================
# 하수관로 노후도
#
# 전북 5개시 하수관로 엑셀의
# '노후도_요약' 시트를 사용한다.
#
# 노후관 비율 자체가 0~100%라는 해석 가능한 척도이므로
# Min-Max 정규화하지 않고 그대로 위험점수로 사용한다.
# ============================================================

sewer = pd.read_excel(
    SEWER_FILE,
    sheet_name="노후도_요약",
    header=3
)


sewer = sewer.iloc[
    :,
    :5
].copy()


sewer.columns = [
    "city",
    "total_sewer_length",
    "old_sewer_length",
    "old_sewer_ratio",
    "average_sewer_age",
]


sewer = sewer[
    sewer["city"].isin(
        TARGET_CITIES
    )
].copy()


sewer[
    "old_sewer_ratio"
] = pd.to_numeric(
    sewer[
        "old_sewer_ratio"
    ],
    errors="coerce"
)


sewer["sewer_score"] = (
    sewer["old_sewer_ratio"]
    * 100
).clip(
    0,
    100
)


# 교통량 데이터에 하수관로 정보 연결
traffic = traffic.merge(
    sewer[
        [
            "city",
            "total_sewer_length",
            "old_sewer_length",
            "old_sewer_ratio",
            "average_sewer_age",
            "sewer_score",
        ]
    ],
    on="city",
    how="left"
)


# ============================================================
# 날씨 데이터
#
# 실제 파일:
# data/weather_history.csv
#
# 주요 컬럼:
# date
# station_name
# min_temp
# max_temp
# precipitation
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
# 동결융해 판정
#
# [근거]
#
# 물의 상변화는 0℃ 부근에서 발생한다.
#
# 따라서
#
# 최저기온 <= 0℃
# 최고기온 > 0℃
#
# 이면 하루 동안 0℃ 경계를 통과했다고 보고
# 동결융해 가능일로 판정한다.
#
# FHWA의 LTPP 등 포장 연구에서도 freeze-thaw는
# 포장 성능에 영향을 주는 환경요인으로 다뤄진다.
#
# 실제 포장 내부 온도를 측정하는 것이 가장 정확하지만
# 현재 프로젝트에는 대기온도 자료가 있으므로
# 최저/최고기온을 동결융해 proxy로 사용한다.
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


# 최근 14일 동결융해 발생일 수
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
# 동결융해 점수
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
# 반복 동결융해가 포장재 열화를 증가시킨다는 연구 근거는 있지만
# 위의 정확한 점수 구간은 정부 공식 포트홀 위험등급이 아니다.
#
# 반복 횟수가 증가할수록 위험을 단계적으로 높이기 위한
# 본 프로젝트의 engineering heuristic이다.
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
# 7일 / 14일 누적강수
#
# 수분이 포장 내부로 침투하면 기층 및 노상의 강도와
# 지지력이 감소할 수 있고, 교통하중 및 동결융해와 결합하면
# 포장손상 위험이 증가할 수 있다.
#
# 하루 강수량만 보는 대신
#
# 최근 7일 = 단기간 수분 유입
# 최근 14일 = 지속적인 습윤 상태
#
# 를 함께 사용한다.
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
# 강수량 위험점수
#
# 포트홀에 대해
# "7일 누적강수 XXmm 이상이면 위험"
# 같은 전국 공통 공식 절대 임계값은 확인되지 않았다.
#
# 따라서 임의의 100mm/150mm를 사용하지 않고
# 프로젝트의 전북 5개시 2024~2025 기상자료에서
# 6~9월 누적강수량 분포를 이용했다.
#
# 7일 누적강수:
#
# P50 = 약 16.98mm
# P75 = 약 30.05mm
# P90 = 약 41.94mm
# P95 = 약 50.95mm
# P99 = 약 64.81mm
#
# 14일 누적강수:
#
# P50 = 약 38.74mm
# P75 = 약 55.03mm
# P90 = 약 73.21mm
# P95 = 약 82.47mm
# P99 = 약 98.25mm
#
# 점수:
#
# 0   → 0점
# P50 → 20점
# P75 → 40점
# P90 → 60점
# P95 → 80점
# P99 → 100점
#
# 따라서 100점은 전북 실제 강수분포의
# 상위 약 1% 수준을 의미한다.
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
# 최종 강수 점수
#
# 7일과 14일 점수의 평균이 아니라 높은 값을 사용한다.
#
# 7일 고강수:
# 단기간 집중적인 수분 유입
#
# 14일 고강수:
# 장기간 지속적인 습윤 상태
#
# 둘 중 하나가 심한 경우 평균으로 위험이 희석되는 것을
# 방지하기 위해 max 값을 사용한다.
#
# max 방식은 공식 정부식이 아니라 본 모델의 설계 방식이다.
# ============================================================

weather[
    "rain_score"
] = (
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
# 날씨 관측소 → 시군 연결
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
# 교통량 + 하수관 + 날씨 결합
# ============================================================

risk = traffic.merge(
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
# 계절별 가중치
#
# 겨울 및 해빙기: 11~4월
#
# 동결융해 35%
# 강수     15%
# 하수관   25%
# ESAL     25%
#
# 장마 및 여름철: 6~9월
#
# 동결융해 5%
# 강수     45%
# 하수관   25%
# ESAL     25%
#
# 전환기: 5월, 10월
#
# 동결융해 15%
# 강수     35%
# 하수관   25%
# ESAL     25%
#
# 동결융해와 수분이 포장 성능에 영향을 준다는 것은
# 포장공학적 근거가 있다.
#
# 다만 정확히 35%, 45%, 25%라는 수치는
# 정부 또는 FHWA의 공식 포트홀 가중치가 아니다.
#
# 현재 실제 포트홀 발생 라벨을 이용해 통계적으로 학습한
# 계수가 없으므로 계절별 손상 메커니즘을 반영한
# 규칙 기반 가중치로 사용한다.
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
# 각 요인의 최종 위험도 기여점수
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
# 기본 위험도
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
# 동결융해 × 강수 복합위험
#
# 수분이 많은 상태에서 동결융해가 반복되면
# 포장손상이 더욱 심해질 가능성이 있으므로
#
# freeze_score >= 60
# rain_score >= 60
#
# 을 동시에 만족하면 추가 10점.
#
# +10점은 공식 정부 기준이 아니라
# 복합위험을 반영하기 위한 모델링 값이다.
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
    10,
    0
)


# ============================================================
# 최종 위험점수
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
).clip(
    0,
    100
)


# ============================================================
# 위험등급
#
# UI에서 위험도를 쉽게 표현하기 위한 프로젝트 기준.
#
# 0~30   안전
# 30~50  주의
# 50~70  위험
# 70~100 매우 위험
#
# 정부 공식 포트홀 위험등급은 아니다.
# ============================================================

risk["risk_level"] = pd.cut(
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
# 가장 크게 기여한 위험요인
#
# 지도에서
#
# "이 도로가 왜 위험한가?"
#
# 를 설명하기 위한 컬럼이다.
# ============================================================

contribution_columns = {
    "freeze_contribution": "동결융해",
    "rain_contribution": "누적강수",
    "sewer_contribution": "하수관로 노후",
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
# 도시 경계구간 처리
#
# 예:
#
# 전주시-익산시
#
# 같은 구간은 전주시 조건과 익산시 조건을 각각 계산한다.
#
# 두 위험도의 평균을 내면 한쪽 도시가 매우 위험한 상황이
# 희석될 수 있으므로 도로 점검 우선순위 관점에서
# 높은 위험도를 대표값으로 사용한다.
# ============================================================

risk = risk.sort_values(
    [
        "road_id",
        "date",
        "risk_score",
    ],
    ascending=[
        True,
        True,
        False,
    ]
)


risk = risk.drop_duplicates(
    subset=[
        "road_id",
        "date",
    ],
    keep="first"
)


# ============================================================
# 결과 저장
# ============================================================

OUTPUT_ALL.parent.mkdir(
    parents=True,
    exist_ok=True
)


# 모든 날짜의 위험도
risk.to_csv(
    OUTPUT_ALL,
    index=False,
    encoding="utf-8-sig"
)


# 가장 최근 날짜만 추출
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
# 실행 결과 확인
# ============================================================

print(
    "도로 위험도 계산 완료"
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
    f"분석 대상 도로 구간 수: {latest_risk['road_id'].nunique()}"
)


# ============================================================
# 대상 5개시와 자동 매칭되지 않은 도로 확인
#
# 예:
#
# 대강면-동계면
#
# 처럼 구간명에 '남원시'라는 글자가 없으면
# 현재 문자열만으로 남원시라고 판단할 수 없다.
#
# 이런 도로를 마지막에 출력하여 추가 매핑 여부를 확인한다.
# ============================================================

unmatched = read_traffic_file(
    TRAFFIC_FILE
)


unmatched[
    "target_city_count"
] = (
    unmatched[
        "구간명"
    ]
    .apply(
        lambda x:
        len(
            extract_target_cities(
                x
            )
        )
    )
)


unmatched = unmatched[
    unmatched[
        "target_city_count"
    ]
    == 0
]


print(
    "\n전북 5개 대상 시와 매칭되지 않은 구간 수:",
    len(
        unmatched
    )
)


if len(
    unmatched
) > 0:

    print(
        unmatched[
            [
                "노선명",
                "구간명",
            ]
        ]
        .head(
            50
        )
        .to_string(
            index=False
        )
    )


# ============================================================
# 위험도 상위 20개 도로 출력
# ============================================================

display_columns = [
    "노선명",
    "구간명",
    "city",
    "date",
    "AADT",
    "total_traffic",
    "heavy_vehicle_ratio",
    "traffic_esal",
    "traffic_score",
    "old_sewer_ratio",
    "sewer_score",
    "freeze_thaw_14d",
    "freeze_score",
    "rain_7d",
    "rain_14d",
    "rain_score",
    "interaction_bonus",
    "risk_score",
    "risk_level",
    "main_risk_factor",
]


print(
    "\n위험도 상위 20개 도로"
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