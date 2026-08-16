import pandas as pd
import numpy as np
from pathlib import Path


TRAFFIC_FILE = Path("data/전북국도교통량(상시조사구간).csv")
WEATHER_FILE = Path("data/weather.csv")
SEWER_FILE = Path("data/전북5개시_하수관로_노후도.xlsx")

OUTPUT_ALL = Path("data/road_risk_all_dates.csv")
OUTPUT_LATEST = Path("data/road_risk_latest.csv")


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


traffic = read_traffic_file(
    TRAFFIC_FILE
)


traffic["cities"] = (
    traffic["구간명"]
    .apply(extract_target_cities)
)


traffic = traffic[
    traffic["cities"].str.len() > 0
].copy()


traffic = traffic.explode(
    "cities"
)


traffic = traffic.rename(
    columns={
        "cities": "city"
    }
)


traffic["road_id"] = (
    traffic["노선명"].astype(str)
    + "|"
    + traffic["구간명"].astype(str)
)


for vehicle, factor in ESAL_FACTORS.items():

    traffic[
        f"{vehicle}_ESAL"
    ] = (
        traffic[vehicle]
        * factor
    )


# ------------------------------------------------------------
# ESAL
#
# ESAL은 서로 다른 차종과 축하중이 포장에 주는 영향을
# 표준 축하중으로 환산하기 위한 포장공학 지표이다.
#
# 단순 교통량보다 대형차 반복하중의 영향을 반영할 수 있기 때문에
# 본 모델에서는 총 차량대수 대신 ESAL을 사용한다.
#
# 차량별 계수는 기존 프로젝트에서 사용한 값을 그대로 유지한다.
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# ESAL 값은 도로별 편차가 매우 클 수 있기 때문에 log1p를 적용한다.
#
# log 변환은 FHWA 공식 위험점수식이 아니라
# 극단적인 대형교통량 도로가 전체 점수분포를 지배하지 않도록 하기 위한
# 본 프로젝트의 전처리 방식이다.
# ------------------------------------------------------------

traffic["traffic_score"] = minmax_score(
    np.log1p(
        traffic["traffic_esal"]
    )
)


# ------------------------------------------------------------
# 하수관로 데이터
#
# 실제 업로드한 엑셀의 '노후도_요약' 시트:
#
# 시군
# 총연장(m)
# 30년이상 노후연장(m)
# 노후관 비율(%)
# 평균경과연수
#
# 노후관 비율은 파일 내부에서는
# 0~1 형태의 비율값으로 저장되어 있으므로 ×100 한다.
#
# 예:
# 익산시 약 0.762
# → 약 76.2점
#
# 노후관 비율 자체가 이미 해석 가능한 0~100 척도이므로
# 별도의 Min-Max 정규화는 적용하지 않는다.
# ------------------------------------------------------------

sewer = pd.read_excel(
    SEWER_FILE,
    sheet_name="노후도_요약",
    header=3
)


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


# ------------------------------------------------------------
# 동결융해 판정
#
# 도로포장 연구와 FHWA 자료에서 freeze-thaw는
# 포장 열화의 주요 환경요인으로 다뤄진다.
#
# 실제 포장 내부 온도가 가장 정확하지만 현재 데이터에는
# 기상관측소의 대기온도만 있으므로 proxy를 사용한다.
#
# 하루 최저기온 <= 0℃
# AND
# 하루 최고기온 > 0℃
#
# 이면 하루 동안 0℃ 경계를 통과했을 가능성이 있다고 보고
# 동결융해 발생일로 판정한다.
#
# 즉 0℃ 기준은 물의 상변화라는 물리적 근거를 갖지만,
# 대기온도 기반 판정은 실제 포장 내부 동결융해의 대리지표이다.
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# 최근 14일 동결융해 점수
#
# 0회       = 0
# 1~2회     = 20
# 3~4회     = 40
# 5~6회     = 60
# 7~9회     = 80
# 10회 이상 = 100
#
# 동결융해가 반복될수록 포장재 열화가 증가한다는 문헌적 근거는 있으나,
# 위의 정확한 20점 단위는 정부 공식 포트홀 등급이 아니다.
#
# 위험지수화를 위해 반복횟수가 증가할수록 단계적으로
# 위험도를 상승시키도록 설계한 engineering heuristic이다.
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# 누적강수량
#
# 하루 강수량보다 최근 일정 기간 동안 도로가
# 수분에 얼마나 지속적으로 노출되었는지가 중요하기 때문에
#
# 최근 7일 누적강수
# 최근 14일 누적강수
#
# 를 모두 계산한다.
#
# 포장 내부 수분은 노상/기층의 강성과 지지력을 감소시키고,
# 반복 교통하중 및 동결융해와 결합하면 포장손상을 증가시킬 수 있다.
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# 강수 위험 임계값
#
# 전국적으로
# "7일 누적강수 XX mm부터 포트홀 위험"
# 이라는 공식 절대 임계값은 없기 때문에
# 임의로 100mm 또는 150mm를 선택하지 않는다.
#
# 업로드한 전북 5개시 2024~2025 기상자료에서
# 강수가 중요한 6~9월의 실제 분포를 이용했다.
#
# 7일 누적강수 분위수:
#
# P50 약 16.98mm
# P75 약 30.05mm
# P90 약 41.94mm
# P95 약 50.95mm
# P99 약 64.81mm
#
# 14일 누적강수 분위수:
#
# P50 약 38.74mm
# P75 약 55.03mm
# P90 약 73.21mm
# P95 약 82.47mm
# P99 약 98.25mm
#
# 각각
#
# 0 → 0점
# P50 → 20점
# P75 → 40점
# P90 → 60점
# P95 → 80점
# P99 이상 → 100점
#
# 으로 연속 보간한다.
#
# 따라서 최고 위험은
# 전북 실제 강수분포에서 상위 약 1% 수준의 누적강수 상태를 의미한다.
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# 7일과 14일 위험점수 중 높은 값을 채택한다.
#
# 이유:
#
# 7일 고강수
# → 최근 집중적인 수분 유입
#
# 14일 고강수
# → 장기간 지속적인 습윤 상태
#
# 둘 중 한 상황만 심해도 위험할 수 있으므로 평균을 사용해
# 위험이 희석되는 것을 막는다.
#
# max 방식은 공식 포트홀 계산식이 아니라
# 서로 다른 시간척도의 위험을 보존하기 위한 모델 설계이다.
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# 계절별 가중치
#
# 겨울/해빙기: 11~4월
#
# 동결융해 35%
# 강수     15%
# 하수관   25%
# ESAL     25%
#
# 장마/여름철: 6~9월
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
# 동결융해·수분·교통하중 등이 포장손상 메커니즘이라는 점은
# 포장공학 및 FHWA 자료에 근거한다.
#
# 다만 위의 정확한 35%, 45%, 25%라는 값은
# 정부나 FHWA 공식 가중치가 아니다.
#
# 아직 실제 포트홀 발생 라벨을 이용한 회귀모델을 학습하지 않았기 때문에
# 계절별 물리 메커니즘을 반영한 규칙 기반 가중치로 사용한다.
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# 동결융해 × 수분 복합효과
#
# 물이 포장 내부에 존재한 상태에서 동결융해가 반복되면
# 각각의 요인을 독립적으로 보는 것보다 높은 위험을 가질 수 있다.
#
# 따라서
#
# freeze_score >= 60
# AND
# rain_score >= 60
#
# 이면 추가 10점을 적용한다.
#
# +10점은 공식 국가기준이 아니라
# 복합위험을 반영하기 위한 모델링 보너스이며
# 최종점수는 최대 100으로 제한한다.
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# 구간명 안에 대상 도시가 2개 모두 들어있는 경우
#
# 예:
# 전주시-익산시
#
# 같은 구간은 두 도시의 날씨/하수관 조건을 각각 계산한 뒤
# 최종 위험도가 높은 쪽을 해당 구간의 대표 위험도로 사용한다.
#
# 평균을 사용하면 한쪽 도시가 매우 위험할 때 위험도가 희석될 수 있으므로
# 도로 점검 우선순위 관점에서 보수적으로 max를 사용한다.
# ------------------------------------------------------------

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


unmatched = (
    read_traffic_file(
        TRAFFIC_FILE
    )
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
            20
        )
    )


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