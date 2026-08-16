import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# 0. 파일 경로
# ============================================================

TRAFFIC_FILE = Path("data/traffic.csv")
WEATHER_FILE = Path("data/weather.csv")
SEWER_FILE = Path("data/전북5개시_하수관로_노후도.xlsx")

OUTPUT_ALL = Path("data/road_risk_all_dates.csv")
OUTPUT_LATEST = Path("data/road_risk_latest.csv")


# ============================================================
# 1. ESAL 계수
# ============================================================
#
# [근거]
# FHWA Traffic Monitoring Guide:
# ESAL(Equivalent Single Axle Load)은 서로 다른 차량/축하중이
# 포장에 미치는 영향을 18,000 lb 단축하중 기준으로 환산하는 지표.
#
# FHWA:
# https://www.fhwa.dot.gov/policyinformation/tmguide/tmg_2013/traffic-data-pavement.cfm
#
# 또한 축하중의 포장 영향은 전통적인 AASHTO 방식에서
# 4승 법칙(fourth power relationship)의 영향을 받는 것으로 사용됨.
#
# 아래 차량별 계수는 기존 프로젝트에서 계산해 사용한 값이며,
# 위험도 모델에서도 동일한 값을 유지한다.
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

def minmax_score(series):
    """
    데이터를 0~100 범위로 변환한다.
    """

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
    ).clip(0, 100)


def piecewise_score(value, thresholds, scores):
    """
    여러 임계값 사이를 선형보간하여
    0~100 사이의 연속형 위험점수를 만든다.
    """

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
# 3. 교통량 + ESAL 계산
# ============================================================

traffic = pd.read_csv(
    TRAFFIC_FILE,
    encoding="utf-8-sig"
)


required_traffic_columns = list(
    ESAL_FACTORS.keys()
)

missing_columns = [
    col
    for col in required_traffic_columns
    if col not in traffic.columns
]

if missing_columns:
    raise ValueError(
        f"traffic.csv에 필요한 컬럼이 없습니다: {missing_columns}\n"
        f"현재 컬럼: {traffic.columns.tolist()}"
    )


for col in ESAL_FACTORS:

    traffic[col] = (
        traffic[col]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
    )

    traffic[col] = pd.to_numeric(
        traffic[col],
        errors="coerce"
    ).fillna(0)


# 차종별 ESAL 계산
for vehicle, factor in ESAL_FACTORS.items():

    traffic[f"{vehicle}_ESAL"] = (
        traffic[vehicle]
        * factor
    )


# 전체 도로하중
traffic["traffic_esal"] = sum(
    traffic[f"{vehicle}_ESAL"]
    for vehicle in ESAL_FACTORS
)


# 총 교통량
traffic["total_traffic"] = (
    traffic[
        list(ESAL_FACTORS.keys())
    ]
    .sum(axis=1)
)


# 중형 + 대형 화물차량
traffic["heavy_traffic"] = (
    traffic["중형화물"]
    + traffic["대형화물"]
)


traffic["heavy_vehicle_ratio"] = (
    traffic["heavy_traffic"]
    / traffic["total_traffic"].replace(
        0,
        np.nan
    )
)


# ============================================================
# ESAL 점수
#
# [근거]
# ESAL 자체가 포장에 미치는 등가 반복하중을 의미하므로
# 단순 차량대수 대신 traffic_esal을 위험변수로 사용한다.
#
# 도로별 ESAL의 차이가 매우 클 수 있으므로 log1p를 적용해
# 극단적으로 큰 교통량 한 개가 전체 0~100점 분포를
# 지배하는 현상을 완화한다.
#
# log 변환 및 0~100 정규화는 공식 FHWA 기준이 아니라
# 본 프로젝트의 위험지수 산정을 위한 데이터 처리 방식이다.
# ============================================================

traffic["traffic_score"] = minmax_score(
    np.log1p(
        traffic["traffic_esal"]
    )
)


# ============================================================
# 4. 하수관로 노후도
# ============================================================
#
# 업로드된 엑셀의 노후도_요약 시트 사용.
#
# 파일에는:
#
# 전주시
# 군산시
# 익산시
# 남원시
# 정읍시
#
# 의 총연장, 노후연장, 노후관 비율 등이 정리되어 있음.
#
# 노후관 비율을 그대로 0~100 위험점수로 사용한다.
#
# 예)
# 노후관 비율 76.2%
# → sewer_score = 76.2
#
# 별도의 Min-Max 정규화를 하지 않는 이유:
# 노후관 비율 자체가 이미 0~100%라는 해석 가능한 절대척도이기 때문.
# ============================================================

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
    "average_sewer_age"
]


sewer = sewer[
    sewer["city"].notna()
].copy()


sewer["old_sewer_ratio"] = pd.to_numeric(
    sewer["old_sewer_ratio"],
    errors="coerce"
)


# 엑셀 값은 예를 들어 0.762009 형태이므로
# 퍼센트 점수로 변환
sewer["sewer_score"] = (
    sewer["old_sewer_ratio"]
    * 100
).clip(0, 100)


# ============================================================
# 5. 도로명 → 시군 연결
# ============================================================

road_mapping = pd.read_excel(
    SEWER_FILE,
    sheet_name="road_name_매핑",
    header=3
)


road_mapping.columns = [
    "road_name",
    "city"
]


road_mapping = road_mapping[
    road_mapping["road_name"].notna()
].copy()


# traffic.csv에 city가 이미 있다면 그대로 사용하고,
# 없다면 road_name을 이용해 city를 붙인다.
if "city" not in traffic.columns:

    if "road_name" not in traffic.columns:
        raise ValueError(
            "traffic.csv에 city 또는 road_name 컬럼이 필요합니다."
        )

    traffic = traffic.merge(
        road_mapping,
        on="road_name",
        how="left"
    )


traffic = traffic.merge(
    sewer[
        [
            "city",
            "old_sewer_ratio",
            "average_sewer_age",
            "sewer_score"
        ]
    ],
    on="city",
    how="left"
)


# ============================================================
# 6. 날씨 데이터 불러오기
# ============================================================
#
# 실제 업로드한 날씨 데이터 컬럼:
#
# date
# station_id
# station_name
# lat
# lon
# avg_temp
# min_temp
# max_temp
# precipitation
# snowfall
# humidity
#
# 따라서 min_temp / max_temp로 동결융해를,
# precipitation으로 7일/14일 누적강수를 계산할 수 있음.
# ============================================================

weather = pd.read_csv(
    WEATHER_FILE,
    encoding="utf-8-sig"
)


weather["date"] = pd.to_datetime(
    weather["date"]
)


numeric_weather_columns = [
    "min_temp",
    "max_temp",
    "precipitation"
]


for col in numeric_weather_columns:

    weather[col] = pd.to_numeric(
        weather[col],
        errors="coerce"
    ).fillna(0)


weather = weather.sort_values(
    [
        "station_name",
        "date"
    ]
).copy()


# ============================================================
# 7. 동결융해 판정
# ============================================================
#
# [외부 근거]
#
# FHWA LTPP 및 포장 연구에서는 freezing/thawing을
# 포장 성능에 영향을 주는 주요 환경요인으로 다룸.
#
# 물의 상변화가 0℃ 부근에서 발생하기 때문에,
# 일 최저기온이 0℃ 이하이고 일 최고기온이 0℃를 초과하면
# 하루 동안 0℃ 경계를 통과한 것으로 보고
# 동결→융해가 발생할 가능성이 있는 날로 정의한다.
#
# 주의:
# 실제 포장 내부온도를 측정하는 것이 가장 정확하지만,
# 본 프로젝트에는 대기온도 자료만 있으므로
# 일 최저/최고기온을 동결융해 proxy(대리지표)로 사용한다.
#
# FHWA LTPP:
# https://www.fhwa.dot.gov/publications/research/infrastructure/pavements/ltpp/06121/appenda.cfm
#
# FHWA는 반복적인 동결/융해와 해빙기 약화가
# 포장 성능 저하와 관련된다는 점을 설명하고 있다.
# ============================================================

weather["freeze_thaw"] = (
    (weather["min_temp"] <= 0)
    &
    (weather["max_temp"] > 0)
).astype(int)


# 최근 14일간 동결융해 발생일 수
weather["freeze_thaw_14d"] = (
    weather
    .groupby("station_name")["freeze_thaw"]
    .transform(
        lambda x: x.rolling(
            window=14,
            min_periods=1
        ).sum()
    )
)


# ============================================================
# 8. 동결융해 점수
# ============================================================
#
# [점수 설계]
#
# FHWA 자료는 반복적인 freeze-thaw cycle의 누적 영향을
# 뒷받침하지만,
#
# "14일 동안 정확히 몇 회면 위험 60점"
#
# 과 같은 공식 포트홀 점수 기준은 존재하지 않는다.
#
# 따라서 다음 구간은 본 프로젝트의 engineering heuristic이다.
#
# 최근 14일 기준:
#
# 0회       → 0점
# 1~2회     → 20점
# 3~4회     → 40점
# 5~6회     → 60점
# 7~9회     → 80점
# 10회 이상 → 100점
#
# 10회 이상은 최근 14일 중 70% 이상에서
# 0℃ 경계를 반복 통과했다는 뜻이므로
# 최고 위험군으로 설정한다.
#
# 따라서:
# - 0℃ 기준 = 물리적/문헌 근거
# - 횟수별 20점 간격 = 본 위험지수 설계
#
# 로 구분해서 발표해야 한다.
# ============================================================

def calculate_freeze_score(count):

    if count <= 0:
        return 0

    elif count <= 2:
        return 20

    elif count <= 4:
        return 40

    elif count <= 6:
        return 60

    elif count <= 9:
        return 80

    else:
        return 100


weather["freeze_score"] = (
    weather["freeze_thaw_14d"]
    .apply(calculate_freeze_score)
)


# ============================================================
# 9. 최근 7일 / 14일 누적강수
# ============================================================
#
# [외부 근거]
#
# FHWA:
# 포장 내부의 과도한 수분은
#
# - 비결합층의 강도와 강성 감소
# - 노상 약화
# - 동적 교통하중 시 간극수압 증가
# - 국부적인 pothole 등 포장 손상
#
# 으로 이어질 수 있다고 설명함.
#
# FHWA:
# https://www.fhwa.dot.gov/engineering/geotech/pubs/05037/03a.cfm
#
# 따라서 하루 강수량보다는 도로가 일정 기간 동안
# 지속적으로 수분에 노출되었는지를 보기 위해
# 7일 및 14일 누적강수를 사용한다.
# ============================================================

weather["rain_7d"] = (
    weather
    .groupby("station_name")["precipitation"]
    .transform(
        lambda x: x.rolling(
            window=7,
            min_periods=1
        ).sum()
    )
)


weather["rain_14d"] = (
    weather
    .groupby("station_name")["precipitation"]
    .transform(
        lambda x: x.rolling(
            window=14,
            min_periods=1
        ).sum()
    )
)


# ============================================================
# 10. 강수량 점수 기준
# ============================================================
#
# [중요]
#
# "7일 누적강수 XX mm 이상이면 포트홀 위험"
# 같은 전국 공통 공식 임계값은 확인되지 않았다.
#
# 따라서 임의의 100mm, 150mm를 정하지 않고
# 실제 프로젝트 데이터의 분포를 기준으로 점수화한다.
#
#
# [데이터 근거]
#
# 업로드한 전북 5개시
# 2024~2025 날씨자료에서
# 강수가 중요한 6~9월 데이터의 분위수를 직접 계산함.
#
# 7일 누적강수:
#
# P50 = 16.98 mm
# P75 = 30.05 mm
# P90 = 41.94 mm
# P95 = 50.95 mm
# P99 = 64.81 mm
#
#
# 14일 누적강수:
#
# P50 = 38.74 mm
# P75 = 55.03 mm
# P90 = 73.21 mm
# P95 = 82.47 mm
# P99 = 98.25 mm
#
#
# 위험점수:
#
# 0       → 0점
# P50     → 20점
# P75     → 40점
# P90     → 60점
# P95     → 80점
# P99↑    → 100점
#
#
# 이 방식의 장점:
#
# "150mm가 위험해 보이니까 100점"
#
# 같은 임의 기준이 아니라
#
# "전북 5개시 실제 과거 강수량 중
# 상위 1% 수준이면 100점"
#
# 이라는 통계적 설명이 가능하다.
#
#
# 참고:
# 기상청 호우주의보는
# 3시간 60mm 또는 12시간 110mm,
#
# 호우경보는
# 3시간 90mm 또는 12시간 180mm 기준이다.
#
# 그러나 이는 '강우 강도 및 재난특보 기준'이지
# 7일/14일 포트홀 기준이 아니므로
# 본 모델의 누적강수 임계값으로 직접 사용하지 않는다.
#
# 기상청:
# https://www.kma.go.kr/kma/biz/forecast03.jsp
# ============================================================


RAIN7_THRESHOLDS = [
    0,
    16.98,
    30.05,
    41.94,
    50.95,
    64.81
]


RAIN14_THRESHOLDS = [
    0,
    38.74,
    55.03,
    73.21,
    82.47,
    98.25
]


RAIN_SCORES = [
    0,
    20,
    40,
    60,
    80,
    100
]


weather["rain_7d_score"] = (
    weather["rain_7d"]
    .apply(
        lambda x: piecewise_score(
            x,
            RAIN7_THRESHOLDS,
            RAIN_SCORES
        )
    )
)


weather["rain_14d_score"] = (
    weather["rain_14d"]
    .apply(
        lambda x: piecewise_score(
            x,
            RAIN14_THRESHOLDS,
            RAIN_SCORES
        )
    )
)


# ============================================================
# 11. 최종 강수 점수
# ============================================================
#
# 7일과 14일을 단순 평균하지 않고 둘 중 큰 값을 사용한다.
#
# 이유:
#
# ① 7일 강수가 높음
#    → 단기간 집중적인 수분 유입
#
# ② 14일 강수가 높음
#    → 장기간 지속된 포장 습윤 상태
#
# 두 상황 모두 포장에는 위험할 수 있다.
#
# 예:
#
# 7일 = 극단적으로 높음
# 14일 = 보통
#
# 이어도 단기간 집중호우 위험을 평균으로 희석시키지 않는다.
#
# max() 사용은 공식 FHWA 공식이 아니라
# 두 시간척도의 위험을 보존하기 위한 모델 설계다.
# ============================================================

weather["rain_score"] = weather[
    [
        "rain_7d_score",
        "rain_14d_score"
    ]
].max(axis=1)


# ============================================================
# 12. 날씨 관측소명을 시군명으로 변경
# ============================================================

CITY_MAP = {
    "전주": "전주시",
    "군산": "군산시",
    "익산": "익산시",
    "남원": "남원시",
    "정읍": "정읍시",
}


weather["city"] = (
    weather["station_name"]
    .map(CITY_MAP)
)


weather = weather[
    weather["city"].notna()
].copy()


# ============================================================
# 13. 도로 데이터 + 날씨 데이터 결합
# ============================================================
#
# 각 도로의 city를 기준으로
# 해당 도시 기상관측값을 연결한다.
#
# 예:
#
# 전주시 도로 → 전주 관측소
# 군산시 도로 → 군산 관측소
# 익산시 도로 → 익산 관측소
# 남원시 도로 → 남원 관측소
# 정읍시 도로 → 정읍 관측소
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
            "rain_score"
        ]
    ],
    on="city",
    how="left"
)


# ============================================================
# 14. 계절별 가중치
# ============================================================
#
# [외부 근거]
#
# FHWA:
# - 추운 기후에서 freeze/thaw가 포장 성능에 영향을 미침
# - 해빙기에는 노상의 지지력이 감소할 수 있음
#
# FHWA:
# https://www.fhwa.dot.gov/publications/research/infrastructure/geotechnical/98139/04.cfm
#
# 또한 과도한 수분은 포장 성능을 악화시키므로
# 장마철에는 강수요인의 중요성을 높인다.
#
#
# [중요]
#
# 아래 35%, 45%, 25% 등의 숫자 자체는
# 국토부/FHWA가 제시한 공식 포트홀 가중치가 아니다.
#
# 현재 포트홀 발생 라벨을 이용해 회귀분석이나
# 머신러닝으로 학습한 계수가 없기 때문에
# 이것을 "통계적으로 검증된 계수"라고 표현하면 안 된다.
#
# 현재 모델에서는 계절별 물리 메커니즘을 반영한
# 전문가 규칙 기반 가중치로 사용한다.
#
#
# 겨울·해빙기 (11~4월)
#
# 동결융해 35%
# 강수     15%
# 하수관   25%
# ESAL     25%
#
# → 동결융해를 가장 높은 기상요인으로 설정
#
#
# 장마·고강수기 (6~9월)
#
# 동결융해 5%
# 강수     45%
# 하수관   25%
# ESAL     25%
#
# → 동결융해 가능성이 거의 없기 때문에 비중 축소
# → 강수 위험을 핵심 기상요인으로 설정
#
#
# 전환기 (5월, 10월)
#
# 동결융해 15%
# 강수     35%
# 하수관   25%
# ESAL     25%
#
# 모든 경우 합계 = 100%
# ============================================================


def get_weights(month):

    # 겨울 + 해빙기
    if month in [
        11,
        12,
        1,
        2,
        3,
        4
    ]:

        return (
            0.35,
            0.15,
            0.25,
            0.25
        )

    # 장마 및 여름철
    elif month in [
        6,
        7,
        8,
        9
    ]:

        return (
            0.05,
            0.45,
            0.25,
            0.25
        )

    # 전환기
    else:

        return (
            0.15,
            0.35,
            0.25,
            0.25
        )


risk["month"] = (
    risk["date"]
    .dt.month
)


weights = (
    risk["month"]
    .apply(get_weights)
)


risk[
    [
        "freeze_weight",
        "rain_weight",
        "sewer_weight",
        "traffic_weight"
    ]
] = pd.DataFrame(
    weights.tolist(),
    index=risk.index
)


# ============================================================
# 15. 기본 위험도 계산
# ============================================================
#
# Risk =
#
# freeze_score × 계절별 동결융해 가중치
# +
# rain_score × 계절별 강수 가중치
# +
# sewer_score × 0.25
# +
# traffic_score × 0.25
#
# 기본점수 범위 = 0~100
# ============================================================

risk["base_risk_score"] = (

    risk["freeze_score"]
    * risk["freeze_weight"]

    +

    risk["rain_score"]
    * risk["rain_weight"]

    +

    risk["sewer_score"]
    * risk["sewer_weight"]

    +

    risk["traffic_score"]
    * risk["traffic_weight"]

)


# ============================================================
# 16. 강수 × 동결융해 복합위험
# ============================================================
#
# [외부 근거]
#
# FHWA는 포장 내부의 수분이 포장 성능을 악화시키고,
# 추운 환경에서는 그 수분이 freeze/thaw cycle을
# 경험할 수 있다고 설명한다.
#
# 또한 해빙기에 포장/노상의 지지력이 크게 감소할 수 있다.
#
# 즉,
#
# 수분 + 동결융해
#
# 가 동시에 높은 상황은 각각을 독립적으로 보는 것보다
# 더 주의할 필요가 있다.
#
#
# [모델 설계]
#
# freeze_score >= 60
# AND
# rain_score >= 60
#
# 인 경우 10점 bonus.
#
# 60점은 각각 위험분포에서 상위 위험구간 진입점으로 사용.
#
# +10이라는 값은 공식 임계값이 아니라
# 복합위험을 반영하기 위한 보수적인 모델링 값이다.
#
# 최종 결과는 100점을 넘지 않도록 clip한다.
# ============================================================

risk["interaction_bonus"] = np.where(
    (
        (risk["freeze_score"] >= 60)
        &
        (risk["rain_score"] >= 60)
    ),
    10,
    0
)


# ============================================================
# 17. 최종 위험점수
# ============================================================

risk["risk_score"] = (
    risk["base_risk_score"]
    + risk["interaction_bonus"]
).clip(
    0,
    100
)


# ============================================================
# 18. 위험등급
# ============================================================
#
# 위험등급은 서비스/UI 표현을 위한 구간.
#
# 0~30   안전
# 30~50  주의
# 50~70  위험
# 70~100 매우 위험
#
# 이 등급 역시 정부 공식 포트홀 등급이 아니라
# 사용자에게 0~100 점수를 이해하기 쉽게 보여주기 위한
# 프로젝트 분류기준이다.
# ============================================================

risk["risk_level"] = pd.cut(
    risk["risk_score"],
    bins=[
        -1,
        30,
        50,
        70,
        100
    ],
    labels=[
        "안전",
        "주의",
        "위험",
        "매우 위험"
    ]
)


# ============================================================
# 19. 주요 위험 원인 계산
# ============================================================
#
# 지도에서
#
# "왜 이 도로가 위험한가?"
#
# 를 보여주기 위한 설명용 컬럼.
# ============================================================

risk["freeze_contribution"] = (
    risk["freeze_score"]
    * risk["freeze_weight"]
)


risk["rain_contribution"] = (
    risk["rain_score"]
    * risk["rain_weight"]
)


risk["sewer_contribution"] = (
    risk["sewer_score"]
    * risk["sewer_weight"]
)


risk["traffic_contribution"] = (
    risk["traffic_score"]
    * risk["traffic_weight"]
)


contribution_columns = {
    "freeze_contribution": "동결융해",
    "rain_contribution": "누적강수",
    "sewer_contribution": "하수관로 노후",
    "traffic_contribution": "교통하중"
}


risk["main_risk_factor"] = (
    risk[
        list(
            contribution_columns.keys()
        )
    ]
    .idxmax(axis=1)
    .map(contribution_columns)
)


# ============================================================
# 20. 결과 저장
# ============================================================

OUTPUT_ALL.parent.mkdir(
    parents=True,
    exist_ok=True
)


# 전체 날짜 결과
risk.to_csv(
    OUTPUT_ALL,
    index=False,
    encoding="utf-8-sig"
)


# 데이터의 가장 최신 날짜
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
# 21. 결과 확인
# ============================================================

print("도로 위험도 계산 완료")

print(
    f"전체 결과: {OUTPUT_ALL}"
)

print(
    f"최신 결과: {OUTPUT_LATEST}"
)

print(
    f"최신 날씨 날짜: {latest_date.date()}"
)


display_columns = [
    "road_name",
    "city",
    "date",
    "freeze_thaw_14d",
    "freeze_score",
    "rain_7d",
    "rain_14d",
    "rain_score",
    "sewer_score",
    "traffic_esal",
    "traffic_score",
    "interaction_bonus",
    "risk_score",
    "risk_level",
    "main_risk_factor"
]


available_columns = [
    col
    for col in display_columns
    if col in latest_risk.columns
]


print(
    latest_risk[
        available_columns
    ].head(20)
)