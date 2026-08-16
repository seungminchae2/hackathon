import pandas as pd
import numpy as np
import joblib
from pathlib import Path


LATEST_RISK_FILE = Path(
    "data/road_risk_latest.csv"
)

TRAINING_FILE = Path(
    "data/pothole_training_data.csv"
)

REPAIRS_FILE = Path(
    "data/repairs.csv"
)

LOGISTIC_MODEL_FILE = Path(
    "models/pothole_logistic.joblib"
)

OUTPUT_FILE = Path(
    "data/final_road_risk.csv"
)


REPAIR_MATCH_RADIUS_M = 300


FEATURES = [
    "freeze_thaw_14d",
    "rain_7d",
    "rain_14d",
    "sewer_old30_ratio",
    "traffic_esal",
    "has_past_repair",
    "days_since_last_repair",
]


FEATURE_NAMES_KR = {
    "freeze_thaw_14d": "동결융해",
    "rain_7d": "7일 누적강수",
    "rain_14d": "14일 누적강수",
    "sewer_old30_ratio": "하수관로 노후도",
    "traffic_esal": "교통하중",
    "has_past_repair": "과거 보수이력",
    "days_since_last_repair": "보수 후 경과기간",
}


def read_csv_auto_encoding(
    path,
    **kwargs
):

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


def get_risk_level(
    percentile
):

    if percentile >= 99:
        return "매우 위험"

    if percentile >= 95:
        return "위험"

    if percentile >= 85:
        return "주의"

    if percentile >= 70:
        return "관심"

    return "일반"


def get_risk_rank_group(
    percentile
):

    if percentile >= 99:
        return "상위 1%"

    if percentile >= 95:
        return "상위 1~5%"

    if percentile >= 85:
        return "상위 5~15%"

    if percentile >= 70:
        return "상위 15~30%"

    return "하위 70%"


def calculate_latest_past_repairs(
    latest,
    repairs
):

    print(
        "\n최신 평가시점 기준 보수이력 재계산 중..."
    )

    repair_lats = (
        repairs[
            "lat"
        ]
        .to_numpy()
    )

    repair_lons = (
        repairs[
            "lon"
        ]
        .to_numpy()
    )

    results = []


    for _, road in latest.iterrows():

        road_lat = road[
            "lat"
        ]

        road_lon = road[
            "lon"
        ]

        evaluation_date = road[
            "date"
        ]


        if (
            pd.isna(
                road_lat
            )
            or pd.isna(
                road_lon
            )
            or pd.isna(
                evaluation_date
            )
        ):

            results.append(
                {
                    "has_past_repair": 0,
                    "latest_repair_id": np.nan,
                    "latest_repair_date": pd.NaT,
                    "latest_repair_distance_m": np.nan,
                    "days_since_last_repair": np.nan,
                }
            )

            continue


        distances = haversine_distance(
            road_lat,
            road_lon,
            repair_lats,
            repair_lons
        )


        spatial_mask = (
            distances
            <= REPAIR_MATCH_RADIUS_M
        )


        date_mask = (
            repairs[
                "repair_date"
            ]
            <= evaluation_date
        ).to_numpy()


        valid_mask = (
            spatial_mask
            &
            date_mask
        )


        if not valid_mask.any():

            results.append(
                {
                    "has_past_repair": 0,
                    "latest_repair_id": np.nan,
                    "latest_repair_date": pd.NaT,
                    "latest_repair_distance_m": np.nan,
                    "days_since_last_repair": np.nan,
                }
            )

            continue


        candidates = (
            repairs.loc[
                valid_mask
            ]
            .copy()
        )


        candidates[
            "distance_m"
        ] = distances[
            valid_mask
        ]


        latest_repair_date = (
            candidates[
                "repair_date"
            ]
            .max()
        )


        latest_candidates = (
            candidates[
                candidates[
                    "repair_date"
                ]
                == latest_repair_date
            ]
            .sort_values(
                "distance_m"
            )
        )


        chosen = (
            latest_candidates
            .iloc[0]
        )


        days_since = (
            evaluation_date
            -
            chosen[
                "repair_date"
            ]
        ).days


        results.append(
            {
                "has_past_repair": 1,

                "latest_repair_id":
                    chosen[
                        "repair_id"
                    ],

                "latest_repair_date":
                    chosen[
                        "repair_date"
                    ],

                "latest_repair_distance_m":
                    float(
                        chosen[
                            "distance_m"
                        ]
                    ),

                "days_since_last_repair":
                    int(
                        days_since
                    ),
            }
        )


    repair_features = pd.DataFrame(
        results
    )


    # ========================================================
    # 기존 road_risk_latest.csv 안에 이미 존재하는
    # 보수 관련 컬럼 제거
    #
    # 이 컬럼들을 제거하지 않고 새 계산결과를 concat하면
    # days_since_last_repair 등이 중복 생성되어
    #
    # latest["days_since_last_repair"]
    #
    # 가 Series가 아닌 DataFrame으로 반환될 수 있다.
    #
    # 그 결과 pd.to_numeric()에서
    #
    # TypeError:
    # arg must be a list, tuple, 1-d array, or Series
    #
    # 오류가 발생한다.
    # ========================================================

    old_repair_columns = [
        "has_past_repair",
        "latest_repair_id",
        "latest_repair_date",
        "latest_repair_distance_m",
        "days_since_last_repair",
    ]


    columns_to_drop = [
        col
        for col in old_repair_columns
        if col in latest.columns
    ]


    if columns_to_drop:

        latest = latest.drop(
            columns=columns_to_drop
        )


    latest = pd.concat(
        [
            latest.reset_index(
                drop=True
            ),

            repair_features.reset_index(
                drop=True
            ),
        ],
        axis=1
    )


    # ========================================================
    # 안전검사:
    # 동일한 컬럼명이 남아 있는지 확인
    # ========================================================

    duplicate_columns = (
        latest.columns[
            latest.columns.duplicated()
        ]
        .tolist()
    )


    if duplicate_columns:

        raise ValueError(
            "보수이력 재계산 후 중복 컬럼이 존재합니다: "
            f"{duplicate_columns}"
        )


    return latest


def calculate_logistic_contributions(
    latest,
    model,
    X_latest
):

    scaler = (
        model
        .named_steps[
            "scaler"
        ]
    )

    logistic = (
        model
        .named_steps[
            "model"
        ]
    )


    scaled_values = (
        scaler.transform(
            X_latest
        )
    )


    coefficients = (
        logistic
        .coef_[0]
    )


    contribution_matrix = (
        scaled_values
        *
        coefficients
    )


    contribution_columns = []


    for i, feature in enumerate(
        FEATURES
    ):

        column_name = (
            f"contribution_{feature}"
        )

        latest[
            column_name
        ] = (
            contribution_matrix[
                :,
                i
            ]
        )

        contribution_columns.append(
            column_name
        )


    def select_main_factor(
        row
    ):

        contributions = {}


        for feature in FEATURES:

            value = row[
                f"contribution_{feature}"
            ]

            contributions[
                feature
            ] = value


        positive = {
            feature: value
            for feature, value
            in contributions.items()
            if value > 0
        }


        if len(
            positive
        ) == 0:

            return (
                "뚜렷한 증가요인 없음"
            )


        main_feature = max(
            positive,
            key=positive.get
        )


        return FEATURE_NAMES_KR[
            main_feature
        ]


    latest[
        "main_risk_factor"
    ] = latest.apply(
        select_main_factor,
        axis=1
    )


    return (
        latest,
        contribution_columns
    )


def main():

    print(
        "최신 도로 위험도 데이터 불러오는 중..."
    )


    latest = read_csv_auto_encoding(
        LATEST_RISK_FILE,
        low_memory=False
    )


    latest[
        "date"
    ] = pd.to_datetime(
        latest[
            "date"
        ],
        errors="coerce"
    )


    latest[
        "lat"
    ] = pd.to_numeric(
        latest[
            "lat"
        ],
        errors="coerce"
    )


    latest[
        "lon"
    ] = pd.to_numeric(
        latest[
            "lon"
        ],
        errors="coerce"
    )


    print(
        "최신 도로 포인트 수:",
        len(latest)
    )


    print(
        "최신 날짜:",
        latest[
            "date"
        ]
        .max()
        .date()
    )


    # ========================================================
    # 기존 규칙 기반 위험도 보존
    # ========================================================

    if (
        "risk_score"
        in latest.columns
    ):

        latest[
            "rule_based_risk_score"
        ] = latest[
            "risk_score"
        ]


    if (
        "risk_level"
        in latest.columns
    ):

        latest[
            "rule_based_risk_level"
        ] = latest[
            "risk_level"
        ].astype(str)


    if (
        "risk_reason"
        in latest.columns
    ):

        latest[
            "rule_based_risk_reason"
        ] = latest[
            "risk_reason"
        ].astype(str)


    # ========================================================
    # repairs.csv 읽기
    # ========================================================

    repairs = read_csv_auto_encoding(
        REPAIRS_FILE
    )


    required_repair_columns = [
        "repair_id",
        "repair_date",
        "lat",
        "lon",
    ]


    missing = [
        col
        for col
        in required_repair_columns
        if col not in repairs.columns
    ]


    if missing:

        raise ValueError(
            "repairs.csv에 필요한 컬럼이 없습니다: "
            f"{missing}"
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
        "전체 보수기록:",
        len(repairs)
    )


    # ========================================================
    # 최신 날짜 기준 보수이력 재계산
    # ========================================================

    latest = (
        calculate_latest_past_repairs(
            latest,
            repairs
        )
    )


    # ========================================================
    # 전체 중복 컬럼 최종 검사
    # ========================================================

    duplicate_columns = (
        latest.columns[
            latest.columns.duplicated()
        ]
        .tolist()
    )


    if duplicate_columns:

        raise ValueError(
            f"중복 컬럼이 존재합니다: {duplicate_columns}"
        )


    print(
        "과거 보수이력 포인트:",
        int(
            latest[
                "has_past_repair"
            ]
            .sum()
        )
    )


    future_leakage = int(
        (
            latest[
                "latest_repair_date"
            ]
            >
            latest[
                "date"
            ]
        )
        .sum()
    )


    print(
        "미래 보수정보 포함:",
        future_leakage
    )


    if (
        future_leakage
        != 0
    ):

        raise ValueError(
            "미래 보수정보가 최종 예측에 포함됐습니다."
        )


    # ========================================================
    # ML 입력 변수 숫자형 변환
    # ========================================================

    for col in [
        "freeze_thaw_14d",
        "rain_7d",
        "rain_14d",
        "sewer_old30_ratio",
        "traffic_esal",
    ]:

        if col not in latest.columns:

            raise ValueError(
                f"{col} 컬럼이 최신 도로 데이터에 없습니다."
            )


        latest[col] = pd.to_numeric(
            latest[col],
            errors="coerce"
        )


    latest[
        "has_past_repair"
    ] = pd.to_numeric(
        latest[
            "has_past_repair"
        ],
        errors="coerce"
    ).fillna(0)


    latest[
        "days_since_last_repair"
    ] = pd.to_numeric(
        latest[
            "days_since_last_repair"
        ],
        errors="coerce"
    )


    # ========================================================
    # 학습 당시 결측 처리 방식 재현
    # ========================================================

    latest.loc[
        latest[
            "has_past_repair"
        ]
        == 0,
        "days_since_last_repair"
    ] = 0


    training = read_csv_auto_encoding(
        TRAINING_FILE,
        low_memory=False
    )


    training[
        "days_since_last_repair"
    ] = pd.to_numeric(
        training[
            "days_since_last_repair"
        ],
        errors="coerce"
    ).fillna(0)


    for col in FEATURES:

        if col not in training.columns:

            raise ValueError(
                f"학습 데이터에 {col} 컬럼이 없습니다."
            )


        training[col] = pd.to_numeric(
            training[col],
            errors="coerce"
        )


    train_medians = (
        training[
            FEATURES
        ]
        .median(
            numeric_only=True
        )
    )


    X_latest = latest[
        FEATURES
    ].copy()


    X_latest = (
        X_latest
        .fillna(
            train_medians
        )
    )


    # ========================================================
    # 최종 모델 입력값 결측 여부 검사
    # ========================================================

    if X_latest.isna().any().any():

        missing_counts = (
            X_latest
            .isna()
            .sum()
        )

        missing_counts = (
            missing_counts[
                missing_counts
                > 0
            ]
        )

        raise ValueError(
            "Logistic 모델 입력값에 결측치가 남아 있습니다:\n"
            f"{missing_counts}"
        )


    # ========================================================
    # Logistic Regression 모델
    # ========================================================

    print(
        "\nLogistic Regression 모델 불러오는 중..."
    )


    model = joblib.load(
        LOGISTIC_MODEL_FILE
    )


    # ========================================================
    # Logistic model score
    #
    # balanced class weight를 적용해 학습했으므로
    # 절대적인 실제 포트홀 발생확률로 해석하지 않고
    # 상대 위험순위 계산에 사용한다.
    # ========================================================

    latest[
        "logistic_score"
    ] = (
        model.predict_proba(
            X_latest
        )[
            :,
            1
        ]
    )


    # ========================================================
    # Logistic 실제 변수 기여도
    #
    # 표준화 값 × Logistic 회귀계수
    # ========================================================

    (
        latest,
        contribution_columns
    ) = calculate_logistic_contributions(
        latest,
        model,
        X_latest
    )


    # ========================================================
    # Logistic score를 680개 포인트 내 백분위로 환산
    # ========================================================

    latest[
        "risk_percentile"
    ] = (
        latest[
            "logistic_score"
        ]
        .rank(
            method="average",
            pct=True
        )
        * 100
    )


    latest[
        "risk_score"
    ] = (
        latest[
            "risk_percentile"
        ]
        .clip(
            0,
            100
        )
    )


    latest[
        "risk_level"
    ] = latest[
        "risk_percentile"
    ].apply(
        get_risk_level
    )


    latest[
        "risk_rank_group"
    ] = latest[
        "risk_percentile"
    ].apply(
        get_risk_rank_group
    )


    # ========================================================
    # 위험도 설명 생성
    # ========================================================

    def make_reason(
        row
    ):

        if (
            row.get(
                "traffic_match_type"
            )
            == "nearest_station"
        ):

            distance = row.get(
                "traffic_distance_m"
            )


            if pd.notna(
                distance
            ):

                traffic_text = (
                    f"실제 조사점 "
                    f"{distance / 1000:.1f}km"
                )

            else:

                traffic_text = (
                    "실제 조사점"
                )


        elif (
            row.get(
                "traffic_match_type"
            )
            == "city_median"
        ):

            traffic_text = (
                "도시 중앙값"
            )


        else:

            traffic_text = (
                "교통량 매칭정보 없음"
            )


        if (
            row[
                "has_past_repair"
            ]
            == 1
        ):

            repair_text = (
                f"최근 보수 "
                f"{row['days_since_last_repair']:.0f}일 전"
                f" / 거리 "
                f"{row['latest_repair_distance_m']:.0f}m"
            )

        else:

            repair_text = (
                "과거 300m 이내 보수이력 없음"
            )


        contribution_parts = []


        for feature in FEATURES:

            value = row[
                f"contribution_{feature}"
            ]

            contribution_parts.append(
                f"{FEATURE_NAMES_KR[feature]}"
                f"={value:+.3f}"
            )


        contribution_text = (
            ", ".join(
                contribution_parts
            )
        )


        top_percent = (
            100
            -
            row[
                "risk_percentile"
            ]
        )


        return (
            f"Logistic 위험순위 상위 "
            f"{top_percent:.1f}%"
            f" | 주요 증가요인="
            f"{row['main_risk_factor']}"
            f" | 동결융해 14일 "
            f"{row['freeze_thaw_14d']:.0f}회"
            f" | 7일 강수 "
            f"{row['rain_7d']:.1f}mm"
            f" | 14일 강수 "
            f"{row['rain_14d']:.1f}mm"
            f" | 하수관 노후비율 "
            f"{row['sewer_old30_ratio'] * 100:.1f}%"
            f" | ESAL "
            f"{row['traffic_esal']:.1f}"
            f" ({traffic_text})"
            f" | {repair_text}"
            f" | 모델기여도["
            f"{contribution_text}"
            f"]"
        )


    latest[
        "risk_reason"
    ] = latest.apply(
        make_reason,
        axis=1
    )


    # ========================================================
    # 최종 위험도 순으로 정렬
    # ========================================================

    latest = latest.sort_values(
        [
            "risk_score",
            "logistic_score",
        ],
        ascending=[
            False,
            False,
        ]
    ).reset_index(
        drop=True
    )


    # ========================================================
    # 결과 저장
    # ========================================================

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    latest.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )


    # ========================================================
    # 최종 검증
    # ========================================================

    print(
        "\n최종 Logistic 도로 위험도 생성 완료"
    )


    print(
        "저장 위치:",
        OUTPUT_FILE
    )


    print(
        "전체 포인트:",
        len(latest)
    )


    print(
        "\n위험등급 분포"
    )


    print(
        latest[
            "risk_level"
        ]
        .value_counts()
    )


    print(
        "\n위험 순위 그룹 분포"
    )


    print(
        latest[
            "risk_rank_group"
        ]
        .value_counts()
    )


    print(
        "\nLogistic score 통계"
    )


    print(
        latest[
            "logistic_score"
        ]
        .describe()
    )


    print(
        "\n최종 risk_score 통계"
    )


    print(
        latest[
            "risk_score"
        ]
        .describe()
    )


    print(
        "\n주요 위험 증가요인 분포"
    )


    print(
        latest[
            "main_risk_factor"
        ]
        .value_counts()
    )


    print(
        "\n보수이력 상태"
    )


    print(
        latest[
            "has_past_repair"
        ]
        .value_counts()
    )


    print(
        "\n중복 컬럼 검사"
    )


    duplicate_columns = (
        latest.columns[
            latest.columns.duplicated()
        ]
        .tolist()
    )


    print(
        "중복 컬럼:",
        duplicate_columns
    )


    print(
        "\n위험도 상위 20개 도로 포인트"
    )


    display_columns = [
        "point_id",
        "road_name",
        "city",
        "lat",
        "lon",
        "date",

        "logistic_score",
        "risk_percentile",
        "risk_score",
        "risk_level",
        "risk_rank_group",

        "freeze_thaw_14d",
        "rain_7d",
        "rain_14d",
        "sewer_old30_ratio",
        "traffic_esal",

        "has_past_repair",
        "latest_repair_id",
        "latest_repair_date",
        "latest_repair_distance_m",
        "days_since_last_repair",

        "main_risk_factor",

        "contribution_freeze_thaw_14d",
        "contribution_rain_7d",
        "contribution_rain_14d",
        "contribution_sewer_old30_ratio",
        "contribution_traffic_esal",
        "contribution_has_past_repair",
        "contribution_days_since_last_repair",

        "rule_based_risk_score",
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


if __name__ == "__main__":
    main()