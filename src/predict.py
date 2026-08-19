from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .common import (
    load_config,
    resolve_path,
)

from .features import (
    OPTIONAL_FEATURE_COLUMNS,
    calculate_priority_components,
    explain_with_contributions,
    prepare_dataset,
    prepare_feature_matrix,
    risk_level_from_percentile,
)

from .train import (
    MODEL_SCHEMA_VERSION,
)


DISPLAY_POTHOLE_FEATURES = [
    "past_potholes_30d",
    "past_potholes_90d",
    "past_potholes_total",
]


DISPLAY_OTHER_FEATURES = [
    "days_since_last_repair",
    "freeze_thaw_7d",
    "freeze_thaw",
    "precip_7d",
    "precip_3d",
    "precipitation",
    "temp_range",
    "snowfall",
]


def percentile_score(
    series: pd.Series,
) -> pd.Series:
    values = pd.to_numeric(
        series,
        errors="coerce",
    )

    if (
        values.notna().sum()
        == 0
    ):
        return pd.Series(
            0.0,
            index=series.index,
            dtype=float,
        )

    if (
        values.nunique(
            dropna=True
        )
        <= 1
    ):
        return pd.Series(
            0.0,
            index=series.index,
            dtype=float,
        )

    values = values.fillna(
        values.median()
    )

    ranks = values.rank(
        method="average",
        pct=True,
    )

    min_rank = float(
        ranks.min()
    )

    max_rank = float(
        ranks.max()
    )

    if (
        max_rank
        <= min_rank
    ):
        return pd.Series(
            0.0,
            index=series.index,
            dtype=float,
        )

    return (
        ranks
        - min_rank
    ) / (
        max_rank
        - min_rank
    )


def load_road_attributes() -> pd.DataFrame:
    attr = pd.read_csv(
        "data/grid_road_attributes_500m.csv",
        dtype={
            "grid_id": str,
            "road_rank": str,
            "road_type": str,
        },
    )

    attr[
        "lanes"
    ] = pd.to_numeric(
        attr[
            "lanes"
        ],
        errors="coerce",
    )

    return attr


def lookup_category_lift(
    value,
    stats: dict,
) -> float:
    categories = stats.get(
        "categories",
        {},
    )

    key = str(
        value
    )

    if key in categories:
        return float(
            categories[
                key
            ].get(
                "lift",
                1.0,
            )
        )

    return 1.0


def normalize_lift_series(
    series: pd.Series,
) -> pd.Series:
    values = pd.to_numeric(
        series,
        errors="coerce",
    ).fillna(1.0)

    if (
        values.nunique()
        <= 1
    ):
        return pd.Series(
            0.5,
            index=series.index,
            dtype=float,
        )

    min_value = float(
        values.min()
    )

    max_value = float(
        values.max()
    )

    if (
        max_value
        <= min_value
    ):
        return pd.Series(
            0.5,
            index=series.index,
            dtype=float,
        )

    return (
        values
        - min_value
    ) / (
        max_value
        - min_value
    )


def calculate_road_structure_score(
    latest: pd.DataFrame,
    bundle: dict,
) -> pd.DataFrame:
    stats = bundle.get(
        "road_structure_statistics"
    )

    if not stats:
        raise ValueError(
            "모델 bundle에 도로 구조 통계가 없습니다. "
            "python -m src.train을 다시 실행하십시오."
        )

    attributes = (
        load_road_attributes()
    )

    result = latest[
        [
            "grid_id"
        ]
    ].merge(
        attributes[
            [
                "grid_id",
                "lanes",
                "road_rank",
            ]
        ],
        on="grid_id",
        how="left",
    )

    lanes_stats = stats[
        "lanes"
    ]

    rank_stats = stats[
        "road_rank"
    ]

    result[
        "lanes_lift"
    ] = result[
        "lanes"
    ].map(
        lambda value:
            lookup_category_lift(
                value,
                lanes_stats,
            )
    )

    result[
        "road_rank_lift"
    ] = result[
        "road_rank"
    ].map(
        lambda value:
            lookup_category_lift(
                value,
                rank_stats,
            )
    )

    result[
        "lanes_risk_component"
    ] = normalize_lift_series(
        result[
            "lanes_lift"
        ]
    )

    result[
        "road_rank_risk_component"
    ] = normalize_lift_series(
        result[
            "road_rank_lift"
        ]
    )

    lanes_weight = float(
        stats.get(
            "lanes_weight",
            0.20,
        )
    )

    rank_weight = float(
        stats.get(
            "road_rank_weight",
            0.80,
        )
    )

    denominator = (
        lanes_weight
        + rank_weight
    )

    if denominator <= 0:
        lanes_weight = 0.20
        rank_weight = 0.80
        denominator = 1.0

    lanes_weight /= (
        denominator
    )

    rank_weight /= (
        denominator
    )

    result[
        "road_structure_score"
    ] = (
        result[
            "lanes_risk_component"
        ]
        * lanes_weight
        + result[
            "road_rank_risk_component"
        ]
        * rank_weight
    )

    return result[
        [
            "grid_id",
            "lanes",
            "road_rank",
            "lanes_lift",
            "road_rank_lift",
            "road_structure_score",
        ]
    ]


def calculate_ai_road_risk_score(
    latest: pd.DataFrame,
    model,
    feature_columns: list[str],
) -> tuple[
    pd.Series,
    pd.DataFrame,
]:
    importances = np.asarray(
        model.feature_importances_,
        dtype=float,
    )

    importance_map = {
        feature:
            float(
                importance
            )
        for (
            feature,
            importance,
        ) in zip(
            feature_columns,
            importances,
        )
    }

    components = pd.DataFrame(
        index=latest.index
    )

    available_pothole_features = [
        feature
        for feature
        in DISPLAY_POTHOLE_FEATURES
        if (
            feature in latest.columns
            and feature
            in importance_map
            and latest[
                feature
            ].notna().any()
        )
    ]

    pothole_parts = []

    for feature in (
        available_pothole_features
    ):
        component = percentile_score(
            latest[
                feature
            ]
        )

        pothole_parts.append(
            component
        )

    if pothole_parts:
        pothole_history_score = (
            pd.concat(
                pothole_parts,
                axis=1,
            )
            .mean(
                axis=1
            )
        )

        pothole_group_importance = max(
            importance_map.get(
                feature,
                0.0,
            )
            for feature
            in available_pothole_features
        )

    else:
        pothole_history_score = pd.Series(
            0.0,
            index=latest.index,
            dtype=float,
        )

        pothole_group_importance = 0.0

    available_other_features = [
        feature
        for feature
        in DISPLAY_OTHER_FEATURES
        if (
            feature
            in latest.columns
            and feature
            in importance_map
            and latest[
                feature
            ].notna().any()
        )
    ]

    group_weights = {
        "pothole_history":
            max(
                pothole_group_importance,
                0.0,
            )
    }

    for feature in (
        available_other_features
    ):
        group_weights[
            feature
        ] = max(
            importance_map.get(
                feature,
                0.0,
            ),
            0.0,
        )

    total_weight = sum(
        group_weights.values()
    )

    if (
        total_weight
        <= 0
    ):
        group_weights = {
            key: 1.0
            for key
            in group_weights
        }

        total_weight = float(
            len(
                group_weights
            )
        )

    group_weights = {
        key:
            value
            / total_weight
        for (
            key,
            value,
        ) in group_weights.items()
    }

    feature_score = (
        pothole_history_score
        * group_weights.get(
            "pothole_history",
            0.0,
        )
    )

    for feature in (
        available_other_features
    ):
        component = percentile_score(
            latest[
                feature
            ]
        )

        feature_score += (
            component
            * group_weights[
                feature
            ]
        )

    model_component = (
        percentile_score(
            latest[
                "risk_score"
            ]
        )
    )

    structure_component = (
        pd.to_numeric(
            latest[
                "road_structure_score"
            ],
            errors="coerce",
        )
        .fillna(0.5)
        .clip(
            lower=0.0,
            upper=1.0,
        )
    )

    # 최종 구성
    #
    # feature 종합지수 65%
    # XGBoost 상대위험 20%
    # 도로 구조 위험 15%
    #
    # 도로 구조는 실제 발생률 기반 smoothing 결과를 사용
    combined_score = (
        feature_score
        * 0.65
        + model_component
        * 0.20
        + structure_component
        * 0.15
    )

    components[
        "display_feature_score"
    ] = feature_score

    components[
        "display_model_score"
    ] = model_component

    components[
        "display_road_structure_score"
    ] = structure_component

    components[
        "display_combined_score"
    ] = combined_score

    if (
        combined_score.nunique(
            dropna=True
        )
        <= 1
    ):
        final_score = pd.Series(
            50.0,
            index=latest.index,
            dtype=float,
        )

    else:
        final_rank = (
            combined_score
            .rank(
                method="average",
                pct=True,
            )
        )

        min_rank = float(
            final_rank.min()
        )

        max_rank = float(
            final_rank.max()
        )

        if (
            max_rank
            > min_rank
        ):
            final_score = (
                (
                    final_rank
                    - min_rank
                )
                / (
                    max_rank
                    - min_rank
                )
                * 100.0
            )

        else:
            final_score = pd.Series(
                50.0,
                index=latest.index,
                dtype=float,
            )

    final_score = (
        final_score
        .clip(
            lower=0.0,
            upper=100.0,
        )
        .round(1)
    )

    print()
    print(
        "=== 화면용 AI 위험점수 구성 ==="
    )

    print(
        "기존 feature 종합지수 : 65%"
    )

    print(
        "XGBoost 상대위험      : 20%"
    )

    print(
        "도로 구조 위험        : 15%"
    )

    print()

    print(
        "도로 구조 내부 비율:"
    )

    print(
        " - 도로등급: 80%"
    )

    print(
        " - 차로수  : 20%"
    )

    return (
        final_score,
        components,
    )


def main(
    config_path: str,
    prediction_date: str | None = None,
) -> None:
    config = load_config(
        config_path
    )

    model_path = resolve_path(
        config,
        "model",
    )

    if not model_path.exists():
        raise FileNotFoundError(
            "학습 모델이 없습니다. "
            "먼저 python -m src.train을 실행하십시오."
        )

    bundle = joblib.load(
        model_path
    )

    if (
        bundle.get(
            "schema_version"
        )
        != MODEL_SCHEMA_VERSION
    ):
        raise ValueError(
            "기존 모델 형식입니다. "
            "python -m src.train을 다시 실행하십시오."
        )

    history = pd.read_csv(
        resolve_path(
            config,
            "weather_history",
        )
    )

    forecast = pd.read_csv(
        resolve_path(
            config,
            "weather_forecast",
        )
    )

    combined = pd.concat(
        [
            history,
            forecast,
        ],
        ignore_index=True,
    )

    combined[
        "date"
    ] = pd.to_datetime(
        combined[
            "date"
        ],
        errors="coerce",
    )

    combined = (
        combined
        .dropna(
            subset=[
                "date"
            ]
        )
        .sort_values(
            [
                "station_id",
                "date",
            ]
        )
        .drop_duplicates(
            [
                "station_id",
                "date",
            ],
            keep="last",
        )
    )

    temp_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".csv",
            delete=False,
        ) as temp_file:
            temp_path = Path(
                temp_file.name
            )

        combined.to_csv(
            temp_path,
            index=False,
        )

        available_dates = (
            pd.to_datetime(
                forecast[
                    "date"
                ],
                errors="coerce",
            )
            .dropna()
        )

        if (
            available_dates.empty
        ):
            raise ValueError(
                "weather_forecast.csv에 "
                "유효한 예보 날짜가 없습니다."
            )

        target_date = (
            pd.Timestamp(
                prediction_date
            ).normalize()
            if prediction_date
            else (
                available_dates
                .max()
                .normalize()
            )
        )

        prepared = prepare_dataset(
            pothole_path=resolve_path(
                config,
                "potholes",
            ),
            repair_path=resolve_path(
                config,
                "repairs",
            ),
            road_path=resolve_path(
                config,
                "roads",
            ),
            weather_path=temp_path,
            grid_size_m=int(
                bundle[
                    "grid_size_m"
                ]
            ),
            start_date=None,
            end_date=str(
                target_date.date()
            ),
            include_target=False,
            target_horizon_days=int(
                bundle[
                    "target_horizon_days"
                ]
            ),
        )

        latest = (
            prepared.panel.loc[
                prepared.panel[
                    "date"
                ].eq(
                    target_date
                )
            ]
            .copy()
        )

        if latest.empty:
            raise ValueError(
                "예측 대상 날짜 데이터가 없습니다: "
                f"{target_date.date()}"
            )

        feature_columns = (
            bundle[
                "features"
            ]
        )

        matrix, _ = (
            prepare_feature_matrix(
                latest,
                feature_columns,
                bundle[
                    "feature_medians"
                ],
            )
        )

        model = (
            bundle[
                "model"
            ]
        )

        latest[
            "risk_score"
        ] = (
            model.predict_proba(
                matrix
            )[:, 1]
        )

        latest[
            "risk_percentile"
        ] = (
            latest[
                "risk_score"
            ]
            .rank(
                method="average",
                pct=True,
            )
        )

        latest[
            "predicted_label"
        ] = (
            latest[
                "risk_score"
            ]
            >= float(
                bundle[
                    "classification_threshold"
                ]
            )
        ).astype(int)

        latest[
            "risk_reason"
        ] = (
            explain_with_contributions(
                model,
                matrix,
                latest,
            )
        )

        priority = (
            calculate_priority_components(
                latest,
                latest[
                    "risk_score"
                ],
                bundle[
                    "recurrence_scales"
                ],
                bundle[
                    "importance_scales"
                ],
            )
        )

        for column in (
            priority.columns
        ):
            latest[
                column
            ] = priority[
                column
            ]

        structure = (
            calculate_road_structure_score(
                latest,
                bundle,
            )
        )

        latest = (
            latest
            .merge(
                structure,
                on="grid_id",
                how="left",
            )
        )

        (
            road_risk_score,
            display_components,
        ) = (
            calculate_ai_road_risk_score(
                latest,
                model,
                feature_columns,
            )
        )

        latest[
            "road_risk_score"
        ] = (
            road_risk_score
        )

        latest[
            "road_risk_percentile"
        ] = (
            latest[
                "road_risk_score"
            ]
            / 100.0
        )

        latest[
            "risk_level"
        ] = (
            latest[
                "road_risk_percentile"
            ]
            .map(
                risk_level_from_percentile
            )
        )

        latest[
            "prediction_date"
        ] = (
            target_date
            .date()
            .isoformat()
        )

        latest = (
            latest
            .sort_values(
                [
                    "road_risk_score",
                    "priority_score",
                    "risk_score",
                    "grid_id",
                ],
                ascending=[
                    False,
                    False,
                    False,
                    True,
                ],
                kind="mergesort",
            )
            .reset_index(
                drop=True
            )
        )

        latest[
            "priority_rank"
        ] = (
            latest.index
            + 1
        )

        output_columns = [
            "prediction_date",

            "grid_id",
            "grid_lat",
            "grid_lon",

            "risk_score",
            "risk_percentile",

            "road_risk_score",
            "road_risk_percentile",

            "risk_level",
            "predicted_label",
            "risk_reason",

            "priority_score",
            "priority_rank",
            "recurrence_score",

            "lanes",
            "road_rank",
            "lanes_lift",
            "road_rank_lift",
            "road_structure_score",

            "avg_temp",
            "min_temp",
            "max_temp",
            "temp_range",

            "precipitation",
            "precip_3d",
            "precip_7d",

            "snowfall",
            "humidity",

            "freeze_thaw",
            "freeze_thaw_7d",

            "past_potholes_30d",
            "past_potholes_90d",
            "past_potholes_total",

            "has_repair_history",
            "days_since_last_repair",

            "month",
            "day_of_year_sin",
            "day_of_year_cos",

        ] + [
            column
            for column
            in OPTIONAL_FEATURE_COLUMNS
            if column
            in latest.columns
        ]

        output_path = (
            resolve_path(
                config,
                "predictions",
            )
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        latest[
            output_columns
        ].to_csv(
            output_path,
            index=False,
            float_format="%.8f",
        )

        print()

        print(
            f"예측 저장 완료: "
            f"{output_path}"
        )

        print()

        print(
            "=== AI 도로 위험점수 ==="
        )

        print(
            "고유값 수:",
            latest[
                "road_risk_score"
            ].nunique(),
        )

        print(
            latest[
                "road_risk_score"
            ].describe()
        )

        print()

        print(
            "=== 위험등급 분포 ==="
        )

        print(
            latest[
                "risk_level"
            ].value_counts()
        )

        print()

        print(
            "=== 도로구조 점수 ==="
        )

        print(
            latest[
                "road_structure_score"
            ].describe()
        )

        print()

        print(
            "=== 상위 20개 ==="
        )

        print(
            latest[
                [
                    "grid_id",
                    "road_risk_score",
                    "risk_score",
                    "road_structure_score",
                    "lanes",
                    "road_rank",
                    "lanes_lift",
                    "road_rank_lift",
                    "past_potholes_total",
                    "days_since_last_repair",
                    "precip_7d",
                    "freeze_thaw_7d",
                ]
            ]
            .head(20)
            .to_string(
                index=False
            )
        )

    finally:
        if (
            temp_path
            and temp_path.exists()
        ):
            temp_path.unlink()


if __name__ == "__main__":
    parser = (
        argparse
        .ArgumentParser()
    )

    parser.add_argument(
        "--config",
        default="config.yaml",
    )

    parser.add_argument(
        "--date",
        default=None,
        help="YYYY-MM-DD",
    )

    args = (
        parser.parse_args()
    )

    main(
        args.config,
        args.date,
    )