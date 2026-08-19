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

    if values.notna().sum() == 0:
        return pd.Series(
            0.0,
            index=series.index,
            dtype=float,
        )

    if values.nunique(dropna=True) <= 1:
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

    if max_rank <= min_rank:
        return pd.Series(
            0.0,
            index=series.index,
            dtype=float,
        )

    return (
        ranks - min_rank
    ) / (
        max_rank - min_rank
    )


def road_risk_level_from_score(
    score: float,
) -> str:
    if score >= 60:
        return "매우 높음"

    if score >= 30:
        return "높음"

    if score >= 20:
        return "보통"

    return "낮음"


def absolute_risk_level_from_score(
    score: float,
) -> str:
    if score >= 80:
        return "매우 높음"

    if score >= 60:
        return "높음"

    if score >= 40:
        return "보통"

    return "낮음"


def load_road_attributes() -> pd.DataFrame:
    attr = pd.read_csv(
        "data/grid_road_attributes_500m.csv",
        dtype={
            "grid_id": str,
            "road_rank": str,
            "road_type": str,
        },
    )

    attr["lanes"] = pd.to_numeric(
        attr["lanes"],
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

    key = str(value)

    if key in categories:
        return float(
            categories[key].get(
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

    if values.nunique() <= 1:
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

    if max_value <= min_value:
        return pd.Series(
            0.5,
            index=series.index,
            dtype=float,
        )

    return (
        values - min_value
    ) / (
        max_value - min_value
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

    attributes = load_road_attributes()

    latest = latest.copy()

    latest["grid_id"] = (
        latest["grid_id"]
        .astype(str)
    )

    attributes["grid_id"] = (
        attributes["grid_id"]
        .astype(str)
    )

    columns_to_add = [
        "grid_id",
        "lanes",
        "road_rank",
        "road_type",
    ]

    existing_columns = [
        column
        for column in [
            "lanes",
            "road_rank",
            "road_type",
        ]
        if column in latest.columns
    ]

    if existing_columns:
        latest = latest.drop(
            columns=existing_columns
        )

    latest = latest.merge(
        attributes[columns_to_add],
        on="grid_id",
        how="left",
    )

    lanes_stats = stats.get(
        "lanes",
        {},
    )

    road_rank_stats = stats.get(
        "road_rank",
        {},
    )

    latest["lanes_lift"] = (
        latest["lanes"]
        .apply(
            lambda value: lookup_category_lift(
                value,
                lanes_stats,
            )
        )
    )

    latest["road_rank_lift"] = (
        latest["road_rank"]
        .apply(
            lambda value: lookup_category_lift(
                value,
                road_rank_stats,
            )
        )
    )

    latest["lanes_structure_component"] = (
        normalize_lift_series(
            latest["lanes_lift"]
        )
    )

    latest["road_rank_structure_component"] = (
        normalize_lift_series(
            latest["road_rank_lift"]
        )
    )

    latest["road_structure_score"] = (
        0.20
        * latest[
            "lanes_structure_component"
        ]
        + 0.80
        * latest[
            "road_rank_structure_component"
        ]
    )

    latest["road_structure_score"] = (
        latest["road_structure_score"]
        .clip(0.0, 1.0)
    )

    return latest


def build_pothole_history_component(
    latest: pd.DataFrame,
) -> pd.Series:
    components = []

    for feature in DISPLAY_POTHOLE_FEATURES:
        if feature not in latest.columns:
            continue

        component = percentile_score(
            latest[feature]
        )

        components.append(
            component
        )

    if not components:
        return pd.Series(
            0.0,
            index=latest.index,
            dtype=float,
        )

    matrix = pd.concat(
        components,
        axis=1,
    )

    return matrix.max(
        axis=1
    )


def build_feature_display_score(
    latest: pd.DataFrame,
    model,
    feature_columns: list[str],
) -> tuple[pd.Series, dict[str, float]]:
    importances = np.asarray(
        model.feature_importances_,
        dtype=float,
    )

    importance_map = {
        feature: float(importance)
        for feature, importance in zip(
            feature_columns,
            importances,
        )
    }

    pothole_importance = sum(
        importance_map.get(
            feature,
            0.0,
        )
        for feature in DISPLAY_POTHOLE_FEATURES
    )

    group_importances = {
        "pothole_history": pothole_importance,
    }

    for feature in DISPLAY_OTHER_FEATURES:
        if feature in latest.columns:
            group_importances[feature] = (
                importance_map.get(
                    feature,
                    0.0,
                )
            )

    positive_total = sum(
        max(
            value,
            0.0,
        )
        for value in group_importances.values()
    )

    if positive_total <= 0:
        weights = {
            key: 1.0 / len(group_importances)
            for key in group_importances
        }
    else:
        weights = {
            key: max(value, 0.0)
            / positive_total
            for key, value
            in group_importances.items()
        }

    feature_score = pd.Series(
        0.0,
        index=latest.index,
        dtype=float,
    )

    pothole_component = (
        build_pothole_history_component(
            latest
        )
    )

    feature_score += (
        weights.get(
            "pothole_history",
            0.0,
        )
        * pothole_component
    )

    for feature in DISPLAY_OTHER_FEATURES:
        if feature not in latest.columns:
            continue

        component = percentile_score(
            latest[feature]
        )

        feature_score += (
            weights.get(
                feature,
                0.0,
            )
            * component
        )

    return (
        feature_score.clip(
            0.0,
            1.0,
        ),
        weights,
    )


def calculate_relative_model_score(
    risk_score: pd.Series,
) -> pd.Series:
    return percentile_score(
        risk_score
    )


def calculate_absolute_feature_component(
    latest: pd.DataFrame,
) -> pd.Series:
    pothole_30 = pd.to_numeric(
        latest.get(
            "past_potholes_30d",
            0,
        ),
        errors="coerce",
    ).fillna(0)

    pothole_90 = pd.to_numeric(
        latest.get(
            "past_potholes_90d",
            0,
        ),
        errors="coerce",
    ).fillna(0)

    pothole_total = pd.to_numeric(
        latest.get(
            "past_potholes_total",
            0,
        ),
        errors="coerce",
    ).fillna(0)

    repair_days = pd.to_numeric(
        latest.get(
            "days_since_last_repair",
            0,
        ),
        errors="coerce",
    ).fillna(0)

    freeze = pd.to_numeric(
        latest.get(
            "freeze_thaw_7d",
            0,
        ),
        errors="coerce",
    ).fillna(0)

    precip_7d = pd.to_numeric(
        latest.get(
            "precip_7d",
            0,
        ),
        errors="coerce",
    ).fillna(0)

    snowfall = pd.to_numeric(
        latest.get(
            "snowfall",
            0,
        ),
        errors="coerce",
    ).fillna(0)

    pothole_component = np.maximum.reduce(
        [
            np.clip(
                pothole_30 / 1.0,
                0,
                1,
            ),
            np.clip(
                pothole_90 / 1.0,
                0,
                1,
            )
            * 0.85,
            np.clip(
                pothole_total / 3.0,
                0,
                1,
            )
            * 0.70,
        ]
    )

    repair_component = np.clip(
        repair_days / 365.0,
        0,
        1,
    )

    freeze_component = np.clip(
        freeze / 5.0,
        0,
        1,
    )

    precip_component = np.clip(
        precip_7d / 80.0,
        0,
        1,
    )

    snow_component = np.clip(
        snowfall / 10.0,
        0,
        1,
    )

    result = (
        0.35 * pothole_component
        + 0.20 * repair_component
        + 0.25 * freeze_component
        + 0.15 * precip_component
        + 0.05 * snow_component
    )

    return pd.Series(
        np.clip(
            result,
            0,
            1,
        ),
        index=latest.index,
    )


def build_risk_trigger(
    row: pd.Series,
) -> str:
    triggers = []

    recent30 = float(
        row.get(
            "past_potholes_30d",
            0,
        )
        or 0
    )

    recent90 = float(
        row.get(
            "past_potholes_90d",
            0,
        )
        or 0
    )

    total = float(
        row.get(
            "past_potholes_total",
            0,
        )
        or 0
    )

    repair_days = float(
        row.get(
            "days_since_last_repair",
            0,
        )
        or 0
    )

    freeze = float(
        row.get(
            "freeze_thaw_7d",
            0,
        )
        or 0
    )

    if (
        recent30 >= 1
        and freeze >= 3
    ):
        triggers.append(
            "최근30일 포트홀+동결융해 3회 이상"
        )

    elif (
        recent90 >= 1
        and freeze >= 3
    ):
        triggers.append(
            "최근90일 포트홀+동결융해 3회 이상"
        )

    elif (
        total >= 1
        and freeze >= 3
    ):
        triggers.append(
            "과거 포트홀 이력+동결융해 3회 이상"
        )

    if (
        total >= 1
        and repair_days >= 365
    ):
        triggers.append(
            "보수 후 365일 이상"
        )

    if not triggers:
        return "없음"

    return "; ".join(
        triggers
    )


def is_strong_trigger(
    trigger: str,
) -> bool:
    trigger = str(trigger)

    return (
        "최근30일 포트홀+동결융해 3회 이상"
        in trigger
        or
        "최근90일 포트홀+동결융해 3회 이상"
        in trigger
    )


def determine_action_level(
    row: pd.Series,
) -> str:
    absolute_score = float(
        row.get(
            "absolute_risk_score",
            0,
        )
    )

    relative_top = float(
        row.get(
            "relative_top_percent",
            100,
        )
    )

    trigger = str(
        row.get(
            "risk_trigger",
            "없음",
        )
    )

    has_trigger = (
        trigger != "없음"
    )

    strong_trigger = (
        is_strong_trigger(
            trigger
        )
    )

    if (
        absolute_score >= 80
        and strong_trigger
        and relative_top <= 5
    ):
        return "예방보수"

    if (
        absolute_score >= 60
        or strong_trigger
    ):
        return "긴급점검"

    if (
        absolute_score >= 40
        or relative_top <= 5
        or has_trigger
    ):
        return "우선점검"

    return "모니터링"


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

    combined["date"] = pd.to_datetime(
        combined["date"],
        errors="coerce",
    )

    combined = (
        combined
        .dropna(
            subset=["date"]
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

        available_dates = pd.to_datetime(
            forecast["date"],
            errors="coerce",
        ).dropna()

        if available_dates.empty:
            raise ValueError(
                "weather_forecast.csv에 유효한 예보 날짜가 없습니다."
            )

        target_date = (
            pd.Timestamp(
                prediction_date
            ).normalize()
            if prediction_date
            else available_dates.max().normalize()
        )

        if target_date not in set(
            available_dates.dt.normalize()
        ):
            raise ValueError(
                f"예측 요청 날짜 {target_date.date()}가 "
                "weather_forecast.csv에 없습니다."
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

        latest = prepared.panel.loc[
            prepared.panel[
                "date"
            ].eq(
                target_date
            )
        ].copy()

        if latest.empty:
            raise ValueError(
                "예측 대상 날짜 데이터가 없습니다: "
                f"{target_date.date()}"
            )

        feature_columns = bundle[
            "features"
        ]

        matrix, _ = prepare_feature_matrix(
            latest,
            feature_columns,
            bundle[
                "feature_medians"
            ],
        )

        model = bundle[
            "model"
        ]

        latest["risk_score"] = (
            model.predict_proba(
                matrix
            )[:, 1]
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

        latest = (
            calculate_road_structure_score(
                latest,
                bundle,
            )
        )

        (
            feature_display_score,
            display_weights,
        ) = build_feature_display_score(
            latest,
            model,
            feature_columns,
        )

        latest[
            "feature_risk_score"
        ] = (
            feature_display_score
        )

        latest[
            "model_relative_score"
        ] = (
            calculate_relative_model_score(
                latest[
                    "risk_score"
                ]
            )
        )

        latest[
            "relative_ai_index"
        ] = (
            0.65
            * latest[
                "feature_risk_score"
            ]
            + 0.20
            * latest[
                "model_relative_score"
            ]
            + 0.15
            * latest[
                "road_structure_score"
            ]
        ).clip(
            0,
            1,
        )

        latest[
            "road_risk_score"
        ] = (
            100
            * np.sqrt(
                latest[
                    "relative_ai_index"
                ]
            )
        ).clip(
            0,
            100,
        ).round(1)

        descending_rank = (
            latest[
                "road_risk_score"
            ]
            .rank(
                method="min",
                ascending=False,
            )
        )

        latest[
            "relative_top_percent"
        ] = (
            descending_rank
            / len(latest)
            * 100
        ).round(3)

        latest[
            "risk_percentile"
        ] = (
            latest[
                "road_risk_score"
            ]
            .rank(
                method="average",
                pct=True,
            )
        )

        latest[
            "risk_level"
        ] = (
            latest[
                "road_risk_score"
            ]
            .apply(
                road_risk_level_from_score
            )
        )

        latest[
            "absolute_feature_score"
        ] = (
            calculate_absolute_feature_component(
                latest
            )
        )

        model_absolute_signal = (
            latest[
                "risk_score"
            ]
            .clip(
                0,
                1,
            )
        )

        latest[
            "absolute_risk_index"
        ] = (
            0.75
            * latest[
                "absolute_feature_score"
            ]
            + 0.15
            * latest[
                "road_structure_score"
            ]
            + 0.10
            * model_absolute_signal
        ).clip(
            0,
            1,
        )

        latest[
            "absolute_risk_score"
        ] = (
            latest[
                "absolute_risk_index"
            ]
            * 100
        ).round(1)

        latest[
            "absolute_risk_level"
        ] = (
            latest[
                "absolute_risk_score"
            ]
            .apply(
                absolute_risk_level_from_score
            )
        )

        latest[
            "risk_trigger"
        ] = (
            latest.apply(
                build_risk_trigger,
                axis=1,
            )
        )

        latest[
            "action_level"
        ] = (
            latest.apply(
                determine_action_level,
                axis=1,
            )
        )

        latest[
            "preventive_repair_candidate"
        ] = (
            latest[
                "action_level"
            ]
            .eq(
                "예방보수"
            )
            .astype(int)
        )

        latest[
            "risk_reason"
        ] = explain_with_contributions(
            model,
            matrix,
            latest,
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

        for column in priority.columns:
            latest[column] = (
                priority[column]
            )

        latest[
            "prediction_date"
        ] = (
            target_date
            .date()
            .isoformat()
        )

        action_priority = {
            "예방보수": 4,
            "긴급점검": 3,
            "우선점검": 2,
            "모니터링": 1,
        }

        latest[
            "_action_priority"
        ] = (
            latest[
                "action_level"
            ]
            .map(
                action_priority
            )
            .fillna(0)
        )

        latest = (
            latest
            .sort_values(
                [
                    "_action_priority",
                    "absolute_risk_score",
                    "road_risk_score",
                    "priority_score",
                    "grid_id",
                ],
                ascending=[
                    False,
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
            "action_rank"
        ] = (
            latest.index + 1
        )

        latest = latest.drop(
            columns=[
                "_action_priority"
            ]
        )

        priority_order = (
            latest[
                "priority_score"
            ]
            .rank(
                method="min",
                ascending=False,
            )
        )

        latest[
            "priority_rank"
        ] = (
            priority_order
            .astype(int)
        )

        output_columns = [
            "prediction_date",
            "grid_id",
            "grid_lat",
            "grid_lon",

            "road_risk_score",
            "risk_level",
            "risk_percentile",
            "relative_top_percent",

            "absolute_risk_score",
            "absolute_risk_level",

            "action_level",
            "action_rank",
            "preventive_repair_candidate",
            "risk_trigger",

            "risk_score",
            "predicted_label",

            "feature_risk_score",
            "model_relative_score",
            "relative_ai_index",

            "absolute_feature_score",
            "absolute_risk_index",

            "road_structure_score",
            "lanes",
            "road_rank",
            "road_type",
            "lanes_lift",
            "road_rank_lift",

            "risk_reason",

            "priority_score",
            "priority_rank",
            "recurrence_score",
            "importance_score",
            "priority_weight_risk",
            "priority_weight_recurrence",
            "priority_weight_importance",

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
        ]

        for column in OPTIONAL_FEATURE_COLUMNS:
            if (
                column in latest.columns
                and column not in output_columns
            ):
                output_columns.append(
                    column
                )

        output_columns = [
            column
            for column in output_columns
            if column in latest.columns
        ]

        # ==================================================
        # CSV 저장
        #
        # 1. 기존 predictions_v2.csv 유지
        # 2. 날짜별 predictions_YYYY-MM-DD.csv 추가 저장
        # ==================================================

        output_path = resolve_path(
            config,
            "predictions",
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_df = latest[
            output_columns
        ].copy()

        # 기존 Streamlit 호환용
        output_df.to_csv(
            output_path,
            index=False,
            float_format="%.8f",
        )

        # 날짜별 예측 파일
        date_string = (
            target_date
            .date()
            .isoformat()
        )

        dated_output_path = (
            output_path.parent
            / f"predictions_{date_string}.csv"
        )

        output_df.to_csv(
            dated_output_path,
            index=False,
            float_format="%.8f",
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

        print()
        print(
            "=== feature 그룹 가중치 ==="
        )

        for (
            feature,
            weight,
        ) in sorted(
            display_weights.items(),
            key=lambda item: item[1],
            reverse=True,
        ):
            print(
                f"{feature:<25}: "
                f"{weight * 100:6.2f}%"
            )

        print()
        print(
            f"기본 예측 저장 완료: "
            f"{output_path}"
        )

        print(
            f"날짜별 예측 저장 완료: "
            f"{dated_output_path}"
        )

        print()
        print(
            "=== 당일 상대 AI 위험점수 ==="
        )

        print(
            latest[
                "road_risk_score"
            ].describe()
        )

        print()
        print(
            "=== 절대 위험점수 ==="
        )

        print(
            latest[
                "absolute_risk_score"
            ].describe()
        )

        print()
        print(
            "=== 상대 위험등급 ==="
        )

        print(
            latest[
                "risk_level"
            ].value_counts()
        )

        print()
        print(
            "=== 절대 위험등급 ==="
        )

        print(
            latest[
                "absolute_risk_level"
            ].value_counts()
        )

        print()
        print(
            "=== 조치 단계 ==="
        )

        print(
            latest[
                "action_level"
            ].value_counts()
        )

        print()
        print(
            "=== 예방보수 후보 ==="
        )

        print(
            int(
                latest[
                    "preventive_repair_candidate"
                ].sum()
            )
        )

        print()
        print(
            "=== Trigger 현황 ==="
        )

        print(
            latest[
                "risk_trigger"
            ].value_counts()
        )

        print()
        print(
            "=== 조치 우선순위 상위 30개 ==="
        )

        debug_columns = [
            "action_rank",
            "grid_id",
            "action_level",
            "absolute_risk_score",
            "road_risk_score",
            "relative_top_percent",
            "risk_trigger",
            "past_potholes_30d",
            "past_potholes_90d",
            "past_potholes_total",
            "days_since_last_repair",
            "freeze_thaw_7d",
            "snowfall",
            "precip_7d",
            "road_structure_score",
        ]

        debug_columns = [
            column
            for column in debug_columns
            if column in latest.columns
        ]

        print(
            latest[
                debug_columns
            ]
            .head(30)
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
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="config.yaml",
    )

    parser.add_argument(
        "--date",
        default=None,
        help="YYYY-MM-DD",
    )

    args = parser.parse_args()

    main(
        args.config,
        args.date,
    )