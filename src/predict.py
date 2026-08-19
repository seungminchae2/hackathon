from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import joblib
import pandas as pd

from .common import load_config, resolve_path
from .features import (
    OPTIONAL_FEATURE_COLUMNS,
    calculate_priority_components,
    explain_with_contributions,
    prepare_dataset,
    prepare_feature_matrix,
    risk_level_from_percentile,
)
from .train import MODEL_SCHEMA_VERSION


def main(config_path: str, prediction_date: str | None = None) -> None:
    config = load_config(config_path)
    model_path = resolve_path(config, "model")
    if not model_path.exists():
        raise FileNotFoundError("학습 모델이 없습니다. 먼저 python -m src.train을 실행하십시오.")
    bundle = joblib.load(model_path)
    if bundle.get("schema_version") != MODEL_SCHEMA_VERSION:
        raise ValueError("기존 모델 형식입니다. 새 구조로 python -m src.train을 다시 실행하십시오.")

    history = pd.read_csv(resolve_path(config, "weather_history"))
    forecast = pd.read_csv(resolve_path(config, "weather_forecast"))
    combined = pd.concat([history, forecast], ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
    combined = (
        combined.dropna(subset=["date"])
        .sort_values(["station_id", "date"])
        .drop_duplicates(["station_id", "date"], keep="last")
    )

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temp_file:
            temp_path = Path(temp_file.name)
        combined.to_csv(temp_path, index=False)

        available_dates = pd.to_datetime(forecast["date"], errors="coerce").dropna()
        target_date = (
            pd.Timestamp(prediction_date).normalize()
            if prediction_date
            else available_dates.max().normalize()
        )
        prepared = prepare_dataset(
            pothole_path=resolve_path(config, "potholes"),
            repair_path=resolve_path(config, "repairs"),
            road_path=resolve_path(config, "roads"),
            weather_path=temp_path,
            grid_size_m=int(bundle["grid_size_m"]),
            start_date=None,
            end_date=str(target_date.date()),
            include_target=False,
            target_horizon_days=int(bundle["target_horizon_days"]),
        )
        latest = prepared.panel.loc[prepared.panel["date"].eq(target_date)].copy()
        if latest.empty:
            raise ValueError(f"예측 대상 날짜 데이터가 없습니다: {target_date.date()}")

        feature_columns = bundle["features"]
        matrix, _ = prepare_feature_matrix(latest, feature_columns, bundle["feature_medians"])
        model = bundle["model"]
        latest["risk_score"] = model.predict_proba(matrix)[:, 1]
        latest["risk_percentile"] = latest["risk_score"].rank(method="average", pct=True)
        latest["risk_level"] = latest["risk_percentile"].map(risk_level_from_percentile)
        latest["predicted_label"] = (
            latest["risk_score"] >= float(bundle["classification_threshold"])
        ).astype(int)
        latest["risk_reason"] = explain_with_contributions(model, matrix, latest)

        priority = calculate_priority_components(
            latest,
            latest["risk_score"],
            bundle["recurrence_scales"],
            bundle["importance_scales"],
        )
        for column in priority.columns:
            latest[column] = priority[column]

        latest["prediction_date"] = target_date.date().isoformat()
        latest = latest.sort_values(
            ["priority_score", "risk_score", "recurrence_score", "grid_id"],
            ascending=[False, False, False, True],
            kind="mergesort",
        ).reset_index(drop=True)
        latest["priority_rank"] = latest.index + 1

        output_columns = [
            "prediction_date",
            "grid_id",
            "grid_lat",
            "grid_lon",
            "risk_score",
            "risk_percentile",
            "risk_level",
            "predicted_label",
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
        ] + [column for column in OPTIONAL_FEATURE_COLUMNS if column in latest.columns]

        output_path = resolve_path(config, "predictions")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        latest[output_columns].to_csv(output_path, index=False, float_format="%.8f")
        print(f"예측 저장 완료: {output_path}")
        print(latest[output_columns].head(10).to_string(index=False))
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--date", default=None, help="YYYY-MM-DD")
    args = parser.parse_args()
    main(args.config, args.date)
