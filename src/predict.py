from __future__ import annotations

import argparse

import joblib
import pandas as pd

from .common import load_config, resolve_path
from .features import FEATURE_COLUMNS, explain_risk, prepare_dataset, risk_level


def main(config_path: str, prediction_date: str | None = None) -> None:
    config = load_config(config_path)
    model_path = resolve_path(config, "model")
    if not model_path.exists():
        raise FileNotFoundError("학습 모델이 없습니다. 먼저 python -m src.train을 실행하십시오.")
    bundle = joblib.load(model_path)

    forecast_path = resolve_path(config, "weather_forecast")
    history_path = resolve_path(config, "weather_history")

    history = pd.read_csv(history_path)
    forecast = pd.read_csv(forecast_path)
    combined_weather_path = history_path.parent / "_combined_weather_for_prediction.csv"
    combined = pd.concat([history, forecast], ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
    combined = combined.dropna(subset=["date"])
    combined = combined.sort_values(["station_id", "date"]).drop_duplicates(
        ["station_id", "date"], keep="last"
    )
    combined.to_csv(combined_weather_path, index=False)

    try:
        available_dates = pd.to_datetime(forecast["date"], errors="coerce").dropna()
        target_date = pd.Timestamp(prediction_date) if prediction_date else available_dates.max().normalize()

        prepared = prepare_dataset(
            pothole_path=resolve_path(config, "potholes"),
            repair_path=resolve_path(config, "repairs"),
            road_path=resolve_path(config, "roads"),
            weather_path=combined_weather_path,
            grid_size_m=int(bundle["grid_size_m"]),
            start_date=None,
            end_date=str(target_date.date()),
            include_target=False,
        )

        latest = prepared.panel[prepared.panel["date"] == target_date].copy()
        if latest.empty:
            raise ValueError(f"예측 대상 날짜 데이터가 없습니다: {target_date.date()}")

        model = bundle["model"]
        latest["risk_score"] = model.predict_proba(latest[FEATURE_COLUMNS])[:, 1]
        latest["risk_level"] = latest["risk_score"].map(risk_level)
        latest["risk_reason"] = latest.apply(explain_risk, axis=1)
        latest = latest.sort_values("risk_score", ascending=False).reset_index(drop=True)
        latest["priority_rank"] = latest.index + 1
        latest["prediction_date"] = target_date.date().isoformat()

        output_columns = [
            "prediction_date",
            "grid_id",
            "grid_lat",
            "grid_lon",
            "risk_score",
            "risk_level",
            "risk_reason",
            "priority_rank",
            "precip_3d",
            "precip_7d",
            "freeze_thaw_7d",
            "past_potholes_90d",
            "days_since_last_repair",
        ]
        output_path = resolve_path(config, "predictions")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        latest[output_columns].to_csv(output_path, index=False)
        print(f"예측 저장 완료: {output_path}")
        print(latest[output_columns].head(10).to_string(index=False))
    finally:
        if combined_weather_path.exists():
            combined_weather_path.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--date", default=None, help="YYYY-MM-DD")
    args = parser.parse_args()
    main(args.config, args.date)
