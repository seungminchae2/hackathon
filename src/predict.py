from __future__ import annotations

import argparse

import joblib
import pandas as pd

from .common import load_config, resolve_path
from .features import FEATURE_COLUMNS, explain_risk, prepare_dataset, risk_level


def main(
    config_path: str,
    prediction_date: str | None = None,
    horizon: int = 3,
) -> None:
    config = load_config(config_path)
    model_path = resolve_path(config, "model")
    if not model_path.exists():
        raise FileNotFoundError(
            "학습 모델이 없습니다. 먼저 python -m src.train을 실행하십시오."
        )
    bundle = joblib.load(model_path)

    forecast_path = resolve_path(config, "weather_forecast")
    history_path = resolve_path(config, "weather_history")

    history = pd.read_csv(history_path)
    forecast = pd.read_csv(forecast_path)

    combined_weather_path = history_path.parent / "_combined_weather_for_prediction.csv"
    combined = pd.concat([history, forecast], ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
    combined = combined.dropna(subset=["date"])
    combined = (
        combined.sort_values(["station_id", "date"])
        .drop_duplicates(["station_id", "date"], keep="last")
    )
    combined.to_csv(combined_weather_path, index=False)

    try:
        forecast_dates = (
            pd.to_datetime(forecast["date"], errors="coerce")
            .dropna()
            .dt.normalize()
            .drop_duplicates()
            .sort_values()
        )

        if forecast_dates.empty:
            raise ValueError("weather_forecast.csv에 유효한 날짜가 없습니다.")

        if prediction_date:
            target_dates = [pd.Timestamp(prediction_date).normalize()]
        else:
            today = pd.Timestamp.today().normalize()
            future_dates = forecast_dates[forecast_dates >= today]
            if future_dates.empty:
                future_dates = forecast_dates
            target_dates = list(future_dates.iloc[: max(int(horizon), 1)])

        max_target_date = max(target_dates)

        prepared = prepare_dataset(
            pothole_path=resolve_path(config, "potholes"),
            repair_path=resolve_path(config, "repairs"),
            road_path=resolve_path(config, "roads"),
            weather_path=combined_weather_path,
            grid_size_m=int(bundle["grid_size_m"]),
            start_date=None,
            end_date=str(max_target_date.date()),
            include_target=False,
        )

        model = bundle["model"]
        outputs: list[pd.DataFrame] = []

        for target_date in target_dates:
            latest = prepared.panel[prepared.panel["date"] == target_date].copy()
            if latest.empty:
                raise ValueError(
                    f"예측 대상 날짜 데이터가 없습니다: {target_date.date()}"
                )

            latest["risk_score"] = model.predict_proba(
                latest[FEATURE_COLUMNS]
            )[:, 1]
            latest["risk_level"] = latest["risk_score"].map(risk_level)
            latest["risk_reason"] = latest.apply(explain_risk, axis=1)
            latest = latest.sort_values(
                "risk_score", ascending=False
            ).reset_index(drop=True)
            latest["priority_rank"] = latest.index + 1
            latest["prediction_date"] = target_date.date().isoformat()
            outputs.append(latest)

        result = pd.concat(outputs, ignore_index=True)

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
        result[output_columns].to_csv(
            output_path,
            index=False,
            encoding="utf-8-sig",
        )

        print(f"예측 저장 완료: {output_path}")
        print(
            "예측 날짜:",
            ", ".join(sorted(result["prediction_date"].unique())),
        )
        print(result[output_columns].head(10).to_string(index=False))
    finally:
        if combined_weather_path.exists():
            combined_weather_path.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--date", default=None, help="YYYY-MM-DD")
    parser.add_argument(
        "--horizon",
        type=int,
        default=3,
        help="--date가 없을 때 예측할 미래 날짜 수",
    )
    args = parser.parse_args()
    main(args.config, args.date, args.horizon)
