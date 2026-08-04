from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score

from .common import load_config, resolve_path
from .features import FEATURE_COLUMNS, downsample_negatives, prepare_dataset


def make_model(config: dict, scale_pos_weight: float):
    try:
        from xgboost import XGBClassifier

        params = config.get("model", {})
        return XGBClassifier(
            n_estimators=int(params.get("n_estimators", 350)),
            max_depth=int(params.get("max_depth", 4)),
            learning_rate=float(params.get("learning_rate", 0.05)),
            subsample=float(params.get("subsample", 0.85)),
            colsample_bytree=float(params.get("colsample_bytree", 0.85)),
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            random_state=int(config["project"].get("random_seed", 42)),
            n_jobs=-1,
        )
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier

        print("[WARN] xgboost가 없어 HistGradientBoostingClassifier를 사용합니다.")
        return HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=6,
            max_iter=300,
            random_state=int(config["project"].get("random_seed", 42)),
        )


def temporal_split(df: pd.DataFrame, validation_ratio: float, test_ratio: float):
    dates = np.array(sorted(df["date"].unique()))
    n_dates = len(dates)
    test_start = dates[max(1, int(n_dates * (1 - test_ratio)))]
    val_start = dates[max(1, int(n_dates * (1 - test_ratio - validation_ratio)))]

    train = df[df["date"] < val_start].copy()
    val = df[(df["date"] >= val_start) & (df["date"] < test_start)].copy()
    test = df[df["date"] >= test_start].copy()
    return train, val, test


def evaluate(model, frame: pd.DataFrame, name: str) -> dict[str, float]:
    y_true = frame["target"].to_numpy()
    y_prob = model.predict_proba(frame[FEATURE_COLUMNS])[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan"),
        "average_precision": float(average_precision_score(y_true, y_prob)),
    }
    print(f"\n[{name}] {metrics}")
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))
    return metrics


def main(config_path: str) -> None:
    config = load_config(config_path)
    grid_size_m = int(config["project"].get("grid_size_m", 500))
    training = config.get("training", {})

    prepared = prepare_dataset(
        pothole_path=resolve_path(config, "potholes"),
        repair_path=resolve_path(config, "repairs"),
        road_path=resolve_path(config, "roads"),
        weather_path=resolve_path(config, "weather_history"),
        grid_size_m=grid_size_m,
        start_date=training.get("start_date"),
        end_date=training.get("end_date"),
        include_target=True,
    )

    sampled = downsample_negatives(
        prepared.panel,
        ratio=int(training.get("negative_downsample_ratio", 8)),
        random_seed=int(config["project"].get("random_seed", 42)),
    )
    train, val, test = temporal_split(
        sampled,
        validation_ratio=float(training.get("validation_ratio", 0.15)),
        test_ratio=float(training.get("test_ratio", 0.20)),
    )

    n_pos = max(int(train["target"].sum()), 1)
    n_neg = max(int((train["target"] == 0).sum()), 1)
    model = make_model(config, scale_pos_weight=n_neg / n_pos)
    model.fit(train[FEATURE_COLUMNS], train["target"])

    metrics = {
        "validation": evaluate(model, val, "validation"),
        "test": evaluate(model, test, "test"),
    }

    model_path = resolve_path(config, "model")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": model,
        "features": FEATURE_COLUMNS,
        "grid_size_m": grid_size_m,
        "grid_catalog": prepared.grid_catalog,
        "station_map": prepared.station_map,
        "metrics": metrics,
        "training_max_date": prepared.panel["date"].max(),
    }
    joblib.dump(bundle, model_path)
    print(f"\n모델 저장 완료: {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)
