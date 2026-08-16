from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, classification_report, f1_score, roc_auc_score

from .common import load_config, resolve_path
from .features import (
    build_priority_scales,
    prepare_dataset,
    prepare_feature_matrix,
    select_feature_columns,
)


MODEL_SCHEMA_VERSION = "2.0"


def make_model(config: dict, scale_pos_weight: float):
    from xgboost import XGBClassifier

    params = config.get("model", {})
    return XGBClassifier(
        n_estimators=int(params.get("n_estimators", 700)),
        max_depth=int(params.get("max_depth", 4)),
        learning_rate=float(params.get("learning_rate", 0.05)),
        min_child_weight=float(params.get("min_child_weight", 5)),
        subsample=float(params.get("subsample", 0.85)),
        colsample_bytree=float(params.get("colsample_bytree", 0.85)),
        reg_alpha=float(params.get("reg_alpha", 0.05)),
        reg_lambda=float(params.get("reg_lambda", 1.0)),
        objective="binary:logistic",
        eval_metric=str(params.get("eval_metric", "logloss")),
        early_stopping_rounds=int(params.get("early_stopping_rounds", 50)),
        scale_pos_weight=scale_pos_weight,
        random_state=int(config["project"].get("random_seed", 42)),
        n_jobs=-1,
        tree_method="hist",
    )


def temporal_split(df: pd.DataFrame, validation_ratio: float, test_ratio: float):
    dates = np.array(sorted(df["date"].unique()))
    if len(dates) < 5:
        raise ValueError("시간 분할을 위해 최소 5개 기준일이 필요합니다.")
    test_position = min(max(2, int(len(dates) * (1 - test_ratio))), len(dates) - 1)
    validation_position = min(
        max(1, int(len(dates) * (1 - test_ratio - validation_ratio))),
        test_position - 1,
    )
    validation_start = dates[validation_position]
    test_start = dates[test_position]
    train = df[df["date"] < validation_start].copy()
    validation = df[(df["date"] >= validation_start) & (df["date"] < test_start)].copy()
    test = df[df["date"] >= test_start].copy()
    for name, frame in [("train", train), ("validation", validation), ("test", test)]:
        if frame.empty or frame["target"].nunique() != 2:
            raise ValueError(f"{name} 구간에 양성과 음성이 모두 필요합니다.")
    return train, validation, test, pd.Timestamp(validation_start), pd.Timestamp(test_start)


def best_f1_threshold(y_true: pd.Series, probabilities: np.ndarray) -> tuple[float, float]:
    candidates = np.unique(np.concatenate(([0.0], probabilities, [1.0])))
    best_threshold = 0.5
    best_score = -1.0
    for threshold in candidates:
        current = f1_score(y_true, probabilities >= threshold, zero_division=0)
        if current > best_score or (
            math.isclose(current, best_score)
            and abs(float(threshold) - 0.5) < abs(best_threshold - 0.5)
        ):
            best_threshold = float(threshold)
            best_score = float(current)
    return best_threshold, best_score


def evaluate(
    model,
    matrix: pd.DataFrame,
    y_true: pd.Series,
    name: str,
    threshold: float,
) -> dict[str, float]:
    probability = model.predict_proba(matrix)[:, 1]
    prediction = (probability >= threshold).astype(int)
    metrics = {
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "pr_auc": float(average_precision_score(y_true, probability)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "threshold": float(threshold),
        "positive_rate": float(y_true.mean()),
    }
    print(f"\n[{name}] {metrics}")
    print(classification_report(y_true, prediction, digits=4, zero_division=0))
    return metrics


def main(config_path: str) -> None:
    config = load_config(config_path)
    grid_size_m = int(config["project"].get("grid_size_m", 500))
    training = config.get("training", {})
    target_horizon_days = int(training.get("target_horizon_days", 30))

    prepared = prepare_dataset(
        pothole_path=resolve_path(config, "potholes"),
        repair_path=resolve_path(config, "repairs"),
        road_path=resolve_path(config, "roads"),
        weather_path=resolve_path(config, "weather_history"),
        grid_size_m=grid_size_m,
        start_date=training.get("start_date"),
        end_date=training.get("end_date"),
        include_target=True,
        target_horizon_days=target_horizon_days,
    )

    train, validation, test, validation_start, test_start = temporal_split(
        prepared.panel,
        validation_ratio=float(training.get("validation_ratio", 0.15)),
        test_ratio=float(training.get("test_ratio", 0.20)),
    )
    feature_columns = select_feature_columns(train)
    x_train, medians = prepare_feature_matrix(train, feature_columns)
    x_validation, _ = prepare_feature_matrix(validation, feature_columns, medians)
    x_test, _ = prepare_feature_matrix(test, feature_columns, medians)
    y_train = train["target"].astype(int)
    y_validation = validation["target"].astype(int)
    y_test = test["target"].astype(int)

    positives = int(y_train.sum())
    negatives = int(len(y_train) - positives)
    scale_pos_weight = negatives / positives
    model = make_model(config, scale_pos_weight)
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_validation, y_validation)],
        verbose=False,
    )

    validation_probability = model.predict_proba(x_validation)[:, 1]
    threshold, _ = best_f1_threshold(y_validation, validation_probability)
    metrics = {
        "validation": evaluate(model, x_validation, y_validation, "validation", threshold),
        "test": evaluate(model, x_test, y_test, "test", threshold),
    }
    recurrence_scales, importance_scales = build_priority_scales(train)
    trained_at = datetime.now(timezone.utc).isoformat()

    model_path = resolve_path(config, "model")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "trained_at_utc": trained_at,
        "model": model,
        "features": feature_columns,
        "feature_medians": medians,
        "classification_threshold": threshold,
        "target_horizon_days": target_horizon_days,
        "grid_size_m": grid_size_m,
        "grid_catalog": prepared.grid_catalog,
        "station_map": prepared.station_map,
        "recurrence_scales": recurrence_scales,
        "importance_scales": importance_scales,
        "metrics": metrics,
        "training_max_date": prepared.panel["date"].max(),
    }
    joblib.dump(bundle, model_path)

    metrics_payload = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "trained_at_utc": trained_at,
        "target": f"기준일 다음 {target_horizon_days}일 내 포트홀 발생 여부",
        "features": feature_columns,
        "split": {
            "validation_start": validation_start.date().isoformat(),
            "test_start": test_start.date().isoformat(),
            "train_rows": len(train),
            "validation_rows": len(validation),
            "test_rows": len(test),
        },
        "class_imbalance": {
            "train_positive": positives,
            "train_negative": negatives,
            "scale_pos_weight": scale_pos_weight,
        },
        "metrics": metrics,
        "notes": [
            "F1 임계값은 validation에서 선택하고 test에는 그대로 적용했습니다.",
            "risk_score는 순위화를 위한 모델 점수이며 별도 확률 보정 전에는 절대 발생확률로 해석하지 않습니다.",
        ],
    }
    metrics_path = resolve_path(config, "metrics")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    importance_path = resolve_path(config, "feature_importance")
    importance_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"feature": feature_columns, "gain_importance": model.feature_importances_}
    ).sort_values("gain_importance", ascending=False).to_csv(importance_path, index=False)

    print(f"\n모델 저장 완료: {model_path}")
    print(f"지표 저장 완료: {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)
