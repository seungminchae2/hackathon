from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)


PREDICTION_PATH = Path("data/pothole_model_predictions.csv")
OUTPUT_PATH = Path("data/logistic_risk_analysis.csv")


def main():
    print("Logistic Regression 예측 결과 불러오는 중...")

    df = pd.read_csv(PREDICTION_PATH)

    print("컬럼 목록")
    print(df.columns.tolist())
    print()

    label_candidates = [
        "pothole_label",
        "label",
        "y_true",
        "actual",
    ]

    probability_candidates = [
        "logistic_probability",
        "logistic_prob",
        "logistic_proba",
        "logistic_score",
        "logistic_prediction_probability",
    ]

    label_col = None
    probability_col = None

    for col in label_candidates:
        if col in df.columns:
            label_col = col
            break

    for col in probability_candidates:
        if col in df.columns:
            probability_col = col
            break

    if label_col is None:
        raise ValueError(
            "실제 라벨 컬럼을 찾지 못했습니다. "
            f"현재 컬럼: {df.columns.tolist()}"
        )

    if probability_col is None:
        raise ValueError(
            "Logistic Regression 확률 컬럼을 찾지 못했습니다. "
            f"현재 컬럼: {df.columns.tolist()}"
        )

    print(f"실제 라벨 컬럼: {label_col}")
    print(f"Logistic 모델 점수 컬럼: {probability_col}")
    print()

    df = df[[label_col, probability_col]].copy()

    df[label_col] = pd.to_numeric(
        df[label_col],
        errors="coerce"
    )

    df[probability_col] = pd.to_numeric(
        df[probability_col],
        errors="coerce"
    )

    df = df.dropna()

    y_true = df[label_col].astype(int)
    y_prob = df[probability_col].astype(float)

    print("========================================")
    print("기본 정보")
    print("========================================")
    print(f"전체 TEST 행 수: {len(df)}")
    print(f"실제 포트홀 양성: {int(y_true.sum())}")
    print(f"실제 양성 비율: {y_true.mean():.4%}")
    print()

    pr_auc = average_precision_score(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)

    print(f"PR-AUC : {pr_auc:.6f}")
    print(f"ROC-AUC: {roc_auc:.6f}")
    print()

    results = []

    thresholds = np.arange(0.10, 0.96, 0.05)

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)

        predicted_positive = int(y_pred.sum())
        true_positive = int(((y_pred == 1) & (y_true == 1)).sum())
        false_positive = int(((y_pred == 1) & (y_true == 0)).sum())
        false_negative = int(((y_pred == 0) & (y_true == 1)).sum())

        precision = precision_score(
            y_true,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_true,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_true,
            y_pred,
            zero_division=0
        )

        results.append({
            "threshold": round(float(threshold), 2),
            "predicted_positive": predicted_positive,
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })

    result_df = pd.DataFrame(results)

    print("========================================")
    print("Threshold별 성능")
    print("========================================")
    print(
        result_df.to_string(
            index=False,
            formatters={
                "precision": "{:.4%}".format,
                "recall": "{:.4%}".format,
                "f1": "{:.6f}".format,
            }
        )
    )

    print()

    recall70 = result_df[
        result_df["recall"] >= 0.70
    ].copy()

    if len(recall70) > 0:
        recommended = recall70.sort_values(
            ["false_positive", "precision"],
            ascending=[True, False]
        ).iloc[0]

        print("========================================")
        print("Recall 70% 이상 조건 추천 Threshold")
        print("========================================")
        print(f"Threshold       : {recommended['threshold']:.2f}")
        print(f"Recall          : {recommended['recall']:.4%}")
        print(f"Precision       : {recommended['precision']:.4%}")
        print(f"F1              : {recommended['f1']:.6f}")
        print(
            f"위험 판정 행 수 : "
            f"{int(recommended['predicted_positive'])}"
        )
        print(
            f"실제 포트홀 적중 : "
            f"{int(recommended['true_positive'])}"
        )
        print(
            f"오탐             : "
            f"{int(recommended['false_positive'])}"
        )
        print(
            f"놓친 포트홀      : "
            f"{int(recommended['false_negative'])}"
        )
        print()

    print("========================================")
    print("상위 위험 백분위 분석")
    print("========================================")

    percentile_results = []

    top_percentages = [
        0.01,
        0.03,
        0.05,
        0.10,
        0.15,
        0.20,
        0.30,
    ]

    total_positive = int(y_true.sum())

    for top_pct in top_percentages:
        cutoff = y_prob.quantile(1 - top_pct)

        selected = df[
            df[probability_col] >= cutoff
        ]

        selected_count = len(selected)
        selected_positive = int(
            selected[label_col].sum()
        )

        precision = (
            selected_positive / selected_count
            if selected_count > 0
            else 0
        )

        recall = (
            selected_positive / total_positive
            if total_positive > 0
            else 0
        )

        lift = (
            precision / y_true.mean()
            if y_true.mean() > 0
            else 0
        )

        percentile_results.append({
            "top_percent": top_pct * 100,
            "probability_cutoff": cutoff,
            "selected_rows": selected_count,
            "captured_potholes": selected_positive,
            "precision": precision,
            "recall": recall,
            "lift_vs_random": lift,
        })

    percentile_df = pd.DataFrame(
        percentile_results
    )

    print(
        percentile_df.to_string(
            index=False,
            formatters={
                "top_percent": "{:.0f}%".format,
                "probability_cutoff": "{:.6f}".format,
                "precision": "{:.4%}".format,
                "recall": "{:.4%}".format,
                "lift_vs_random": "{:.2f}x".format,
            }
        )
    )

    result_df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    percentile_output = Path(
        "data/logistic_percentile_analysis.csv"
    )

    percentile_df.to_csv(
        percentile_output,
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print("========================================")
    print("분석 완료")
    print("========================================")
    print(f"Threshold 결과: {OUTPUT_PATH}")
    print(f"백분위 결과: {percentile_output}")


if __name__ == "__main__":
    main()