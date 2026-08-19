import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
)


DATA_FILE = Path(
    "data/pothole_training_data.csv"
)


FEATURES = [
    "freeze_thaw_14d",
    "rain_7d",
    "rain_14d",
    "sewer_old30_ratio",
    "traffic_esal",
    "has_past_repair",
    "days_since_last_repair",
]


TARGET = "pothole_label"


FOLDS = [
    {
        "name": "Fold 1",
        "train_start": "2024-01-01",
        "train_end": "2024-06-23",
        "test_start": "2024-07-01",
        "test_end": "2024-09-30",
    },
    {
        "name": "Fold 2",
        "train_start": "2024-01-01",
        "train_end": "2024-09-23",
        "test_start": "2024-10-01",
        "test_end": "2024-12-31",
    },
    {
        "name": "Fold 3",
        "train_start": "2024-01-01",
        "train_end": "2024-12-24",
        "test_start": "2025-01-01",
        "test_end": "2025-03-31",
    },
    {
        "name": "Fold 4",
        "train_start": "2024-01-01",
        "train_end": "2025-03-24",
        "test_start": "2025-04-01",
        "test_end": "2025-06-30",
    },
    {
        "name": "Fold 5",
        "train_start": "2024-01-01",
        "train_end": "2025-06-23",
        "test_start": "2025-07-01",
        "test_end": "2025-12-20",
    },
]


def main():

    print(
        "학습 데이터 불러오는 중..."
    )

    df = pd.read_csv(
        DATA_FILE,
        encoding="utf-8-sig",
        low_memory=False
    )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    for col in FEATURES:

        if col not in df.columns:

            raise ValueError(
                f"{col} 컬럼이 없습니다."
            )

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df[TARGET] = pd.to_numeric(
        df[TARGET],
        errors="coerce"
    )

    df[
        "days_since_last_repair"
    ] = (
        df[
            "days_since_last_repair"
        ]
        .fillna(0)
    )

    df = df.dropna(
        subset=[
            "date",
            TARGET,
        ]
    ).copy()

    df[TARGET] = (
        df[TARGET]
        .astype(int)
    )

    print(
        "전체 행 수:",
        len(df)
    )

    print(
        "전체 양성:",
        int(
            df[TARGET].sum()
        )
    )

    print(
        "전체 날짜:",
        df["date"].min().date(),
        "~",
        df["date"].max().date()
    )

    results = []

    for fold in FOLDS:

        name = fold["name"]

        train_start = pd.Timestamp(
            fold["train_start"]
        )

        train_end = pd.Timestamp(
            fold["train_end"]
        )

        test_start = pd.Timestamp(
            fold["test_start"]
        )

        test_end = pd.Timestamp(
            fold["test_end"]
        )

        train = df[
            (
                df["date"]
                >= train_start
            )
            &
            (
                df["date"]
                <= train_end
            )
        ].copy()

        test = df[
            (
                df["date"]
                >= test_start
            )
            &
            (
                df["date"]
                <= test_end
            )
        ].copy()

        print(
            "\n========================================"
        )

        print(
            name
        )

        print(
            "========================================"
        )

        print(
            "TRAIN:",
            train_start.date(),
            "~",
            train_end.date()
        )

        print(
            "TEST :",
            test_start.date(),
            "~",
            test_end.date()
        )

        print(
            "TRAIN 행:",
            len(train)
        )

        print(
            "TRAIN 양성:",
            int(
                train[TARGET].sum()
            )
        )

        print(
            "TEST 행:",
            len(test)
        )

        print(
            "TEST 양성:",
            int(
                test[TARGET].sum()
            )
        )

        if len(train) == 0 or len(test) == 0:

            print(
                "데이터가 없어 Fold 제외"
            )

            continue

        if (
            train[TARGET].nunique()
            < 2
        ):

            print(
                "TRAIN에 label 0/1이 모두 존재하지 않아 제외"
            )

            continue

        if (
            test[TARGET].nunique()
            < 2
        ):

            print(
                "TEST에 label 0/1이 모두 존재하지 않아 제외"
            )

            continue

        X_train = train[
            FEATURES
        ].copy()

        y_train = train[
            TARGET
        ].copy()

        X_test = test[
            FEATURES
        ].copy()

        y_test = test[
            TARGET
        ].copy()

        train_medians = (
            X_train
            .median(
                numeric_only=True
            )
        )

        X_train = X_train.fillna(
            train_medians
        )

        X_test = X_test.fillna(
            train_medians
        )

        model = Pipeline(
            steps=[
                (
                    "scaler",
                    StandardScaler()
                ),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=42,
                    )
                ),
            ]
        )

        model.fit(
            X_train,
            y_train
        )

        probability = (
            model
            .predict_proba(
                X_test
            )[:, 1]
        )

        roc_auc = (
            roc_auc_score(
                y_test,
                probability
            )
        )

        pr_auc = (
            average_precision_score(
                y_test,
                probability
            )
        )

        positive_rate = (
            y_test.mean()
        )

        if positive_rate > 0:

            pr_lift = (
                pr_auc
                /
                positive_rate
            )

        else:

            pr_lift = np.nan

        print(
            f"양성 비율 : "
            f"{positive_rate * 100:.4f}%"
        )

        print(
            f"ROC-AUC   : "
            f"{roc_auc:.6f}"
        )

        print(
            f"PR-AUC    : "
            f"{pr_auc:.6f}"
        )

        print(
            f"PR 기준선 : "
            f"{positive_rate:.6f}"
        )

        print(
            f"PR Lift   : "
            f"{pr_lift:.2f}x"
        )

        results.append(
            {
                "fold":
                    name,

                "train_start":
                    train_start.date(),

                "train_end":
                    train_end.date(),

                "test_start":
                    test_start.date(),

                "test_end":
                    test_end.date(),

                "train_rows":
                    len(train),

                "train_positive":
                    int(
                        y_train.sum()
                    ),

                "test_rows":
                    len(test),

                "test_positive":
                    int(
                        y_test.sum()
                    ),

                "positive_rate":
                    positive_rate,

                "roc_auc":
                    roc_auc,

                "pr_auc":
                    pr_auc,

                "pr_lift_vs_random":
                    pr_lift,
            }
        )

    result_df = pd.DataFrame(
        results
    )

    print(
        "\n========================================"
    )

    print(
        "시간순 교차검증 결과"
    )

    print(
        "========================================"
    )

    if len(result_df) == 0:

        print(
            "사용 가능한 Fold가 없습니다."
        )

        return

    print(
        result_df[
            [
                "fold",
                "test_positive",
                "positive_rate",
                "roc_auc",
                "pr_auc",
                "pr_lift_vs_random",
            ]
        ]
        .to_string(
            index=False
        )
    )

    print(
        "\n========================================"
    )

    print(
        "평균 성능"
    )

    print(
        "========================================"
    )

    print(
        f"평균 ROC-AUC : "
        f"{result_df['roc_auc'].mean():.6f}"
    )

    print(
        f"ROC-AUC 표준편차 : "
        f"{result_df['roc_auc'].std():.6f}"
    )

    print(
        f"평균 PR-AUC : "
        f"{result_df['pr_auc'].mean():.6f}"
    )

    print(
        f"평균 PR Lift : "
        f"{result_df['pr_lift_vs_random'].mean():.2f}x"
    )

    OUTPUT_FILE = Path(
        "data/logistic_timeseries_validation.csv"
    )

    result_df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        "\n검증 결과 저장:"
    )

    print(
        OUTPUT_FILE
    )


if __name__ == "__main__":
    main()