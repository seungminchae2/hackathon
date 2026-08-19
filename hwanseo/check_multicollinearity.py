import pandas as pd
import numpy as np
from pathlib import Path


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


def calculate_vif(df):

    results = []

    X = df.astype(float).copy()

    for target in X.columns:

        y = X[target].to_numpy()

        other_columns = [
            col
            for col in X.columns
            if col != target
        ]

        X_other = X[
            other_columns
        ].to_numpy()

        X_other = np.column_stack(
            [
                np.ones(
                    len(X_other)
                ),
                X_other,
            ]
        )

        beta, _, _, _ = np.linalg.lstsq(
            X_other,
            y,
            rcond=None
        )

        y_hat = (
            X_other
            @ beta
        )

        ss_res = np.sum(
            (
                y
                -
                y_hat
            ) ** 2
        )

        ss_tot = np.sum(
            (
                y
                -
                np.mean(y)
            ) ** 2
        )

        if ss_tot == 0:

            r_squared = 1.0

        else:

            r_squared = (
                1
                -
                ss_res
                / ss_tot
            )


        if (
            1
            -
            r_squared
            <= 1e-12
        ):

            vif = np.inf

        else:

            vif = (
                1
                /
                (
                    1
                    -
                    r_squared
                )
            )


        results.append(
            {
                "feature": target,
                "VIF": vif,
            }
        )


    return pd.DataFrame(
        results
    )


def main():

    print(
        "학습 데이터 불러오는 중..."
    )


    df = pd.read_csv(
        DATA_FILE,
        encoding="utf-8-sig",
        low_memory=False
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


    df[
        "days_since_last_repair"
    ] = (
        df[
            "days_since_last_repair"
        ]
        .fillna(0)
    )


    analysis = (
        df[
            FEATURES
        ]
        .dropna()
        .copy()
    )


    print(
        "분석 행 수:",
        len(analysis)
    )


    print(
        "\n========================================"
    )

    print(
        "Pearson 상관계수"
    )

    print(
        "========================================"
    )


    correlation = (
        analysis
        .corr()
    )


    print(
        correlation
        .round(4)
        .to_string()
    )


    rain_corr = (
        correlation.loc[
            "rain_7d",
            "rain_14d"
        ]
    )


    print(
        "\n"
        "rain_7d ↔ rain_14d 상관계수:"
    )

    print(
        f"{rain_corr:.4f}"
    )


    print(
        "\n========================================"
    )

    print(
        "VIF"
    )

    print(
        "========================================"
    )


    vif_df = calculate_vif(
        analysis
    )


    vif_df = (
        vif_df
        .sort_values(
            "VIF",
            ascending=False
        )
    )


    print(
        vif_df
        .to_string(
            index=False
        )
    )


    rain7_vif = (
        vif_df.loc[
            vif_df[
                "feature"
            ]
            == "rain_7d",
            "VIF"
        ]
        .iloc[0]
    )


    rain14_vif = (
        vif_df.loc[
            vif_df[
                "feature"
            ]
            == "rain_14d",
            "VIF"
        ]
        .iloc[0]
    )


    print(
        "\n========================================"
    )

    print(
        "강수 변수 해석"
    )

    print(
        "========================================"
    )


    print(
        f"rain_7d VIF  : {rain7_vif:.4f}"
    )


    print(
        f"rain_14d VIF : {rain14_vif:.4f}"
    )


    if (
        abs(
            rain_corr
        )
        >= 0.8
    ):

        print(
            "\n강수 변수 간 상관관계가 매우 높음."
        )

    elif (
        abs(
            rain_corr
        )
        >= 0.6
    ):

        print(
            "\n강수 변수 간 상관관계가 높은 편임."
        )

    else:

        print(
            "\n강수 변수 간 상관관계가 아주 높지는 않음."
        )


    if (
        rain7_vif
        >= 10
        or
        rain14_vif
        >= 10
    ):

        print(
            "VIF 기준으로 다중공선성 문제가 강하게 의심됨."
        )

    elif (
        rain7_vif
        >= 5
        or
        rain14_vif
        >= 5
    ):

        print(
            "VIF 기준으로 다중공선성을 주의해서 볼 필요가 있음."
        )

    else:

        print(
            "VIF 기준으로 심각한 다중공선성은 확인되지 않음."
        )


if __name__ == "__main__":
    main()