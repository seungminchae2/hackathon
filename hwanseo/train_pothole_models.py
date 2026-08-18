import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)

import joblib


# ============================================================
# 파일 경로
# ============================================================

INPUT_FILE = Path(
    "data/pothole_training_data.csv"
)

MODEL_DIR = Path(
    "models"
)

LOGISTIC_MODEL_FILE = (
    MODEL_DIR
    / "pothole_logistic.joblib"
)

RF_MODEL_FILE = (
    MODEL_DIR
    / "pothole_random_forest.joblib"
)

RESULT_FILE = Path(
    "data/pothole_model_predictions.csv"
)

IMPORTANCE_FILE = Path(
    "data/pothole_feature_importance.csv"
)


# ============================================================
# 시간순 Train / Test 분할
#
# 예측 목표가 "향후 7일 포트홀 발생"이므로
# 학습과 테스트 사이에도 7일 간격을 둔다.
#
# TRAIN:
# 2024-01-01 ~ 2025-06-23
#
# 2025-06-24 ~ 2025-06-30:
# 학습/테스트 양쪽에서 제외
#
# TEST:
# 2025-07-01 ~ 2025-12-20
#
# 이렇게 하는 이유:
# 6월 30일의 label은 7월 초 발생 포트홀을 볼 수 있으므로,
# Train과 Test의 미래 관측기간이 겹치는 것을 방지한다.
# ============================================================

TRAIN_END = pd.Timestamp(
    "2025-06-23"
)

TEST_START = pd.Timestamp(
    "2025-07-01"
)


# ============================================================
# 머신러닝 입력 변수
#
# 우리가 직접 만든 risk_score나
# freeze_score / rain_score / sewer_score 등은 사용하지 않는다.
#
# 모델이 원변수와 실제 포트홀 발생 사이의 관계를
# 직접 학습하도록 한다.
# ============================================================

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


# ============================================================
# 데이터 불러오기
# ============================================================

print(
    "학습 데이터 불러오는 중..."
)


df = pd.read_csv(
    INPUT_FILE,
    encoding="utf-8-sig",
    low_memory=False
)


df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)


df = df.dropna(
    subset=[
        "date",
        TARGET,
    ]
).copy()


# ============================================================
# 필요한 컬럼 존재 여부 검사
# ============================================================

missing_features = [
    feature
    for feature in FEATURES
    if feature not in df.columns
]


if missing_features:

    raise ValueError(
        "학습 데이터에 필요한 컬럼이 없습니다: "
        f"{missing_features}"
    )


# ============================================================
# 숫자형 변환
# ============================================================

for col in FEATURES:

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )


df[TARGET] = pd.to_numeric(
    df[TARGET],
    errors="coerce"
)


df = df.dropna(
    subset=[
        TARGET,
    ]
).copy()


df[TARGET] = (
    df[TARGET]
    .astype(int)
)


# ============================================================
# 보수이력이 없는 경우 days_since_last_repair 처리
#
# has_past_repair = 0
# → 과거 보수 자체가 없으므로
# days_since_last_repair가 NaN인 것이 정상이다.
#
# 머신러닝 모델에서는 숫자가 필요하므로
# 0으로 채우되,
# has_past_repair 변수를 함께 사용해서
#
# "보수 없음"
# 과
# "당일 보수"
#
# 를 모델이 구분할 수 있도록 한다.
# ============================================================

df[
    "days_since_last_repair"
] = (
    df[
        "days_since_last_repair"
    ]
    .fillna(0)
)


# ============================================================
# 다른 수치형 결측치 처리
#
# 혹시 일부 데이터에 결측치가 있을 경우
# 전체 데이터를 삭제하지 않고
# Train 데이터 중앙값으로 채우기 위해
# 분할 후 처리한다.
# ============================================================

print(
    "\n전체 데이터 행 수:",
    len(df)
)


print(
    "전체 양성:",
    int(
        df[TARGET].sum()
    )
)


print(
    "전체 양성 비율:",
    f"{df[TARGET].mean() * 100:.4f}%"
)


# ============================================================
# 시간순 Train / Test 분할
# ============================================================

train_df = df[
    df["date"]
    <= TRAIN_END
].copy()


test_df = df[
    df["date"]
    >= TEST_START
].copy()


print(
    "\nTRAIN 기간:",
    train_df["date"].min().date(),
    "~",
    train_df["date"].max().date()
)


print(
    "TEST 기간:",
    test_df["date"].min().date(),
    "~",
    test_df["date"].max().date()
)


print(
    "\nTRAIN 행 수:",
    len(train_df)
)


print(
    "TRAIN 양성:",
    int(
        train_df[TARGET].sum()
    )
)


print(
    "TRAIN 양성 비율:",
    f"{train_df[TARGET].mean() * 100:.4f}%"
)


print(
    "\nTEST 행 수:",
    len(test_df)
)


print(
    "TEST 양성:",
    int(
        test_df[TARGET].sum()
    )
)


print(
    "TEST 양성 비율:",
    f"{test_df[TARGET].mean() * 100:.4f}%"
)


if train_df[TARGET].nunique() < 2:

    raise ValueError(
        "TRAIN 데이터에 label 0/1이 모두 존재하지 않습니다."
    )


if test_df[TARGET].nunique() < 2:

    raise ValueError(
        "TEST 데이터에 label 0/1이 모두 존재하지 않습니다."
    )


# ============================================================
# X / y 생성
# ============================================================

X_train = train_df[
    FEATURES
].copy()


y_train = train_df[
    TARGET
].copy()


X_test = test_df[
    FEATURES
].copy()


y_test = test_df[
    TARGET
].copy()


# ============================================================
# Train 중앙값으로 결측치 처리
#
# Test 데이터의 정보를 이용해
# Train 결측값을 채우면 data leakage가 되므로
#
# 반드시 Train 기준 중앙값만 사용한다.
# ============================================================

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


# ============================================================
# Logistic Regression
#
# StandardScaler:
# 변수마다 단위 차이가 매우 크기 때문에 표준화한다.
#
# 예:
# sewer_old30_ratio → 0~1
# traffic_esal → 수천 단위
# rain_7d → 수십 mm
#
# class_weight="balanced":
# 현재 label=1 비율이 약 0.15%이므로
# 극심한 클래스 불균형을 보정한다.
# ============================================================

logistic_model = Pipeline(
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


print(
    "\nLogistic Regression 학습 중..."
)


logistic_model.fit(
    X_train,
    y_train
)


# ============================================================
# Random Forest
#
# 비선형 관계 및 변수 간 복합적인 상호작용을
# 학습할 수 있는 모델.
#
# class_weight="balanced_subsample":
# 각 트리의 bootstrap sample마다
# 클래스 불균형을 보정한다.
#
# min_samples_leaf를 크게 두어
# 극소수 양성 데이터를 과도하게 암기하는 것을
# 조금 완화한다.
# ============================================================

rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=10,
    class_weight="balanced_subsample",
    n_jobs=-1,
    random_state=42,
)


print(
    "Random Forest 학습 중..."
)


rf_model.fit(
    X_train,
    y_train
)


# ============================================================
# 모델 평가 함수
#
# Accuracy는 출력하지 않는다.
#
# 현재 음성 데이터가 99.8% 이상이기 때문에
# 전부 0으로 예측해도 Accuracy가 매우 높게 나올 수 있다.
#
# 핵심 지표:
#
# PR-AUC:
# 불균형 데이터에서 특히 중요.
#
# ROC-AUC:
# 전체적인 양성/음성 구분 능력.
#
# Recall:
# 실제 포트홀 위험 중 얼마나 잡았는가.
#
# Precision:
# 모델이 위험하다고 한 것 중 실제 위험 비율.
#
# F1:
# Precision과 Recall의 조화평균.
# ============================================================

def evaluate_model(
    model_name,
    y_true,
    probabilities,
    threshold=0.5
):

    predictions = (
        probabilities
        >= threshold
    ).astype(int)


    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )


    recall = recall_score(
        y_true,
        predictions,
        zero_division=0
    )


    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0
    )


    roc_auc = roc_auc_score(
        y_true,
        probabilities
    )


    pr_auc = average_precision_score(
        y_true,
        probabilities
    )


    cm = confusion_matrix(
        y_true,
        predictions
    )


    print(
        "\n========================================"
    )


    print(
        model_name
    )


    print(
        "========================================"
    )


    print(
        f"Threshold : {threshold:.2f}"
    )


    print(
        f"PR-AUC    : {pr_auc:.6f}"
    )


    print(
        f"ROC-AUC   : {roc_auc:.6f}"
    )


    print(
        f"Precision : {precision:.6f}"
    )


    print(
        f"Recall    : {recall:.6f}"
    )


    print(
        f"F1-score  : {f1:.6f}"
    )


    print(
        "\nConfusion Matrix"
    )


    print(
        cm
    )


    print(
        "\nClassification Report"
    )


    print(
        classification_report(
            y_true,
            predictions,
            digits=6,
            zero_division=0
        )
    )


    return {
        "model":
            model_name,

        "pr_auc":
            pr_auc,

        "roc_auc":
            roc_auc,

        "precision":
            precision,

        "recall":
            recall,

        "f1":
            f1,
    }


# ============================================================
# 확률 예측
# ============================================================

logistic_probability = (
    logistic_model
    .predict_proba(
        X_test
    )[:, 1]
)


rf_probability = (
    rf_model
    .predict_proba(
        X_test
    )[:, 1]
)


# ============================================================
# 기본 Threshold = 0.5 성능
# ============================================================

logistic_metrics = evaluate_model(
    "Logistic Regression",
    y_test,
    logistic_probability,
    threshold=0.5
)


rf_metrics = evaluate_model(
    "Random Forest",
    y_test,
    rf_probability,
    threshold=0.5
)


# ============================================================
# 두 모델 종합 비교
# ============================================================

metrics_df = pd.DataFrame(
    [
        logistic_metrics,
        rf_metrics,
    ]
)


print(
    "\n========================================"
)


print(
    "모델 성능 비교"
)


print(
    "========================================"
)


print(
    metrics_df
    .sort_values(
        "pr_auc",
        ascending=False
    )
    .to_string(
        index=False
    )
)


# ============================================================
# Logistic Regression 계수
#
# StandardScaler 이후 표준화된 변수 기준 계수.
#
# 양수:
# 해당 변수가 증가할수록
# 포트홀 발생확률 증가 방향.
#
# 음수:
# 해당 변수가 증가할수록
# 포트홀 발생확률 감소 방향.
#
# 절댓값이 클수록
# Logistic Regression 내 영향력이 큰 편이다.
# ============================================================

logistic_coefficients = (
    logistic_model
    .named_steps[
        "model"
    ]
    .coef_[0]
)


logistic_importance = pd.DataFrame(
    {
        "feature":
            FEATURES,

        "logistic_coefficient":
            logistic_coefficients,

        "logistic_abs_coefficient":
            np.abs(
                logistic_coefficients
            ),
    }
)


logistic_importance = (
    logistic_importance
    .sort_values(
        "logistic_abs_coefficient",
        ascending=False
    )
)


print(
    "\n========================================"
)


print(
    "Logistic Regression 변수 영향"
)


print(
    "========================================"
)


print(
    logistic_importance
    .to_string(
        index=False
    )
)


# ============================================================
# Random Forest Feature Importance
#
# 값이 클수록 Random Forest가
# 예측에 많이 활용한 변수.
#
# 단, feature_importances_는
# 인과관계를 의미하지 않는다.
# ============================================================

rf_importance = pd.DataFrame(
    {
        "feature":
            FEATURES,

        "random_forest_importance":
            rf_model.feature_importances_,
    }
)


rf_importance = (
    rf_importance
    .sort_values(
        "random_forest_importance",
        ascending=False
    )
)


print(
    "\n========================================"
)


print(
    "Random Forest 변수 중요도"
)


print(
    "========================================"
)


print(
    rf_importance
    .to_string(
        index=False
    )
)


# ============================================================
# 두 모델 변수 중요도 합치기
# ============================================================

importance = (
    logistic_importance
    .merge(
        rf_importance,
        on="feature",
        how="outer"
    )
)


importance.to_csv(
    IMPORTANCE_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 테스트 데이터 예측 결과 저장
#
# 지도나 추가 분석에서 사용할 수 있도록
#
# point_id
# date
# 실제 label
# Logistic 발생확률
# RF 발생확률
#
# 을 저장한다.
# ============================================================

prediction_output = test_df[
    [
        "point_id",
        "date",
        "road_name",
        "city",
        "lat",
        "lon",
        TARGET,
    ]
].copy()


prediction_output[
    "logistic_probability"
] = logistic_probability


prediction_output[
    "random_forest_probability"
] = rf_probability


prediction_output.to_csv(
    RESULT_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 학습된 모델 저장
#
# 이후 Streamlit 등에서 다시 학습하지 않고
# 바로 불러와 예측할 수 있다.
# ============================================================

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


joblib.dump(
    logistic_model,
    LOGISTIC_MODEL_FILE
)


joblib.dump(
    rf_model,
    RF_MODEL_FILE
)


# ============================================================
# 최종 결과
# ============================================================

print(
    "\n========================================"
)


print(
    "모델 학습 완료"
)


print(
    "========================================"
)


print(
    "Logistic 모델:",
    LOGISTIC_MODEL_FILE
)


print(
    "Random Forest 모델:",
    RF_MODEL_FILE
)


print(
    "예측 결과:",
    RESULT_FILE
)


print(
    "변수 중요도:",
    IMPORTANCE_FILE
)


best_model = (
    metrics_df
    .sort_values(
        "pr_auc",
        ascending=False
    )
    .iloc[0]
)


print(
    "\nPR-AUC 기준 우수 모델:"
)


print(
    best_model[
        "model"
    ]
)


print(
    "PR-AUC:",
    f"{best_model['pr_auc']:.6f}"
)