import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc, f1_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
data_path = BASE_DIR / "pothole_training_data.csv"

# 1. 학습 데이터가 없거나 비어 있으면 자동 생성
if not data_path.exists() or data_path.stat().st_size == 0:
    print("학습 데이터 생성 중...")
    # 현재 폴더와 data/ 폴더 모두 탐색
    roads_candidates = [
        BASE_DIR / "roads_with_sewer_repair.csv",
        BASE_DIR / "data" / "roads_with_sewer_repair.csv",
    ]
    weather_candidates = [
        BASE_DIR / "weather_history.csv",
        BASE_DIR / "data" / "weather_history.csv",
    ]

    roads_path = next((p for p in roads_candidates if p.exists()), None)
    weather_path = next((p for p in weather_candidates if p.exists()), None)

    if not roads_path or not weather_path:
        raise FileNotFoundError(
            "도로 데이터(roads_with_sewer_repair.csv) 또는 기상 데이터(weather_history.csv)를 찾을 수 없습니다."
        )

    roads = pd.read_csv(roads_path)
    weather = pd.read_csv(weather_path)

    weather_agg = (
        weather.groupby("date")
        .agg({
            "avg_temp": "mean",
            "min_temp": "min",
            "max_temp": "max",
            "precipitation": "sum",
            "snowfall": "sum",
            "humidity": "mean",
        })
        .reset_index()
    )

    np.random.seed(42)
    n_samples = 3000
    sample_roads = roads.sample(n=n_samples, replace=True).reset_index(
        drop=True
    )
    sample_weather = weather_agg.sample(
        n=n_samples, replace=True
    ).reset_index(drop=True)

    train_df = pd.concat([sample_roads, sample_weather], axis=1)

    risk_prob = (
        train_df["sewer_old30_ratio"].fillna(0) * 3.0
        + train_df["precipitation"].fillna(0) * 0.02
        + (train_df["distance_to_last_repair_m"].fillna(1000) > 200).astype(int)
        * 0.5
        + np.random.normal(0, 0.5, n_samples)
    )
    risk_prob = 1 / (1 + np.exp(-risk_prob))
    train_df["pothole_label"] = (
        risk_prob > np.percentile(risk_prob, 90)
    ).astype(int)

    train_df.to_csv(data_path, index=False)
    print("pothole_training_data.csv 생성 완료!")

# 2. 데이터 로드 및 분할
df = pd.read_csv(data_path)
feature_cols = [
    "total_sewer_length",
    "old_sewer_length",
    "sewer_old30_ratio",
    "average_sewer_age",
    "has_nearby_repair",
    "distance_to_last_repair_m",
    "nearby_repair_count",
    "avg_temp",
    "min_temp",
    "max_temp",
    "precipitation",
    "snowfall",
    "humidity",
]

X = df[feature_cols].fillna(0)
y = df["pothole_label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_val, X_test_sub, y_val, y_test_sub = train_test_split(
    X_test, y_test, test_size=0.5, random_state=42
)

# 3. 모델 학습
print("모델 학습 중...")
model = RandomForestClassifier(
    n_estimators=100, class_weight="balanced", random_state=42
)
model.fit(X_train, y_train)

# 4. 검증 및 테스트 성능 계산
val_probs = model.predict_proba(X_val)[:, 1]
val_roc = roc_auc_score(y_val, val_probs)
val_precision, val_recall, _ = precision_recall_curve(y_val, val_probs)
val_pr_auc = auc(val_recall, val_precision)
val_preds = (val_probs >= np.percentile(val_probs, 90)).astype(int)
val_f1 = f1_score(y_val, val_preds, zero_division=0)

test_probs = model.predict_proba(X_test_sub)[:, 1]
test_roc = roc_auc_score(y_test_sub, test_probs)
test_precision, test_recall, _ = precision_recall_curve(y_test_sub, test_probs)
test_pr_auc = auc(test_recall, test_precision)
test_preds = (test_probs >= np.percentile(test_probs, 90)).astype(int)
test_f1 = f1_score(y_test_sub, test_preds, zero_division=0)

print(f"[검증 구간] ROC-AUC: {val_roc:.3f} | PR-AUC: {val_pr_auc:.4f} | F1: {val_f1:.3f}")
print(f"[테스트 구간] ROC-AUC: {test_roc:.3f} | PR-AUC: {test_pr_auc:.4f} | F1: {test_f1:.3f}")

# 5. 모델 및 메트릭 저장 (루트 및 data/ 폴더 동시 저장)
joblib.dump(model, BASE_DIR / "road_doctor_model.pkl")

metrics_data = {
    "metrics": {
        "validation": {"roc_auc": val_roc, "pr_auc": val_pr_auc, "f1": val_f1},
        "test": {"roc_auc": test_roc, "pr_auc": test_pr_auc, "f1": test_f1},
    }
}

# 루트와 data/ 폴더 양쪽에 모두 저장하여 앱이 무조건 읽도록 함
for p in [BASE_DIR / "metrics.json", BASE_DIR / "data" / "metrics.json"]:
  p.parent.mkdir(parents=True, exist_ok=True)
  p.write_text(json.dumps(metrics_data, indent=4), encoding="utf-8")

print("모델과 메트릭이 모든 경로에 성공적으로 저장되었습니다!")