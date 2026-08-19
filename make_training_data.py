from pathlib import Path
import numpy as np
import pandas as pd

# 파일 경로 설정
RISK_FILE = Path("data/road_risk_latest.csv")  # 또는 road_risk_all_dates.csv
POTHOLE_FILE = Path("data/potholes.csv")
REPAIRS_FILE = Path("data/repairs.csv")
SEWER_EXCEL_FILE = Path("전북5개시_하수관로_노후도.xlsx")
OUTPUT_FILE = Path("data/pothole_training_data.csv")

POTHOLE_MATCH_RADIUS_M = 200
PREDICTION_DAYS = 7
REPAIR_MATCH_RADIUS_M = 300


def haversine_distance(lat1, lon1, lat2, lon2):
  earth_radius_m = 6371000
  lat1 = np.radians(lat1)
  lon1 = np.radians(lon1)
  lat2 = np.radians(lat2)
  lon2 = np.radians(lon2)

  dlat = lat2 - lat1
  dlon = lon2 - lon1

  a = (
      np.sin(dlat / 2) ** 2
      + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
  )
  c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
  return earth_radius_m * c


print("1. 도로 위험도 및 기상 데이터 불러오는 중...")
if not RISK_FILE.exists():
  alt_risk = Path("data/final_road_risk.csv")
  if alt_risk.exists():
    RISK_FILE = alt_risk
  else:
    raise FileNotFoundError(
        f"위험도 데이터 파일을 찾을 수 없습니다: {RISK_FILE}"
    )

risk = pd.read_csv(RISK_FILE, encoding="utf-8-sig", low_memory=False)

# 필수 컬럼 검증 및 정제
if "date" not in risk.columns:
  risk["date"] = pd.to_datetime(
      "2025-01-01"
  )  # 고정일자 필요시 (# 기호 추가 완료)
else:
  risk["date"] = pd.to_datetime(risk["date"], errors="coerce")

risk["lat"] = pd.to_numeric(risk["lat"], errors="coerce")
risk["lon"] = pd.to_numeric(risk["lon"], errors="coerce")

if "point_id" not in risk.columns:
  risk["point_id"] = range(len(risk))

risk = risk.dropna(subset=["point_id", "date", "lat", "lon"]).copy()
risk["point_id"] = risk["point_id"].astype(int)

# ============================================================
# 2. 하수관로 노후도 데이터 병합
# ============================================================
print("2. 하수관로 노후도 데이터 병합 중...")
if SEWER_EXCEL_FILE.exists():
  try:
    sewer_summary = pd.read_excel(
        SEWER_EXCEL_FILE, sheet_name="노후도_요약", skiprows=2
    )
    sewer_summary.columns = [
        "city",
        "total_len",
        "old_len",
        "sewer_old30_ratio",
        "avg_age",
    ]

    if "city" in risk.columns:
      risk = pd.merge(
          risk,
          sewer_summary[["city", "sewer_old30_ratio"]],
          on="city",
          how="left",
      )
    else:
      risk["sewer_old30_ratio"] = 0.44
  except Exception as e:
    print(f"하수관로 노후도 병합 중 예외 발생 (기본값 처리): {e}")
    risk["sewer_old30_ratio"] = 0.0
else:
  risk["sewer_old30_ratio"] = 0.0

# ============================================================
# 3. 고유 도로 포인트 추출
# ============================================================
points = (
    risk[["point_id", "lat", "lon"]]
    .drop_duplicates(subset=["point_id"])
    .sort_values("point_id")
    .reset_index(drop=True)
)
point_lats = points["lat"].to_numpy()
point_lons = points["lon"].to_numpy()

# ============================================================
# 4. 보수이력 매칭 (Data Leakage 방지)
# ============================================================
print("3. 보수 이력 매칭 및 경과일 계산 중...")
if REPAIRS_FILE.exists():
  repairs = pd.read_csv(REPAIRS_FILE, encoding="utf-8-sig")
  repairs["repair_date"] = pd.to_datetime(
      repairs["repair_date"], errors="coerce"
  )
  repairs["lat"] = pd.to_numeric(repairs["lat"], errors="coerce")
  repairs["lon"] = pd.to_numeric(repairs["lon"], errors="coerce")
  repairs = repairs.dropna(subset=["repair_date", "lat", "lon"]).copy()

  repair_matches = []
  for _, repair in repairs.iterrows():
    distances = haversine_distance(
        repair["lat"], repair["lon"], point_lats, point_lons
    )
    nearby_positions = np.where(distances <= REPAIR_MATCH_RADIUS_M)[0]
    for pos in nearby_positions:
      pt = points.iloc[pos]
      repair_matches.append({
          "point_id": int(pt["point_id"]),
          "repair_date": repair["repair_date"],
          "repair_distance_m": float(distances[pos]),
      })

  repair_history = pd.DataFrame(repair_matches)
  if not repair_history.empty:
    repair_history = repair_history.sort_values(
        ["point_id", "repair_date"]
    ).drop_duplicates(subset=["point_id", "repair_date"], keep="first")

    merged_groups = []
    for pid, group in risk.groupby("point_id", sort=False):
      group = group.sort_values("date").copy()
      prep = repair_history[repair_history["point_id"] == pid].sort_values(
          "repair_date"
      )
      if prep.empty:
        group["repair_date"] = pd.NaT
        merged_groups.append(group)
        continue
      merged = pd.merge_asof(
          group,
          prep[["repair_date"]],
          left_on="date",
          right_on="repair_date",
          direction="backward",
          allow_exact_matches=True,
      )
      merged_groups.append(merged)
    risk = pd.concat(merged_groups, ignore_index=True)
  else:
    risk["repair_date"] = pd.NaT
else:
  risk["repair_date"] = pd.NaT

risk["has_past_repair"] = risk["repair_date"].notna().astype(int)
risk["days_since_last_repair"] = (risk["date"] - risk["repair_date"]).dt.days

# ============================================================
# 5. 포트홀 발생 데이터 매칭 및 7일 후행 라벨(Label) 생성
# ============================================================
print("4. 포트홀 라벨링 (향후 7일 예측 대상) 생성 중...")
if POTHOLE_FILE.exists():
  potholes = pd.read_csv(POTHOLE_FILE, encoding="utf-8-sig")
  potholes["event_date"] = pd.to_datetime(
      potholes["event_date"], errors="coerce"
  )
  potholes["lat"] = pd.to_numeric(potholes["lat"], errors="coerce")
  potholes["lon"] = pd.to_numeric(potholes["lon"], errors="coerce")
  potholes = potholes.dropna(subset=["event_date", "lat", "lon"]).copy()

  pothole_matches = []
  for _, ph in potholes.iterrows():
    distances = haversine_distance(ph["lat"], ph["lon"], point_lats, point_lons)
    nearby_positions = np.where(distances <= POTHOLE_MATCH_RADIUS_M)[0]
    for pos in nearby_positions:
      pt = points.iloc[pos]
      pothole_matches.append({
          "event_date": ph["event_date"],
          "point_id": int(pt["point_id"]),
      })

  label_counts = {}
  for m in pothole_matches:
    pid = m["point_id"]
    ed = m["event_date"]
    for d in range(1, PREDICTION_DAYS + 1):
      pdate = ed - pd.Timedelta(days=d)
      label_counts[(pid, pdate)] = label_counts.get((pid, pdate), 0) + 1

  risk_keys = list(zip(risk["point_id"], risk["date"]))
  risk["pothole_count_7d"] = [label_counts.get(k, 0) for k in risk_keys]
  risk["pothole_label"] = (risk["pothole_count_7d"] > 0).astype(int)

  last_ph_date = potholes["event_date"].max()
  if pd.notna(last_ph_date):
    valid_end = last_ph_date - pd.Timedelta(days=PREDICTION_DAYS)
    risk = risk[risk["date"] <= valid_end].copy()
else:
  risk["pothole_label"] = 0

# ============================================================
# 6. 최종 학습용 컬럼 선택 및 저장
# ============================================================
candidate_cols = [
    "point_id",
    "date",
    "lat",
    "lon",
    "freeze_thaw_14d",
    "rain_7d",
    "rain_14d",
    "sewer_old30_ratio",
    "has_past_repair",
    "days_since_last_repair",
    "pothole_label",
]

training_columns = [c for c in candidate_cols if c in risk.columns]
training = risk[training_columns].copy()

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
training.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

print("\n머신러닝 학습 데이터 생성 완료!")
print(f"저장 위치: {OUTPUT_FILE}")
print(f"전체 행 수: {len(training):,}")
print(f"양성(포트홀 발생) 샘플 수: {int(training['pothole_label'].sum()):,}")

# make_training_data.py의 13번 항목(학습 피처 선택) 부분을 아래 내용으로 교체하세요.

# [추가할 피처 생성 로직]
risk['rain_change_rate'] = risk['rain_7d'] / (risk['rain_14d'] / 2 + 1e-6)
risk['freeze_thaw_spike'] = (risk['freeze_thaw_14d'] > 5).astype(int)

# [최종 학습 피처 리스트]
training_columns = [
    "point_id", "date", "lat", "lon",
    "freeze_thaw_14d", "freeze_thaw_spike",  # 보강된 동결융해
    "rain_7d", "rain_14d", "rain_change_rate", # 보강된 강수
    "sewer_old30_ratio",
    "has_past_repair", "days_since_last_repair",
    "pothole_label"
]