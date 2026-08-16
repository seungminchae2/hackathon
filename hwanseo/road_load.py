import pandas as pd
from pathlib import Path

INPUT_FILE = Path("data/traffic.csv")
OUTPUT_FILE = Path("data/traffic_with_esal.csv")

ESAL_FACTORS = {
    "승용차": 0.0002,
    "버스": 0.852,
    "소형화물": 0.004,
    "중형화물": 1.735,
    "대형화물": 3.169,
}

df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")

missing_columns = [
    col for col in ESAL_FACTORS
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"CSV에 다음 컬럼이 없습니다: {missing_columns}\n"
        f"현재 컬럼: {df.columns.tolist()}"
    )

for col in ESAL_FACTORS:
    df[col] = (
        df[col]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
    )

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    ).fillna(0)

for vehicle, factor in ESAL_FACTORS.items():
    df[f"{vehicle}_ESAL"] = df[vehicle] * factor

df["traffic_esal"] = sum(
    df[f"{vehicle}_ESAL"]
    for vehicle in ESAL_FACTORS
)

df["total_traffic"] = (
    df[list(ESAL_FACTORS.keys())]
    .sum(axis=1)
)

df["heavy_traffic"] = (
    df["중형화물"]
    + df["대형화물"]
)

df["heavy_vehicle_ratio"] = (
    df["heavy_traffic"]
    / df["total_traffic"].replace(0, pd.NA)
)

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print("ESAL 계산 완료")
print(f"저장 위치: {OUTPUT_FILE}")

print(
    df[
        [
            "승용차",
            "버스",
            "소형화물",
            "중형화물",
            "대형화물",
            "total_traffic",
            "heavy_vehicle_ratio",
            "traffic_esal",
        ]
    ].head()
)