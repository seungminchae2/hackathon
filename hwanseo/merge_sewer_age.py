import pandas as pd
from pathlib import Path


ROADS_FILE = Path("data/roads.csv")
SEWER_FILE = Path("data/전북5개시_하수관로_노후도.xlsx")
OUTPUT_FILE = Path("data/roads_with_sewer_age.csv")


TARGET_CITIES = [
    "전주시",
    "군산시",
    "익산시",
    "남원시",
    "정읍시",
]


def normalize_city(value):
    if pd.isna(value):
        return None

    text = str(value).strip()

    city_aliases = {
        "전주": "전주시",
        "전주시": "전주시",
        "군산": "군산시",
        "군산시": "군산시",
        "익산": "익산시",
        "익산시": "익산시",
        "남원": "남원시",
        "남원시": "남원시",
        "정읍": "정읍시",
        "정읍시": "정읍시",
    }

    return city_aliases.get(text)


def find_city_in_text(value):
    if pd.isna(value):
        return None

    text = str(value)

    aliases = {
        "전주시": ["전주시", "전주"],
        "군산시": ["군산시", "군산"],
        "익산시": ["익산시", "익산"],
        "남원시": ["남원시", "남원"],
        "정읍시": ["정읍시", "정읍"],
    }

    for city, keywords in aliases.items():
        for keyword in keywords:
            if keyword in text:
                return city

    return None


roads = pd.read_csv(
    ROADS_FILE,
    encoding="utf-8-sig"
)


sewer = pd.read_excel(
    SEWER_FILE,
    sheet_name="노후도_요약",
    header=3
)

sewer = sewer.iloc[:, :5].copy()

sewer.columns = [
    "city",
    "total_sewer_length",
    "old_sewer_length",
    "sewer_old30_ratio",
    "average_sewer_age",
]

sewer = sewer[
    sewer["city"].isin(TARGET_CITIES)
].copy()

sewer["sewer_old30_ratio"] = pd.to_numeric(
    sewer["sewer_old30_ratio"],
    errors="coerce"
)

sewer["average_sewer_age"] = pd.to_numeric(
    sewer["average_sewer_age"],
    errors="coerce"
)


road_mapping = pd.read_excel(
    SEWER_FILE,
    sheet_name="road_name_매핑",
    header=3
)

road_mapping = road_mapping.iloc[:, :2].copy()

road_mapping.columns = [
    "road_name",
    "mapped_city",
]

road_mapping = road_mapping[
    road_mapping["road_name"].notna()
].copy()


if "city" in roads.columns:
    roads["city"] = roads["city"].apply(normalize_city)

else:
    roads["city"] = None


if "road_name" in roads.columns:
    roads = roads.merge(
        road_mapping,
        on="road_name",
        how="left"
    )

    roads["mapped_city"] = (
        roads["mapped_city"]
        .apply(normalize_city)
    )

    roads["city"] = roads["city"].fillna(
        roads["mapped_city"]
    )

    roads["city"] = roads["city"].fillna(
        roads["road_name"].apply(find_city_in_text)
    )

    roads = roads.drop(
        columns=["mapped_city"]
    )


text_columns = [
    col
    for col in [
        "road_name",
        "name",
        "address",
        "구간명",
        "도로명",
    ]
    if col in roads.columns
]


for col in text_columns:

    missing = roads["city"].isna()

    roads.loc[
        missing,
        "city"
    ] = (
        roads.loc[
            missing,
            col
        ]
        .apply(find_city_in_text)
    )


roads = roads.merge(
    sewer[
        [
            "city",
            "total_sewer_length",
            "old_sewer_length",
            "sewer_old30_ratio",
            "average_sewer_age",
        ]
    ],
    on="city",
    how="left"
)


roads["sewer_match_status"] = (
    roads["sewer_old30_ratio"]
    .notna()
    .map({
        True: "matched",
        False: "unmatched",
    })
)


OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

roads.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


print("도로 포인트 + 하수관로 노후도 결합 완료")
print(f"저장 위치: {OUTPUT_FILE}")
print(f"전체 도로 포인트 수: {len(roads)}")

print(
    "하수관로 매칭 성공:",
    (roads["sewer_match_status"] == "matched").sum()
)

print(
    "하수관로 매칭 실패:",
    (roads["sewer_match_status"] == "unmatched").sum()
)


print("\n시별 포인트 수")

print(
    roads["city"]
    .value_counts(
        dropna=False
    )
)


print("\n하수관로 데이터 확인")

check_columns = [
    col
    for col in [
        "road_name",
        "city",
        "lat",
        "lon",
        "sewer_old30_ratio",
        "average_sewer_age",
        "sewer_match_status",
    ]
    if col in roads.columns
]

print(
    roads[
        check_columns
    ]
    .head(20)
    .to_string(index=False)
)