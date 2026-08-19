import pandas as pd
import streamlit as st
import pydeck as pdk
from pathlib import Path


DATA_FILE = Path(
    "data/final_road_risk.csv"
)


st.set_page_config(
    page_title="Road Doctor",
    page_icon="🛣️",
    layout="wide",
)


@st.cache_data
def load_data():

    df = pd.read_csv(
        DATA_FILE,
        encoding="utf-8-sig",
        low_memory=False,
    )

    df["lat"] = pd.to_numeric(
        df["lat"],
        errors="coerce"
    )

    df["lon"] = pd.to_numeric(
        df["lon"],
        errors="coerce"
    )

    df["risk_score"] = pd.to_numeric(
        df["risk_score"],
        errors="coerce"
    )

    df["logistic_score"] = pd.to_numeric(
        df["logistic_score"],
        errors="coerce"
    )

    df = df.dropna(
        subset=[
            "lat",
            "lon",
            "risk_score",
        ]
    ).copy()

    return df


def get_color(
    risk_level
):

    color_map = {
        "매우 위험": [220, 30, 30, 220],
        "위험": [255, 100, 30, 210],
        "주의": [255, 190, 30, 200],
        "관심": [80, 160, 255, 190],
        "일반": [80, 190, 120, 170],
    }

    return color_map.get(
        risk_level,
        [150, 150, 150, 170]
    )


df = load_data()


df["map_color"] = (
    df["risk_level"]
    .apply(
        get_color
    )
)


st.title(
    "🛣️ Road Doctor"
)

st.caption(
    "포트홀 발생 데이터를 학습한 Logistic Regression 기반 도로 위험도"
)


# ============================================================
# 사이드바 필터
# ============================================================

st.sidebar.header(
    "지도 필터"
)


cities = sorted(
    df[
        "city"
    ]
    .dropna()
    .unique()
)


selected_cities = st.sidebar.multiselect(
    "지역",
    options=cities,
    default=cities,
)


risk_levels = [
    "매우 위험",
    "위험",
    "주의",
    "관심",
    "일반",
]


selected_levels = st.sidebar.multiselect(
    "위험 등급",
    options=risk_levels,
    default=risk_levels,
)


minimum_score = st.sidebar.slider(
    "최소 위험도 점수",
    min_value=0,
    max_value=100,
    value=0,
    step=1,
)


filtered = df[
    (
        df["city"]
        .isin(
            selected_cities
        )
    )
    &
    (
        df["risk_level"]
        .isin(
            selected_levels
        )
    )
    &
    (
        df["risk_score"]
        >= minimum_score
    )
].copy()


# ============================================================
# 상단 지표
# ============================================================

col1, col2, col3, col4 = st.columns(
    4
)


col1.metric(
    "전체 도로 포인트",
    f"{len(df):,}"
)


col2.metric(
    "현재 표시 포인트",
    f"{len(filtered):,}"
)


danger_count = len(
    filtered[
        filtered[
            "risk_level"
        ]
        .isin(
            [
                "매우 위험",
                "위험",
            ]
        )
    ]
)


col3.metric(
    "위험 이상",
    f"{danger_count:,}"
)


very_danger_count = len(
    filtered[
        filtered[
            "risk_level"
        ]
        == "매우 위험"
    ]
)


col4.metric(
    "매우 위험",
    f"{very_danger_count:,}"
)


# ============================================================
# 지도 중심
# ============================================================

if len(filtered) > 0:

    center_lat = (
        filtered[
            "lat"
        ]
        .mean()
    )

    center_lon = (
        filtered[
            "lon"
        ]
        .mean()
    )

else:

    center_lat = (
        df[
            "lat"
        ]
        .mean()
    )

    center_lon = (
        df[
            "lon"
        ]
        .mean()
    )


# ============================================================
# PyDeck 레이어
# ============================================================

scatter_layer = pdk.Layer(
    "ScatterplotLayer",
    data=filtered,
    get_position=[
        "lon",
        "lat",
    ],
    get_fill_color="map_color",
    get_line_color=[
        20,
        20,
        20,
        180,
    ],
    get_radius=(
        "80 + risk_score * 2"
    ),
    radius_min_pixels=4,
    radius_max_pixels=14,
    line_width_min_pixels=1,
    stroked=True,
    filled=True,
    pickable=True,
    auto_highlight=True,
)


view_state = pdk.ViewState(
    latitude=center_lat,
    longitude=center_lon,
    zoom=8.5,
    pitch=0,
)


tooltip = {
    "html": """
        <b>{city}</b><br/>
        위험도: <b>{risk_score}</b>점<br/>
        등급: <b>{risk_level}</b><br/>
        위험순위: {risk_rank_group}<br/>
        주요 원인: {main_risk_factor}<br/>
        Logistic score: {logistic_score}
    """,
    "style": {
        "backgroundColor": "rgba(20, 20, 20, 0.9)",
        "color": "white",
    },
}


deck = pdk.Deck(
    layers=[
        scatter_layer
    ],
    initial_view_state=view_state,
    tooltip=tooltip,
)


st.pydeck_chart(
    deck,
    use_container_width=True,
)


# ============================================================
# 위험도 범례
# ============================================================

st.markdown(
    """
### 위험도 등급

🔴 **매우 위험** — 상위 1%  
🟠 **위험** — 상위 1~5%  
🟡 **주의** — 상위 5~15%  
🔵 **관심** — 상위 15~30%  
🟢 **일반** — 하위 70%
"""
)


# ============================================================
# 위험도 상위 구간
# ============================================================

st.subheader(
    "위험도 상위 도로 포인트"
)


table_columns = [
    "point_id",
    "city",
    "road_name",
    "risk_score",
    "risk_level",
    "risk_rank_group",
    "main_risk_factor",
    "freeze_thaw_14d",
    "rain_7d",
    "rain_14d",
    "sewer_old30_ratio",
    "traffic_esal",
    "days_since_last_repair",
]


table_columns = [
    col
    for col in table_columns
    if col in filtered.columns
]


top_df = (
    filtered[
        table_columns
    ]
    .sort_values(
        "risk_score",
        ascending=False
    )
    .head(30)
)


st.dataframe(
    top_df,
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# 지역별 요약
# ============================================================

st.subheader(
    "지역별 위험도"
)


city_summary = (
    filtered
    .groupby(
        "city"
    )
    .agg(
        평균_위험도=(
            "risk_score",
            "mean"
        ),
        최대_위험도=(
            "risk_score",
            "max"
        ),
        위험_이상=(
            "risk_level",
            lambda x:
                x.isin(
                    [
                        "매우 위험",
                        "위험",
                    ]
                ).sum()
        ),
        포인트수=(
            "point_id",
            "count"
        ),
    )
    .reset_index()
)


city_summary[
    "평균_위험도"
] = (
    city_summary[
        "평균_위험도"
    ]
    .round(1)
)


city_summary[
    "최대_위험도"
] = (
    city_summary[
        "최대_위험도"
    ]
    .round(1)
)


st.dataframe(
    city_summary,
    use_container_width=True,
    hide_index=True,
)