from __future__ import annotations

from pathlib import Path

import pandas as pd
import pydeck as pdk
import streamlit as st
import yaml


st.set_page_config(page_title="Road Doctor", page_icon="🛣️", layout="wide")
st.title("Road Doctor — 전북 포트홀 위험 예측")
st.caption("과거 민원·보수 이력과 기상정보를 결합한 500m 격자별 상대 위험도")

BASE_DIR = Path(__file__).resolve().parent
with (BASE_DIR / "config.yaml").open("r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
prediction_path = BASE_DIR / config["paths"]["predictions"]

if not prediction_path.exists():
    st.error("예측 파일이 없습니다. `python -m src.predict --config config.yaml`을 먼저 실행하십시오.")
    st.stop()

pred = pd.read_csv(prediction_path)
pred["risk_score"] = pd.to_numeric(pred["risk_score"], errors="coerce").fillna(0)

max_risk = float(pred["risk_score"].max())
high_count = int((pred["risk_score"] >= 0.5).sum())
very_high_count = int((pred["risk_score"] >= 0.75).sum())

c1, c2, c3, c4 = st.columns(4)
c1.metric("예측 기준일", str(pred["prediction_date"].iloc[0]))
c2.metric("최고 위험도", f"{max_risk:.1%}")
c3.metric("고위험 격자", f"{high_count}개")
c4.metric("매우 고위험", f"{very_high_count}개")

left, right = st.columns([2.1, 1])
with left:
    st.subheader("포트홀 위험도 지도")
    view_state = pdk.ViewState(
        latitude=float(pred["grid_lat"].mean()),
        longitude=float(pred["grid_lon"].mean()),
        zoom=9.2,
        pitch=35,
    )
    layers = [
        pdk.Layer(
            "HeatmapLayer",
            data=pred,
            get_position="[grid_lon, grid_lat]",
            get_weight="risk_score",
            radius_pixels=45,
            intensity=1.2,
            threshold=0.05,
        ),
        pdk.Layer(
            "ScatterplotLayer",
            data=pred.head(50),
            get_position="[grid_lon, grid_lat]",
            get_radius="80 + risk_score * 250",
            get_fill_color="[255, 80 * (1-risk_score), 20, 180]",
            pickable=True,
        ),
    ]
    tooltip = {
        "html": "<b>위험도:</b> {risk_score}<br/><b>등급:</b> {risk_level}<br/><b>원인:</b> {risk_reason}<br/><b>우선순위:</b> {priority_rank}",
        "style": {"backgroundColor": "#222", "color": "white"},
    }
    st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view_state, tooltip=tooltip))

with right:
    st.subheader("우선 보수 대상")
    top_n = st.slider("표시 개수", min_value=5, max_value=30, value=10)
    table = pred.head(top_n)[
        ["priority_rank", "risk_score", "risk_level", "risk_reason", "grid_lat", "grid_lon"]
    ].copy()
    table["risk_score"] = table["risk_score"].map(lambda x: f"{x:.1%}")
    st.dataframe(table, hide_index=True, use_container_width=True)

st.subheader("위험 요인 분포")
reason_counts = (
    pred.assign(reason=pred["risk_reason"].str.split(", "))
    .explode("reason")
    .groupby("reason", as_index=False)
    .size()
    .sort_values("size", ascending=False)
)
st.bar_chart(reason_counts.set_index("reason"))

with st.expander("원본 예측 데이터"):
    st.dataframe(pred, hide_index=True, use_container_width=True)
