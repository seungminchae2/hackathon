from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.common import load_config, resolve_path
from src.features import risk_level_from_percentile
from src.kakao_route_component import show_kakao_map
from src.routing import KakaoApiError, build_safe_route_plan


BASE_DIR = Path(__file__).resolve().parent


def load_app_env(path: Path) -> None:
    """실행 중 .env가 바뀌어도 앱 키를 즉시 갱신합니다."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ[key.strip()] = value.strip().strip('"').strip("'")


load_app_env(BASE_DIR / ".env")
config = load_config(BASE_DIR / "config.yaml")

st.set_page_config(page_title="Road Doctor", page_icon="🛣️", layout="wide")
st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1500px;}
    [data-testid="stMetric"] {background:#f7f9f6; border:1px solid #e2e8e3; padding:14px 16px; border-radius:14px;}
    [data-testid="stMetric"] * {color:#10231c !important;}
    h1 {letter-spacing:-0.04em;}
    </style>
    """,
    unsafe_allow_html=True,
)


def normalize_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    """기존 CSV도 화면에서 열리도록 새 컬럼의 안전한 기본값을 만듭니다."""
    out = frame.copy()
    numeric_columns = [
        "grid_lat",
        "grid_lon",
        "risk_score",
        "risk_percentile",
        "priority_score",
        "priority_rank",
        "recurrence_score",
        "importance_score",
        "precip_3d",
        "precip_7d",
        "freeze_thaw_7d",
        "past_potholes_90d",
        "past_potholes_total",
        "has_repair_history",
        "days_since_last_repair",
    ]
    for column in numeric_columns:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")

    out["risk_score"] = out.get("risk_score", pd.Series(0.0, index=out.index)).fillna(0).clip(0, 1)
    if "risk_percentile" not in out:
        out["risk_percentile"] = out["risk_score"].rank(method="average", pct=True)
    if "risk_level" not in out:
        out["risk_level"] = out["risk_percentile"].map(risk_level_from_percentile)
    if "past_potholes_total" not in out:
        out["past_potholes_total"] = out.get("past_potholes_90d", 0)
    if "recurrence_score" not in out:
        source = out["past_potholes_90d"] if "past_potholes_90d" in out else pd.Series(0.0, index=out.index)
        recurrence = pd.to_numeric(source, errors="coerce").fillna(0)
        scale = max(float(recurrence.quantile(0.95)), 1.0)
        out["recurrence_score"] = recurrence.div(scale).clip(upper=1)
    if "importance_score" not in out:
        out["importance_score"] = np.nan
    if "priority_score" not in out:
        out["priority_score"] = (out["risk_score"] * 0.75 + out["recurrence_score"] * 0.15) / 0.90
    out = out.sort_values(
        ["priority_score", "risk_score", "grid_id"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    if "priority_rank" not in frame.columns:
        out["priority_rank"] = out.index + 1
    return out


prediction_path = resolve_path(config, "predictions")
if not prediction_path.exists():
    st.error("예측 파일이 없습니다. `python -m src.predict --config config.yaml`을 먼저 실행하십시오.")
    st.stop()

pred = normalize_predictions(pd.read_csv(prediction_path))
if pred.empty:
    st.error("예측 파일에 데이터가 없습니다.")
    st.stop()

key_env_name = config.get("kakao", {}).get("app_key_env", "KAKAO_MAP_APP_KEY")
kakao_key = os.getenv(key_env_name, "").strip()
rest_key_env_name = config.get("kakao", {}).get("rest_api_key_env", "KAKAO_REST_API_KEY")
kakao_rest_key = os.getenv(rest_key_env_name, "").strip()

st.title("Road Doctor")
st.caption("전북 500m 격자별 포트홀 상대 위험도 · 모델 위험 + 재발 위험 + 도로 중요도 기반 보수 우선순위")

very_high_count = int(pred["risk_level"].eq("매우 높음").sum())
high_count = int(pred["risk_level"].isin(["매우 높음", "높음"]).sum())
c1, c2, c3, c4 = st.columns(4)
c1.metric("예측 기준일", str(pred.get("prediction_date", pd.Series(["-"])).iloc[0]))
c2.metric("최고 모델 위험도", f"{float(pred['risk_score'].max()):.1%}")
c3.metric("고위험 격자", f"{high_count:,}개")
c4.metric("매우 고위험", f"{very_high_count:,}개")

if not kakao_key:
    st.warning(
        f"카카오 지도를 표시하려면 프로젝트 루트의 `.env`에 "
        f"`{key_env_name}=발급받은_JavaScript_키`를 입력하십시오. 목록은 키 없이도 동작합니다."
    )

st.subheader("포트홀 위험 회피 경로")
with st.form("safe_route_form", border=True):
    route_col1, route_col2, route_col3 = st.columns([1, 1, 0.32], vertical_alignment="bottom")
    with route_col1:
        origin_query = st.text_input(
            "출발지",
            placeholder="예: 전북대학교 또는 전주시 덕진구 백제대로 567",
        )
    with route_col2:
        destination_query = st.text_input(
            "도착지",
            placeholder="예: 전주역 또는 전주시 덕진구 동부대로 680",
        )
    with route_col3:
        route_submitted = st.form_submit_button("안전 경로 찾기", type="primary", width="stretch")

if route_submitted:
    if not kakao_rest_key:
        st.error(
            f"길찾기에는 REST API 키가 필요합니다. `.env`에 "
            f"`{rest_key_env_name}=발급받은_REST_API_키`를 추가하십시오."
        )
    else:
        try:
            with st.spinner("대안 경로와 포트홀 위험 구간을 비교하고 있습니더..."):
                st.session_state["safe_route_plan"] = build_safe_route_plan(
                    origin_query,
                    destination_query,
                    kakao_rest_key,
                    pred,
                )
        except KakaoApiError as exc:
            st.session_state.pop("safe_route_plan", None)
            st.error(str(exc))

route_plan = st.session_state.get("safe_route_plan")
if route_plan:
    recommended = route_plan["recommended"]
    if route_plan["is_detour"]:
        st.success(f"안전 우회 경로 추천: {route_plan['message']}")
    else:
        st.info(route_plan["message"])
    route_m1, route_m2, route_m3, route_m4 = st.columns(4)
    route_m1.metric("추천 경로", recommended["label"])
    route_m2.metric("예상 시간", f"{recommended['duration_min']:.0f}분")
    route_m3.metric("이동 거리", f"{recommended['distance_km']:.1f}km")
    route_m4.metric("고위험 격자", f"{recommended['high_risk_count']}개")
    road_names = recommended.get("road_names") or []
    if road_names:
        st.caption("주요 통과 도로: " + " · ".join(road_names))

show_kakao_map(pred, kakao_key, height=730, route_plan=route_plan)

tab_components, tab_model, tab_raw = st.tabs(["우선순위 구성", "모델 검증", "전체 예측 데이터"])
with tab_components:
    st.caption("교통량/도로중요도가 없으면 0.75:0.15 가중치를 합이 1이 되도록 재정규화합니다.")
    display_columns = [
        "priority_rank",
        "grid_id",
        "priority_score",
        "risk_score",
        "recurrence_score",
        "importance_score",
        "risk_reason",
    ]
    st.dataframe(
        pred[[column for column in display_columns if column in pred.columns]].head(30),
        hide_index=True,
        width="stretch",
        column_config={
            "priority_score": st.column_config.ProgressColumn("우선순위", min_value=0, max_value=1, format="%.3f"),
            "risk_score": st.column_config.ProgressColumn("모델 위험", min_value=0, max_value=1, format="%.3f"),
            "recurrence_score": st.column_config.ProgressColumn("재발 위험", min_value=0, max_value=1, format="%.3f"),
        },
    )

with tab_model:
    metrics_path = resolve_path(config, "metrics")
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        test_metrics = metrics.get("metrics", {}).get("test", {})
        validation_metrics = metrics.get("metrics", {}).get("validation", {})
        m1, m2, m3 = st.columns(3)
        m1.metric("Test ROC-AUC", f"{test_metrics.get('roc_auc', 0):.3f}")
        m2.metric("Test PR-AUC", f"{test_metrics.get('pr_auc', 0):.3f}")
        m3.metric("Test F1", f"{test_metrics.get('f1', 0):.3f}")
        st.caption(
            f"검증 구간에서 선택한 분류 임계값 {validation_metrics.get('threshold', 0.5):.4f}를 "
            "테스트 구간에 그대로 적용했습니다. risk_score는 보정 전 상대 위험 점수입니다."
        )
        st.json(metrics, expanded=False)
    else:
        st.info("새 모델을 학습하면 `outputs/metrics_v2.json`에 검증 지표가 저장됩니다.")

with tab_raw:
    st.dataframe(pred, hide_index=True, width="stretch")
    st.download_button(
        "예측 CSV 내려받기",
        pred.to_csv(index=False).encode("utf-8-sig"),
        file_name="road_doctor_predictions.csv",
        mime="text/csv",
    )
