from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.common import load_config, resolve_path
from src.features import risk_level_from_percentile
from src.kakao_route_component import show_kakao_map


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

st.set_page_config(page_title="전북 특별 민원창구", page_icon="🛣️", layout="wide")
st.markdown(
    """
    <style>
    .block-container {padding-top: 3rem; padding-bottom: 3rem; padding-left: 2rem; padding-right: 2rem; max-width: 100%;}
    [data-testid="stMetric"] {background:#f7f9f6; border:1px solid #e2e8e3; padding:14px 16px; border-radius:14px;}
    [data-testid="stMetric"] * {color:#10231c !important;}
    .st-key-rd-top-metrics [data-testid="stMetric"] {text-align:right;}
    .st-key-rd-top-metrics [data-testid="stMetricValue"] {font-size:29px;}
    .st-key-rd-top-metrics [data-testid="stMetricLabel"] {font-size:11px; justify-content:flex-end;}
    h1 {letter-spacing:-0.04em;}

    .rd-header {margin-top: 0.2rem; margin-bottom: 1.2rem; line-height: 1.6;}
    .rd-header .rd-title {font-size: 1.35rem; font-weight: 800; letter-spacing: -0.03em; color:#10231c; display:block;}
    .rd-header .rd-caption {font-size: 0.78rem; color:#5b6b63; display:block; margin-top:4px;}

    .rd-board-title {font-size: 1.0rem; font-weight: 700; margin-bottom: 0.4rem; color:#10231c;}
    .rd-board-sub {font-size: 0.75rem; color:#7a8a82; margin-bottom: 0.6rem;}
    .rd-item {padding: 8px 4px; border-bottom: 1px solid #ecefec; animation: rd-slide-in 0.45s cubic-bezier(.2,.8,.3,1) both;}
    .rd-item:last-child {border-bottom: none;}

    @keyframes rd-slide-in {
        0% {transform: translateY(16px); opacity: 0;}
        100% {transform: translateY(0); opacity: 1;}
    }
    .rd-item:nth-child(1) {animation-delay: 0s;}
    .rd-item:nth-child(2) {animation-delay: .04s;}
    .rd-item:nth-child(3) {animation-delay: .08s;}
    .rd-item:nth-child(4) {animation-delay: .12s;}
    .rd-item:nth-child(5) {animation-delay: .16s;}
    .rd-item:nth-child(6) {animation-delay: .2s;}
    .rd-item:nth-child(n+7) {animation-delay: .24s;}
    .rd-item .rd-item-date {font-size: 0.72rem; color:#7a8a82; font-weight:600;}
    .rd-item .rd-item-address {font-size: 0.86rem; color:#10231c; font-weight:600; margin: 1px 0;}
    .rd-item .rd-item-meta {font-size: 0.75rem; color:#5b6b63;}
    .rd-badge {display:inline-block; font-size:0.68rem; font-weight:700; padding:1px 7px; border-radius:999px; margin-right:5px;}
    .rd-badge-alert {background:#fdeceb; color:#c0392b;}
    .rd-badge-done {background:#e8f4ee; color:#13795b;}
    .rd-badge-check {background:#eaf1fb; color:#2c5aa0;}
    .rd-badge-critical {background:#e0332a; color:#fff;}

    @keyframes rd-flash {
        0%, 100% {background:#ffe0dd; border-color:#e0332a; box-shadow:0 0 0 0 rgba(224,51,42,.35);}
        50% {background:#fff6f5; border-color:#f3aca7; box-shadow:0 0 10px 2px rgba(224,51,42,.15);}
    }
    .rd-item-critical {
        animation: rd-slide-in 0.45s cubic-bezier(.2,.8,.3,1) both, rd-flash 1s ease-in-out 0.45s infinite;
        border: 1.5px solid #e0332a;
        border-radius: 10px;
        padding: 8px 10px;
        margin-bottom: 6px;
    }
    .rd-item-critical .rd-item-address {color:#a91e17;}
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
        "avg_temp",
        "min_temp",
        "max_temp",
        "temp_range",
        "precipitation",
        "precip_3d",
        "precip_7d",
        "snowfall",
        "humidity",
        "freeze_thaw",
        "freeze_thaw_7d",
        "past_potholes_30d",
        "past_potholes_90d",
        "past_potholes_total",
        "has_repair_history",
        "days_since_last_repair",
        "month",
        "day_of_year_sin",
        "day_of_year_cos",
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
    if "address" in out.columns:
        missing_address = out["address"].isna() | out["address"].astype(str).str.strip().eq("")
        out.loc[missing_address, "address"] = out.loc[missing_address, "grid_id"]
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


def attach_nearest_address(events: pd.DataFrame, grid: pd.DataFrame) -> pd.DataFrame:
    """이벤트 위경도에서 가장 가까운 격자의 주소를 붙입니다."""
    out = events.copy()
    if out.empty or grid.empty or "grid_lat" not in grid.columns:
        out["address"] = ""
        return out

    from src.common import haversine_distance_matrix

    distances = haversine_distance_matrix(
        out["lat"].to_numpy(),
        out["lon"].to_numpy(),
        grid["grid_lat"].to_numpy(),
        grid["grid_lon"].to_numpy(),
    )
    nearest_idx = distances.argmin(axis=1)
    address_source = grid["address"] if "address" in grid.columns else grid["grid_id"]
    out["address"] = address_source.to_numpy()[nearest_idx]
    return out


def load_recent_events(path: Path, date_column: str, grid: pd.DataFrame, limit: int = 60) -> pd.DataFrame:
    """포트홀/보수 이력 CSV를 최신순으로 불러오고 주소를 붙입니다."""
    if not path.exists():
        return pd.DataFrame()
    events = pd.read_csv(path)
    if events.empty:
        return events
    events[date_column] = pd.to_datetime(events[date_column], errors="coerce")
    events = events.dropna(subset=[date_column]).sort_values(date_column, ascending=False).head(limit)
    return attach_nearest_address(events, grid)


def render_board(title: str, subtitle: str, events: pd.DataFrame, date_column: str, badge: str, badge_class: str, height: int, empty_message: str) -> None:
    st.markdown(f'<div class="rd-board-title">{title}</div><div class="rd-board-sub">{subtitle}</div>', unsafe_allow_html=True)
    with st.container(height=height, border=True):
        if events.empty:
            st.caption(empty_message)
            return
        rows_html = []
        for _, row in events.iterrows():
            event_date = row[date_column].strftime("%Y-%m-%d") if pd.notna(row[date_column]) else "-"
            address = row.get("address", "") or "-"
            meta = row.get("meta_text", "")
            is_critical = bool(row.get("critical", False))
            item_class = "rd-item rd-item-critical" if is_critical else "rd-item"
            item_badge = "🔥 95점 초과" if is_critical else badge
            item_badge_class = "rd-badge-critical" if is_critical else badge_class
            rows_html.append(
                f'<div class="{item_class}">'
                f'<span class="rd-badge {item_badge_class}">{item_badge}</span>'
                f'<span class="rd-item-date">{event_date}</span>'
                f'<div class="rd-item-address">{address}</div>'
                f'<div class="rd-item-meta">{meta}</div>'
                f'</div>'
            )
        st.markdown("".join(rows_html), unsafe_allow_html=True)


def attach_road_authority(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    """주소의 시군명을 기준으로 관할 보수 담당 부서·연락처를 붙입니다."""
    out = df.copy()
    out["road_authority_dept"] = None
    out["road_authority_phone"] = None
    if not path.exists() or "address" not in out.columns:
        return out
    authorities = pd.read_csv(path)
    address = out["address"].astype(str)
    for row in authorities.itertuples(index=False):
        mask = address.str.contains(row.sigungu_name, na=False, regex=False)
        out.loc[mask, "road_authority_dept"] = f"{row.sigungu_name} {row.dept_name}"
        out.loc[mask, "road_authority_phone"] = row.phone_number
    return out


prediction_path = resolve_path(config, "predictions")
if not prediction_path.exists():
    st.error("예측 파일이 없습니다. `python -m src.predict --config config.yaml`을 먼저 실행하십시오.")
    st.stop()

pred = normalize_predictions(pd.read_csv(prediction_path))
if pred.empty:
    st.error("예측 파일에 데이터가 없습니다.")
    st.stop()

current_prediction_date = str(pred.get("prediction_date", pd.Series([""])).iloc[0])
if current_prediction_date != date.today().isoformat():
    banner_col, button_col = st.columns([5, 1], vertical_alignment="center")
    banner_col.info(f"예측 기준일이 {current_prediction_date}로 오늘({date.today().isoformat()})보다 오래됐습니다.")
    if button_col.button("오늘 날짜로 갱신", width="stretch"):
        with st.spinner("오늘 날짜 기준으로 예측을 갱신하는 중입니다... (몇 분 걸릴 수 있습니다. 이 탭을 닫지 마십시오)"):
            try:
                result = subprocess.run(
                    [sys.executable, str(BASE_DIR / "scripts" / "refresh_daily.py")],
                    cwd=BASE_DIR,
                    capture_output=True,
                    text=True,
                    timeout=1800,
                )
            except subprocess.TimeoutExpired:
                st.error("갱신이 30분 넘게 걸려 중단했습니다. 터미널에서 `python scripts/refresh_daily.py`를 직접 실행해 보십시오.")
                st.stop()
        if result.returncode == 0:
            st.success("갱신 완료. 최신 데이터를 불러옵니다.")
            st.rerun()
        else:
            st.error("갱신에 실패했습니다. 터미널에서 `python scripts/refresh_daily.py`를 직접 실행해 오류를 확인하십시오.")

pred = attach_road_authority(pred, BASE_DIR / "data" / "road_authorities.csv")

key_env_name = config.get("kakao", {}).get("app_key_env", "KAKAO_MAP_APP_KEY")
kakao_key = os.getenv(key_env_name, "").strip()

st.markdown(
    '<div class="rd-header">'
    '<span class="rd-title">전북 특별 민원창구</span>'
    '<span class="rd-caption">전북 500m 격자별 포트홀 상대 위험도 · 모델 위험 + 재발 위험 + 도로 중요도 기반 보수 우선순위</span>'
    "</div>",
    unsafe_allow_html=True,
)

high_count = int(pred["risk_level"].isin(["매우 높음", "높음"]).sum())

if not kakao_key:
    st.warning(
        f"카카오 지도를 표시하려면 프로젝트 루트의 `.env`에 "
        f"`{key_env_name}=발급받은_JavaScript_키`를 입력하십시오. 목록은 키 없이도 동작합니다."
    )

pothole_events = load_recent_events(BASE_DIR / "data" / "potholes.csv", "event_date", pred)
if not pothole_events.empty:
    severity_text = pothole_events.get("severity", pd.Series(dtype=object)).apply(
        lambda value: f"심각도 {int(value)}" if pd.notna(value) else ""
    )
    road_text = pothole_events.get("road_name", pd.Series(dtype=object)).fillna("")
    pothole_events["meta_text"] = (severity_text + " · " + road_text).str.strip(" ·")
pothole_events["critical"] = False

# TODO: 위험도 점수 체계 도입 후 실제 데이터(위험도 95점 초과)로 교체할 예시 항목입니다.
critical_example = pd.DataFrame(
    [
        {
            "event_date": pd.Timestamp.now(),
            "address": "전북특별자치도 전주시 덕진구 백제대로 567 (예시)",
            "meta_text": "위험도 점수 97점 · 실제 데이터 연동 전 예시 항목입니다",
            "critical": True,
        }
    ]
)
pothole_events = pd.concat([critical_example, pothole_events], ignore_index=True)

repair_events = load_recent_events(BASE_DIR / "data" / "repairs.csv", "repair_date", pred)
if not repair_events.empty:
    repair_events["meta_text"] = repair_events.get("repair_type", pd.Series(dtype=object)).fillna("")

board_col, map_col = st.columns([1, 2.3], gap="medium")

with board_col:
    render_board(
        "🔔 보수 필요 알림",
        f"최근 포트홀 발생 {len(pothole_events)}건 · 날짜순",
        pothole_events,
        "event_date",
        "발생",
        "rd-badge-alert",
        475,
        "등록된 포트홀 발생 이력이 없습니다.",
    )
    render_board(
        "✅ 보수완료",
        f"최근 보수완료 {len(repair_events)}건",
        repair_events,
        "repair_date",
        "완료",
        "rd-badge-done",
        127,
        "등록된 보수완료 이력이 없습니다.",
    )
    render_board(
        "🔍 점검완료",
        "점검 이력 연동 예정",
        pd.DataFrame(),
        "inspection_date",
        "점검",
        "rd-badge-check",
        127,
        "등록된 점검 이력이 없습니다. 데이터 연동 예정입니다.",
    )

with map_col:
    with st.container(border=True, key="rd-top-metrics"):
        m1, m2, m3 = st.columns(3)
        m1.metric("예측 기준일", str(pred.get("prediction_date", pd.Series(["-"])).iloc[0]))
        m2.metric("최고 모델 위험도", f"{float(pred['risk_score'].max()):.1%}")
        m3.metric("고위험 격자", f"{high_count:,}개")

    show_kakao_map(pred, kakao_key, height=800)

tab_components, tab_model, tab_raw = st.tabs(["우선순위 구성", "모델 검증", "전체 예측 데이터"])
with tab_components:
    st.caption("교통량/도로중요도가 없으면 0.75:0.15 가중치를 합이 1이 되도록 재정규화합니다.")
    display_columns = [
        "priority_rank",
        "grid_id",
        "address",
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
            "address": st.column_config.TextColumn("주소"),
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
        split = metrics.get("split", {})
        imbalance = metrics.get("class_imbalance", {})

        st.caption("테스트 구간 (실제 성능 확인용)")
        m1, m2, m3 = st.columns(3)
        m1.metric("Test ROC-AUC", f"{test_metrics.get('roc_auc', 0):.3f}")
        m2.metric("Test PR-AUC", f"{test_metrics.get('pr_auc', 0):.4f}")
        m3.metric("Test F1", f"{test_metrics.get('f1', 0):.3f}")

        st.caption("검증 구간 (분류 임계값을 정한 구간)")
        v1, v2, v3 = st.columns(3)
        v1.metric("Validation ROC-AUC", f"{validation_metrics.get('roc_auc', 0):.3f}")
        v2.metric("Validation PR-AUC", f"{validation_metrics.get('pr_auc', 0):.4f}")
        v3.metric("Validation F1", f"{validation_metrics.get('f1', 0):.3f}")

        train_positive = imbalance.get("train_positive", 0)
        train_negative = imbalance.get("train_negative", 0)
        st.warning(
            f"학습 구간의 양성(포트홀 발생) 샘플이 {train_positive:,}개뿐이고 음성은 {train_negative:,}개라, "
            "PR-AUC·F1이 0에 가깝게 나옵니다. ROC-AUC는 높아 보여도 이런 극단적 불균형에서는 "
            "실제 변별력을 의미하지 않습니다 — 원본 포트홀 이력 데이터 자체가 적기 때문입니다."
        )
        st.caption(
            f"학습 {split.get('train_rows', 0):,}행 · 검증 {split.get('validation_rows', 0):,}행 · "
            f"테스트 {split.get('test_rows', 0):,}행 · "
            f"검증 구간에서 고른 분류 임계값 {validation_metrics.get('threshold', 0.5):.4f}를 테스트 구간에 그대로 적용했습니다. "
            "risk_score는 보정 전 상대 위험 점수입니다."
        )
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
