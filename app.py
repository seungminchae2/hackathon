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

        os.environ[key.strip()] = (
            value.strip().strip('"').strip("'")
        )


load_app_env(BASE_DIR / ".env")
config = load_config(BASE_DIR / "config.yaml")


st.set_page_config(
    page_title="Road Doctor",
    page_icon="🛣️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }

    [data-testid="stMetric"] {
        background:#f7f9f6;
        border:1px solid #e2e8e3;
        padding:14px 16px;
        border-radius:14px;
    }

    [data-testid="stMetric"] * {
        color:#10231c !important;
    }

    h1 {
        letter-spacing:-0.04em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def normalize_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    """
    기존 CSV도 화면에서 열리도록
    새 컬럼의 안전한 기본값을 만듭니다.
    """

    out = frame.copy()

    numeric_columns = [
        "grid_lat",
        "grid_lon",
        "risk_score",
        "road_risk_score",
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
            out[column] = pd.to_numeric(
                out[column],
                errors="coerce",
            )

    # XGBoost 원본 예측값
    out["risk_score"] = (
        out.get(
            "risk_score",
            pd.Series(0.0, index=out.index),
        )
        .fillna(0)
        .clip(0, 1)
    )

    # 위험 백분위
    if "risk_percentile" not in out:
        out["risk_percentile"] = (
            out["risk_score"]
            .rank(
                method="average",
                pct=True,
            )
        )

    # --------------------------------------------------
    # 화면 표시용 도로 위험점수
    # 최저 = 0점 / 최고 = 100점
    #
    # 새 predictions_v2.csv에 road_risk_score가 있으면
    # 해당 값을 그대로 사용.
    #
    # 구버전 CSV라서 컬럼이 없으면 앱에서 자동 계산.
    # --------------------------------------------------

    if "road_risk_score" not in out.columns:
        min_risk = float(out["risk_score"].min())
        max_risk = float(out["risk_score"].max())

        if max_risk > min_risk:
            out["road_risk_score"] = (
                (
                    out["risk_score"] - min_risk
                )
                / (max_risk - min_risk)
                * 100
            ).round(1)
        else:
            out["road_risk_score"] = 0.0

    else:
        out["road_risk_score"] = (
            pd.to_numeric(
                out["road_risk_score"],
                errors="coerce",
            )
            .fillna(0.0)
            .clip(0, 100)
        )

    # 위험 등급
    if "risk_level" not in out:
        out["risk_level"] = (
            out["risk_percentile"]
            .map(risk_level_from_percentile)
        )

    # 과거 포트홀
    if "past_potholes_total" not in out:
        out["past_potholes_total"] = out.get(
            "past_potholes_90d",
            0,
        )

    # 재발 위험
    if "recurrence_score" not in out:
        source = (
            out["past_potholes_90d"]
            if "past_potholes_90d" in out
            else pd.Series(0.0, index=out.index)
        )

        recurrence = pd.to_numeric(
            source,
            errors="coerce",
        ).fillna(0)

        scale = max(
            float(recurrence.quantile(0.95)),
            1.0,
        )

        out["recurrence_score"] = (
            recurrence
            .div(scale)
            .clip(upper=1)
        )

    # 도로 중요도
    if "importance_score" not in out:
        out["importance_score"] = np.nan

    # 주소
    if "address" in out.columns:
        missing_address = (
            out["address"].isna()
            | out["address"]
            .astype(str)
            .str.strip()
            .eq("")
        )

        out.loc[
            missing_address,
            "address",
        ] = out.loc[
            missing_address,
            "grid_id",
        ]

    # 구버전 CSV 대응용 우선순위 점수
    if "priority_score" not in out:
        out["priority_score"] = (
            out["risk_score"] * 0.75
            + out["recurrence_score"] * 0.15
        ) / 0.90

    # 보수 우선순위 정렬
    out = out.sort_values(
        [
            "priority_score",
            "risk_score",
            "grid_id",
        ],
        ascending=[
            False,
            False,
            True,
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    if "priority_rank" not in frame.columns:
        out["priority_rank"] = out.index + 1

    return out


def attach_road_authority(
    df: pd.DataFrame,
    path: Path,
) -> pd.DataFrame:
    """
    주소의 시군명을 기준으로
    관할 보수 담당 부서·연락처를 붙입니다.
    """

    out = df.copy()

    out["road_authority_dept"] = None
    out["road_authority_phone"] = None

    if not path.exists() or "address" not in out.columns:
        return out

    authorities = pd.read_csv(path)

    address = out["address"].astype(str)

    for row in authorities.itertuples(index=False):
        mask = address.str.contains(
            row.sigungu_name,
            na=False,
            regex=False,
        )

        out.loc[
            mask,
            "road_authority_dept",
        ] = f"{row.sigungu_name} {row.dept_name}"

        out.loc[
            mask,
            "road_authority_phone",
        ] = row.phone_number

    return out


prediction_path = resolve_path(
    config,
    "predictions",
)

if not prediction_path.exists():
    st.error(
        "예측 파일이 없습니다. "
        "`python -m src.predict --config config.yaml`을 "
        "먼저 실행하십시오."
    )
    st.stop()


pred = normalize_predictions(
    pd.read_csv(prediction_path)
)

if pred.empty:
    st.error(
        "예측 파일에 데이터가 없습니다."
    )
    st.stop()


current_prediction_date = str(
    pred.get(
        "prediction_date",
        pd.Series([""]),
    ).iloc[0]
)


if current_prediction_date != date.today().isoformat():

    banner_col, button_col = st.columns(
        [5, 1],
        vertical_alignment="center",
    )

    banner_col.info(
        f"예측 기준일이 {current_prediction_date}로 "
        f"오늘({date.today().isoformat()})보다 오래됐습니다."
    )

    if button_col.button(
        "오늘 날짜로 갱신",
        width="stretch",
    ):

        with st.spinner(
            "오늘 날짜 기준으로 예측을 갱신하는 중입니다... "
            "(몇 분 걸릴 수 있습니다. 이 탭을 닫지 마십시오)"
        ):

            try:
                result = subprocess.run(
                    [
                        sys.executable,
                        str(
                            BASE_DIR
                            / "scripts"
                            / "refresh_daily.py"
                        ),
                    ],
                    cwd=BASE_DIR,
                    capture_output=True,
                    text=True,
                    timeout=1800,
                )

            except subprocess.TimeoutExpired:

                st.error(
                    "갱신이 30분 넘게 걸려 중단했습니다. "
                    "터미널에서 "
                    "`python scripts/refresh_daily.py`를 "
                    "직접 실행해 보십시오."
                )

                st.stop()

        if result.returncode == 0:

            st.success(
                "갱신 완료. 최신 데이터를 불러옵니다."
            )

            st.rerun()

        else:

            st.error(
                "갱신에 실패했습니다. "
                "터미널에서 "
                "`python scripts/refresh_daily.py`를 "
                "직접 실행해 오류를 확인하십시오."
            )


pred = attach_road_authority(
    pred,
    BASE_DIR
    / "data"
    / "road_authorities.csv",
)


key_env_name = (
    config
    .get("kakao", {})
    .get(
        "app_key_env",
        "KAKAO_MAP_APP_KEY",
    )
)

kakao_key = os.getenv(
    key_env_name,
    "",
).strip()


# ==========================================================
# 메인 화면
# ==========================================================

st.title("Road Doctor")

st.caption(
    "전북 500m 격자별 포트홀 상대 위험도 · "
    "AI 도로 위험점수 + 재발 위험 + "
    "도로 중요도 기반 보수 우선순위"
)


very_high_count = int(
    pred["risk_level"]
    .eq("매우 높음")
    .sum()
)

high_count = int(
    pred["risk_level"]
    .isin(
        [
            "매우 높음",
            "높음",
        ]
    )
    .sum()
)


c1, c2, c3, c4 = st.columns(4)


c1.metric(
    "예측 기준일",
    str(
        pred.get(
            "prediction_date",
            pd.Series(["-"]),
        ).iloc[0]
    ),
)


# ----------------------------------------------------------
# 기존
# 최고 모델 위험도 52.5%
#
# 변경
# 최고 도로 위험점수 100점
# ----------------------------------------------------------

c2.metric(
    "최고 도로 위험점수",
    f"{float(pred['road_risk_score'].max()):.0f}점",
)


c3.metric(
    "고위험 격자",
    f"{high_count:,}개",
)


c4.metric(
    "매우 고위험",
    f"{very_high_count:,}개",
)


if not kakao_key:

    st.warning(
        f"카카오 지도를 표시하려면 프로젝트 루트의 "
        f"`.env`에 "
        f"`{key_env_name}=발급받은_JavaScript_키`를 "
        f"입력하십시오. 목록은 키 없이도 동작합니다."
    )


show_kakao_map(
    pred,
    kakao_key,
    height=730,
)


tab_components, tab_model, tab_raw = st.tabs(
    [
        "우선순위 구성",
        "모델 검증",
        "전체 예측 데이터",
    ]
)


# ==========================================================
# 우선순위 구성
# ==========================================================

with tab_components:

    st.caption(
        "교통량/도로중요도가 없으면 "
        "0.75:0.15 가중치를 합이 1이 되도록 "
        "재정규화합니다."
    )

    display_columns = [
        "priority_rank",
        "grid_id",
        "address",
        "priority_score",
        "road_risk_score",
        "risk_score",
        "recurrence_score",
        "importance_score",
        "risk_reason",
    ]

    st.dataframe(
        pred[
            [
                column
                for column in display_columns
                if column in pred.columns
            ]
        ].head(30),
        hide_index=True,
        width="stretch",
        column_config={
            "address": st.column_config.TextColumn(
                "주소"
            ),

            "priority_score":
                st.column_config.ProgressColumn(
                    "우선순위",
                    min_value=0,
                    max_value=1,
                    format="%.3f",
                ),

            "road_risk_score":
                st.column_config.ProgressColumn(
                    "도로 위험점수",
                    min_value=0,
                    max_value=100,
                    format="%.1f점",
                ),

            "risk_score":
                st.column_config.NumberColumn(
                    "모델 원본값",
                    format="%.4f",
                ),

            "recurrence_score":
                st.column_config.ProgressColumn(
                    "재발 위험",
                    min_value=0,
                    max_value=1,
                    format="%.3f",
                ),
        },
    )


# ==========================================================
# 모델 검증
# ==========================================================

with tab_model:

    metrics_path = resolve_path(
        config,
        "metrics",
    )

    if metrics_path.exists():

        metrics = json.loads(
            metrics_path.read_text(
                encoding="utf-8"
            )
        )

        test_metrics = (
            metrics
            .get("metrics", {})
            .get("test", {})
        )

        validation_metrics = (
            metrics
            .get("metrics", {})
            .get("validation", {})
        )

        split = metrics.get(
            "split",
            {},
        )

        imbalance = metrics.get(
            "class_imbalance",
            {},
        )


        st.caption(
            "테스트 구간 (실제 성능 확인용)"
        )

        m1, m2, m3 = st.columns(3)

        m1.metric(
            "Test ROC-AUC",
            f"{test_metrics.get('roc_auc', 0):.3f}",
        )

        m2.metric(
            "Test PR-AUC",
            f"{test_metrics.get('pr_auc', 0):.4f}",
        )

        m3.metric(
            "Test F1",
            f"{test_metrics.get('f1', 0):.3f}",
        )


        st.caption(
            "검증 구간 (분류 임계값을 정한 구간)"
        )

        v1, v2, v3 = st.columns(3)

        v1.metric(
            "Validation ROC-AUC",
            f"{validation_metrics.get('roc_auc', 0):.3f}",
        )

        v2.metric(
            "Validation PR-AUC",
            f"{validation_metrics.get('pr_auc', 0):.4f}",
        )

        v3.metric(
            "Validation F1",
            f"{validation_metrics.get('f1', 0):.3f}",
        )


        train_positive = imbalance.get(
            "train_positive",
            0,
        )

        train_negative = imbalance.get(
            "train_negative",
            0,
        )


        st.warning(
            f"학습 구간의 양성(포트홀 발생) 샘플이 "
            f"{train_positive:,}개뿐이고 "
            f"음성은 {train_negative:,}개라, "
            f"PR-AUC·F1이 낮게 나타날 수 있습니다. "
            f"따라서 Road Doctor는 개별 도로의 "
            f"절대 발생확률보다 상대적 위험순위와 "
            f"Top-K 포착 성능을 중심으로 활용합니다."
        )


        st.caption(
            f"학습 {split.get('train_rows', 0):,}행 · "
            f"검증 {split.get('validation_rows', 0):,}행 · "
            f"테스트 {split.get('test_rows', 0):,}행 · "
            f"검증 구간에서 고른 분류 임계값 "
            f"{validation_metrics.get('threshold', 0.5):.4f}를 "
            f"테스트 구간에 그대로 적용했습니다. "
            f"도로 위험점수는 모델 예측값을 "
            f"동일 예측일 내 0~100점으로 정규화한 "
            f"상대 위험 지표입니다."
        )

    else:

        st.info(
            "새 모델을 학습하면 "
            "`outputs/metrics_v2.json`에 "
            "검증 지표가 저장됩니다."
        )


# ==========================================================
# 전체 예측 데이터
# ==========================================================

with tab_raw:

    st.dataframe(
        pred,
        hide_index=True,
        width="stretch",
    )

    st.download_button(
        "예측 CSV 내려받기",
        pred.to_csv(
            index=False
        ).encode(
            "utf-8-sig"
        ),
        file_name="road_doctor_predictions.csv",
        mime="text/csv",
    )