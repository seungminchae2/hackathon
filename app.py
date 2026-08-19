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
from src.kakao_route_component import show_kakao_map


BASE_DIR = Path(__file__).resolve().parent


# ==========================================================
# 환경변수
# ==========================================================

def load_app_env(path: Path) -> None:
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

config = load_config(
    BASE_DIR / "config.yaml"
)


# ==========================================================
# Streamlit 설정
# ==========================================================

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
        background: #f7f9f6;
        border: 1px solid #e2e8e3;
        padding: 14px 16px;
        border-radius: 14px;
    }

    [data-testid="stMetric"] * {
        color: #10231c !important;
    }

    h1 {
        letter-spacing: -0.04em;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================================
# 보조 함수
# ==========================================================

def relative_risk_level(score: float) -> str:
    """
    당일 상대 AI 위험점수 등급.

    0 ~ 19.9  : 낮음
    20 ~ 29.9 : 보통
    30 ~ 59.9 : 높음
    60 이상   : 매우 높음
    """

    if score >= 60:
        return "매우 높음"

    if score >= 30:
        return "높음"

    if score >= 20:
        return "보통"

    return "낮음"


def absolute_risk_level(score: float) -> str:
    """
    절대 위험점수 등급.

    현재 predict.py에서 생성된 값을 우선 사용하며,
    구버전 CSV 대응용 fallback입니다.
    """

    if score >= 80:
        return "매우 높음"

    if score >= 60:
        return "높음"

    if score >= 40:
        return "보통"

    return "낮음"


def normalize_action_level(value: object) -> str:
    """
    조치단계 문자열을 안전하게 정규화합니다.
    """

    text = str(value).strip()

    valid = {
        "모니터링",
        "우선점검",
        "긴급점검",
        "예방보수",
    }

    if text in valid:
        return text

    return "모니터링"


def normalize_bool(value: object) -> bool:
    """
    CSV에서 bool이 문자열/숫자로 읽혀도 처리합니다.
    """

    if pd.isna(value):
        return False

    if isinstance(value, bool):
        return value

    text = str(value).strip().lower()

    return text in {
        "1",
        "1.0",
        "true",
        "yes",
        "y",
        "예",
    }


# ==========================================================
# prediction CSV 정규화
# ==========================================================

def normalize_predictions(
    frame: pd.DataFrame,
) -> pd.DataFrame:

    out = frame.copy()

    numeric_columns = [
        "grid_lat",
        "grid_lon",

        "risk_score",
        "risk_percentile",

        "road_risk_score",
        "absolute_risk_score",

        "relative_top_percent",

        "priority_score",
        "priority_rank",

        "action_rank",

        "recurrence_score",
        "importance_score",

        "road_structure_score",

        "lanes",
        "lanes_lift",
        "road_rank_lift",

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


    # ------------------------------------------------------
    # 모델 원본 risk_score
    # ------------------------------------------------------

    if "risk_score" not in out.columns:

        out["risk_score"] = 0.0

    out["risk_score"] = (
        out["risk_score"]
        .fillna(0.0)
        .clip(0, 1)
    )


    # ------------------------------------------------------
    # risk percentile
    # ------------------------------------------------------

    if "risk_percentile" not in out.columns:

        out["risk_percentile"] = (
            out["risk_score"]
            .rank(
                method="average",
                pct=True,
            )
        )

    else:

        out["risk_percentile"] = (
            out["risk_percentile"]
            .fillna(0.0)
            .clip(0, 1)
        )


    # ------------------------------------------------------
    # 당일 상대 AI 위험점수
    # ------------------------------------------------------

    if "road_risk_score" not in out.columns:

        out["road_risk_score"] = (
            100
            * np.sqrt(
                out["risk_percentile"]
                .clip(0, 1)
            )
        ).round(1)

    else:

        out["road_risk_score"] = (
            out["road_risk_score"]
            .fillna(0.0)
            .clip(0, 100)
        )


    # ------------------------------------------------------
    # 상대 위험등급
    # predict.py 결과가 있으면 그대로 사용
    # ------------------------------------------------------

    if "risk_level" not in out.columns:

        out["risk_level"] = (
            out["road_risk_score"]
            .map(relative_risk_level)
        )

    else:

        missing = (
            out["risk_level"].isna()
            | out["risk_level"]
            .astype(str)
            .str.strip()
            .eq("")
        )

        out.loc[
            missing,
            "risk_level",
        ] = (
            out.loc[
                missing,
                "road_risk_score",
            ]
            .map(relative_risk_level)
        )


    # ------------------------------------------------------
    # 절대 위험점수
    # ------------------------------------------------------

    if "absolute_risk_score" not in out.columns:

        out["absolute_risk_score"] = (
            out["road_risk_score"]
        )

    out["absolute_risk_score"] = (
        pd.to_numeric(
            out["absolute_risk_score"],
            errors="coerce",
        )
        .fillna(0.0)
        .clip(0, 100)
    )


    # ------------------------------------------------------
    # 절대 위험등급
    # ------------------------------------------------------

    if "absolute_risk_level" not in out.columns:

        out["absolute_risk_level"] = (
            out["absolute_risk_score"]
            .map(absolute_risk_level)
        )

    else:

        missing = (
            out["absolute_risk_level"].isna()
            | out["absolute_risk_level"]
            .astype(str)
            .str.strip()
            .eq("")
        )

        out.loc[
            missing,
            "absolute_risk_level",
        ] = (
            out.loc[
                missing,
                "absolute_risk_score",
            ]
            .map(absolute_risk_level)
        )


    # ------------------------------------------------------
    # 상대 상위 %
    # ------------------------------------------------------

    if "relative_top_percent" not in out.columns:

        out["relative_top_percent"] = (
            (
                1.0
                - out["road_risk_score"]
                .rank(
                    method="min",
                    pct=True,
                    ascending=True,
                )
            )
            * 100
        )

    out["relative_top_percent"] = (
        pd.to_numeric(
            out["relative_top_percent"],
            errors="coerce",
        )
        .fillna(100.0)
        .clip(0, 100)
    )


    # ------------------------------------------------------
    # 과거 포트홀
    # ------------------------------------------------------

    for column in [
        "past_potholes_30d",
        "past_potholes_90d",
        "past_potholes_total",
    ]:

        if column not in out.columns:
            out[column] = 0.0

        out[column] = (
            pd.to_numeric(
                out[column],
                errors="coerce",
            )
            .fillna(0.0)
        )


    # ------------------------------------------------------
    # 보수 이력
    # ------------------------------------------------------

    if "days_since_last_repair" not in out.columns:

        out["days_since_last_repair"] = 0.0

    out["days_since_last_repair"] = (
        pd.to_numeric(
            out["days_since_last_repair"],
            errors="coerce",
        )
        .fillna(0.0)
    )


    # ------------------------------------------------------
    # Trigger
    # ------------------------------------------------------

    if "risk_trigger" not in out.columns:

        out["risk_trigger"] = "없음"

    out["risk_trigger"] = (
        out["risk_trigger"]
        .fillna("없음")
        .astype(str)
    )


    # ------------------------------------------------------
    # AI 설명
    # ------------------------------------------------------

    if "risk_reason" not in out.columns:

        out["risk_reason"] = "-"

    out["risk_reason"] = (
        out["risk_reason"]
        .fillna("-")
        .astype(str)
    )


    # ------------------------------------------------------
    # 예방보수
    # ------------------------------------------------------

    if "preventive_repair" not in out.columns:

        out["preventive_repair"] = False

    out["preventive_repair"] = (
        out["preventive_repair"]
        .map(normalize_bool)
    )


    # ------------------------------------------------------
    # 조치 단계
    # ------------------------------------------------------

    if "action_level" not in out.columns:

        out["action_level"] = "모니터링"

        out.loc[
            out["road_risk_score"] >= 30,
            "action_level",
        ] = "우선점검"

        out.loc[
            out["road_risk_score"] >= 60,
            "action_level",
        ] = "긴급점검"

        out.loc[
            out["preventive_repair"],
            "action_level",
        ] = "예방보수"

    else:

        out["action_level"] = (
            out["action_level"]
            .map(normalize_action_level)
        )


    # ------------------------------------------------------
    # 재발 위험 fallback
    # ------------------------------------------------------

    if "recurrence_score" not in out.columns:

        recurrence = (
            out["past_potholes_90d"]
        )

        scale = max(
            float(
                recurrence.quantile(
                    0.95
                )
            ),
            1.0,
        )

        out["recurrence_score"] = (
            recurrence
            .div(scale)
            .clip(upper=1)
        )


    # ------------------------------------------------------
    # 도로 중요도 fallback
    # ------------------------------------------------------

    if "importance_score" not in out.columns:

        out["importance_score"] = np.nan


    # ------------------------------------------------------
    # 주소
    # ------------------------------------------------------

    if "address" not in out.columns:

        out["address"] = out["grid_id"]

    else:

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


    # ------------------------------------------------------
    # 조치 우선순위
    # ------------------------------------------------------

    action_order = {
        "예방보수": 4,
        "긴급점검": 3,
        "우선점검": 2,
        "모니터링": 1,
    }

    out["_action_order"] = (
        out["action_level"]
        .map(action_order)
        .fillna(1)
    )


    # predict.py에서 action_rank가 생성돼도
    # 화면에서는 현재 데이터를 기준으로 정렬을 보장
    out = out.sort_values(
        [
            "_action_order",
            "absolute_risk_score",
            "road_risk_score",
            "risk_score",
            "grid_id",
        ],
        ascending=[
            False,
            False,
            False,
            False,
            True,
        ],
        kind="mergesort",
    ).reset_index(drop=True)


    out["action_rank"] = (
        out.index + 1
    )


    out = out.drop(
        columns=[
            "_action_order",
        ]
    )


    return out


# ==========================================================
# 관할기관
# ==========================================================

def attach_road_authority(
    df: pd.DataFrame,
    path: Path,
) -> pd.DataFrame:

    out = df.copy()

    out["road_authority_dept"] = None
    out["road_authority_phone"] = None

    if (
        not path.exists()
        or "address" not in out.columns
    ):
        return out

    authorities = pd.read_csv(
        path
    )

    address = (
        out["address"]
        .astype(str)
    )

    for row in authorities.itertuples(
        index=False
    ):

        mask = address.str.contains(
            str(row.sigungu_name),
            na=False,
            regex=False,
        )

        out.loc[
            mask,
            "road_authority_dept",
        ] = (
            f"{row.sigungu_name} "
            f"{row.dept_name}"
        )

        out.loc[
            mask,
            "road_authority_phone",
        ] = row.phone_number

    return out


# ==========================================================
# 예측 데이터 로딩
# ==========================================================

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
    pd.read_csv(
        prediction_path
    )
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


# ==========================================================
# 오늘 날짜 갱신
# ==========================================================

if (
    current_prediction_date
    != date.today().isoformat()
):

    banner_col, button_col = st.columns(
        [5, 1],
        vertical_alignment="center",
    )

    banner_col.info(
        f"예측 기준일이 "
        f"{current_prediction_date}로 "
        f"오늘({date.today().isoformat()})보다 "
        f"오래됐습니다."
    )

    if button_col.button(
        "오늘 날짜로 갱신",
        width="stretch",
    ):

        with st.spinner(
            "오늘 날짜 기준으로 예측을 "
            "갱신하는 중입니다... "
            "(몇 분 걸릴 수 있습니다. "
            "이 탭을 닫지 마십시오)"
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
                    "갱신이 30분 넘게 걸려 "
                    "중단했습니다. "
                    "터미널에서 "
                    "`python scripts/refresh_daily.py`를 "
                    "직접 실행해 보십시오."
                )

                st.stop()


        if result.returncode == 0:

            st.success(
                "갱신 완료. "
                "최신 데이터를 불러옵니다."
            )

            st.rerun()

        else:

            st.error(
                "갱신에 실패했습니다. "
                "터미널에서 "
                "`python scripts/refresh_daily.py`를 "
                "직접 실행해 오류를 확인하십시오."
            )

            if result.stderr:

                st.code(
                    result.stderr
                )


# ==========================================================
# 관할기관 결합
# ==========================================================

pred = attach_road_authority(
    pred,
    BASE_DIR
    / "data"
    / "road_authorities.csv",
)


# ==========================================================
# Kakao key
# ==========================================================

key_env_name = (
    config
    .get(
        "kakao",
        {},
    )
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

st.title(
    "Road Doctor"
)


st.caption(
    "전북 500m 도로 격자별 AI 포트홀 위험 예측 · "
    "상대 위험도 + 절대 위험도 + 위험 Trigger를 "
    "종합한 예방적 도로 유지보수 의사결정 시스템"
)


# ==========================================================
# KPI
# ==========================================================

emergency_count = int(
    pred["action_level"]
    .eq("긴급점검")
    .sum()
)


preventive_count = int(
    pred["action_level"]
    .eq("예방보수")
    .sum()
)


priority_inspection_count = int(
    pred["action_level"]
    .eq("우선점검")
    .sum()
)


monitor_count = int(
    pred["action_level"]
    .eq("모니터링")
    .sum()
)


c1, c2, c3, c4 = st.columns(
    4
)


c1.metric(
    "예측 기준일",
    current_prediction_date,
)


c2.metric(
    "최고 상대 위험도",
    (
        f"{float(pred['road_risk_score'].max()):.1f}점"
    ),
)


c3.metric(
    "긴급점검",
    f"{emergency_count:,}개",
)


c4.metric(
    "예방보수",
    f"{preventive_count:,}개",
)


# ==========================================================
# 절대 위험도 보조 KPI
# ==========================================================

k1, k2, k3, k4 = st.columns(
    4
)


k1.metric(
    "최고 절대 위험도",
    (
        f"{float(pred['absolute_risk_score'].max()):.1f}점"
    ),
)


k2.metric(
    "우선점검",
    f"{priority_inspection_count:,}개",
)


k3.metric(
    "모니터링",
    f"{monitor_count:,}개",
)


strong_trigger_count = int(
    pred["risk_trigger"]
    .ne("없음")
    .sum()
)


k4.metric(
    "위험 Trigger 감지",
    f"{strong_trigger_count:,}개",
)


# ==========================================================
# 시스템 설명
# ==========================================================

with st.expander(
    "📌 AI 점검·보수 단계 선정 기준"
):

    st.markdown(
        """
### Road Doctor 조치 단계 선정 기준

Road Doctor는 단순히 위험점수 하나만으로 조치 단계를 결정하지 않습니다.

**① 절대 위험도  
② 당일 상대 위험순위  
③ 실증 기반 위험 Trigger**

세 가지를 종합하여 최종 조치 단계를 결정합니다.

| 조치 단계 | 판정 기준 | 대응 |
|---|---|---|
| 🟢 **모니터링** | 별도의 위험조건에 해당하지 않는 도로 | AI 위험도 지속 관찰 |
| 🟡 **우선점검** | 절대위험 **40점 이상** 또는 당일 위험도 **상위 5%** 또는 위험 Trigger 존재 | 현장 점검 우선 배정 |
| 🟠 **긴급점검** | 절대위험 **60점 이상** 또는 강한 위험 Trigger 발생 | 신속한 현장 확인 |
| 🔴 **예방보수** | 절대위험 **80점 이상** + 강한 위험 Trigger + 당일 위험도 **상위 5%** | 포트홀 발생 전 선제적 보수 |

---

### 🚨 강한 위험 Trigger

현재 코드에서 강한 위험 Trigger로 판단하는 조건은 다음 두 가지입니다.

- **최근 30일 포트홀 이력 + 최근 7일 동결·융해 3회 이상**
- **최근 90일 포트홀 이력 + 최근 7일 동결·융해 3회 이상**

과거 데이터 분석에서  
**최근 30일 포트홀 이력 + 동결·융해 3회 이상** 조건은  
전체 평균 대비 약 **309배의 Lift**가 확인되었습니다.

따라서 절대 위험점수가 60점에 도달하지 않더라도
이러한 강한 위험 Trigger가 발생하면
**긴급점검 대상으로 상향**합니다.

---

### 우선점검은 왜 필요한가?

절대 위험도가 아직 높지 않더라도

- 당일 전체 도로 중 **상위 5%**
- 과거 포트홀 이력과 동결·융해 등 **위험 Trigger 발생**

중 하나에 해당하면 일반 도로보다 먼저 확인할 필요가 있으므로
**우선점검** 대상으로 분류합니다.

---

### 상대 위험도와 절대 위험도를 왜 같이 사용하나요?

**상대 위험도**  
→ 오늘 어떤 도로부터 먼저 확인할 것인지 판단

**절대 위험도**  
→ 현재 도로 자체의 위험조건이 실제로 얼마나 누적됐는지 판단

따라서 여름처럼 전체 위험도가 낮은 날에도
당일 위험도로를 선별할 수 있고,

겨울철 동결·융해와 강설 등으로 실제 위험수준이 상승하면
절대 위험도까지 높아져
**긴급점검 또는 예방보수 단계로 자동 상향**될 수 있습니다.

---

### 최종 대응 흐름

**모니터링 → 우선점검 → 긴급점검 → 예방보수**

Road Doctor는 단순한 위험지도 제공을 넘어
현장 점검과 선제적 도로보수 의사결정을 지원합니다.
        """
    )


# ==========================================================
# Kakao 지도
# ==========================================================

if not kakao_key:

    st.warning(
        f"카카오 지도를 표시하려면 "
        f"프로젝트 루트의 `.env`에 "
        f"`{key_env_name}=발급받은_JavaScript_키`를 "
        f"입력하십시오."
    )


show_kakao_map(
    pred,
    kakao_key,
    height=730,
)


# ==========================================================
# Tabs
# ==========================================================

tab_action, tab_risk, tab_model, tab_raw = st.tabs(
    [
        "조치 우선순위",
        "위험도 분석",
        "모델 검증",
        "전체 예측 데이터",
    ]
)


# ==========================================================
# 조치 우선순위
# ==========================================================

with tab_action:

    st.subheader(
        "도로 유지보수 조치 우선순위"
    )

    st.caption(
        "상대 위험도뿐 아니라 절대 위험도와 "
        "위험 Trigger를 함께 고려하여 "
        "실제 점검·보수 순서를 결정합니다."
    )


    action_columns = [
        "action_rank",
        "grid_id",
        "address",

        "action_level",

        "road_risk_score",
        "absolute_risk_score",

        "risk_level",
        "absolute_risk_level",

        "relative_top_percent",

        "risk_trigger",
        "risk_reason",

        "preventive_repair",

        "past_potholes_30d",
        "past_potholes_90d",
        "past_potholes_total",

        "days_since_last_repair",

        "freeze_thaw_7d",
        "precip_7d",

        "road_structure_score",

        "road_authority_dept",
        "road_authority_phone",
    ]


    available_action_columns = [
        column
        for column in action_columns
        if column in pred.columns
    ]


    st.dataframe(
        pred[
            available_action_columns
        ].head(100),

        hide_index=True,
        width="stretch",

        column_config={

            "action_rank":
                st.column_config.NumberColumn(
                    "조치 순위",
                    format="%d",
                ),

            "address":
                st.column_config.TextColumn(
                    "주소"
                ),

            "action_level":
                st.column_config.TextColumn(
                    "권고 조치"
                ),

            "road_risk_score":
                st.column_config.ProgressColumn(
                    "AI 상대 위험도",
                    min_value=0,
                    max_value=100,
                    format="%.1f점",
                ),

            "absolute_risk_score":
                st.column_config.ProgressColumn(
                    "절대 위험도",
                    min_value=0,
                    max_value=100,
                    format="%.1f점",
                ),

            "relative_top_percent":
                st.column_config.NumberColumn(
                    "당일 상위 %",
                    format="%.3f%%",
                ),

            "risk_trigger":
                st.column_config.TextColumn(
                    "위험 Trigger"
                ),

            "risk_reason":
                st.column_config.TextColumn(
                    "AI 위험 근거"
                ),

            "preventive_repair":
                st.column_config.CheckboxColumn(
                    "예방보수"
                ),

            "road_authority_dept":
                st.column_config.TextColumn(
                    "관할 부서"
                ),

            "road_authority_phone":
                st.column_config.TextColumn(
                    "연락처"
                ),
        },
    )


# ==========================================================
# 위험도 분석
# ==========================================================

with tab_risk:

    st.subheader(
        "현재 도로 위험도 현황"
    )


    r1, r2 = st.columns(
        2
    )


    with r1:

        st.markdown(
            "#### 조치 단계"
        )

        action_summary = (
            pred["action_level"]
            .value_counts()
            .reindex(
                [
                    "예방보수",
                    "긴급점검",
                    "우선점검",
                    "모니터링",
                ],
                fill_value=0,
            )
            .rename_axis(
                "조치 단계"
            )
            .reset_index(
                name="격자 수"
            )
        )

        st.dataframe(
            action_summary,
            hide_index=True,
            width="stretch",
        )


    with r2:

        st.markdown(
            "#### 절대 위험등급"
        )

        absolute_summary = (
            pred[
                "absolute_risk_level"
            ]
            .value_counts()
            .reindex(
                [
                    "매우 높음",
                    "높음",
                    "보통",
                    "낮음",
                ],
                fill_value=0,
            )
            .rename_axis(
                "절대 위험등급"
            )
            .reset_index(
                name="격자 수"
            )
        )

        st.dataframe(
            absolute_summary,
            hide_index=True,
            width="stretch",
        )


    st.markdown(
        "#### 위험 Trigger 현황"
    )


    trigger_summary = (
        pred["risk_trigger"]
        .value_counts()
        .rename_axis(
            "위험 Trigger"
        )
        .reset_index(
            name="격자 수"
        )
    )


    st.dataframe(
        trigger_summary,
        hide_index=True,
        width="stretch",
    )


    st.markdown(
        "#### 상위 위험도로"
    )


    risk_columns = [
        "action_rank",
        "grid_id",
        "address",

        "road_risk_score",
        "absolute_risk_score",

        "action_level",

        "risk_trigger",

        "past_potholes_30d",
        "past_potholes_90d",
        "past_potholes_total",

        "days_since_last_repair",

        "freeze_thaw_7d",
        "snowfall",
        "precip_7d",

        "road_structure_score",
    ]


    st.dataframe(
        pred[
            [
                column
                for column in risk_columns
                if column in pred.columns
            ]
        ].head(30),

        hide_index=True,
        width="stretch",
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
            .get(
                "metrics",
                {},
            )
            .get(
                "test",
                {},
            )
        )


        validation_metrics = (
            metrics
            .get(
                "metrics",
                {},
            )
            .get(
                "validation",
                {},
            )
        )


        split = metrics.get(
            "split",
            {},
        )


        imbalance = metrics.get(
            "class_imbalance",
            {},
        )


        st.subheader(
            "XGBoost 모델 검증"
        )


        st.caption(
            "테스트 구간"
        )


        m1, m2, m3 = st.columns(
            3
        )


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
            "검증 구간"
        )


        v1, v2, v3 = st.columns(
            3
        )


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
            f"학습 데이터는 포트홀 발생 양성 "
            f"{train_positive:,}건, "
            f"음성 {train_negative:,}건으로 "
            f"극심한 클래스 불균형이 존재합니다. "
            f"따라서 개별 도로의 모델 출력값을 "
            f"절대 발생확률로 해석하지 않고, "
            f"위험 도로의 상대적 선별과 "
            f"Top-K 포착 능력을 중심으로 활용합니다."
        )


        st.info(
            "Road Doctor의 최종 도로 위험도는 "
            "XGBoost 모델 출력값 하나만으로 결정되지 않습니다. "
            "기상·포트홀 이력·보수 이력·도로 구조와 "
            "과거 데이터에서 확인한 고위험 Trigger를 "
            "함께 사용하여 실제 점검 및 예방보수 "
            "의사결정에 활용합니다."
        )


        st.caption(
            f"학습 {split.get('train_rows', 0):,}행 · "
            f"검증 {split.get('validation_rows', 0):,}행 · "
            f"테스트 {split.get('test_rows', 0):,}행 · "
            f"Validation에서 선택한 분류 임계값 "
            f"{validation_metrics.get('threshold', 0.5):.4f}"
        )


    else:

        st.info(
            "새 모델을 학습하면 "
            "`outputs/metrics_v2.json`에 "
            "검증 지표가 저장됩니다."
        )


# ==========================================================
# 전체 데이터
# ==========================================================

with tab_raw:

    st.subheader(
        "전체 예측 데이터"
    )


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

        file_name=(
            "road_doctor_predictions.csv"
        ),

        mime="text/csv",
    )