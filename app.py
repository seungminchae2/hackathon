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

from src.common import (
    load_config,
    resolve_path,
)

from src.kakao_route_component import (
    show_kakao_map,
)


BASE_DIR = Path(__file__).resolve().parent


# ==========================================================
# 환경변수
# ==========================================================

def load_app_env(
    path: Path,
) -> None:

    if not path.exists():
        return

    for raw_line in path.read_text(
        encoding="utf-8"
    ).splitlines():

        line = raw_line.strip()

        if (
            not line
            or line.startswith("#")
            or "=" not in line
        ):
            continue

        key, value = line.split(
            "=",
            1,
        )

        os.environ[key.strip()] = (
            value
            .strip()
            .strip('"')
            .strip("'")
        )


load_app_env(
    BASE_DIR / ".env"
)


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
# 위험등급
# ==========================================================

def relative_risk_level(
    score: float,
) -> str:

    if score >= 85:
        return "매우 높음"

    if score >= 70:
        return "높음"

    if score >= 40:
        return "보통"

    return "낮음"


def absolute_risk_level(
    score: float,
) -> str:

    if score >= 80:
        return "매우 높음"

    if score >= 60:
        return "높음"

    if score >= 40:
        return "보통"

    return "낮음"


# ==========================================================
# 정규화
# ==========================================================

def normalize_action_level(
    value: object,
) -> str:

    text = str(
        value
    ).strip()

    valid = {
        "모니터링",
        "우선점검",
        "긴급점검",
        "예방보수",
    }

    if text in valid:
        return text

    return "모니터링"


def normalize_bool(
    value: object,
) -> bool:

    if pd.isna(
        value
    ):
        return False

    if isinstance(
        value,
        bool,
    ):
        return value

    text = str(
        value
    ).strip().lower()

    return text in {
        "1",
        "1.0",
        "true",
        "yes",
        "y",
        "예",
    }


# ==========================================================
# 예측 데이터 정규화
# ==========================================================

def normalize_predictions(
    frame: pd.DataFrame,
) -> pd.DataFrame:

    out = frame.copy()

    if "grid_id" not in out.columns:

        raise ValueError(
            "예측 CSV에 grid_id 컬럼이 없습니다."
        )

    out["grid_id"] = (
        out[
            "grid_id"
        ]
        .astype(str)
    )

    numeric_columns = [
        "grid_lat",
        "grid_lon",

        "risk_score",
        "risk_percentile",

        "road_risk_score",
        "relative_top_percent",

        "absolute_risk_score",

        "action_rank",

        "priority_score",
        "priority_rank",

        "recurrence_score",
        "importance_score",

        "feature_risk_score",
        "model_relative_score",
        "relative_ai_index",

        "absolute_feature_score",
        "absolute_risk_index",

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
    # XGBoost 원본값
    # ------------------------------------------------------

    if "risk_score" not in out.columns:

        out[
            "risk_score"
        ] = 0.0

    out[
        "risk_score"
    ] = (
        out[
            "risk_score"
        ]
        .fillna(0.0)
        .clip(
            0.0,
            1.0,
        )
    )

    # ------------------------------------------------------
    # 모델 percentile fallback
    # ------------------------------------------------------

    if (
        "risk_percentile"
        not in out.columns
    ):

        out[
            "risk_percentile"
        ] = (
            out[
                "risk_score"
            ]
            .rank(
                method="average",
                pct=True,
            )
        )

    # ------------------------------------------------------
    # 상대 위험점수
    # ------------------------------------------------------

    if (
        "road_risk_score"
        not in out.columns
    ):

        out[
            "road_risk_score"
        ] = (
            100.0
            * np.sqrt(
                out[
                    "risk_percentile"
                ]
                .fillna(0.0)
                .clip(
                    0.0,
                    1.0,
                )
            )
        ).round(1)

    else:

        out[
            "road_risk_score"
        ] = (
            out[
                "road_risk_score"
            ]
            .fillna(0.0)
            .clip(
                0.0,
                100.0,
            )
        )

    # ------------------------------------------------------
    # 상대 위험등급
    #
    # 85 이상 : 매우 높음
    # 70 이상 : 높음
    # 40 이상 : 보통
    # 40 미만 : 낮음
    # ------------------------------------------------------

    out[
        "risk_level"
    ] = (
        out[
            "road_risk_score"
        ]
        .map(
            relative_risk_level
        )
    )

    # ------------------------------------------------------
    # 상대 상위 %
    # ------------------------------------------------------

    if (
        "relative_top_percent"
        not in out.columns
    ):

        descending_rank = (
            out[
                "road_risk_score"
            ]
            .rank(
                method="min",
                ascending=False,
            )
        )

        out[
            "relative_top_percent"
        ] = (
            descending_rank
            / max(
                len(out),
                1,
            )
            * 100.0
        )

    out[
        "relative_top_percent"
    ] = (
        out[
            "relative_top_percent"
        ]
        .fillna(100.0)
        .clip(
            0.0,
            100.0,
        )
    )

    # ------------------------------------------------------
    # 절대 위험점수
    # ------------------------------------------------------

    if (
        "absolute_risk_score"
        not in out.columns
    ):

        out[
            "absolute_risk_score"
        ] = (
            out[
                "road_risk_score"
            ]
        )

    out[
        "absolute_risk_score"
    ] = (
        out[
            "absolute_risk_score"
        ]
        .fillna(0.0)
        .clip(
            0.0,
            100.0,
        )
    )

    # ------------------------------------------------------
    # 절대 위험등급
    #
    # 80 이상 : 매우 높음
    # 60 이상 : 높음
    # 40 이상 : 보통
    # 40 미만 : 낮음
    # ------------------------------------------------------

    out[
        "absolute_risk_level"
    ] = (
        out[
            "absolute_risk_score"
        ]
        .map(
            absolute_risk_level
        )
    )

    # ------------------------------------------------------
    # 포트홀 이력
    # ------------------------------------------------------

    for column in [
        "past_potholes_30d",
        "past_potholes_90d",
        "past_potholes_total",
    ]:

        if column not in out.columns:

            out[
                column
            ] = 0.0

        out[
            column
        ] = (
            pd.to_numeric(
                out[
                    column
                ],
                errors="coerce",
            )
            .fillna(0.0)
        )

    # ------------------------------------------------------
    # 보수 경과
    # ------------------------------------------------------

    if (
        "days_since_last_repair"
        not in out.columns
    ):

        out[
            "days_since_last_repair"
        ] = 0.0

    out[
        "days_since_last_repair"
    ] = (
        pd.to_numeric(
            out[
                "days_since_last_repair"
            ],
            errors="coerce",
        )
        .fillna(0.0)
    )

    # ------------------------------------------------------
    # Trigger
    # ------------------------------------------------------

    if (
        "risk_trigger"
        not in out.columns
    ):

        out[
            "risk_trigger"
        ] = "없음"

    out[
        "risk_trigger"
    ] = (
        out[
            "risk_trigger"
        ]
        .fillna("없음")
        .astype(str)
    )

    # ------------------------------------------------------
    # AI 위험근거
    # ------------------------------------------------------

    if (
        "risk_reason"
        not in out.columns
    ):

        out[
            "risk_reason"
        ] = "-"

    out[
        "risk_reason"
    ] = (
        out[
            "risk_reason"
        ]
        .fillna("-")
        .astype(str)
    )

    # ------------------------------------------------------
    # 예방보수 여부
    # ------------------------------------------------------

    if (
        "preventive_repair_candidate"
        not in out.columns
    ):

        out[
            "preventive_repair_candidate"
        ] = 0

    out[
        "preventive_repair_candidate"
    ] = (
        out[
            "preventive_repair_candidate"
        ]
        .map(
            normalize_bool
        )
    )

    # ------------------------------------------------------
    # 조치 단계
    # ------------------------------------------------------

    if (
        "action_level"
        not in out.columns
    ):

        out[
            "action_level"
        ] = "모니터링"

        out.loc[
            (
                out[
                    "absolute_risk_score"
                ] >= 40
            )
            |
            (
                out[
                    "relative_top_percent"
                ] <= 5
            )
            |
            (
                out[
                    "risk_trigger"
                ].ne(
                    "없음"
                )
            ),
            "action_level",
        ] = "우선점검"

        out.loc[
            (
                out[
                    "absolute_risk_score"
                ] >= 60
            ),
            "action_level",
        ] = "긴급점검"

        out.loc[
            out[
                "preventive_repair_candidate"
            ],
            "action_level",
        ] = "예방보수"

    else:

        out[
            "action_level"
        ] = (
            out[
                "action_level"
            ]
            .map(
                normalize_action_level
            )
        )

    # ------------------------------------------------------
    # 재발 위험 fallback
    # ------------------------------------------------------

    if (
        "recurrence_score"
        not in out.columns
    ):

        source = (
            out[
                "past_potholes_90d"
            ]
        )

        scale = max(
            float(
                source.quantile(
                    0.95
                )
            ),
            1.0,
        )

        out[
            "recurrence_score"
        ] = (
            source
            .div(
                scale
            )
            .clip(
                upper=1.0
            )
        )

    # ------------------------------------------------------
    # importance fallback
    # ------------------------------------------------------

    if (
        "importance_score"
        not in out.columns
    ):

        out[
            "importance_score"
        ] = np.nan

    # ------------------------------------------------------
    # 주소 fallback
    # ------------------------------------------------------

    if (
        "address"
        not in out.columns
    ):

        out[
            "address"
        ] = (
            out[
                "grid_id"
            ]
        )

    else:

        missing_address = (
            out[
                "address"
            ].isna()
            |
            out[
                "address"
            ]
            .astype(str)
            .str.strip()
            .eq("")
        )

        out.loc[
            missing_address,
            "address",
        ] = (
            out.loc[
                missing_address,
                "grid_id",
            ]
        )

    # ------------------------------------------------------
    # action_rank
    # ------------------------------------------------------

    if (
        "action_rank"
        not in out.columns
    ):

        action_order = {
            "예방보수": 4,
            "긴급점검": 3,
            "우선점검": 2,
            "모니터링": 1,
        }

        out[
            "_action_order"
        ] = (
            out[
                "action_level"
            ]
            .map(
                action_order
            )
            .fillna(1)
        )

        out = (
            out
            .sort_values(
                [
                    "_action_order",
                    "absolute_risk_score",
                    "road_risk_score",
                    "grid_id",
                ],
                ascending=[
                    False,
                    False,
                    False,
                    True,
                ],
                kind="mergesort",
            )
            .reset_index(
                drop=True
            )
        )

        out[
            "action_rank"
        ] = (
            out.index + 1
        )

        out = (
            out
            .drop(
                columns=[
                    "_action_order"
                ]
            )
        )

    else:

        out = (
            out
            .sort_values(
                [
                    "action_rank",
                    "grid_id",
                ],
                ascending=[
                    True,
                    True,
                ],
                kind="mergesort",
            )
            .reset_index(
                drop=True
            )
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

    out[
        "road_authority_dept"
    ] = None

    out[
        "road_authority_phone"
    ] = None

    if (
        not path.exists()
        or "address" not in out.columns
    ):
        return out

    authorities = pd.read_csv(
        path
    )

    address = (
        out[
            "address"
        ]
        .astype(str)
    )

    for row in authorities.itertuples(
        index=False
    ):

        mask = (
            address
            .str.contains(
                str(
                    row.sigungu_name
                ),
                na=False,
                regex=False,
            )
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
        ] = (
            row.phone_number
        )

    return out


# ==========================================================
# predictions 로딩
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


try:

    pred = normalize_predictions(
        pd.read_csv(
            prediction_path
        )
    )

except Exception as exc:

    st.error(
        "예측 데이터를 읽는 중 오류가 발생했습니다."
    )

    st.exception(
        exc
    )

    st.stop()


if pred.empty:

    st.error(
        "예측 파일에 데이터가 없습니다."
    )

    st.stop()


current_prediction_date = str(
    pred.get(
        "prediction_date",
        pd.Series(
            [
                "-"
            ]
        ),
    ).iloc[0]
)


# ==========================================================
# 오늘 날짜 자동 갱신
# ==========================================================

if (
    current_prediction_date
    and current_prediction_date != "-"
    and current_prediction_date
    != date.today().isoformat()
):

    banner_col, button_col = (
        st.columns(
            [
                5,
                1,
            ],
            vertical_alignment="center",
        )
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
            "기상청 최신 단기예보를 불러오고 "
            "오늘 기준 위험도를 갱신하는 중입니다..."
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
                    "중단했습니다."
                )

                st.stop()

        if result.returncode == 0:

            st.success(
                "기상청 단기예보 반영 및 "
                "위험도 갱신이 완료되었습니다."
            )

            st.rerun()

        else:

            st.error(
                "예측 갱신에 실패했습니다."
            )

            if result.stderr:

                st.code(
                    result.stderr
                )


# ==========================================================
# 관할기관 부착
# ==========================================================

pred = attach_road_authority(
    pred,
    BASE_DIR
    / "data"
    / "road_authorities.csv",
)


# ==========================================================
# Kakao Map key
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


kakao_key = (
    os.getenv(
        key_env_name,
        "",
    )
    .strip()
)


# ==========================================================
# 메인
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
    pred[
        "action_level"
    ]
    .eq(
        "긴급점검"
    )
    .sum()
)


preventive_count = int(
    pred[
        "action_level"
    ]
    .eq(
        "예방보수"
    )
    .sum()
)


priority_count = int(
    pred[
        "action_level"
    ]
    .eq(
        "우선점검"
    )
    .sum()
)


monitor_count = int(
    pred[
        "action_level"
    ]
    .eq(
        "모니터링"
    )
    .sum()
)


trigger_count = int(
    pred[
        "risk_trigger"
    ]
    .ne(
        "없음"
    )
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
    f"{float(pred['road_risk_score'].max()):.1f}점",
)


c3.metric(
    "긴급점검",
    f"{emergency_count:,}개",
)


c4.metric(
    "예방보수",
    f"{preventive_count:,}개",
)


k1, k2, k3, k4 = st.columns(
    4
)


k1.metric(
    "최고 절대 위험도",
    f"{float(pred['absolute_risk_score'].max()):.1f}점",
)


k2.metric(
    "우선점검",
    f"{priority_count:,}개",
)


k3.metric(
    "모니터링",
    f"{monitor_count:,}개",
)


k4.metric(
    "위험 Trigger 감지",
    f"{trigger_count:,}개",
)


# ==========================================================
# 조치 기준 설명
# ==========================================================

with st.expander(
    "📌 AI 점검·보수 단계 선정 기준"
):

    st.markdown(
        """
### Road Doctor 조치 단계 선정 기준

Road Doctor는 단순히 위험점수 하나만으로
조치 단계를 결정하지 않습니다.

**① 절대 위험도  
② 당일 상대 위험순위  
③ 실증 기반 위험 Trigger**

세 가지를 종합하여 최종 조치 단계를 결정합니다.

| 조치 단계 | 판정 기준 | 대응 |
|---|---|---|
| 🟢 **모니터링** | 별도의 위험조건에 해당하지 않는 도로 | AI 위험도 지속 관찰 |
| 🟡 **우선점검** | 절대위험 40점 이상 또는 당일 위험도 상위 5% 또는 위험 Trigger 존재 | 현장 점검 우선 배정 |
| 🟠 **긴급점검** | 절대위험 60점 이상 또는 강한 위험 Trigger 발생 | 신속한 현장 확인 |
| 🔴 **예방보수** | 절대위험 80점 이상 + 강한 위험 Trigger + 당일 위험도 상위 5% | 포트홀 발생 전 선제적 보수 |

---

### 상대 위험도

**오늘 다른 도로와 비교했을 때 어느 도로가 더 위험한가?**

- 85점 이상 → 매우 높음
- 70점 이상 → 높음
- 40점 이상 → 보통
- 40점 미만 → 낮음

### 절대 위험도

**현재 해당 도로 자체의 위험요인이 얼마나 누적됐는가?**

- 80점 이상 → 매우 높음
- 60점 이상 → 높음
- 40점 이상 → 보통
- 40점 미만 → 낮음

따라서 예를 들어,

**상대 위험도 85점 / 매우 높음**이면서  
**절대 위험도 35점 / 낮음**

일 수 있습니다.

이는 현재 전체 도로 중에서는 상대적으로 매우 위험하지만,
절대적인 위험요인 누적 수준은 아직 낮다는 의미입니다.
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

tab_action, tab_risk, tab_model, tab_raw = (
    st.tabs(
        [
            "조치 우선순위",
            "위험도 분석",
            "모델 검증",
            "전체 예측 데이터",
        ]
    )
)


# ==========================================================
# 조치 우선순위
# ==========================================================

with tab_action:

    st.subheader(
        "도로 유지보수 조치 우선순위"
    )

    st.caption(
        "당일 상대 위험도뿐 아니라 "
        "절대 위험도와 위험 Trigger를 함께 고려해 "
        "실제 점검·보수 순서를 결정합니다."
    )

    display_columns = [
        "action_rank",
        "grid_id",
        "address",

        "action_level",

        "road_risk_score",
        "risk_level",

        "absolute_risk_score",
        "absolute_risk_level",

        "relative_top_percent",

        "risk_trigger",
        "risk_reason",

        "preventive_repair_candidate",

        "past_potholes_30d",
        "past_potholes_90d",
        "past_potholes_total",

        "days_since_last_repair",

        "freeze_thaw_7d",
        "precip_7d",

        "road_authority_dept",
        "road_authority_phone",
    ]

    display_columns = [
        column
        for column in display_columns
        if column in pred.columns
    ]

    st.dataframe(
        pred[
            display_columns
        ].head(
            100
        ),

        hide_index=True,
        width="stretch",

        column_config={

            "action_rank":
                st.column_config.NumberColumn(
                    "조치 순위",
                    format="%d",
                ),

            "grid_id":
                st.column_config.TextColumn(
                    "격자 ID"
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
                    "당일 상대 위험도",
                    min_value=0,
                    max_value=100,
                    format="%.1f점",
                ),

            "risk_level":
                st.column_config.TextColumn(
                    "상대 위험등급"
                ),

            "absolute_risk_score":
                st.column_config.ProgressColumn(
                    "절대 위험도",
                    min_value=0,
                    max_value=100,
                    format="%.1f점",
                ),

            "absolute_risk_level":
                st.column_config.TextColumn(
                    "절대 위험등급"
                ),

            "relative_top_percent":
                st.column_config.NumberColumn(
                    "당일 상위",
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

            "preventive_repair_candidate":
                st.column_config.CheckboxColumn(
                    "예방보수 대상"
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
            "#### 상대 위험등급"
        )

        relative_summary = (
            pred[
                "risk_level"
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
                "상대 위험등급"
            )
            .reset_index(
                name="격자 수"
            )
        )

        st.dataframe(
            relative_summary,
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
        "#### 조치 단계"
    )

    action_summary = (
        pred[
            "action_level"
        ]
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

    st.markdown(
        "#### 위험 Trigger 현황"
    )

    trigger_summary = (
        pred[
            "risk_trigger"
        ]
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
        "#### 조치 우선순위 상위 30개"
    )

    risk_columns = [
        "action_rank",
        "grid_id",
        "address",

        "action_level",

        "road_risk_score",
        "risk_level",

        "absolute_risk_score",
        "absolute_risk_level",

        "relative_top_percent",

        "risk_trigger",

        "past_potholes_30d",
        "past_potholes_90d",
        "past_potholes_total",

        "days_since_last_repair",

        "freeze_thaw_7d",
        "snowfall",
        "precip_7d",

        "road_structure_score",

        "road_authority_dept",
    ]

    risk_columns = [
        column
        for column in risk_columns
        if column in pred.columns
    ]

    st.dataframe(
        pred[
            risk_columns
        ].head(
            30
        ),

        hide_index=True,
        width="stretch",
    )


# ==========================================================
# 모델 검증
# ==========================================================

with tab_model:

    st.subheader(
        "AI 모델 검증"
    )

    st.caption(
        "포트홀은 전체 도로·날짜 표본에서 매우 드물게 발생하므로 "
        "일반적인 분류 성능과 실제 활용 목적에 맞는 "
        "위험구간 선별 성능을 함께 평가합니다."
    )

    metrics_path = resolve_path(
        config,
        "metrics",
    )

    topk_path = (
        BASE_DIR
        / "outputs"
        / "topk_metrics_v2.json"
    )

    test_metrics = {}
    validation_metrics = {}
    split = {}

    if metrics_path.exists():

        try:

            metrics = json.loads(
                metrics_path.read_text(
                    encoding="utf-8"
                )
            )

        except Exception as exc:

            st.error(
                "모델 검증 지표를 읽지 못했습니다."
            )

            st.exception(
                exc
            )

        else:

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

            st.markdown(
                "### 1. 테스트 구간 성능"
            )

            st.caption(
                "학습 및 임계값 결정에 사용하지 않은 "
                "미래 시점 Test 데이터에 대한 최종 성능입니다."
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

            st.markdown(
                "### 2. 검증 구간 성능"
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

            train_positive = int(
                imbalance.get(
                    "train_positive",
                    0,
                )
            )

            train_negative = int(
                imbalance.get(
                    "train_negative",
                    0,
                )
            )

            total_train = (
                train_positive
                + train_negative
            )

            positive_rate = (
                train_positive
                / total_train
                if total_train > 0
                else 0.0
            )

            st.warning(
                f"학습 데이터는 포트홀 발생 양성 "
                f"{train_positive:,}건, "
                f"음성 {train_negative:,}건으로 "
                f"양성 비율이 약 "
                f"{positive_rate:.4%}에 불과한 "
                f"희귀사건 데이터입니다. "
                f"따라서 F1만으로 모델을 평가하지 않고 "
                f"Top-K Capture와 Lift를 함께 확인합니다."
            )

    else:

        st.warning(
            "`outputs/metrics_v2.json`이 없습니다."
        )

    st.divider()

    st.markdown(
        "### 3. 위험구간 선별 성능"
    )

    if topk_path.exists():

        try:

            topk_data = json.loads(
                topk_path.read_text(
                    encoding="utf-8"
                )
            )

        except Exception as exc:

            st.error(
                "Top-K 검증 결과를 읽지 못했습니다."
            )

            st.exception(
                exc
            )

        else:

            top_k = topk_data.get(
                "top_k",
                {},
            )

            test_rows = int(
                topk_data.get(
                    "test_rows",
                    0,
                )
            )

            test_positive = int(
                topk_data.get(
                    "test_positive",
                    0,
                )
            )

            base_rate = float(
                topk_data.get(
                    "base_rate",
                    0.0,
                )
            )

            st.info(
                f"Test 구간 {test_rows:,}개 grid-day 중 "
                f"실제 포트홀 발생 양성은 "
                f"{test_positive:,}개이며, "
                f"기본 발생률은 "
                f"{base_rate:.5%}입니다."
            )

            top1 = top_k.get(
                "top_1",
                {},
            )

            top5 = top_k.get(
                "top_5",
                {},
            )

            top10 = top_k.get(
                "top_10",
                {},
            )

            t1, t2, t3 = st.columns(
                3
            )

            t1.metric(
                "Top 1% 포착률",
                f"{float(top1.get('capture_rate', 0)):.1%}",
                delta=(
                    f"Lift "
                    f"{float(top1.get('lift', 0)):.1f}×"
                ),
            )

            t2.metric(
                "Top 5% 포착률",
                f"{float(top5.get('capture_rate', 0)):.1%}",
                delta=(
                    f"Lift "
                    f"{float(top5.get('lift', 0)):.1f}×"
                ),
            )

            t3.metric(
                "Top 10% 포착률",
                f"{float(top10.get('capture_rate', 0)):.1%}",
                delta=(
                    f"Lift "
                    f"{float(top10.get('lift', 0)):.1f}×"
                ),
            )

            topk_rows = []

            for key in [
                "top_1",
                "top_5",
                "top_10",
                "top_20",
                "top_30",
            ]:

                item = top_k.get(
                    key
                )

                if not item:
                    continue

                pct_value = int(
                    round(
                        float(
                            item.get(
                                "percent",
                                0,
                            )
                        )
                        * 100
                    )
                )

                captured = int(
                    item.get(
                        "captured",
                        0,
                    )
                )

                total_positive_item = int(
                    item.get(
                        "total_positive",
                        test_positive,
                    )
                )

                topk_rows.append(
                    {
                        "선별 범위":
                            f"상위 {pct_value}%",

                        "점검 대상":
                            int(
                                item.get(
                                    "rows",
                                    0,
                                )
                            ),

                        "실제 포트홀 포착":
                            (
                                f"{captured:,}"
                                f" / "
                                f"{total_positive_item:,}"
                            ),

                        "포착률":
                            float(
                                item.get(
                                    "capture_rate",
                                    0,
                                )
                            )
                            * 100,

                        "Lift":
                            float(
                                item.get(
                                    "lift",
                                    0,
                                )
                            ),
                    }
                )

            topk_frame = pd.DataFrame(
                topk_rows
            )

            st.dataframe(
                topk_frame,

                hide_index=True,
                width="stretch",

                column_config={

                    "선별 범위":
                        st.column_config.TextColumn(
                            "위험도 상위 구간"
                        ),

                    "점검 대상":
                        st.column_config.NumberColumn(
                            "선별 grid-day",
                            format="%d",
                        ),

                    "실제 포트홀 포착":
                        st.column_config.TextColumn(
                            "실제 포트홀 포착"
                        ),

                    "포착률":
                        st.column_config.NumberColumn(
                            "포착률",
                            format="%.1f%%",
                        ),

                    "Lift":
                        st.column_config.NumberColumn(
                            "Lift",
                            format="%.2f×",
                        ),
                },
            )

            st.success(
                "Top 1% 기준으로 모델이 선정한 도로만 점검했을 때 "
                f"전체 실제 포트홀 발생의 "
                f"{float(top1.get('capture_rate', 0)):.1%}를 "
                f"포착했습니다. 이는 무작위 점검 대비 "
                f"약 {float(top1.get('lift', 0)):.1f}배 높은 "
                f"포트홀 밀도입니다."
            )

    else:

        st.warning(
            "`outputs/topk_metrics_v2.json`이 없습니다."
        )

    st.divider()

    st.markdown(
        "### 4. 모델 결과 해석"
    )

    st.markdown(
        """
**Road Doctor는 '포트홀이 반드시 발생한다'고 판정하는 시스템이 아닙니다.**

모델의 핵심 역할은 수많은 도로 중
향후 포트홀 발생 가능성이 상대적으로 높은 구간을
우선적으로 선별하는 것입니다.

실제 운영에서는

**XGBoost 위험 예측  
→ 상대·절대 위험도 산정  
→ 위험 Trigger 확인  
→ 우선점검 / 긴급점검 / 예방보수**

순서로 활용합니다.

포트홀처럼 실제 발생 빈도가 매우 낮은 희귀사건에서는
PR-AUC와 F1뿐 아니라
**Top-K Capture와 Lift를 함께 보는 것이 중요합니다.**
        """
    )

    if metrics_path.exists():

        st.caption(
            f"시간순 데이터 분할 · "
            f"학습 "
            f"{int(split.get('train_rows', 0)):,}행 · "
            f"검증 "
            f"{int(split.get('validation_rows', 0)):,}행 · "
            f"테스트 "
            f"{int(split.get('test_rows', 0)):,}행"
        )


# ==========================================================
# 전체 예측 데이터
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