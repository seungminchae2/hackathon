# Road Doctor

전북 도로를 약 500m 격자로 나누고, 각 기준일의 **다음 30일 내 포트홀 발생 상대 위험도**와 보수 우선순위를 산출하는 해커톤용 프로젝트입니다.

## 설계 요약

- 모델: XGBoost 이진 분류
- 검증: 시간 순서 기준 train / validation / test 분할
- 불균형 처리: train 구간의 음성/양성 비율을 `scale_pos_weight`에 반영
- 평가: ROC-AUC, PR-AUC, F1
- 위험 설명: XGBoost feature contribution을 의미 그룹별로 합산한 양(+) 기여 상위 2~3개
- 지도: 카카오 지도 JavaScript SDK + 보수 우선순위 목록

`risk_score`는 보정된 절대 발생확률이 아니라 격자 간 순위를 위한 상대 위험 점수입니다. 위험 등급도 해당 예측일의 백분위로 나눕니다.

## 입력 데이터

기본 CSV 위치는 `data/`입니다.

| 파일 | 필수 주요 컬럼 |
|---|---|
| `potholes.csv` | `event_date`, `lat`, `lon` |
| `repairs.csv` | `repair_date`, `lat`, `lon` |
| `roads.csv` | `lat`, `lon` |
| `weather_history.csv` | `date`, `station_id`, `lat`, `lon`, 기온·강수 컬럼 |
| `weather_forecast.csv` | 위와 동일 |

`roads.csv`에 `traffic_volume` 또는 `road_importance` 숫자 컬럼이 있으면 자동으로 모델과 우선순위에 반영됩니다. 없으면 관련 가중치를 자동 재정규화합니다.

### 학습 feature

- 격자 중심 위도·경도
- 최근 3일/7일 누적강수
- 최근 7일 동결·융해 횟수
- 과거 90일 포트홀 수와 전체 누적 포트홀 수
- `has_repair_history`
- `days_since_last_repair`
- 선택: 교통량, 도로중요도

보수이력이 없는 격자는 `has_repair_history=0`, `days_since_last_repair=0`으로 둡니다. 0일을 “방금 보수함”으로 오해하지 않도록 두 feature를 항상 함께 학습시킵니다.

## 설치와 실행

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

모델 학습과 예측:

```bash
python -m src.train --config config.yaml
python -m src.predict --config config.yaml
```

특정 날짜를 예측하려면:

```bash
python -m src.predict --config config.yaml --date 2026-01-03
```

대시보드 실행:

```bash
streamlit run app.py
```

## 카카오 지도 설정

프로젝트 루트의 `.env`에 카카오 디벨로퍼스 **JavaScript 키**와 **REST API 키**를 입력합니다.

```dotenv
KAKAO_MAP_APP_KEY=발급받은_JavaScript_키
KAKAO_REST_API_KEY=발급받은_REST_API_키
```

JavaScript 키는 지도 표시, REST API 키는 주소·장소 검색과 자동차 길찾기에 사용합니다. REST 키는 Streamlit 서버에서만 요청 헤더에 넣고 브라우저 컴포넌트에는 전달하지 않습니다.

카카오 디벨로퍼스 앱 설정에서 실행 주소도 Web 플랫폼 사이트 도메인으로 등록해야 합니다. 로컬 실행 기본 주소는 보통 `http://localhost:8501`입니다. `.env`는 Git에서 제외되며, 공유용 형식은 `.env.example`에 있습니다.

## 포트홀 위험 회피 경로

1. 출발지와 도착지를 주소 또는 장소명으로 입력합니다.
2. 카카오 자동차 길찾기의 추천·최단시간·최단거리 및 대안 경로를 최대 3개 수집합니다.
3. 경로를 약 150m 간격으로 표본화하고 500m 위험 격자 중심과 360m 이내인지 계산합니다.
4. 기본 경로에 `높음` 또는 `매우 높음` 격자가 있으면, 예상 시간이 기본 경로 대비 30% 이내인 후보 중 위험 노출이 낮은 우회 경로를 추천합니다.
5. 추천 경로는 초록색, 기본 경로는 주황색, 다른 대안은 회색으로 지도에 표시합니다.

위험 회피 결과는 포트홀 예측 격자와의 공간적 근접도를 이용한 해커톤용 의사결정 보조값입니다. 실제 도로 통제나 안전 운행 지시를 대체하지 않습니다.

## 우선순위 산식

교통량 또는 도로중요도가 있는 경우:

```text
priority_score = risk_score × 0.75
               + recurrence_score × 0.15
               + importance_score × 0.10
```

둘 다 없으면 0.10을 버리지 않고 남은 항목에 재정규화합니다.

```text
priority_score = risk_score × 0.8333...
               + recurrence_score × 0.1666...
```

`priority_rank`는 이 별도 점수를 기준으로 결정하므로 단순 `risk_score` 정렬과 다릅니다.

## 산출물

- `models/road_doctor_v2.joblib`: 모델, feature 목록, 전처리 기준, 분류 임계값
- `outputs/metrics_v2.json`: split 정보, 클래스 불균형 값, validation/test 지표
- `outputs/feature_importance_v2.csv`: 모델 feature 중요도
- `outputs/predictions_v2.csv`: 최종 격자별 예측

주요 예측 CSV 컬럼:

```text
prediction_date, grid_id, grid_lat, grid_lon,
risk_score, risk_percentile, risk_level, predicted_label, risk_reason,
priority_score, priority_rank, recurrence_score, importance_score,
priority_weight_risk, priority_weight_recurrence, priority_weight_importance,
precip_3d, precip_7d, freeze_thaw_7d,
past_potholes_90d, past_potholes_total,
has_repair_history, days_since_last_repair
```

## 테스트

```bash
python -m unittest discover -s tests -v
```
