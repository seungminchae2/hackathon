# Road Doctor MVP

전북 지역의 포트홀 민원·보수 이력과 기상 데이터를 결합해 **500m 격자별 포트홀 발생 위험도**를 예측하는 해커톤용 최소 구현입니다.

## 1. 기능

- 포트홀 좌표를 500m 격자로 변환
- 과거 1·3·7일 강수량, 일교차, 동결·융해 횟수 생성
- 동일 격자의 과거 포트홀 발생 횟수와 최근 보수 경과일 생성
- XGBoost로 위험도 학습
- 오늘·내일 예보를 넣어 격자별 위험도 산출
- Streamlit 지도에서 위험 구간과 원인 표시

## 2. 프로젝트 구조

```text
road_doctor_mvp/
├── app.py
├── config.yaml
├── requirements.txt
├── data/
│   ├── potholes.csv
│   ├── repairs.csv
│   ├── roads.csv
│   ├── weather_history.csv
│   └── weather_forecast.csv
├── models/
├── outputs/
├── scripts/
│   ├── make_sample_data.py
│   └── shp_to_road_points.py
└── src/
    ├── common.py
    ├── features.py
    ├── train.py
    └── predict.py
```

## 3. 실행

```bash
cd road_doctor_mvp
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 샘플 데이터 생성
python scripts/make_sample_data.py

# 모델 학습
python -m src.train --config config.yaml

# 최신 위험도 예측
python -m src.predict --config config.yaml

# 화면 실행
streamlit run app.py
```

## 4. 실제 데이터 CSV 형식

### `data/potholes.csv`

| 컬럼 | 설명 |
|---|---|
| event_id | 민원 또는 포트홀 ID |
| event_date | 신고일 또는 발견일, `YYYY-MM-DD` |
| lat | 위도 |
| lon | 경도 |
| severity | 심각도, 없으면 1 |
| road_name | 도로명, 선택 |

### `data/repairs.csv`

| 컬럼 | 설명 |
|---|---|
| repair_id | 보수 ID |
| repair_date | 보수일 |
| lat | 위도 |
| lon | 경도 |
| repair_type | 보수 방식, 선택 |

### `data/roads.csv`

전북 도로 위의 점들을 100~250m 간격으로 샘플링한 파일입니다.

| 컬럼 | 설명 |
|---|---|
| road_point_id | 점 ID |
| lat | 위도 |
| lon | 경도 |
| road_name | 도로명, 선택 |

SHP만 있을 경우 아래 명령으로 변환합니다.

```bash
python scripts/shp_to_road_points.py \
  --input /path/to/road.shp \
  --output data/roads.csv \
  --spacing-m 200
```

### `data/weather_history.csv`

| 컬럼 | 설명 |
|---|---|
| date | 관측일 |
| station_id | 관측소 ID |
| station_name | 관측소명 |
| lat | 관측소 위도 |
| lon | 관측소 경도 |
| avg_temp | 평균기온 |
| min_temp | 최저기온 |
| max_temp | 최고기온 |
| precipitation | 일강수량 mm |
| snowfall | 일적설량 cm, 없으면 0 |
| humidity | 평균습도 %, 없으면 결측 허용 |

### `data/weather_forecast.csv`

형식은 `weather_history.csv`와 동일하며, 미래 날짜의 예보값을 넣습니다.

## 5. 실제 데이터 연결 시 수정할 부분

공공데이터 API 응답 컬럼을 위 CSV 형식으로 한 번만 변환하면 나머지 코드는 그대로 쓸 수 있습니다. 해커톤에서는 API를 매번 직접 호출하기보다, 먼저 CSV로 저장한 뒤 모델을 돌리는 방식이 안정적입니다.

## 6. 모델 출력

`outputs/predictions.csv`

| 컬럼 | 설명 |
|---|---|
| prediction_date | 예측 대상일 |
| grid_id | 500m 격자 ID |
| grid_lat | 격자 중심 위도 |
| grid_lon | 격자 중심 경도 |
| risk_score | 포트홀 위험도 0~1 |
| risk_level | 낮음/보통/높음/매우 높음 |
| risk_reason | 주요 위험 요인 |
| priority_rank | 보수 우선순위 |

## 7. 발표 시 표현

데이터가 민원 중심이면 “포트홀 실제 발생 확률”보다는 아래처럼 표현하는 게 안전합니다.

> 과거 포트홀 민원·보수 이력과 기상정보를 결합한 전북 도로 구간별 상대 위험도 예측
