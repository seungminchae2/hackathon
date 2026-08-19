"""
전북 포트홀 프로젝트 - 격자점마다 행정동(관할 구역) 정보 추가

roads_with_sewer_age.csv의 680개 격자점 좌표를 카카오 좌표->행정구역 API에
하나씩 물어봐서, 각 지점이 어느 시/구/동 관할인지 컬럼으로 추가한다.

사용법:
  1. .env 또는 환경변수에 KAKAO_REST_API_KEY 설정해두기
  2. python add_district.py

주의: 카카오 API를 680번 호출하니까 몇 분 걸릴 수 있음.
      너무 빨리 연속 호출하면 카카오가 차단할 수 있어서 요청 사이에 살짝 딜레이를 둠.
"""
import csv
import os
import time
import requests

INPUT_CSV = "data/roads.csv"                       # 지금 있는 원본 파일 그대로 사용
OUTPUT_CSV = "data/roads_with_district.csv"

VALID_SIDO = {"전북특별자치도", "전라북도"}  # 이 범위 밖으로 나오면 오류로 간주하고 재확인 필요


def get_district(lat, lon, rest_key):
    url = "https://dapi.kakao.com/v2/local/geo/coord2regioncode.json"
    headers = {"Authorization": f"KakaoAK {rest_key}"}
    params = {"x": lon, "y": lat}  # 카카오는 x=경도, y=위도 순서 주의
    try:
        r = requests.get(url, headers=headers, params=params, timeout=5)
        if r.status_code != 200:
            # 진짜 원인을 알기 위해 카카오가 보낸 응답 내용까지 출력
            print(f"  경고: ({lat},{lon}) 조회 실패 - status={r.status_code}, body={r.text}")
            return None
        docs = r.json().get("documents", [])
        # region_type "H"(행정동) 우선, 없으면 "B"(법정동)
        h = next((d for d in docs if d["region_type"] == "H"), None)
        d = h or (docs[0] if docs else None)
        if d is None:
            return None
        return {
            "sido": d.get("region_1depth_name", ""),
            "sigungu": d.get("region_2depth_name", ""),
            "dong": d.get("region_3depth_name", ""),
        }
    except Exception as e:
        print(f"  경고: ({lat},{lon}) 조회 실패 - {e}")
        return None


def load_env_file(path=".env"):
    """.env 파일을 직접 읽어서 환경변수로 등록. python-dotenv 같은 별도 설치 없이 동작."""
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ[key.strip()] = value.strip().strip('"').strip("'")


def main():
    load_env_file(".env")
    rest_key = os.getenv("KAKAO_REST_API_KEY", "").strip()
    if not rest_key:
        print("에러: KAKAO_REST_API_KEY 환경변수가 없습니다. .env 확인하세요.")
        print("  .env 파일이 add_district.py와 같은 폴더(C:\\hackathon)에 있는지도 확인하세요.")
        return

    with open(INPUT_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames) + ["행정동_시도", "행정동_시군구", "행정동_읍면동"]

    print(f"총 {len(rows)}개 지점 조회 시작...")

    # 전체 680개 다 돌리기 전에, 첫 번째 지점으로 먼저 테스트해서
    # 설정 자체가 잘못됐으면 빨리 알아채고 멈추게 함 (시간 낭비 방지)
    test_lat, test_lon = float(rows[0]["lat"]), float(rows[0]["lon"])
    test_info = get_district(test_lat, test_lon, rest_key)
    if test_info is None:
        print("\n첫 번째 지점 테스트부터 실패했습니다. 전체를 돌리지 않고 여기서 멈춥니다.")
        print("위에 나온 status/body 내용을 그대로 복사해서 확인 요청하세요.")
        return
    print(f"테스트 성공: {rows[0].get('road_point_id')} -> {test_info}\n")

    suspicious = []
    for i, row in enumerate(rows):
        lat, lon = float(row["lat"]), float(row["lon"])
        info = get_district(lat, lon, rest_key)
        if info is None:
            row["행정동_시도"] = row["행정동_시군구"] = row["행정동_읍면동"] = ""
        else:
            row["행정동_시도"] = info["sido"]
            row["행정동_시군구"] = info["sigungu"]
            row["행정동_읍면동"] = info["dong"]
            if info["sido"] not in VALID_SIDO:
                suspicious.append((row.get("road_point_id"), info))
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(rows)} 완료...")
        time.sleep(0.05)  # 너무 빠른 연속 호출 방지

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"완료: {OUTPUT_CSV}")
    if suspicious:
        print(f"경고: 전북 범위 밖으로 나온 지점 {len(suspicious)}개 있음, 확인 필요:")
        for pid, info in suspicious[:10]:
            print(f"  {pid}: {info}")


if __name__ == "__main__":
    main()