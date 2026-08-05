import pandas as pd
import json

file_name = '전북 포트홀&도로불량 민원 데이터.csv'

try:
    df = pd.read_csv(file_name, encoding='utf-8-sig')
except UnicodeDecodeError:
    df = pd.read_csv(file_name, encoding='cp949')

def extract_sigungu(location):
    if not isinstance(location, str):
        return '기타'
    parts = location.split()
    for part in parts:
        if part.endswith(('시', '군', '구')) and part not in ['전라북도', '전북', '전북특별자치도']:
            return part
    return '기타'

df['시군구'] = df['민원위치'].apply(extract_sigungu)

keywords = ['포터홀', '포트홀', '파임', '파손', '침하', '구멍', '보수', '포장', '단차', '도로']
pattern = '|'.join(keywords)
df['is_pothole_type'] = df['민원내용'].str.contains(pattern, na=False)

clean_df = df.dropna(subset=['위도', '경도']).copy()

clean_df.to_csv('pothole_complaints_clean.csv', index=False, encoding='utf-8-sig')

complaint_list = clean_df.to_dict(orient='records')
with open('pothole_complaints.json', 'w', encoding='utf-8') as f:
    json.dump(complaint_list, f, ensure_ascii=False, indent=4)

print("'pothole_complaints_clean.csv' 및 'pothole_complaints.json' 생성 완료")
print(f"총 {len(clean_df)}건의 민원 위치 좌표 데이터 정제 완료")