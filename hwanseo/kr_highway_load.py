import pandas as pd
import json

file_path = "jeonbuk_traffic_1.csv"
df = pd.read_csv(file_path, encoding='utf-8-sig')

jeonbuk_tcs = [
    '고창', '군산', '금산사', '김제', '내장산', '덕유산', '동군산', '동전주', '무주', 
    '부안', '삼례', '상관', '서김제', '서전주', '선운산', '소양', '완주', '익산', 
    '장수', '전주', '정읍', '줄포', '진안', '태인'
]

df_jb = df[df['출발영업소명'].isin(jeonbuk_tcs) | df['도착영업소명'].isin(jeonbuk_tcs)].copy()

KPRG_LEF = {
    '1종': 0.0008,
    '2종': 0.2800,
    '3종': 1.1500,
    '4종': 3.2000,
    '5종': 4.8000,
    '6종': 0.0001
}

df_jb['KPRG_ESAL'] = (
    df_jb['도착지방향1종교통량'] * KPRG_LEF['1종'] +
    df_jb['도착지방향2종교통량'] * KPRG_LEF['2종'] +
    df_jb['도착지방향3종교통량'] * KPRG_LEF['3종'] +
    df_jb['도착지방향4종교통량'] * KPRG_LEF['4종'] +
    df_jb['도착지방향5종교통량'] * KPRG_LEF['5종'] +
    df_jb['도착지방향6종교통량'] * KPRG_LEF['6종']
)

route_summary = df_jb.groupby(['출발영업소명', '도착영업소명'])[[
    '도착지방향총교통량', '도착지방향1종교통량', '도착지방향2종교통량', 
    '도착지방향3종교통량', '도착지방향4종교통량', '도착지방향5종교통량', 'KPRG_ESAL'
]].sum().reset_index()

top30 = route_summary.sort_values(by='KPRG_ESAL', ascending=False).head(30)

top30['risk_score'] = (top30['KPRG_ESAL'] / top30['KPRG_ESAL'].max() * 100).round(1)

top30_list = top30.to_dict(orient='records')

with open('kprg_top30_risk.json', 'w', encoding='utf-8') as f:
    json.dump(top30_list, f, ensure_ascii=False, indent=4)

top30.to_csv('kprg_top30_risk.csv', index=False, encoding='utf-8-sig')

print("'kprg_top30_risk.json' 및 'kprg_top30_risk.csv' 생성 완료")