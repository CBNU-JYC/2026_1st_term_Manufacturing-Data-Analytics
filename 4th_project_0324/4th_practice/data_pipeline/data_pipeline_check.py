import asyncio
import pandas as pd
import numpy as np
from asyncua import Client
from datetime import datetime
import random

# 품질 검증 함수 (5대 지표)
def evaluate_data_quality(df, stage_name):
    print(f"\n[{stage_name}] 데이터 품질 5대 지표 평가 결과:")
    print("-" * 50)
    
    if len(df) == 0:
        print("데이터가 없습니다.")
        return

    # 1. 유일성: 중복 레코드 비율
    duplicate_ratio = df.duplicated(subset=['Timestamp']).mean()
    print(f" 1. 유일성 평가: 중복 데이터 비율 {duplicate_ratio:.2%}")

    # 2. 완전성: 결측치 비율
    missing_ratio = df[['Temperature', 'Pressure']].isnull().mean().max()
    print(f" 2. 완전성 평가: 최대 결측치 비율 {missing_ratio:.2%}")

    # 3. 유효성: 정상 범위(온도 0~100, 압력 0~5) 이탈 비율
    invalid_temp = ~df['Temperature'].between(0, 100, inclusive='both') & df['Temperature'].notnull()
    invalid_pressure = ~df['Pressure'].between(0, 5, inclusive='both') & df['Pressure'].notnull()
    validity_error_ratio = (invalid_temp | invalid_pressure).mean()
    print(f" 3. 유효성 평가: 범위를 이탈한 데이터 비율 {validity_error_ratio:.2%}")

    # 4. 일관성: 논리적 모순 확인
    if 'Status_Label' in df.columns:
        rule_violation = df[
            ((df['Temperature'] >= 28.0) | (df['Pressure'] >= 1.4)) != (df['Status_Label'] == 1)
        ]
        consistency_error_ratio = len(rule_violation) / len(df)
        print(f" 4. 일관성 평가: 논리적 모순 데이터 비율 {consistency_error_ratio:.2%}")
    else:
        print(" 4. 일관성 평가: 라벨링 전이므로 평가 보류")

    # 5. 정확성: 데이터 타입 확인
    is_time_acc = pd.api.types.is_datetime64_any_dtype(df['Timestamp'])
    is_num_acc = pd.api.types.is_numeric_dtype(df['Temperature']) and pd.api.types.is_numeric_dtype(df['Pressure'])
    accuracy_status = "Pass" if is_time_acc and is_num_acc else "Fail"
    print(f" 5. 정확성 평가: 데이터 타입 및 포맷 적합성 [{accuracy_status}]")
    print("-" * 50)


async def main():
    url = "opc.tcp://127.0.0.1:4840/freeopcua/server/"
    
    # [1단계~2단계] 데이터 수집
    print("데이터 수집을 시작합니다...")
    collected_data = []
    
    async with Client(url=url) as client:
        uri = "http://manufacturing.example.com"
        idx = await client.get_namespace_index(uri)
        machine = await client.nodes.objects.get_child([f"{idx}:Machine_A"])
        temp_node = await machine.get_child([f"{idx}:Temperature"])
        pressure_node = await machine.get_child([f"{idx}:Pressure"])

        for i in range(20): 
            temp_val = await temp_node.read_value()
            pressure_val = await pressure_node.read_value()
            timestamp = datetime.now()
            
            # 의도적 불량 데이터 발생
            if i == 5: temp_val = np.nan
            if i == 10: pressure_val = 999.9
            
            row = {"Timestamp": timestamp, "Machine_ID": "Machine_A", "Temperature": temp_val, "Pressure": pressure_val}
            collected_data.append(row)
            
            # 의도적 중복 레코드 삽입
            if i == 15: collected_data.append(row.copy())
            
            await asyncio.sleep(0.5)

    df_raw = pd.DataFrame(collected_data)
    
    # 정제 전 원시 데이터(Raw Data) 별도 저장
    raw_filename = "raw_sensor_dataset.csv"
    df_raw.to_csv(raw_filename, index=False, encoding='utf-8-sig')
    print(f"\n[저장 완료] 정제 전 원시 데이터가 '{raw_filename}'로 저장되었습니다.")
    
    # --- 원시 데이터 품질 평가 ---
    evaluate_data_quality(df_raw, "정제 전 원시 데이터")

    # [3단계] 정제 및 [4단계] 라벨링
    print("\n데이터 정제 및 라벨링을 수행합니다...")
    df_clean = df_raw.copy()
    df_clean = df_clean.drop_duplicates(subset=['Timestamp'])
    df_clean = df_clean.dropna()
    df_clean = df_clean[(df_clean['Temperature'].between(0, 100)) & (df_clean['Pressure'].between(0, 5))]
    
    df_clean['Temperature'] = df_clean['Temperature'].round(2)
    df_clean['Pressure'] = df_clean['Pressure'].round(2)
    
    df_clean['Status_Label'] = np.where((df_clean['Temperature'] >= 28.0) | (df_clean['Pressure'] >= 1.4), 1, 0)

    # --- 최종 데이터 품질 평가 ---
    evaluate_data_quality(df_clean, "정제 및 라벨링 완료 데이터")

    # ==========================================
    # 정제 후 최종 데이터(Clean Data) 저장
    # ==========================================
    clean_filename = "verified_sensor_dataset.csv"
    df_clean.to_csv(clean_filename, index=False, encoding='utf-8-sig')
    print(f"\n[저장 완료] 최종 데이터셋이 '{clean_filename}'로 저장되었습니다.")

if __name__ == "__main__":
    asyncio.run(main())