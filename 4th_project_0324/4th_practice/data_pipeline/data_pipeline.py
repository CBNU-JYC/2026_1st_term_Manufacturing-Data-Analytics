import asyncio
import pandas as pd
import numpy as np
from asyncua import Client
from datetime import datetime
import random

async def main():
    url = "opc.tcp://127.0.0.1:4840/freeopcua/server/"
    
    # ==========================================
    # 2단계: 획득 (Acquisition) - 원시 데이터 수집
    # ==========================================
    print("[1/4] 데이터 획득 중...")
    collected_data = []
    
    async with Client(url=url) as client:
        uri = "http://manufacturing.example.com"
        idx = await client.get_namespace_index(uri)
        
        machine = await client.nodes.objects.get_child([f"{idx}:Machine_A"])
        temp_node = await machine.get_child([f"{idx}:Temperature"])
        pressure_node = await machine.get_child([f"{idx}:Pressure"])

        for _ in range(20): # 15초간 데이터 수집
            temp_val = await temp_node.read_value()
            pressure_val = await pressure_node.read_value()
            timestamp = datetime.now()
            
            # 실습(정제 단계)을 위해 의도적으로 10% 확률로 결측치(NaN) 발생
            if random.random() < 0.1:
                temp_val = np.nan
            
            # 메타정보(Machine_ID) 매핑하여 원시 데이터 적재
            collected_data.append({
                "Timestamp": timestamp,
                "Machine_ID": "Machine_A",
                "Temperature": temp_val,
                "Pressure": pressure_val
            })
            await asyncio.sleep(1)

    # 원시 데이터프레임 생성
    df_raw = pd.DataFrame(collected_data)
    print(">>> 획득 완료: 총 {}건의 원시 데이터 수집\n".format(len(df_raw)))

    # ==========================================
    # 3단계: 정제 (Cleansing) - 비식별화, 결측치/중복 처리
    # ==========================================
    print("[2/4] 데이터 정제 중...")
    df_clean = df_raw.copy()
    
    # 1. 결측치 제거 (혹은 보간법 사용 가능)
    df_clean = df_clean.dropna()
    
    # 2. 중복 데이터 제거
    df_clean = df_clean.drop_duplicates()
    
    # 3. 데이터 포맷 변환 (소수점 둘째 자리 반올림)
    df_clean['Temperature'] = df_clean['Temperature'].round(2)
    df_clean['Pressure'] = df_clean['Pressure'].round(2)
    
    print(">>> 정제 완료: 결측치/중복 제거 후 {}건 남음\n".format(len(df_clean)))

    # ==========================================
    # 4단계: 라벨링 (Labeling) - 참값 부여
    # ==========================================
    print("[3/4] 데이터 라벨링 중...")
    
    # 준비 단계에서 수립한 지침에 따라 어노테이션 수행
    def assign_label(row):
        if row['Temperature'] >= 28.0 or row['Pressure'] >= 1.4:
            return 1 # 이상 (Anomaly)
        else:
            return 0 # 정상 (Normal)

    df_clean['Status_Label'] = df_clean.apply(assign_label, axis=1)
    print(">>> 라벨링 완료: 'Status_Label' 컬럼 추가\n")

    # ==========================================
    # 5단계: 품질 검증 (Quality Verification)
    # ==========================================
    print("[4/4] 데이터 품질 검증 중...")
    
    # 1. 유효성 및 데이터 타입 평가
    print("-" * 30)
    print("1. 데이터셋 기본 정보 (Null 여부 및 타입):")
    df_clean.info()
    
    # 2. 라벨 분포 검사 (클래스 불균형 확인)
    print("-" * 30)
    print("2. 라벨 분포 검증 (0: 정상, 1: 이상):")
    label_counts = df_clean['Status_Label'].value_counts()
    print(label_counts)
    print("-" * 30)

    # 최종 데이터셋 저장
    final_filename = "lifecycle_optimized_dataset.csv"
    df_clean.to_csv(final_filename, index=False, encoding='utf-8-sig')
    print(f"\n모든 파이프라인 완료! 최종 데이터셋이 '{final_filename}'로 저장되었습니다.")

if __name__ == "__main__":
    asyncio.run(main())