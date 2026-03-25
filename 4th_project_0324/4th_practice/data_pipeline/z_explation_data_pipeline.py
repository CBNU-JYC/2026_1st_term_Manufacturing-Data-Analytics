"""
이 파일은 `data_pipeline.py`를 학습용으로 아주 자세히 풀어쓴 설명 버전입니다.

이 코드의 전체 목표:
1. OPC-UA 서버에서 센서 데이터를 읽어 온다.
2. 읽어 온 데이터를 표(DataFrame) 형태로 정리한다.
3. 결측치와 중복을 제거해서 데이터를 정제한다.
4. 규칙에 따라 정상/이상 라벨을 붙인다.
5. 최종 결과를 CSV 파일로 저장한다.

즉, "센서 데이터 수집 -> 정제 -> 라벨링 -> 검증 -> 저장" 흐름을 한 번에 보여주는 예제입니다.
"""

import asyncio
from datetime import datetime
import random

import numpy as np
import pandas as pd
from asyncua import Client


def assign_label(row):
    """
    한 행(row)의 센서값을 보고 상태 라벨을 붙이는 함수입니다.

    매개변수:
    - row: pandas DataFrame의 한 줄

    반환값:
    - 1: 이상(Anomaly)
    - 0: 정상(Normal)

    현재 규칙은 매우 단순합니다.
    - 온도 >= 28.0 이면 이상
    - 압력 >= 1.4 이면 이상
    - 둘 다 아니면 정상

    이런 방식은 "규칙 기반(rule-based) 라벨링"이라고 볼 수 있습니다.
    """
    if row["Temperature"] >= 28.0 or row["Pressure"] >= 1.4:
        return 1
    return 0


async def main():
    """
    전체 데이터 파이프라인을 실행하는 메인 함수입니다.

    왜 async 함수인가?
    - OPC-UA 서버에서 값을 읽는 작업은 네트워크/입출력(I/O) 작업입니다.
    - 이런 작업은 await와 함께 비동기 방식으로 처리하는 것이 자연스럽습니다.
    """
    # OPC-UA 서버 주소입니다.
    # 다른 파일(opc_server_mfg.py)이 이 주소로 서버를 열어 두고 있어야 접속할 수 있습니다.
    url = "opc.tcp://127.0.0.1:4840/freeopcua/server/"

    # ==========================================
    # 1단계: 데이터 획득(Acquisition)
    # ==========================================
    print("[1/4] 데이터 획득 중...")

    # 읽어 온 센서 데이터를 임시로 저장할 리스트입니다.
    # 나중에 이 리스트를 pandas DataFrame으로 바꿉니다.
    collected_data = []

    # Client 객체로 OPC-UA 서버에 접속합니다.
    # async with 구문을 쓰면 작업이 끝난 뒤 연결이 깔끔하게 닫힙니다.
    async with Client(url=url) as client:
        # 서버 내부에서 제조 설비 관련 데이터가 속한 네임스페이스 URI입니다.
        uri = "http://manufacturing.example.com"

        # 문자열 URI를 서버 내부 숫자 인덱스로 바꿉니다.
        # OPC-UA에서는 보통 "네임스페이스 인덱스 + 노드 이름"으로 노드를 찾습니다.
        idx = await client.get_namespace_index(uri)

        # Objects 폴더 아래에서 Machine_A 장비 객체를 찾습니다.
        machine = await client.nodes.objects.get_child([f"{idx}:Machine_A"])

        # Machine_A 아래에서 Temperature와 Pressure 센서 노드를 찾습니다.
        temp_node = await machine.get_child([f"{idx}:Temperature"])
        pressure_node = await machine.get_child([f"{idx}:Pressure"])

        # 총 20번 센서 값을 읽습니다.
        # 아래에서 1초씩 쉬므로 대략 20초 동안 수집한다고 이해하면 됩니다.
        for _ in range(20):
            # 현재 온도와 압력 값을 읽어 옵니다.
            temp_val = await temp_node.read_value()
            pressure_val = await pressure_node.read_value()

            # 값을 읽은 시각도 함께 저장합니다.
            # 데이터 분석에서 "언제 측정된 값인지"는 매우 중요합니다.
            timestamp = datetime.now()

            # 실습용으로 일부러 결측치(NaN)를 만들어 봅니다.
            # 약 10% 확률로 온도 값을 비워 두어,
            # 뒤의 정제 단계(dropna)가 실제로 어떤 역할을 하는지 확인할 수 있습니다.
            if random.random() < 0.1:
                temp_val = np.nan

            # 한 번 측정한 결과를 딕셔너리 한 줄로 저장합니다.
            # Machine_ID 같은 메타정보도 함께 넣어 두면 나중에 여러 장비를 구분하기 쉽습니다.
            collected_data.append(
                {
                    "Timestamp": timestamp,
                    "Machine_ID": "Machine_A",
                    "Temperature": temp_val,
                    "Pressure": pressure_val,
                }
            )

            # 너무 빠르게 읽지 않도록 1초 대기합니다.
            await asyncio.sleep(1)

    # 리스트 형태의 수집 결과를 표 형태(DataFrame)로 변환합니다.
    # 이 시점의 데이터는 아직 정제 전이므로 "원시 데이터(raw data)"입니다.
    df_raw = pd.DataFrame(collected_data)
    print(f">>> 획득 완료: 총 {len(df_raw)}건의 원시 데이터 수집\n")

    # ==========================================
    # 2단계: 데이터 정제(Cleansing)
    # ==========================================
    print("[2/4] 데이터 정제 중...")

    # 원본(df_raw)을 보존하기 위해 복사본에서 작업합니다.
    # 실무에서도 원시 데이터는 가능한 그대로 남겨 두는 편이 좋습니다.
    df_clean = df_raw.copy()

    # 1. 결측치 제거
    # NaN이 있는 행은 분석과 모델 학습에 문제를 일으킬 수 있으므로 제거합니다.
    # 실제 현업에서는 평균값 대체, 보간, 예측 모델 보정 등을 쓰기도 하지만,
    # 여기서는 가장 단순하게 제거합니다.
    df_clean = df_clean.dropna()

    # 2. 중복 데이터 제거
    # 완전히 같은 행이 여러 번 존재하면 데이터가 왜곡될 수 있으므로 제거합니다.
    df_clean = df_clean.drop_duplicates()

    # 3. 숫자 포맷 정리
    # 센서값이 소수점 아래 너무 길면 보기 어렵기 때문에 둘째 자리까지 반올림합니다.
    df_clean["Temperature"] = df_clean["Temperature"].round(2)
    df_clean["Pressure"] = df_clean["Pressure"].round(2)

    print(f">>> 정제 완료: 결측치/중복 제거 후 {len(df_clean)}건 남음\n")

    # ==========================================
    # 3단계: 데이터 라벨링(Labeling)
    # ==========================================
    print("[3/4] 데이터 라벨링 중...")

    # apply(..., axis=1)은 DataFrame의 각 행을 하나씩 함수에 넣는다는 뜻입니다.
    # 즉, 각 행마다 assign_label()이 실행되어 결과가 Status_Label 컬럼에 저장됩니다.
    df_clean["Status_Label"] = df_clean.apply(assign_label, axis=1)
    print(">>> 라벨링 완료: 'Status_Label' 컬럼 추가\n")

    # ==========================================
    # 4단계: 품질 검증(Quality Verification)
    # ==========================================
    print("[4/4] 데이터 품질 검증 중...")

    # DataFrame 기본 정보 확인
    # 컬럼 수, null 여부, 자료형(dtype) 등을 빠르게 볼 수 있습니다.
    print("-" * 30)
    print("1. 데이터셋 기본 정보 (Null 여부 및 타입):")
    df_clean.info()

    # 라벨 분포 확인
    # 정상(0)과 이상(1)이 각각 몇 개인지 확인합니다.
    # 한쪽 라벨만 너무 많으면 데이터 불균형 문제가 생길 수 있습니다.
    print("-" * 30)
    print("2. 라벨 분포 검증 (0: 정상, 1: 이상):")
    label_counts = df_clean["Status_Label"].value_counts()
    print(label_counts)
    print("-" * 30)

    # 최종 데이터셋 저장
    # index=False: 왼쪽 인덱스 번호는 파일에 저장하지 않음
    # utf-8-sig: 엑셀에서 한글이 깨질 가능성을 줄이기 위한 인코딩
    final_filename = "lifecycle_optimized_dataset.csv"
    df_clean.to_csv(final_filename, index=False, encoding="utf-8-sig")
    print(f"\n모든 파이프라인 완료! 최종 데이터셋이 '{final_filename}'로 저장되었습니다.")


if __name__ == "__main__":
    # 파이썬 파일을 직접 실행했을 때만 main()을 시작합니다.
    asyncio.run(main())
