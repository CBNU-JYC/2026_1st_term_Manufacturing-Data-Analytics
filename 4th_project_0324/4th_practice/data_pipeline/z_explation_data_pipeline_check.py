"""
이 파일은 `data_pipeline_check.py`를 쉽게 이해할 수 있도록 다시 풀어쓴 설명용 버전입니다.

이 예제의 핵심은 단순히 데이터를 모으는 것에 그치지 않고,
"정제 전과 정제 후의 데이터 품질을 비교"해 보는 데 있습니다.

즉, 이 코드는 다음 흐름을 보여줍니다.
1. OPC-UA 서버에서 센서 데이터를 읽는다.
2. 일부러 불량 데이터(결측치, 이상치, 중복)를 넣는다.
3. 원시 데이터의 품질을 먼저 평가한다.
4. 데이터를 정제하고 라벨을 붙인다.
5. 정제 후 품질이 얼마나 좋아졌는지 다시 평가한다.
"""

import asyncio
from datetime import datetime

import numpy as np
import pandas as pd
from asyncua import Client


def evaluate_data_quality(df, stage_name):
    """
    데이터 품질을 5가지 관점에서 평가하는 함수입니다.

    매개변수:
    - df: 평가할 pandas DataFrame
    - stage_name: 현재 어떤 단계의 데이터인지 설명하는 문자열

    평가 항목:
    1. 유일성(Uniqueness): 중복 데이터가 있는가?
    2. 완전성(Completeness): 결측치가 있는가?
    3. 유효성(Validity): 허용 범위를 벗어난 값이 있는가?
    4. 일관성(Consistency): 라벨 규칙과 실제 라벨이 맞는가?
    5. 정확성(Accuracy): 데이터 타입이 기대한 형식인가?
    """
    print(f"\n[{stage_name}] 데이터 품질 5대 지표 평가 결과:")
    print("-" * 50)

    # 데이터가 한 줄도 없으면 품질 평가 자체가 의미 없으므로 종료합니다.
    if len(df) == 0:
        print("데이터가 없습니다.")
        return

    # 1. 유일성 평가
    # 같은 Timestamp를 가진 행을 "중복 가능성"이 있는 데이터로 봅니다.
    # duplicated(...) 결과는 True/False이고,
    # mean()을 쓰면 True=1, False=0으로 계산되어 비율이 됩니다.
    duplicate_ratio = df.duplicated(subset=["Timestamp"]).mean()
    print(f" 1. 유일성 평가: 중복 데이터 비율 {duplicate_ratio:.2%}")

    # 2. 완전성 평가
    # Temperature, Pressure 컬럼 중 결측치 비율이 가장 큰 값을 대표값으로 사용합니다.
    missing_ratio = df[["Temperature", "Pressure"]].isnull().mean().max()
    print(f" 2. 완전성 평가: 최대 결측치 비율 {missing_ratio:.2%}")

    # 3. 유효성 평가
    # 업무 규칙상 정상 범위를 정해 두고, 그 범위를 벗어나는지 확인합니다.
    # 여기서는 다음처럼 가정합니다.
    # - 온도: 0 ~ 100
    # - 압력: 0 ~ 5
    invalid_temp = ~df["Temperature"].between(0, 100, inclusive="both") & df["Temperature"].notnull()
    invalid_pressure = ~df["Pressure"].between(0, 5, inclusive="both") & df["Pressure"].notnull()

    # 온도 또는 압력 둘 중 하나라도 잘못되면 오류 행으로 봅니다.
    validity_error_ratio = (invalid_temp | invalid_pressure).mean()
    print(f" 3. 유효성 평가: 범위를 이탈한 데이터 비율 {validity_error_ratio:.2%}")

    # 4. 일관성 평가
    # 라벨이 존재할 때만 검사할 수 있습니다.
    # 현재 예제의 라벨링 규칙:
    # - Temperature >= 28.0 또는 Pressure >= 1.4 이면 Status_Label == 1
    # - 아니면 Status_Label == 0
    if "Status_Label" in df.columns:
        rule_violation = df[
            ((df["Temperature"] >= 28.0) | (df["Pressure"] >= 1.4)) != (df["Status_Label"] == 1)
        ]
        consistency_error_ratio = len(rule_violation) / len(df)
        print(f" 4. 일관성 평가: 논리적 모순 데이터 비율 {consistency_error_ratio:.2%}")
    else:
        print(" 4. 일관성 평가: 라벨링 전이므로 평가 보류")

    # 5. 정확성 평가
    # Timestamp가 시간형인지, 센서값 컬럼이 숫자형인지 검사합니다.
    is_time_acc = pd.api.types.is_datetime64_any_dtype(df["Timestamp"])
    is_num_acc = pd.api.types.is_numeric_dtype(df["Temperature"]) and pd.api.types.is_numeric_dtype(
        df["Pressure"]
    )
    accuracy_status = "Pass" if is_time_acc and is_num_acc else "Fail"
    print(f" 5. 정확성 평가: 데이터 타입 및 포맷 적합성 [{accuracy_status}]")
    print("-" * 50)


async def main():
    """
    센서 데이터를 수집하고,
    일부러 품질 문제를 넣은 뒤,
    정제 전/후 품질을 비교하는 메인 함수입니다.
    """
    url = "opc.tcp://127.0.0.1:4840/freeopcua/server/"

    # [1단계~2단계] 데이터 수집
    print("데이터 수집을 시작합니다...")

    # 읽어 온 데이터를 순서대로 저장할 리스트입니다.
    collected_data = []

    async with Client(url=url) as client:
        # 제조 설비 관련 네임스페이스 조회
        uri = "http://manufacturing.example.com"
        idx = await client.get_namespace_index(uri)

        # Machine_A 객체와 센서 노드 찾기
        machine = await client.nodes.objects.get_child([f"{idx}:Machine_A"])
        temp_node = await machine.get_child([f"{idx}:Temperature"])
        pressure_node = await machine.get_child([f"{idx}:Pressure"])

        # 총 20번 데이터를 수집합니다.
        for i in range(20):
            temp_val = await temp_node.read_value()
            pressure_val = await pressure_node.read_value()
            timestamp = datetime.now()

            # 일부러 품질이 나쁜 데이터를 삽입합니다.
            # i == 5  : 온도 결측치 만들기
            # i == 10 : 압력 이상치 만들기
            if i == 5:
                temp_val = np.nan
            if i == 10:
                pressure_val = 999.9

            # 한 번 읽은 센서값을 한 행의 딕셔너리로 구성합니다.
            row = {
                "Timestamp": timestamp,
                "Machine_ID": "Machine_A",
                "Temperature": temp_val,
                "Pressure": pressure_val,
            }
            collected_data.append(row)

            # 일부러 중복 데이터도 하나 만들어 넣습니다.
            # 이렇게 해야 중복 제거와 유일성 평가가 실제로 동작하는지 확인할 수 있습니다.
            if i == 15:
                collected_data.append(row.copy())

            # 0.5초 간격으로 데이터를 읽습니다.
            await asyncio.sleep(0.5)

    # 리스트를 DataFrame으로 변환합니다.
    df_raw = pd.DataFrame(collected_data)

    # 정제 전 원시 데이터를 따로 저장합니다.
    # 나중에 원본 상태와 정제 결과를 비교하기 좋습니다.
    raw_filename = "raw_sensor_dataset.csv"
    df_raw.to_csv(raw_filename, index=False, encoding="utf-8-sig")
    print(f"\n[저장 완료] 정제 전 원시 데이터가 '{raw_filename}'로 저장되었습니다.")

    # --- 원시 데이터 품질 평가 ---
    # 아직 아무 처리도 하지 않은 상태에서 품질이 어떤지 먼저 측정합니다.
    evaluate_data_quality(df_raw, "정제 전 원시 데이터")

    # [3단계] 정제 및 [4단계] 라벨링
    print("\n데이터 정제 및 라벨링을 수행합니다...")

    # 원본 보존을 위해 복사본에서 정제 작업을 합니다.
    df_clean = df_raw.copy()

    # 1. 중복 제거
    # 같은 Timestamp를 가진 행은 하나만 남깁니다.
    df_clean = df_clean.drop_duplicates(subset=["Timestamp"])

    # 2. 결측치 제거
    df_clean = df_clean.dropna()

    # 3. 허용 범위를 벗어난 이상치 제거
    df_clean = df_clean[
        (df_clean["Temperature"].between(0, 100)) & (df_clean["Pressure"].between(0, 5))
    ]

    # 4. 숫자 포맷 정리
    df_clean["Temperature"] = df_clean["Temperature"].round(2)
    df_clean["Pressure"] = df_clean["Pressure"].round(2)

    # 5. 라벨링
    # np.where(조건, 참일 때 값, 거짓일 때 값)
    # 규칙:
    # - 온도 >= 28.0 또는 압력 >= 1.4 -> 1(이상)
    # - 그 외 -> 0(정상)
    df_clean["Status_Label"] = np.where(
        (df_clean["Temperature"] >= 28.0) | (df_clean["Pressure"] >= 1.4),
        1,
        0,
    )

    # --- 최종 데이터 품질 평가 ---
    # 정제 후 품질이 얼마나 개선되었는지 다시 측정합니다.
    evaluate_data_quality(df_clean, "정제 및 라벨링 완료 데이터")

    # 정제 후 최종 데이터 저장
    clean_filename = "verified_sensor_dataset.csv"
    df_clean.to_csv(clean_filename, index=False, encoding="utf-8-sig")
    print(f"\n[저장 완료] 최종 데이터셋이 '{clean_filename}'로 저장되었습니다.")


if __name__ == "__main__":
    asyncio.run(main())
