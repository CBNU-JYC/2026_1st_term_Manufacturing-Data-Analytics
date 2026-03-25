"""
이 파일은 `opc_client_mfg.py`를 자세히 설명한 학습용 버전입니다.

이 코드의 목적:
1. 제조 설비 OPC-UA 서버에 접속한다.
2. Machine_A 장비 아래의 Temperature, Pressure 노드를 찾는다.
3. 일정 시간 동안 센서값을 읽어 온다.
4. 읽어 온 데이터를 pandas DataFrame으로 만든다.
5. 최종적으로 CSV 파일로 저장한다.

즉, 이 파일은 "센서 데이터를 수집해서 데이터셋으로 만드는 클라이언트" 예제입니다.
"""

import asyncio
from datetime import datetime

import pandas as pd
from asyncua import Client


async def main():
    """
    제조 설비 OPC-UA 서버에 접속해서 센서 데이터를 수집하는 메인 함수입니다.
    """
    # 1. OPC-UA 서버 엔드포인트 연결
    url = "opc.tcp://127.0.0.1:4840/freeopcua/server/"

    async with Client(url=url) as client:
        print("OPC-UA 서버에 연결되었습니다. 데이터 수집을 시작합니다...")

        # 2. 네임스페이스 및 노드 찾기
        uri = "http://manufacturing.example.com"
        idx = await client.get_namespace_index(uri)

        # Objects 아래에서 Machine_A 장비를 찾고,
        # 그 아래에서 Temperature, Pressure 센서 노드를 찾습니다.
        objects = client.nodes.objects
        machine = await objects.get_child([f"{idx}:Machine_A"])
        temp_node = await machine.get_child([f"{idx}:Temperature"])
        pressure_node = await machine.get_child([f"{idx}:Pressure"])

        # 수집한 센서 데이터를 저장할 리스트
        collected_data = []

        # 3. 데이터 수집 루프
        # 총 10번 읽고, 1초마다 한 번씩 수집합니다.
        for i in range(10):
            # 현재 온도/압력 값을 서버에서 읽습니다.
            temp_val = await temp_node.read_value()
            pressure_val = await pressure_node.read_value()

            # 값을 읽은 시각도 함께 저장합니다.
            timestamp = datetime.now()

            # 한 번 읽은 데이터를 딕셔너리 한 줄로 정리합니다.
            collected_data.append(
                {
                    "Timestamp": timestamp,
                    "Machine_ID": "Machine_A",
                    "Temperature": round(temp_val, 2),
                    "Pressure": round(pressure_val, 2),
                }
            )

            # 현재 읽은 값을 화면에 출력
            print(
                f"[{timestamp.strftime('%H:%M:%S')}] 온도: {temp_val:.2f}°C, 압력: {pressure_val:.2f}bar"
            )

            # 1초 주기로 샘플링
            await asyncio.sleep(1)

        # 4. 수집된 데이터를 DataFrame으로 변환
        # DataFrame은 표 형태라서 분석, 저장, 시각화에 편리합니다.
        df = pd.DataFrame(collected_data)

        # 5. CSV 파일로 저장
        csv_filename = "manufacturing_sensor_data.csv"
        df.to_csv(csv_filename, index=False, encoding="utf-8-sig")

        print(f"\n데이터 수집 완료! '{csv_filename}' 파일이 성공적으로 생성되었습니다.")


if __name__ == "__main__":
    asyncio.run(main())
