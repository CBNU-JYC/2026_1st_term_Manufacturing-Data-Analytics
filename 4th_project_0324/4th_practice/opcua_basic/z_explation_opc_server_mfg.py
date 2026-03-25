"""
이 파일은 `opc_server_mfg.py`를 쉽게 이해할 수 있도록 자세히 설명한 버전입니다.

이 코드는 제조 설비를 흉내 낸 OPC-UA 서버 예제입니다.
서버 안에는 Machine_A라는 장비가 있고,
그 장비 아래에 Temperature와 Pressure 센서 변수가 있습니다.

즉, 이 파일은 "제조 설비 센서 데이터를 만들어 내는 서버" 역할을 합니다.
"""

import asyncio
import random

from asyncua import Server


async def main():
    """
    제조 설비용 OPC-UA 서버를 생성하고 실행하는 메인 함수입니다.

    주요 작업:
    1. 서버 생성 및 초기화
    2. 접속 주소 설정
    3. 네임스페이스 등록
    4. Machine_A 객체 생성
    5. Temperature, Pressure 변수 생성
    6. 1초마다 랜덤 센서값 생성 및 기록
    """
    # 1. OPC-UA 서버 초기화
    server = Server()
    await server.init()

    # 클라이언트가 접속할 서버 주소
    server.set_endpoint("opc.tcp://127.0.0.1:4840/freeopcua/server/")

    # 2. 네임스페이스 등록
    uri = "http://manufacturing.example.com"
    idx = await server.register_namespace(uri)

    # 3. 객체(장비) 및 변수(센서) 노드 생성
    # Objects 아래에 Machine_A라는 장비 객체를 만듭니다.
    myobj = await server.nodes.objects.add_object(idx, "Machine_A")

    # Machine_A 아래에 두 개의 센서 변수를 만듭니다.
    # 초기값:
    # - Temperature = 20.0
    # - Pressure = 1.0
    temp = await myobj.add_variable(idx, "Temperature", 20.0)
    pressure = await myobj.add_variable(idx, "Pressure", 1.0)

    # 변수 값을 읽고 쓸 수 있도록 설정
    await temp.set_writable()
    await pressure.set_writable()

    print("가상 제조 설비 OPC-UA 서버가 시작되었습니다...")

    # 4. 서버 실행 및 실시간 데이터 업데이트
    async with server:
        # 무한 반복으로 센서 값을 계속 바꿉니다.
        while True:
            # 온도는 20~30 범위의 랜덤 값
            new_temp = 20.0 + random.uniform(0, 10)

            # 압력은 1.0~1.5 범위의 랜덤 값
            new_pressure = 1.0 + random.uniform(0, 0.5)

            # 새 값을 서버 노드에 기록
            await temp.write_value(new_temp)
            await pressure.write_value(new_pressure)

            # 1초에 한 번씩 센서값 갱신
            await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(main())
