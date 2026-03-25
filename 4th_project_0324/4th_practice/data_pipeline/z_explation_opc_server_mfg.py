"""
이 파일은 `opc_server_mfg.py`를 학습용으로 자세히 설명한 버전입니다.

이 코드의 역할은 "가상의 제조 설비 서버"를 만드는 것입니다.
다른 파이프라인 코드들은 이 서버에 접속해서 Temperature, Pressure 값을 읽어 갑니다.

즉, 이 파일은 데이터 파이프라인 전체에서 "데이터를 공급하는 쪽"입니다.
"""

import asyncio
import random

from asyncua import Server


async def main():
    """
    OPC-UA 서버를 만들고 실행하는 메인 함수입니다.

    이 함수에서 하는 일:
    1. 서버 객체 생성
    2. 서버 초기화
    3. 접속 주소(endpoint) 설정
    4. 네임스페이스 등록
    5. Machine_A 객체와 센서 변수 생성
    6. 온도/압력 값을 1초마다 갱신
    """
    # 1. OPC-UA 서버 객체 생성
    server = Server()

    # 2. 서버 초기화
    # 내부적으로 필요한 설정을 준비합니다.
    await server.init()

    # 3. 클라이언트가 접속할 주소를 지정합니다.
    # data_pipeline.py 쪽의 Client(url=...) 주소와 정확히 같아야 연결됩니다.
    server.set_endpoint("opc.tcp://127.0.0.1:4840/freeopcua/server/")

    # 4. 네임스페이스 등록
    # OPC-UA에서는 노드들을 논리적으로 구분하기 위해 네임스페이스를 사용합니다.
    uri = "http://manufacturing.example.com"

    # URI를 등록하면 해당 네임스페이스 번호(index)를 돌려줍니다.
    idx = await server.register_namespace(uri)

    # 5. 객체(설비)와 변수(센서) 생성
    # Objects 아래에 Machine_A라는 장비를 하나 만듭니다.
    myobj = await server.nodes.objects.add_object(idx, "Machine_A")

    # Machine_A 장비 아래에 센서 변수 두 개를 만듭니다.
    # 초기값:
    # - Temperature = 20.0
    # - Pressure = 1.0
    temp = await myobj.add_variable(idx, "Temperature", 20.0)
    pressure = await myobj.add_variable(idx, "Pressure", 1.0)

    # 변수값을 읽고 쓸 수 있게 설정합니다.
    # 여기서는 서버가 주기적으로 값을 써 넣고,
    # 클라이언트는 그 값을 읽어 가는 구조입니다.
    await temp.set_writable()
    await pressure.set_writable()

    print("가상 제조 설비 OPC-UA 서버가 시작되었습니다...")

    # 6. 서버 실행 및 값 갱신
    # async with server 블록에 들어가 있는 동안 서버가 실제로 동작합니다.
    async with server:
        # 무한 반복으로 센서값을 계속 바꿉니다.
        while True:
            # 온도는 20.0 ~ 30.0 범위의 랜덤 값으로 만듭니다.
            # random.uniform(0, 10)은 0 이상 10 이하의 실수를 만듭니다.
            new_temp = 20.0 + random.uniform(0, 10)

            # 압력은 1.0 ~ 1.5 범위의 랜덤 값으로 만듭니다.
            new_pressure = 1.0 + random.uniform(0, 0.5)

            # 방금 만든 값을 OPC-UA 변수 노드에 기록합니다.
            await temp.write_value(new_temp)
            await pressure.write_value(new_pressure)

            # 1초마다 한 번씩 새로운 센서값을 생성합니다.
            await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(main())
