"""
이 파일은 `advanced_server.py`를 아주 쉽게 이해할 수 있도록 자세히 풀어쓴 설명용 버전입니다.

이 예제는 일반적인 OPC-UA 서버 예제보다 조금 더 심화된 기능을 보여 줍니다.
한 서버 안에서 다음 기능들을 함께 다룹니다.

1. DA (Data Access)
   - 실시간 센서 값을 읽고 쓰는 기능

2. HA (Historical Access)
   - 과거의 센서 값 이력을 저장하고 나중에 조회할 수 있게 하는 기능

3. AC (Alarms & Conditions)
   - 특정 조건이 만족되었을 때 이벤트/알람을 발생시키는 기능

4. Prog (Programs / Method Call)
   - 클라이언트가 서버의 메서드를 원격으로 호출하는 기능

즉, 이 서버는 "실시간 데이터 제공 + 이력 저장 + 이벤트 발생 + 원격 제어 메서드 제공"을
한 번에 보여주는 심화 예제입니다.
"""

import asyncio
import random
from datetime import timedelta

from asyncua import Server, uamethod, ua


@uamethod
def emergency_stop(parent, reason):
    """
    클라이언트가 원격으로 호출할 수 있는 OPC-UA 메서드입니다.

    매개변수:
    - parent: 이 메서드가 속한 객체(보통 자동으로 전달됨)
    - reason: 클라이언트가 전달한 긴급 정지 사유 문자열

    반환값:
    - Boolean True

    @uamethod 데코레이터를 붙이면,
    이 함수는 일반 파이썬 함수가 아니라 OPC-UA 메서드처럼 동작할 수 있습니다.

    실제 공장 환경이라면 이 안에서 다음과 같은 작업이 들어갈 수 있습니다.
    - 모터 정지
    - PLC 출력 차단
    - 알람 로그 남기기
    - 운영자 화면에 경고 표시
    """
    print(f"\n[서버 제어 수신] 긴급 정지 명령 수신! 사유: {reason}")

    # OPC-UA 메서드 반환값은 ua.Variant 형태로 감싸 주는 경우가 많습니다.
    # 여기서는 True를 반환해서 "명령 처리 성공" 의미로 사용합니다.
    return [ua.Variant(True, ua.VariantType.Boolean)]


async def main():
    """
    심화 OPC-UA 서버를 생성하고 실행하는 메인 함수입니다.

    이 함수에서 하는 일:
    1. 서버 생성 및 초기화
    2. 접속 주소 설정
    3. 네임스페이스 등록
    4. Machine_B 객체와 Temperature 변수 생성
    5. 이벤트 발생기 준비
    6. EmergencyStop 메서드 추가
    7. 온도 이력 저장 활성화
    8. 1초마다 온도값을 갱신하고, 조건에 따라 이벤트 발생
    """
    # 1. OPC-UA 서버 객체 생성
    server = Server()

    # 내부 초기화 수행
    await server.init()

    # 2. 서버 엔드포인트 설정
    # 클라이언트는 이 주소로 접속하게 됩니다.
    server.set_endpoint("opc.tcp://127.0.0.1:4840/freeopcua/server/")

    # 3. 네임스페이스 등록
    # 제조 설비 도메인용 URI를 등록하고, 그 인덱스를 받아 옵니다.
    uri = "http://manufacturing.example.com"
    idx = await server.register_namespace(uri)

    # 4. 객체 및 변수 생성 (DA 실습용)
    # Objects 아래에 Machine_B라는 장비 객체를 만듭니다.
    machine = await server.nodes.objects.add_object(idx, "Machine_B")

    # Machine_B 아래에 Temperature 변수 생성
    # 초기값은 20.0입니다.
    temp_node = await machine.add_variable(idx, "Temperature", 20.0)

    # 서버와 클라이언트가 이 변수 값을 변경할 수 있도록 writable 설정
    await temp_node.set_writable()

    # 5. 이벤트 발생기 생성 (AC 실습용)
    # 나중에 온도가 너무 높아졌을 때 이벤트를 발생시키는 데 사용됩니다.
    custom_event = await server.get_event_generator()

    # 6. 메서드(원격 제어) 추가 (Prog 실습용)
    # 클라이언트가 Machine_B 객체에 대해 EmergencyStop 메서드를 호출할 수 있게 합니다.
    #
    # 인자 타입:
    # - String 1개
    #
    # 반환 타입:
    # - Boolean 1개
    await machine.add_method(
        idx,
        "EmergencyStop",
        emergency_stop,
        [ua.VariantType.String],
        [ua.VariantType.Boolean],
    )

    print("OPC-UA 심화 서버가 시작되었습니다...")

    # async with server 블록 안에 있는 동안 서버가 실제 동작합니다.
    async with server:
        # 7. 이력 데이터 저장 설정 (HA 실습용)
        # Temperature 노드의 값이 바뀔 때 그 기록을 서버 메모리에 저장합니다.
        #
        # period=timedelta(days=1)
        #   -> 최대 1일치 보관
        #
        # count=100
        #   -> 최대 100개 기록 보관
        await server.historize_node_data_change(
            temp_node,
            period=timedelta(days=1),
            count=100,
        )

        # 무한 반복으로 실시간 데이터를 갱신합니다.
        while True:
            # 8-1. DA: 실시간 온도값 생성
            # 20.0 ~ 35.0 사이의 랜덤 값을 만듭니다.
            current_temp = 20.0 + random.uniform(0, 15)

            # 생성한 값을 Temperature 노드에 씁니다.
            await temp_node.write_value(current_temp)
            print(f"현재 온도: {current_temp:.2f}")

            # 8-2. AC: 특정 조건이면 이벤트 발생
            # 온도가 33도를 넘으면 "고온 경고" 이벤트를 발생시킵니다.
            if current_temp > 33.0:
                print(">>> 온도 초과! 알람 이벤트를 발생시킵니다.")
                await custom_event.trigger(
                    message=f"High Temperature Warning: {current_temp:.2f}°C"
                )

            # 1초마다 한 번씩 값 갱신
            await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(main())
