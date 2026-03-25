"""
이 파일은 `opc_server.py`를 아주 쉽게 이해할 수 있도록 자세히 주석을 단 설명용 버전입니다.

이 코드의 목적:
1. OPC-UA 서버를 하나 만든다.
2. 서버 안에 Temperature라는 변수를 만든다.
3. 1초마다 Temperature 값을 조금씩 증가시킨다.

즉, 이 파일은 "데이터를 제공하는 쪽(서버)" 예제입니다.
"""

import asyncio

from asyncua import Server


async def main():
    """
    OPC-UA 서버를 생성하고 실행하는 메인 함수입니다.

    이 함수 안에서 하는 일:
    1. 서버 객체 생성
    2. 서버 초기화
    3. 접속 주소(endpoint) 설정
    4. 서버 이름 설정
    5. 네임스페이스 등록
    6. 객체와 변수 생성
    7. 변수 값을 1초마다 업데이트
    """
    # 1. 서버 인스턴스 생성
    # Server()는 OPC-UA 서버를 파이썬 코드 안에서 만들 수 있게 해 주는 객체입니다.
    server = Server()

    # 2. 서버 초기화
    # 내부적으로 필요한 준비 작업을 수행합니다.
    await server.init()

    # 3. 클라이언트가 접속할 주소(endpoint)를 설정합니다.
    # 나중에 클라이언트는 이 주소를 통해 서버에 연결합니다.
    server.set_endpoint("opc.tcp://127.0.0.1:4840/freeopcua/server/")

    # 서버의 이름을 설정합니다.
    # OPC-UA 클라이언트 툴에서 서버를 식별할 때 도움이 됩니다.
    server.set_server_name("Grad_Lab_OPCUA_Server")

    # 4. 네임스페이스 등록
    # 네임스페이스는 쉽게 말해 "이름 충돌을 피하기 위한 구분 공간"입니다.
    uri = "http://manufacturing.example.com"

    # register_namespace를 호출하면 이 URI에 대응하는 번호(index)를 돌려줍니다.
    idx = await server.register_namespace(uri)

    # 5. 객체(Object) 및 변수(Variable) 생성
    # Objects 폴더 아래에 SensorSystem이라는 객체를 하나 생성합니다.
    myobj = await server.nodes.objects.add_object(idx, "SensorSystem")

    # SensorSystem 아래에 Temperature 변수 하나를 만듭니다.
    # 초기값은 20.0입니다.
    myvar = await myobj.add_variable(idx, "Temperature", 20.0)

    # 클라이언트가 이 변수에 값을 쓸 수 있도록 허용합니다.
    # 이 예제에서는 서버가 직접 값을 바꾸지만, 쓰기 가능 설정 예시로 함께 넣어 둔 것입니다.
    await myvar.set_writable()

    print(f"OPC-UA 서버가 시작되었습니다: {server.endpoint.geturl()}")

    # 6. 서버 실행 및 데이터 시뮬레이션
    # async with server 구문 안에 있는 동안 서버가 실제로 동작합니다.
    async with server:
        # 시작 온도값
        count = 20.0

        # 무한 반복으로 값을 계속 갱신합니다.
        while True:
            # 1초마다 한 번씩 값을 바꾸기 위해 잠시 대기합니다.
            await asyncio.sleep(1)

            # 온도를 0.5씩 증가시킵니다.
            count += 0.5

            # 새 값을 Temperature 변수에 기록합니다.
            await myvar.write_value(count)

            # 서버 콘솔에 현재 값 출력
            print(f"서버 데이터 업데이트 -> 온도: {count}")


if __name__ == "__main__":
    # 파일을 직접 실행했을 때만 main()을 시작합니다.
    asyncio.run(main())
