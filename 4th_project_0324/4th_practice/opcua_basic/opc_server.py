import asyncio
from asyncua import Server

async def main():
    # 1. 서버 인스턴스 생성 및 설정
    server = Server()
    await server.init()
    server.set_endpoint("opc.tcp://127.0.0.1:4840/freeopcua/server/")
    server.set_server_name("Grad_Lab_OPCUA_Server")

    # 2. 네임스페이스(Namespace) 등록: 변수 충돌 방지를 위한 고유 공간
    uri = "http://manufacturing.example.com"
    idx = await server.register_namespace(uri)

    # 3. 객체(Object) 및 변수(Variable) 생성 (정보 모델링)
    # Root > Objects 폴더 아래에 'SensorSystem' 객체 생성
    myobj = await server.nodes.objects.add_object(idx, "SensorSystem")
    
    # SensorSystem 객체 아래에 'Temperature' 변수 생성 (초기값 20.0)
    myvar = await myobj.add_variable(idx, "Temperature", 20.0)
    
    # 클라이언트가 이 변수의 값을 쓸 수 있도록 권한 부여
    await myvar.set_writable()

    try:
        print(f"OPC-UA 서버가 시작되었습니다: {server.endpoint.geturl()}")

        # 4. 서버 실행 및 데이터 시뮬레이션
        async with server:
            count = 20.0
            while True:
                await asyncio.sleep(1) # 1초마다 데이터 업데이트
                count += 0.5
                await myvar.write_value(count) # 변수 값 쓰기
                print(f"서버 데이터 업데이트 -> 온도: {count}")
    except OSError as exc:
        print("\n[서버 시작 실패]")
        print(f"- 주소: {server.endpoint.geturl()}")
        print(f"- 원인: {exc}")
        print("- 같은 포트를 다른 프로세스가 이미 사용 중인지 확인하세요.")
        print("- 실습은 서버를 먼저 실행한 뒤 클라이언트를 실행해야 합니다.")

if __name__ == "__main__":
    asyncio.run(main())
