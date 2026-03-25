import asyncio
import random
from asyncua import Server

async def main():
    # 1. OPC-UA 서버 초기화 및 엔드포인트 설정
    server = Server()
    await server.init()
    server.set_endpoint("opc.tcp://127.0.0.1:4840/freeopcua/server/")

    # 2. 네임스페이스 등록
    uri = "http://manufacturing.example.com"
    idx = await server.register_namespace(uri)

    # 3. 객체(설비) 및 변수(센서) 노드 생성
    myobj = await server.nodes.objects.add_object(idx, "Machine_A")
    temp = await myobj.add_variable(idx, "Temperature", 20.0)
    pressure = await myobj.add_variable(idx, "Pressure", 1.0)

    # 클라이언트에서 값을 읽을 수 있도록 설정
    await temp.set_writable()
    await pressure.set_writable()

    print("가상 제조 설비 OPC-UA 서버가 시작되었습니다...")
    
    # 4. 서버 실행 및 실시간 데이터 업데이트
    async with server:
        while True:
            # 온도 20~30도, 압력 1.0~1.5 범위의 가상 데이터 생성
            new_temp = 20.0 + random.uniform(0, 10)
            new_pressure = 1.0 + random.uniform(0, 0.5)
            
            await temp.write_value(new_temp)
            await pressure.write_value(new_pressure)
            
            await asyncio.sleep(1) # 1초 대기

if __name__ == "__main__":
    asyncio.run(main())