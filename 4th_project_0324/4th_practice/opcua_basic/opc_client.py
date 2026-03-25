import asyncio
from asyncua import Client

# 데이터 변경 시 호출될 핸들러 클래스
class SubHandler(object):
    def datachange_notification(self, node, val, data):
        print(f"[데이터 수신] 노드: {node}, 변경된 온도 값: {val}")

async def main():
    url = "opc.tcp://127.0.0.1:4840/freeopcua/server/"
    
    print(f"서버에 연결 중: {url} ...")
    try:
        async with Client(url=url) as client:
            print("연결 성공!")

            # 1. 네임스페이스 인덱스 찾기
            uri = "http://manufacturing.example.com"
            idx = await client.get_namespace_index(uri)

            # 2. 노드 경로를 통해 변수 노드 찾기 (Browse)
            # Root > Objects > 2:SensorSystem > 2:Temperature
            myvar = await client.nodes.root.get_child(["0:Objects", f"{idx}:SensorSystem", f"{idx}:Temperature"])
            
            # 3. 현재 값 한 번 읽기
            current_val = await myvar.read_value()
            print(f"현재 온도 초기값: {current_val}")

            # 4. 구독(Subscription) 생성 (주기: 500ms)
            handler = SubHandler()
            sub = await client.create_subscription(500, handler)
            
            # 구독할 노드 등록
            handle = await sub.subscribe_data_change(myvar)
            print("온도 데이터 구독을 시작합니다. (Ctrl+C로 종료)")

            # 10초 동안 대기하면서 구독된 데이터 수신
            await asyncio.sleep(10)

            # 5. 구독 해제 및 종료
            await sub.unsubscribe(handle)
            await sub.delete()
            print("구독 종료 및 클라이언트 연결 해제")
    except ConnectionRefusedError:
        print("\n[연결 실패] OPC-UA 서버가 실행 중이 아니거나 아직 포트 4840에서 대기 중이 아닙니다.")
        print("먼저 다른 터미널에서 `python3 opc_server.py`를 실행한 뒤, 이 클라이언트를 다시 실행하세요.")
    except TimeoutError:
        print("\n[연결 실패] 서버 응답 시간이 초과되었습니다.")
        print("서버 주소와 포트가 맞는지, 방화벽 또는 보안 소프트웨어가 연결을 막지 않는지 확인하세요.")

if __name__ == "__main__":
    asyncio.run(main())
