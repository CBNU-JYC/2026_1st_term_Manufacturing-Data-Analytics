import asyncio
from asyncua import Client, ua
import datetime

# 알람(이벤트)을 수신할 핸들러 클래스 (AC 실습용)
class EventSubHandler:
    def event_notification(self, event):
        print(f"\n[알람 수신 - AC] 설비 이벤트 발생: {event.Message}")

async def main():
    url = "opc.tcp://127.0.0.1:4840/freeopcua/server/"
    
    async with Client(url=url) as client:
        uri = "http://manufacturing.example.com"
        idx = await client.get_namespace_index(uri)
        machine = await client.nodes.objects.get_child([f"{idx}:Machine_B"])
        temp_node = await machine.get_child([f"{idx}:Temperature"])

        # 1. 알람 구독 설정 (AC)
        # 이벤트 핸들러를 등록하고 서버의 이벤트를 구독합니다.
        handler = EventSubHandler()
        sub = await client.create_subscription(500, handler)
        await sub.subscribe_events()
        print("이벤트(알람) 구독을 시작했습니다.\n")

        # 2. 실시간 데이터 읽기 (DA)
        print("--- [DA 실습] 실시간 데이터 모니터링 ---")
        for _ in range(5):
            val = await temp_node.read_value()
            print(f"실시간 읽기 (DA): {val:.2f}°C")
            await asyncio.sleep(1)
        
        # 3. 이력 데이터 조회 (HA)
        print("\n--- [HA 실습] 과거 이력 데이터 조회 ---")
        
        # 수정됨: now() 대신 utcnow()를 사용하여 OPC UA 표준(UTC)에 맞춤
        end_time = datetime.datetime.utcnow() 
        start_time = end_time - datetime.timedelta(seconds=10)
        
        try:
            # numvalues=5 는 '최대 5개'를 가져오라는 의미입니다.
            history = await temp_node.read_raw_history(start_time, end_time, numvalues=5)
            print(f"조회된 이력 데이터 개수: {len(history)}개")
            for record in history:
                # 출력 시에는 다시 보기 편하게 KST(+9시간)로 변환해서 출력할 수도 있습니다.
                kst_time = record.SourceTimestamp + datetime.timedelta(hours=9)
                print(f" - 시간(KST): {kst_time.strftime('%H:%M:%S')}, 값: {record.Value.Value:.2f}")
        except Exception as e:
            print("이력 데이터 조회 실패:", e)

        # 4. 원격 제어 (Prog)
        print("\n--- [Prog 실습] 설비 원격 제어 명령 하달 ---")
        print("클라이언트에서 '긴급 정지' 메서드를 호출합니다.")
        # machine 객체의 EmergencyStop 메서드를 호출하고 인자("온도 센서 이상")를 전달
        result = await machine.call_method(f"{idx}:EmergencyStop", "온도 센서 이상 패턴 감지")
        print(f"제어 명령 실행 결과: {'성공' if result else '실패'}")

        print("\n실습이 완료되었습니다. 서버와의 연결을 종료합니다.")

if __name__ == "__main__":
    asyncio.run(main())