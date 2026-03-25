"""
이 파일은 `advanced_client.py`를 아주 자세히 설명한 학습용 버전입니다.

이 클라이언트는 심화 OPC-UA 서버에 접속해서 다음 기능들을 차례대로 실습합니다.

1. AC (Alarms & Conditions)
   - 서버가 발생시키는 이벤트/알람을 구독한다.

2. DA (Data Access)
   - Temperature 값을 실시간으로 읽는다.

3. HA (Historical Access)
   - 과거의 Temperature 이력 데이터를 조회한다.

4. Prog (Programs / Method Call)
   - 서버의 EmergencyStop 메서드를 원격으로 호출한다.

즉, 이 코드는 "심화 OPC-UA 기능을 실제로 사용하는 클라이언트" 예제입니다.
"""

import asyncio
import datetime

from asyncua import Client, ua


class EventSubHandler:
    """
    서버 이벤트(알람)를 수신할 때 호출되는 핸들러 클래스입니다.

    서버에서 이벤트가 발생하면 event_notification() 메서드가 자동으로 실행됩니다.
    """

    def event_notification(self, event):
        """
        이벤트 수신 시 호출됩니다.

        매개변수:
        - event: 서버가 보낸 이벤트 객체
        """
        print(f"\n[알람 수신 - AC] 설비 이벤트 발생: {event.Message}")


async def main():
    """
    심화 OPC-UA 서버에 접속해서
    이벤트 구독, 실시간 데이터 읽기, 이력 조회, 메서드 호출을 수행하는 메인 함수입니다.
    """
    # 접속할 서버 주소
    url = "opc.tcp://127.0.0.1:4840/freeopcua/server/"

    # Client로 서버 접속
    async with Client(url=url) as client:
        # 서버의 제조 도메인 네임스페이스 인덱스 조회
        uri = "http://manufacturing.example.com"
        idx = await client.get_namespace_index(uri)

        # Objects 아래에서 Machine_B 객체와 Temperature 노드를 찾습니다.
        machine = await client.nodes.objects.get_child([f"{idx}:Machine_B"])
        temp_node = await machine.get_child([f"{idx}:Temperature"])

        # 1. 알람 구독 설정 (AC)
        # 이벤트가 발생했을 때 자동으로 처리할 핸들러를 준비합니다.
        handler = EventSubHandler()

        # 500ms 주기로 이벤트를 확인하는 subscription 생성
        sub = await client.create_subscription(500, handler)

        # 서버 이벤트 전체를 구독합니다.
        await sub.subscribe_events()
        print("이벤트(알람) 구독을 시작했습니다.\n")

        # 2. 실시간 데이터 읽기 (DA)
        print("--- [DA 실습] 실시간 데이터 모니터링 ---")

        # 5번 반복하며 현재 온도값을 읽습니다.
        for _ in range(5):
            val = await temp_node.read_value()
            print(f"실시간 읽기 (DA): {val:.2f}°C")
            await asyncio.sleep(1)

        # 3. 이력 데이터 조회 (HA)
        print("\n--- [HA 실습] 과거 이력 데이터 조회 ---")

        # OPC UA의 시간 기준은 UTC를 사용하는 경우가 많으므로 utcnow()를 사용합니다.
        end_time = datetime.datetime.utcnow()
        start_time = end_time - datetime.timedelta(seconds=10)

        try:
            # 최근 10초 구간의 이력 데이터 중 최대 5개를 가져옵니다.
            history = await temp_node.read_raw_history(
                start_time,
                end_time,
                numvalues=5,
            )

            print(f"조회된 이력 데이터 개수: {len(history)}개")

            for record in history:
                # SourceTimestamp는 UTC일 가능성이 높으므로,
                # 여기서는 보기 편하게 한국 시간(KST = UTC+9)으로 바꿔 출력합니다.
                kst_time = record.SourceTimestamp + datetime.timedelta(hours=9)
                print(
                    f" - 시간(KST): {kst_time.strftime('%H:%M:%S')}, 값: {record.Value.Value:.2f}"
                )
        except Exception as e:
            print("이력 데이터 조회 실패:", e)

        # 4. 원격 제어 메서드 호출 (Prog)
        print("\n--- [Prog 실습] 설비 원격 제어 명령 하달 ---")
        print("클라이언트에서 '긴급 정지' 메서드를 호출합니다.")

        # Machine_B 객체가 가진 EmergencyStop 메서드를 호출합니다.
        # 인자로는 정지 사유 문자열을 전달합니다.
        result = await machine.call_method(
            f"{idx}:EmergencyStop",
            "온도 센서 이상 패턴 감지",
        )

        print(f"제어 명령 실행 결과: {'성공' if result else '실패'}")
        print("\n실습이 완료되었습니다. 서버와의 연결을 종료합니다.")


if __name__ == "__main__":
    asyncio.run(main())
