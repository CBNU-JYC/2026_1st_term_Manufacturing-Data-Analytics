"""
이 파일은 `opc_client.py`를 학습용으로 자세히 설명한 버전입니다.

이 코드의 목적:
1. OPC-UA 서버에 접속한다.
2. 서버 안의 Temperature 변수를 찾는다.
3. 현재 값을 한 번 읽는다.
4. 값이 바뀔 때마다 자동으로 알림을 받도록 구독(subscription)한다.

즉, 이 파일은 "데이터를 읽어 가는 쪽(클라이언트)" 예제입니다.
"""

import asyncio

from asyncua import Client


class SubHandler(object):
    """
    데이터 변경 알림을 받을 때 호출되는 핸들러 클래스입니다.

    OPC-UA 구독(subscription)을 만들면,
    서버 값이 바뀔 때 이 클래스의 메서드가 자동으로 호출됩니다.
    """

    def datachange_notification(self, node, val, data):
        """
        구독한 노드의 값이 바뀌었을 때 실행됩니다.

        매개변수:
        - node: 값이 바뀐 노드
        - val: 바뀐 후의 실제 값
        - data: 추가 메타정보
        """
        print(f"[데이터 수신] 노드: {node}, 변경된 온도 값: {val}")


async def main():
    """
    OPC-UA 서버에 접속해서 Temperature 값을 읽고 구독하는 메인 함수입니다.
    """
    # 접속할 OPC-UA 서버 주소
    url = "opc.tcp://127.0.0.1:4840/freeopcua/server/"

    print(f"서버에 연결 중: {url} ...")

    # Client로 서버에 접속합니다.
    async with Client(url=url) as client:
        print("연결 성공!")

        # 1. 네임스페이스 인덱스 찾기
        # 서버와 같은 URI를 기준으로 네임스페이스 번호를 조회합니다.
        uri = "http://manufacturing.example.com"
        idx = await client.get_namespace_index(uri)

        # 2. 노드 경로를 따라가며 Temperature 변수 찾기
        # 경로 의미:
        # Root
        #  -> Objects
        #    -> SensorSystem
        #      -> Temperature
        myvar = await client.nodes.root.get_child(
            ["0:Objects", f"{idx}:SensorSystem", f"{idx}:Temperature"]
        )

        # 3. 현재 값 한 번 읽기
        # 서버가 지금 어떤 온도값을 가지고 있는지 즉시 확인합니다.
        current_val = await myvar.read_value()
        print(f"현재 온도 초기값: {current_val}")

        # 4. 구독(subscription) 생성
        # 500ms 주기로 서버 변경 사항을 감시합니다.
        handler = SubHandler()
        sub = await client.create_subscription(500, handler)

        # Temperature 노드의 값이 바뀔 때 알림을 받도록 등록합니다.
        handle = await sub.subscribe_data_change(myvar)
        print("온도 데이터 구독을 시작합니다. (Ctrl+C로 종료)")

        # 10초 동안 기다리며 변경 알림을 받습니다.
        # 이 시간 동안 서버가 값을 갱신하면 handler가 자동 실행됩니다.
        await asyncio.sleep(10)

        # 5. 구독 해제 및 종료
        # 등록했던 감시를 중단하고 구독 객체를 삭제합니다.
        await sub.unsubscribe(handle)
        await sub.delete()
        print("구독 종료 및 클라이언트 연결 해제")


if __name__ == "__main__":
    asyncio.run(main())
