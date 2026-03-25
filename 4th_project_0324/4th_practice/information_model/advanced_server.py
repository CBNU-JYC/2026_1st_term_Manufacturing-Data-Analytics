import asyncio
import random
from datetime import timedelta
from xmlrpc import server
from asyncua import Server, uamethod, ua

# 클라이언트에서 호출할 원격 제어 메서드 정의 (Prog 실습용)
@uamethod
def emergency_stop(parent, reason):
    print(f"\n[서버 제어 수신] 긴급 정지 명령 수신! 사유: {reason}")
    # 실제 환경에서는 여기서 설비 제어 로직(PLC I/O 차단 등)이 실행됩니다.
    return [ua.Variant(True, ua.VariantType.Boolean)]

async def main():
    server = Server()
    await server.init()
    server.set_endpoint("opc.tcp://127.0.0.1:4840/freeopcua/server/")
    
    uri = "http://manufacturing.example.com"
    idx = await server.register_namespace(uri)
    
    # 1. 객체 및 변수 생성 (DA 실습용)
    machine = await server.nodes.objects.add_object(idx, "Machine_B")
    temp_node = await machine.add_variable(idx, "Temperature", 20.0)
    await temp_node.set_writable()
    
    # 3. 알람/이벤트 발생기 생성 (AC 실습용)
    custom_event = await server.get_event_generator()
    
    # 4. 메서드(원격 제어) 추가 (Prog 실습용)
    # 인자: 입력값(String), 반환값(Boolean)
    await machine.add_method(idx, "EmergencyStop", emergency_stop, 
                             [ua.VariantType.String], [ua.VariantType.Boolean])

    print("OPC-UA 심화 서버가 시작되었습니다...")
    
    async with server:
        # 2. 이력 데이터 저장 설정 (HA 실습용)
        # 서버 메모리에 노드의 변경 이력을 저장하도록 설정

        await server.historize_node_data_change(temp_node, period=timedelta(days=1), count=100)        
        
        while True:
            # DA: 실시간 데이터 업데이트
            current_temp = 20.0 + random.uniform(0, 15)
            await temp_node.write_value(current_temp)
            print(f"현재 온도: {current_temp:.2f}")
            
            # AC: 온도가 33도를 초과하면 알람 이벤트 발생
            if current_temp > 33.0:
                print(">>> 온도 초과! 알람 이벤트를 발생시킵니다.")
                await custom_event.trigger(message=f"High Temperature Warning: {current_temp:.2f}°C")
            
            await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())