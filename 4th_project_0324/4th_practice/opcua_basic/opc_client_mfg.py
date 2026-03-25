import asyncio
import pandas as pd
from asyncua import Client
from datetime import datetime

async def main():
    # 1. OPC-UA 서버 엔드포인트 연결
    url = "opc.tcp://127.0.0.1:4840/freeopcua/server/"
    
    async with Client(url=url) as client:
        print("OPC-UA 서버에 연결되었습니다. 데이터 수집을 시작합니다...")
        
        # 2. 네임스페이스 및 노드 찾기
        uri = "http://manufacturing.example.com"
        idx = await client.get_namespace_index(uri)
        
        objects = client.nodes.objects
        machine = await objects.get_child([f"{idx}:Machine_A"])
        temp_node = await machine.get_child([f"{idx}:Temperature"])
        pressure_node = await machine.get_child([f"{idx}:Pressure"])

        collected_data = []
        
        # 3. 데이터 수집 루프 (예: 10초 동안 수집)
        for i in range(10):
            temp_val = await temp_node.read_value()
            pressure_val = await pressure_node.read_value()
            timestamp = datetime.now()
            
            # 수집된 데이터를 딕셔너리 형태로 리스트에 추가
            collected_data.append({
                "Timestamp": timestamp,
                "Machine_ID": "Machine_A",
                "Temperature": round(temp_val, 2),
                "Pressure": round(pressure_val, 2)
            })
            
            print(f"[{timestamp.strftime('%H:%M:%S')}] 온도: {temp_val:.2f}°C, 압력: {pressure_val:.2f}bar")
            await asyncio.sleep(1) # 1초 주기로 샘플링

        # 4. 수집된 데이터를 Pandas 데이터프레임으로 변환
        df = pd.DataFrame(collected_data)
        
        # 5. CSV 파일로 저장 (데이터셋 생성)
        csv_filename = "manufacturing_sensor_data.csv"
        df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
        
        print(f"\n데이터 수집 완료! '{csv_filename}' 파일이 성공적으로 생성되었습니다.")

if __name__ == "__main__":
    asyncio.run(main())