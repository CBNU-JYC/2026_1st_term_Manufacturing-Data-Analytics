# 라이브러리 임포트 및 저장된 객체 불러오기
import torch  # 모델 가중치 로드와 텐서 계산을 위한 PyTorch입니다.
import torch.nn as nn  # 학습 때 사용한 신경망 구조를 다시 만들기 위해 사용합니다.
import pandas as pd  # 실시간 입력 데이터를 데이터프레임으로 다루기 위해 사용합니다.
import numpy as np  # 숫자 배열을 float32로 바꾸는 데 사용합니다.
import joblib  # 저장된 스케일러 객체를 불러오기 위해 사용합니다.
import os  # 현재 파일에서는 직접 사용이 크지 않지만 운영체제 관련 처리용으로 자주 같이 불러옵니다.
from pathlib import Path  # 현재 스크립트 기준 경로를 만들기 위해 사용합니다.

BASE_DIR = Path(__file__).resolve().parent  # 이 파이썬 파일이 있는 폴더 경로를 구합니다.

print("실시간 추론(Inference) 환경 준비 중...")  # 추론 준비 시작 메시지를 출력합니다.

# 학습 장치(Device) 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')  # 가능한 경우 GPU를 사용하고, 아니면 CPU를 사용합니다.

# 모델 아키텍처 재정의 (학습 시와 동일해야 함)
class FaultDiagnosisMLP(nn.Module):  # 학습 때 사용한 MLP 구조를 그대로 다시 정의합니다.
    def __init__(self, input_dim):
        super(FaultDiagnosisMLP, self).__init__()  # 부모 클래스 초기화를 수행합니다.

        self.network = nn.Sequential(  # 여러 층을 순서대로 연결한 신경망입니다.
            nn.Linear(input_dim, 32),  # 입력 특징을 32차원 은닉층으로 변환합니다.
            nn.ReLU(),  # 비선형성을 주기 위해 ReLU 활성화 함수를 적용합니다.
            nn.BatchNorm1d(32),  # 배치 정규화를 적용합니다.
            nn.Dropout(0.3),  # 드롭아웃 30%를 적용합니다.

            nn.Linear(32, 16),  # 두 번째 은닉층으로 차원을 줄입니다.
            nn.ReLU(),  # 다시 ReLU를 적용합니다.
            nn.BatchNorm1d(16),  # 두 번째 배치 정규화를 적용합니다.
            nn.Dropout(0.3),  # 두 번째 드롭아웃을 적용합니다.

            nn.Linear(16, 1)  # 최종 출력은 고장 여부 판단을 위한 로짓 1개입니다.
        )

    def forward(self, x):
        return self.network(x)  # 입력 x를 네트워크에 통과시켜 결과를 반환합니다.


# 파일 경로 설정 (train_model.py에서 저장한 경로와 일치)
model_path = BASE_DIR / 'models' / 'fault_diagnosis_mlp.pth'  # 저장된 모델 가중치 파일 경로입니다.
scaler_path = BASE_DIR / 'models' / 'sensor_scaler.pkl'  # 저장된 스케일러 파일 경로입니다.

# 1. 스케일러 로드
try:
    scaler = joblib.load(scaler_path)  # 학습 때 저장한 스케일러를 불러옵니다.
    print(f"스케일러 로드 완료: {scaler_path}")  # 스케일러 로드 성공 메시지입니다.
except FileNotFoundError:
    print(f"에러: '{scaler_path}' 파일을 찾을 수 없습니다. 모델 학습을 먼저 진행해주세요.")  # 스케일러 파일이 없으면 사용자에게 안내합니다.
    exit()  # 전처리가 불가능하므로 프로그램을 종료합니다.

# 2. 모델 가중치 로드
# 입력 차원(7개: 연속형 5개 + 범주형 원핫 2개)
INPUT_DIM = 7  # 모델이 기대하는 입력 특징 수는 7개입니다.
model = FaultDiagnosisMLP(INPUT_DIM).to(device)  # 동일한 구조의 모델을 만들고 장치로 이동시킵니다.

try:
    # weights_only=True를 통해 보안 경고 방지 및 안전한 로드 수행
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))  # 저장된 모델 가중치를 현재 장치 기준으로 불러옵니다.
    model.eval()  # 추론 모드로 바꿔 BatchNorm과 Dropout이 추론용으로 동작하게 합니다.
    print(f"모델 로드 및 평가 모드(eval) 전환 완료: {model_path}")  # 모델 로드 성공 메시지를 출력합니다.
except FileNotFoundError:
    print(f"에러: '{model_path}' 파일을 찾을 수 없습니다.")  # 모델 파일이 없으면 사용자에게 안내합니다.
    exit()  # 추론이 불가능하므로 종료합니다.

# 현장 설비 실시간 센서 데이터 수집 (시뮬레이션)
# 현장의 PLC나 OPC-UA 서버를 통해 실시간으로 1건의 데이터가 들어왔다고 가정
incoming_data = {
    'Type': 'H',                       # 제품 등급입니다. 가능한 값은 L, M, H입니다.
    'Air temperature [K]': 302.5,      # 공기 온도 센서값입니다.
    'Process temperature [K]': 311.2,  # 공정 온도 센서값입니다.
    'Rotational speed [rpm]': 1350,    # 회전 속도입니다.
    'Torque [Nm]': 70.0,               # 토크값입니다.
    'Tool wear [min]': 215             # 공구 마모 시간입니다.
}

df_new = pd.DataFrame([incoming_data])  # 실시간 입력 1건을 데이터프레임 한 줄로 변환합니다.
print("\n[수집된 실시간 센서 데이터]")  # 입력 데이터 출력 제목입니다.
# 환경에 따라 display가 없으면 print로 대체
print(df_new) if 'display' in globals() else print(df_new)  # 주피터 환경이 아니면 일반 print로 출력합니다.

# 추론을 위한 데이터 전처리 (파이프라인)
# 학습 모델이 기대하는 7개의 피처와 순서를 정확히 맞춰야 합니다.

# 1. 범주형 변수(Type) One-Hot Encoding 수동 처리
df_new['Type_L'] = 1 if incoming_data['Type'] == 'L' else 0  # 제품 등급이 L이면 Type_L을 1로 만듭니다.
df_new['Type_M'] = 1 if incoming_data['Type'] == 'M' else 0  # 제품 등급이 M이면 Type_M을 1로 만듭니다.
df_new = df_new.drop(columns=['Type'])  # 원래 문자열 열 Type은 제거합니다.

# 2. 컬럼 순서 재배치 (학습 데이터와 100% 동일한 순서)
expected_cols = [
    'Air temperature [K]', 'Process temperature [K]',
    'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]',
    'Type_L', 'Type_M'
]  # 학습 시 모델에 들어간 정확한 입력 열 순서입니다.
df_new = df_new[expected_cols]  # 실시간 입력 데이터의 열 순서를 학습 때와 똑같이 맞춥니다.

# 3. 연속형 센서 데이터 스케일링
num_cols = expected_cols[:5]  # 앞의 5개 열만 연속형 센서 데이터입니다.
df_new[num_cols] = scaler.transform(df_new[num_cols])  # 저장된 스케일러 기준으로 센서값을 표준화합니다.

# 4. PyTorch 텐서 변환
X_tensor = torch.tensor(df_new.astype(np.float32).values, dtype=torch.float32).to(device)  # 전처리된 입력을 float32 텐서로 바꾸고 장치로 이동시킵니다.
print("\n데이터 전처리 및 텐서 변환 완료")  # 입력 준비 완료 메시지입니다.

# AI 모델 결함 진단 수행
with torch.no_grad():  # 추론 시에는 기울기 계산이 필요 없으므로 비활성화합니다.
    output = model(X_tensor)  # 입력 1건에 대한 모델 출력을 계산합니다.
    prob = torch.sigmoid(output).item()  # 로짓을 0~1 확률값으로 바꾸고 파이썬 숫자로 꺼냅니다.
    is_fault = prob >= 0.5  # 확률이 0.5 이상이면 고장으로 판정합니다.

print("\n" + "="*40)  # 결과 화면 위쪽 구분선입니다.
print("[AI 설비 상태 판별 결과] ")  # 결과 제목입니다.
print("="*40)  # 제목 아래 구분선입니다.
print(f"▶ 결함 발생 확률: {prob * 100:.2f}%")  # 예측된 고장 확률을 백분율로 출력합니다.

if is_fault:
    print("[경고] 비정상 패턴 감지! 즉시 설비 점검이 필요합니다.")  # 고장으로 판정되면 경고 메시지를 출력합니다.
else:
    print("[정상] 설비가 안정적으로 가동 중입니다.")  # 정상으로 판정되면 정상 메시지를 출력합니다.
print("="*40)  # 결과 화면 아래 구분선입니다.
