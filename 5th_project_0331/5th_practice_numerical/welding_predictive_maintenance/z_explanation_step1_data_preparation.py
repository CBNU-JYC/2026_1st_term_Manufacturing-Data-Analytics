import pandas as pd  # 표 형태의 CSV 데이터를 읽고 다루기 위한 라이브러리입니다.
import numpy as np  # 숫자 배열 계산을 쉽게 처리하기 위한 라이브러리입니다.
import torch  # 최종 전처리 결과를 PyTorch 텐서 형태로 저장하기 위해 사용합니다.
from sklearn.preprocessing import MinMaxScaler  # 데이터를 0~1 범위로 정규화하는 도구입니다.
from pathlib import Path  # 현재 스크립트 파일의 폴더 위치를 안정적으로 찾기 위해 사용합니다.

BASE_DIR = Path(__file__).resolve().parent  # 이 파이썬 파일이 들어 있는 실제 폴더 경로를 구합니다.

print("1. 데이터 로딩 중...", flush=True)  # 현재 어떤 단계가 실행 중인지 화면에 바로 출력합니다.
normal = pd.read_csv(BASE_DIR / 'normal_data.csv', index_col=0)  # 정상 데이터 CSV를 읽고, 첫 번째 열은 인덱스로 사용합니다.
outlier = pd.read_csv(BASE_DIR / 'outlier_data.csv')  # 이상치 데이터 CSV를 읽습니다.

normal_data = normal.copy()  # 원본 정상 데이터를 보호하기 위해 복사본을 만듭니다.
outlier_data = outlier.copy()  # 원본 이상치 데이터도 보호하기 위해 복사본을 만듭니다.
normal_data['FIN_JGMT'] = 0  # 정상 데이터에는 정답 라벨 0을 추가합니다.
outlier_data['FIN_JGMT'] = 1  # 이상 데이터에는 정답 라벨 1을 추가합니다.

use_col = ['DV_R', 'DA_R', 'AV_R', 'AA_R', 'PM_R']  # 모델 학습에 사용할 센서/특징 열 이름 목록입니다.
X_normal = normal_data[use_col]  # 정상 데이터에서 입력 특징만 따로 꺼냅니다.
y_normal = normal_data['FIN_JGMT']  # 정상 데이터의 라벨 열만 따로 꺼냅니다.
X_anomaly = outlier_data[use_col]  # 이상 데이터에서 입력 특징만 따로 꺼냅니다.
y_anomaly = outlier_data['FIN_JGMT']  # 이상 데이터의 라벨 열만 따로 꺼냅니다.

# 가이드북 기준: 정상 데이터 60,000개 분리
X_train_normal = X_normal[:60000]  # 정상 데이터 앞부분 60,000개를 학습용 입력으로 사용합니다.
y_train_normal = y_normal[:60000]  # 위 학습용 입력에 대응하는 라벨입니다.
X_test_normal = X_normal[60000:]  # 남은 정상 데이터는 검증/테스트용 후보로 남깁니다.
y_test_normal = y_normal[60000:]  # 남은 정상 데이터의 라벨입니다.
X_test_anomaly = X_anomaly  # 이상 데이터는 전체를 검증/테스트용 후보로 사용합니다.
y_test_anomaly = y_anomaly  # 이상 데이터 전체의 라벨입니다.

print("2. 데이터 스케일링 적용 중...", flush=True)  # 정규화 단계 시작을 알립니다.
scaler = MinMaxScaler()  # 최소값은 0, 최대값은 1이 되도록 맞추는 스케일러를 준비합니다.
X_train_scaled = scaler.fit_transform(X_train_normal)  # 학습용 정상 데이터로 스케일 기준을 학습하고 변환합니다.
X_test_normal_scaled = scaler.transform(X_test_normal)  # 같은 기준으로 테스트용 정상 데이터를 변환합니다.
X_test_anomaly_scaled = scaler.transform(X_test_anomaly)  # 같은 기준으로 이상 데이터도 변환합니다.

y_train_normal = np.array(y_train_normal)  # 판다스 시리즈를 넘파이 배열로 바꿔 이후 인덱싱을 쉽게 합니다.
y_test_normal = np.array(y_test_normal)  # 정상 테스트 라벨도 넘파이 배열로 변환합니다.
y_test_anomaly = np.array(y_test_anomaly)  # 이상 테스트 라벨도 넘파이 배열로 변환합니다.

print("3. 시계열 시퀀스(Sequence) 생성 중...", flush=True)  # 시계열 입력/목표 시퀀스 생성 단계 시작을 알립니다.
sequence = 20  # 한 번에 사용할 시퀀스 길이를 20 step으로 설정합니다.


def create_sequences(X_scaled, y_data):
    X_seq, Y_seq, Y_idx = [], [], []  # 입력 시퀀스, 정답 시퀀스, 해당 시점 라벨을 담을 빈 리스트입니다.
    for index in range(len(X_scaled) - sequence - 40):  # 미래 구간까지 확보 가능한 위치만 순회합니다.
        # Input: 현재부터 20-step (0~6초)
        X_seq.append(X_scaled[index : index + sequence])  # 현재 시점부터 20개 길이의 입력 구간을 잘라 저장합니다.
        # Target: 일정 시간 뒤의 20-step (12~18초) 예측
        Y_seq.append(X_scaled[index + sequence + 20 : index + sequence + 40])  # 중간 간격을 건너뛴 뒤 미래 20개 구간을 정답으로 저장합니다.
        Y_idx.append(y_data[index + sequence + 40])  # 그 미래 구간 이후 시점의 정상/이상 라벨을 함께 기록합니다.
    return np.array(X_seq), np.array(Y_seq), np.array(Y_idx)  # 리스트를 넘파이 배열로 바꿔 반환합니다.


X_train, Y_train, Y_tr_index = create_sequences(X_train_scaled, y_train_normal)  # 학습용 정상 데이터 시퀀스를 생성합니다.
X_test_normal_arr, Y_test_normal_arr, Y_te_index_1 = create_sequences(X_test_normal_scaled, y_test_normal)  # 테스트용 정상 데이터 시퀀스를 생성합니다.
X_test_anomal_arr, Y_test_anomal_arr, Y_te_index_2 = create_sequences(X_test_anomaly_scaled, y_test_anomaly)  # 테스트용 이상 데이터 시퀀스를 생성합니다.

# 가이드북 기준 검증(Validation) 데이터셋 분리
X_valid_normal, Y_valid_normal = X_test_normal_arr[:4000], Y_test_normal_arr[:4000]  # 정상 시퀀스 중 앞 4,000개를 검증용으로 분리합니다.
Y_val_index_1 = Y_te_index_1[:4000]  # 위 정상 검증 데이터의 라벨입니다.
X_test_normal_arr, Y_test_normal_arr = X_test_normal_arr[4000:], Y_test_normal_arr[4000:]  # 분리하고 남은 정상 데이터만 테스트셋으로 유지합니다.
Y_te_index_1 = Y_te_index_1[4000:]  # 정상 테스트 라벨도 같은 기준으로 남깁니다.

X_valid_anomal, Y_valid_anomal = X_test_anomal_arr[:1200], Y_test_anomal_arr[:1200]  # 이상 시퀀스 중 앞 1,200개를 검증용으로 분리합니다.
Y_val_index_2 = Y_te_index_2[:1200]  # 위 이상 검증 데이터의 라벨입니다.
X_test_anomal_arr, Y_test_anomal_arr = X_test_anomal_arr[1200:], Y_test_anomal_arr[1200:]  # 분리 후 남은 이상 데이터는 테스트셋으로 둡니다.
Y_te_index_2 = Y_te_index_2[1200:]  # 이상 테스트 라벨도 같은 기준으로 남깁니다.

X_valid = np.vstack((X_valid_normal, X_valid_anomal))  # 정상 검증 데이터와 이상 검증 데이터를 위아래로 합칩니다.
Y_valid = np.vstack((Y_valid_normal, Y_valid_anomal))  # 검증용 정답 시퀀스도 같은 순서로 합칩니다.
Y_val_index = np.hstack((Y_val_index_1, Y_val_index_2))  # 검증용 라벨 배열도 좌우로 이어 붙입니다.

X_test = np.vstack((X_test_normal_arr, X_test_anomal_arr))  # 최종 테스트 입력을 정상/이상 데이터 합쳐서 만듭니다.
Y_test = np.vstack((Y_test_normal_arr, Y_test_anomal_arr))  # 최종 테스트 정답 시퀀스도 합칩니다.
Y_te_index = np.hstack((Y_te_index_1, Y_te_index_2))  # 최종 테스트 라벨도 합칩니다.

# 학습용 검증 데이터 (정상 데이터만)
X_valid_0 = X_valid[Y_val_index == 0]  # 검증셋 중 정상 라벨만 골라 학습 중 검증용으로 사용합니다.
Y_valid_0 = Y_valid[Y_val_index == 0]  # 위 정상 검증 입력에 대응하는 정답 시퀀스입니다.

print("4. 전처리된 데이터를 PyTorch 형식으로 저장합니다...", flush=True)  # 전처리 결과 저장 단계 시작을 알립니다.
torch.save({  # 여러 배열을 하나의 사전 형태로 묶어 `.pt` 파일로 저장합니다.
    'X_train': torch.FloatTensor(X_train), 'Y_train': torch.FloatTensor(Y_train),  # 학습용 입력/정답을 float 텐서로 변환해 저장합니다.
    'X_valid_0': torch.FloatTensor(X_valid_0), 'Y_valid_0': torch.FloatTensor(Y_valid_0),  # 정상 검증셋을 텐서로 저장합니다.
    'X_valid': torch.FloatTensor(X_valid), 'Y_valid': torch.FloatTensor(Y_valid), 'Y_val_index': Y_val_index,  # 전체 검증셋과 라벨을 저장합니다.
    'X_test': torch.FloatTensor(X_test), 'Y_test': torch.FloatTensor(Y_test), 'Y_te_index': Y_te_index  # 전체 테스트셋과 라벨을 저장합니다.
}, BASE_DIR / 'processed_dataset.pt')  # 저장 위치는 현재 스크립트와 같은 폴더입니다.
print("데이터셋 준비 완료! (processed_dataset.pt 생성)", flush=True)  # 전처리 완료 메시지를 바로 출력합니다.
