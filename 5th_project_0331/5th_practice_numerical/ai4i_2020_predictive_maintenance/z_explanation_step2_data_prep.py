import pandas as pd  # 표 형식 데이터를 읽고 가공하기 위한 판다스입니다.
import numpy as np  # 수치 계산과 배열 처리를 위한 넘파이입니다.
from pathlib import Path  # 현재 스크립트 기준 경로를 안정적으로 만들기 위한 도구입니다.
from sklearn.model_selection import train_test_split  # 학습용/테스트용 데이터 분할 함수입니다.
from sklearn.preprocessing import StandardScaler  # 연속형 센서 데이터를 표준화하기 위한 스케일러입니다.
import torch  # PyTorch 텐서와 딥러닝 데이터 처리를 위해 사용합니다.
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler  # 데이터셋, 배치 로더, 클래스 불균형 샘플러를 불러옵니다.

BASE_DIR = Path(__file__).resolve().parent  # 이 파이썬 파일이 들어 있는 폴더 경로를 구합니다.
print("라이브러리 로드 완료")  # 필요한 라이브러리 준비 완료 메시지입니다.

# 데이터 로드 (EDA 단계와 동일한 원본 데이터)
df = pd.read_csv(BASE_DIR / 'ai4i2020.csv')  # 현재 폴더의 원본 CSV 파일을 읽어옵니다.
print(f"원본 데이터 로드 완료: {df.shape}")  # 읽어온 데이터의 행/열 크기를 출력합니다.

# 데이터 전처리 (식별자 제거, 인코딩, 데이터 누수 방지)
print("\n데이터 전처리 시작...")  # 전처리 단계 시작을 알립니다.

# 1. 불필요한 식별자 컬럼 제거
df_processed = df.drop(columns=['UDI', 'Product ID'])  # 예측에 직접 필요 없는 식별자 열을 제거합니다.

# 2. 범주형 변수(Type) One-Hot Encoding
df_processed = pd.get_dummies(df_processed, columns=['Type'], drop_first=True)  # Type 열을 숫자형 더미 변수로 변환합니다.

# 3. 피처(X)와 타겟(y) 분리 및 데이터 누수(Leakage) 방지
target_col = 'Machine failure'  # 최종적으로 예측할 목표 열 이름입니다.
leakage_cols = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']  # 고장 여부를 직접 알려주는 세부 고장 모드 열로, 입력에 넣으면 누수가 생깁니다.

X = df_processed.drop(columns=[target_col] + leakage_cols)  # 입력 변수 X에서는 타깃과 누수 열을 제외합니다.
y = df_processed[target_col]  # 타깃 y는 Machine failure 열입니다.

# 4. Train / Test 분할 (stratify 적용하여 클래스 비율 유지)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)  # 전체 데이터를 8:2로 나누되, 클래스 비율이 유지되도록 stratify를 적용합니다.

# 5. 연속형 센서 변수 스케일링 (Train 데이터 기준으로만 fit 적용)
scaler = StandardScaler()  # 평균 0, 표준편차 1 기준으로 맞추는 스케일러를 생성합니다.
num_cols = [  # 표준화를 적용할 연속형 센서 변수 목록입니다.
    'Air temperature [K]', 'Process temperature [K]',
    'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]'
]

X_train[num_cols] = scaler.fit_transform(X_train[num_cols])  # 학습 데이터로 기준을 학습한 뒤 학습 데이터를 변환합니다.
X_test[num_cols] = scaler.transform(X_test[num_cols])  # 같은 기준으로 테스트 데이터도 변환합니다.

print(f"Train 피처 형태: {X_train.shape}")  # 학습용 입력 데이터의 크기를 출력합니다.
print(f"Test 피처 형태: {X_test.shape}")  # 테스트용 입력 데이터의 크기를 출력합니다.

# PyTorch Dataset 정의
class ManufacturingDataset(Dataset):  # 판다스 데이터를 PyTorch 데이터셋으로 바꿔주는 사용자 정의 클래스입니다.
    def __init__(self, features, labels):
        # DataFrame을 PyTorch FloatTensor로 변환
        self.X = torch.tensor(features.astype(np.float32).values, dtype=torch.float32)  # 입력 특징을 float32 텐서로 변환합니다.
        # 손실 함수(BCE Loss) 계산을 위해 타겟 형태를 [batch_size, 1]로 맞춤
        self.y = torch.tensor(labels.astype(np.float32).values, dtype=torch.float32).unsqueeze(1)  # 라벨도 float32 텐서로 바꾸고 차원 하나를 추가합니다.

    def __len__(self):
        return len(self.X)  # 데이터셋 전체 길이를 반환합니다.

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]  # 지정 인덱스의 입력과 라벨을 한 쌍으로 반환합니다.


train_dataset = ManufacturingDataset(X_train, y_train)  # 학습 데이터를 PyTorch 데이터셋 객체로 만듭니다.
test_dataset = ManufacturingDataset(X_test, y_test)  # 테스트 데이터도 같은 방식으로 만듭니다.

print("\nPyTorch Dataset 생성 완료")  # 데이터셋 객체 준비 완료 메시지입니다.

# 클래스 불균형 처리를 위한 DataLoader 구축
batch_size = 64  # 한 번에 모델에 넣을 데이터 개수를 64개로 설정합니다.

# 1. 클래스별 가중치 계산 (정상 vs 고장)
class_counts = y_train.value_counts().sort_index()  # 학습 데이터에서 클래스별 개수를 셉니다.
class_weights = 1.0 / class_counts.values  # 적은 클래스일수록 더 큰 가중치를 주기 위해 역수를 취합니다.

# 2. 각 샘플별 가중치 할당
sample_weights = [class_weights[int(label)] for label in y_train]  # 각 샘플에 자기 클래스의 가중치를 부여합니다.
sample_weights = torch.tensor(sample_weights, dtype=torch.float32)  # 샘플 가중치를 PyTorch 텐서로 변환합니다.

# 3. 오버샘플링을 수행하는 WeightedRandomSampler 정의
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)  # 적은 클래스를 더 자주 뽑아 클래스 불균형을 완화하는 샘플러입니다.

# 4. DataLoader 생성
# Train Loader: sampler를 통해 고장 데이터를 균형 있게 추출
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, sampler=sampler
)  # 학습용 DataLoader는 가중 샘플러를 사용해 배치를 만듭니다.

# Test Loader: 실제 분포를 평가하기 위해 섞거나 샘플링을 조작하지 않음
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False
)  # 테스트용 DataLoader는 실제 분포를 유지하기 위해 순서대로 읽습니다.

print("DataLoader 구축 완료 (WeightedRandomSampler 적용)")  # DataLoader 준비 완료 메시지입니다.

# 배치 데이터 추출 테스트 및 오버샘플링 효과 확인
for batch_X, batch_y in train_loader:  # 학습용 로더에서 첫 번째 배치 하나를 꺼내 봅니다.
    print(f"\n[Train DataLoader 배치 확인]")  # 배치 확인용 제목입니다.
    print(f"- X 텐서 형태: {batch_X.shape}")  # 입력 텐서의 모양을 출력합니다.
    print(f"- y 텐서 형태: {batch_y.shape}")  # 라벨 텐서의 모양을 출력합니다.
    print(f"- 1개 배치({batch_size}개) 내 고장(1) 데이터 개수: {int(batch_y.sum().item())}개")  # 한 배치 안에 고장 샘플이 몇 개 있는지 확인합니다.
    break  # 확인용이므로 첫 번째 배치만 보고 종료합니다.
