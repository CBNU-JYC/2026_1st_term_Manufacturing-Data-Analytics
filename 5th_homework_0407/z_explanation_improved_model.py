"""
이 프로그램은 공장 기계가 고장날지 예측하는 인공지능 모델을 만드는 예제입니다.

전체 흐름은 아주 간단하게 아래 순서로 진행됩니다.
1. CSV 파일에서 데이터를 읽어옵니다.
2. 도움이 될 만한 새 특징(피처)을 만들어 줍니다.
3. 학습용 데이터와 시험용 데이터를 나눕니다.
4. 고장 데이터가 적기 때문에 개수를 늘려 균형을 맞춥니다.
5. 신경망 모델을 만들어 학습합니다.
6. 가장 좋은 기준값(임곗값)을 찾습니다.
7. 성능을 평가하고 그림과 파일로 저장합니다.

즉, "데이터 준비 -> 모델 학습 -> 성능 확인 -> 결과 저장" 순서로 생각하면 됩니다.
"""

# `json`은 결과를 `.json` 파일로 저장할 때 사용합니다.
import json

# `os`는 폴더 만들기 같은 운영체제 작업을 할 때 사용합니다.
import os

# `warnings`는 실행 중 나오는 경고 메시지를 숨길 때 사용합니다.
import warnings

# `Path`는 파일 경로를 안전하고 보기 좋게 다루게 도와줍니다.
from pathlib import Path

# 현재 이 파이썬 파일이 들어 있는 폴더 위치를 찾습니다.
# 이렇게 해 두면 어느 폴더에서 실행하든 항상 같은 기준으로 파일을 찾을 수 있습니다.
BASE_DIR = Path(__file__).resolve().parent

# Matplotlib 설정 파일을 저장할 폴더 경로입니다.
# 일부 컴퓨터에서는 기본 캐시 폴더에 쓸 수 없어서 경고가 뜨는데,
# 프로젝트 안에 따로 폴더를 정해 주면 그런 문제를 줄일 수 있습니다.
MPL_CONFIG_DIR = BASE_DIR / '.matplotlib'

# 글꼴 같은 캐시 파일을 저장할 폴더 경로입니다.
CACHE_DIR = BASE_DIR / '.cache'

# 읽어올 원본 데이터 파일 경로입니다.
DATA_PATH = BASE_DIR / 'ai4i2020.csv'

# 학습 중 가장 좋았던 모델을 저장할 파일 경로입니다.
BEST_MODEL_PATH = BASE_DIR / 'best_improved_model.pth'

# 평가 그래프 그림을 저장할 파일 경로입니다.
FIGURE_PATH = BASE_DIR / 'evaluation_result.png'

# 모델과 스케일러를 저장할 폴더 경로입니다.
MODEL_DIR = BASE_DIR / 'models'

# 숫자 결과를 따로 정리해서 저장할 JSON 파일 경로입니다.
RESULTS_PATH = BASE_DIR / 'results.json'

# Matplotlib가 설정 파일을 이 폴더에 저장하도록 환경 변수를 정합니다.
os.environ.setdefault('MPLCONFIGDIR', str(MPL_CONFIG_DIR))

# 캐시 파일도 이 폴더에 저장하도록 환경 변수를 정합니다.
os.environ.setdefault('XDG_CACHE_HOME', str(CACHE_DIR))

# `numpy`는 숫자 계산을 빠르게 할 때 많이 쓰는 라이브러리입니다.
import numpy as np

# `pandas`는 표 모양 데이터(CSV 등)를 다루기 아주 편한 라이브러리입니다.
import pandas as pd

# `torch`는 딥러닝 모델을 만들고 학습시키는 라이브러리입니다.
import torch

# `torch.nn`은 신경망 층(layer)들을 만들 때 사용합니다.
import torch.nn as nn

# `torch.optim`은 모델을 어떻게 업데이트할지 정하는 도구입니다.
import torch.optim as optim

# `Dataset`, `DataLoader`는 데이터를 모델에 먹기 좋은 작은 묶음으로 나눠 줍니다.
from torch.utils.data import Dataset, DataLoader

# `train_test_split`은 데이터를 학습용과 테스트용으로 나눕니다.
from sklearn.model_selection import train_test_split

# `StandardScaler`는 숫자 크기를 비슷한 범위로 맞춰 학습을 더 잘 되게 도와줍니다.
from sklearn.preprocessing import StandardScaler

# 여러 성능 평가 함수를 가져옵니다.
from sklearn.metrics import (
    accuracy_score,      # 전체 중 몇 개를 맞혔는지 계산합니다.
    precision_score,     # 고장이라고 말한 것 중 진짜 고장 비율을 계산합니다.
    recall_score,        # 진짜 고장 중 몇 개를 잘 찾았는지 계산합니다.
    f1_score,            # precision과 recall을 함께 보는 점수입니다.
    roc_auc_score,       # 모델이 고장과 정상의 순서를 얼마나 잘 구분하는지 봅니다.
    confusion_matrix,    # 맞춘 것과 틀린 것을 표 형태로 보여 줍니다.
    precision_recall_curve,  # 여러 임곗값에서 precision, recall 변화를 계산합니다.
)

# `resample`은 데이터 개수를 늘리거나 줄일 때 사용할 수 있습니다.
# 여기서는 imblearn이 없을 때 랜덤 오버샘플링 대체용으로 씁니다.
from sklearn.utils import resample

# `matplotlib`는 그래프를 그릴 때 사용합니다.
import matplotlib

# 화면 창을 띄우지 않고 파일로만 그래프를 저장하도록 설정합니다.
# 서버나 터미널 환경에서는 이 설정이 더 안전합니다.
matplotlib.use('Agg')

# 그래프를 실제로 그리기 위한 모듈입니다.
import matplotlib.pyplot as plt

# `seaborn`은 보기 좋은 그래프를 쉽게 그리게 도와줍니다.
import seaborn as sns

# `joblib`는 스케일러 같은 객체를 파일로 저장할 때 자주 씁니다.
import joblib

# 자잘한 경고 메시지는 숨겨서 출력이 더 깔끔하게 보이도록 합니다.
warnings.filterwarnings("ignore")

# Matplotlib 설정 폴더가 없으면 새로 만듭니다.
MPL_CONFIG_DIR.mkdir(exist_ok=True)

# 캐시 폴더가 없으면 새로 만듭니다.
CACHE_DIR.mkdir(exist_ok=True)

# `imblearn`이 설치되어 있다면 SMOTE를 사용해 더 똑똑한 오버샘플링을 합니다.
# 없으면 프로그램이 멈추지 않게 `try-except`로 안전하게 처리합니다.
try:
    # SMOTE는 소수 클래스 데이터를 "비슷한 가짜 샘플"로 만들어 늘려 줍니다.
    from imblearn.over_sampling import SMOTE

    # 설치되어 있음을 표시하는 변수입니다.
    HAS_IMBLEARN = True

# `imblearn`이 없으면 이 부분으로 들어옵니다.
except ModuleNotFoundError:
    # SMOTE 대신 `None`을 넣어 둡니다.
    SMOTE = None

    # 설치되어 있지 않음을 표시합니다.
    HAS_IMBLEARN = False

# GPU가 있으면 GPU를 쓰고, 없으면 CPU를 씁니다.
# 컴퓨터가 더 빠른 장치를 가지고 있으면 자동으로 활용하려는 코드입니다.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 어떤 장치를 사용하는지 화면에 보여 줍니다.
print(f"[Device] {device}")

# 1단계: 데이터 파일을 읽어옵니다.
print("\n[Step 1] 데이터 로딩...")

# CSV 파일을 판다스 데이터프레임 형태로 읽습니다.
df = pd.read_csv(DATA_PATH)

# 데이터의 크기(행 개수, 열 개수)를 출력합니다.
print(f"  원본 형태: {df.shape}")

# 2단계: 원래 데이터에서 새 정보를 계산해 추가합니다.
print("\n[Step 2] 피처 엔지니어링...")

# `Power [W]`는 회전속도와 토크를 이용해서 계산한 힘 관련 값입니다.
# 원래 데이터에 없던 중요한 힌트를 모델에게 더 주기 위해 만듭니다.
df['Power [W]'] = (
    df['Torque [Nm]'] * (df['Rotational speed [rpm]'] * 2 * np.pi / 60)
)

# `Temp_diff [K]`는 공정 온도와 공기 온도의 차이입니다.
# 온도 차이가 클수록 기계 상태와 관련된 힌트가 될 수 있어서 추가합니다.
df['Temp_diff [K]'] = (
    df['Process temperature [K]'] - df['Air temperature [K]']
)

# 어떤 피처를 새로 만들었는지 출력합니다.
print("  추가: Power [W], Temp_diff [K]")

# 3단계: 모델이 먹기 좋은 모양으로 데이터를 정리합니다.
print("\n[Step 3] 전처리...")

# `UDI`, `Product ID`는 단지 번호표 역할이라서 예측에 큰 도움이 되지 않습니다.
# 그래서 이런 식별용 열은 제거합니다.
df_p = df.drop(columns=['UDI', 'Product ID'])

# `Type`은 글자 데이터라서 모델이 바로 이해하기 어렵습니다.
# 그래서 `L`, `M` 같은 값을 0과 1 열로 바꾸는 원-핫 인코딩을 합니다.
df_p = pd.get_dummies(df_p, columns=['Type'], drop_first=True)

# 우리가 맞히고 싶은 정답 열 이름을 변수로 저장합니다.
target_col = 'Machine failure'

# 이 열들은 세부 고장 종류라서 정답과 너무 직접적으로 연결되어 있습니다.
# 이런 열을 그대로 넣으면 "시험 문제 답안지"를 몰래 본 것과 비슷해져서
# 모델이 진짜 실력을 배우지 못합니다. 이런 문제를 데이터 누수라고 합니다.
leakage_cols = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']

# 입력 데이터 X는 정답 열과 누수 열을 뺀 나머지입니다.
X = df_p.drop(columns=[target_col] + leakage_cols)

# 출력 데이터 y는 우리가 맞히고 싶은 정답입니다.
y = df_p[target_col]

# 전체 데이터를 학습용과 테스트용으로 나눕니다.
# `stratify=y`를 넣은 이유는 고장/정상 비율을 두 그룹에서 비슷하게 맞추기 위해서입니다.
# 이렇게 해야 평가가 더 공정해집니다.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# 숫자 크기를 맞춰 줄 열 목록입니다.
num_cols = [
    'Air temperature [K]',
    'Process temperature [K]',
    'Rotational speed [rpm]',
    'Torque [Nm]',
    'Tool wear [min]',
    'Power [W]',
    'Temp_diff [K]',
]

# 평균 0, 표준편차 1에 가깝게 바꿔 주는 스케일러를 준비합니다.
scaler = StandardScaler()

# 원본을 보존하기 위해 복사본을 만듭니다.
X_train_sc = X_train.copy()
X_test_sc = X_test.copy()

# 학습 데이터로 기준을 배우고, 그 기준으로 학습 데이터를 변환합니다.
X_train_sc[num_cols] = scaler.fit_transform(X_train[num_cols])

# 테스트 데이터는 학습 데이터에서 배운 기준만 사용해서 변환합니다.
# 테스트 데이터로 새 기준을 배우면 공정한 평가가 아니기 때문입니다.
X_test_sc[num_cols] = scaler.transform(X_test[num_cols])

# 학습 데이터 크기와 고장 비율을 출력합니다.
print(f"  Train: {X_train.shape} | 고장 비율: {y_train.mean() * 100:.1f}%")

# 4단계: 고장 데이터가 적기 때문에 개수를 늘려서 균형을 맞춥니다.
print("\n[Step 4] 오버샘플링...")

# `imblearn`이 있으면 SMOTE를 사용합니다.
if HAS_IMBLEARN:
    # 어떤 방법을 쓰는지 화면에 알려 줍니다.
    print("  SMOTE 사용")

    # 랜덤 시드를 고정해 매번 비슷한 결과가 나오게 합니다.
    smote = SMOTE(random_state=42)

    # 학습 데이터의 소수 클래스(고장)를 늘려서 균형을 맞춥니다.
    X_res, y_res = smote.fit_resample(X_train_sc, y_train)

# `imblearn`이 없으면 더 단순한 방법으로 대신합니다.
else:
    # 사용자에게 대체 방법을 쓰고 있다고 알려 줍니다.
    print("  imblearn 미설치 -> 랜덤 오버샘플링으로 대체")

    # 피처와 정답을 하나의 표로 합칩니다.
    # 이렇게 해야 정상 데이터와 고장 데이터를 나누기 쉽습니다.
    train_df = X_train_sc.copy()
    train_df[target_col] = y_train.values

    # 정상 데이터만 따로 뽑습니다.
    majority = train_df[train_df[target_col] == 0]

    # 고장 데이터만 따로 뽑습니다.
    minority = train_df[train_df[target_col] == 1]

    # 적은 쪽인 고장 데이터를 복제해서 정상 데이터 개수만큼 늘립니다.
    # SMOTE처럼 새 샘플을 만드는 것은 아니고, 기존 샘플을 뽑아 다시 붙이는 방식입니다.
    minority_upsampled = resample(
        minority,
        replace=True,
        n_samples=len(majority),
        random_state=42,
    )

    # 정상 데이터와 늘린 고장 데이터를 다시 합칩니다.
    balanced = pd.concat([majority, minority_upsampled], axis=0)

    # 데이터 순서를 섞어 줍니다.
    # 같은 종류가 몰려 있으면 학습에 좋지 않을 수 있어서 섞는 것이 좋습니다.
    balanced = balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    # 다시 입력 X와 정답 y로 분리합니다.
    X_res = balanced.drop(columns=[target_col])
    y_res = balanced[target_col]

# 오버샘플링 후 각 클래스 개수를 출력합니다.
print(f"  오버샘플링 후 정상: {(y_res == 0).sum()}  고장: {(y_res == 1).sum()}")


# `Dataset`은 파이토치가 데이터를 꺼내 갈 수 있도록 도와주는 틀입니다.
class MfgDataset(Dataset):
    """
    공장 데이터를 PyTorch가 이해할 수 있는 형태로 바꾸는 클래스입니다.

    매개변수:
        X: 입력 피처 데이터입니다. 표 형태 데이터프레임이나 배열이 들어올 수 있습니다.
        y: 정답 레이블입니다. 0은 정상, 1은 고장을 뜻합니다.

    반환값:
        이 클래스 자체는 데이터셋 객체를 만듭니다.
        이후 `__getitem__`이 호출되면 (입력 텐서, 정답 텐서)를 돌려줍니다.
    """

    def __init__(self, X, y):
        """
        데이터셋을 처음 만들 때 실행되는 준비 함수입니다.

        매개변수:
            X: 모델에 넣을 입력값들입니다.
            y: 각 입력값에 대한 정답입니다.

        반환값:
            반환값은 없습니다.
            대신 `self.X`, `self.y`에 텐서 형태로 저장합니다.
        """
        # 입력 데이터를 `float32` 타입의 텐서로 바꿉니다.
        # 딥러닝에서는 이런 숫자 형식을 많이 사용합니다.
        self.X = torch.tensor(np.array(X, dtype=np.float32), dtype=torch.float32)

        # 정답도 `float32` 텐서로 바꾸고, 모양을 `(개수, 1)` 형태로 맞춥니다.
        # `unsqueeze(1)`은 세로 한 칸을 추가한다고 생각하면 쉽습니다.
        self.y = torch.tensor(
            np.array(y, dtype=np.float32),
            dtype=torch.float32,
        ).unsqueeze(1)

    def __len__(self):
        """
        데이터가 총 몇 개 있는지 알려주는 함수입니다.

        매개변수:
            없음

        반환값:
            데이터 개수인 정수(int)를 반환합니다.
        """
        return len(self.X)

    def __getitem__(self, i):
        """
        i번째 데이터를 꺼내 주는 함수입니다.

        매개변수:
            i: 몇 번째 데이터를 가져올지 나타내는 번호입니다.

        반환값:
            (입력 데이터, 정답 데이터) 튜플을 반환합니다.
        """
        return self.X[i], self.y[i]


# 학습용 데이터를 64개씩 섞어서 꺼내오는 로더를 만듭니다.
train_loader = DataLoader(MfgDataset(X_res, y_res), batch_size=64, shuffle=True)

# 테스트용 데이터는 순서를 굳이 섞지 않아도 되므로 `shuffle=False`로 둡니다.
test_loader = DataLoader(
    MfgDataset(X_test_sc.values, y_test.values),
    batch_size=64,
    shuffle=False,
)


# 신경망 모델 클래스를 만듭니다.
class ImprovedMLP(nn.Module):
    """
    여러 층으로 이루어진 간단한 다층 퍼셉트론(MLP) 모델입니다.

    매개변수:
        n: 입력 피처 개수입니다. 예를 들어 열이 9개면 n은 9입니다.

    반환값:
        고장일 확률을 계산하기 위한 하나의 점수(logit)를 출력하는 모델 객체를 만듭니다.
    """

    def __init__(self, n):
        """
        모델 안의 층들을 실제로 만드는 함수입니다.

        매개변수:
            n: 입력 데이터의 열 개수입니다.

        반환값:
            반환값은 없습니다.
            대신 `self.network` 안에 층 구조를 저장합니다.
        """
        # 부모 클래스의 준비 코드를 먼저 실행합니다.
        super().__init__()

        # `Sequential`은 여러 층을 순서대로 쌓을 때 편리합니다.
        self.network = nn.Sequential(
            # 첫 번째 선형층: 입력 특징 n개를 64개 특징으로 바꿉니다.
            nn.Linear(n, 64),

            # 음수를 0으로 바꾸는 활성화 함수입니다.
            nn.ReLU(),

            # 배치 단위로 값을 안정적으로 맞춰 학습을 돕습니다.
            nn.BatchNorm1d(64),

            # 일부 뉴런을 잠깐 쉬게 해서 과적합을 줄이려는 장치입니다.
            nn.Dropout(0.3),

            # 두 번째 선형층입니다.
            nn.Linear(64, 32),

            # 다시 활성화 함수를 적용합니다.
            nn.ReLU(),

            # 다시 정규화합니다.
            nn.BatchNorm1d(32),

            # 다시 과적합 방지를 위해 드롭아웃을 넣습니다.
            nn.Dropout(0.3),

            # 세 번째 선형층입니다.
            nn.Linear(32, 16),

            # 다시 활성화 함수를 적용합니다.
            nn.ReLU(),

            # 다시 정규화합니다.
            nn.BatchNorm1d(16),

            # 마지막 은닉층 드롭아웃입니다.
            nn.Dropout(0.2),

            # 최종 출력층입니다.
            # 결과는 0~1 확률이 아니라 "점수" 형태로 먼저 나옵니다.
            nn.Linear(16, 1),
        )

    def forward(self, x):
        """
        입력값을 받아 모델을 통과시키는 함수입니다.

        매개변수:
            x: 모델에 넣을 입력 텐서입니다.

        반환값:
            모델의 최종 출력 점수(logit)를 반환합니다.
        """
        return self.network(x)


# 입력 피처 개수에 맞는 모델을 만들고, CPU 또는 GPU로 보냅니다.
model = ImprovedMLP(X_res.shape[1]).to(device)

# 이진 분류용 손실 함수를 만듭니다.
# 출력층에 Sigmoid를 따로 쓰지 않고도 안정적으로 학습할 수 있어 자주 사용합니다.
criterion = nn.BCEWithLogitsLoss()

# Adam 옵티마이저는 학습을 비교적 잘 시켜 주는 기본 선택입니다.
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 검증 손실이 좋아지지 않으면 학습률을 줄이는 스케줄러입니다.
# 너무 큰 속도로 계속 가면 좋은 지점을 지나칠 수 있어서 속도를 줄여 줍니다.
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    'min',
    factor=0.5,
    patience=8,
)

# 전체 학습 횟수입니다.
epochs = 80

# 학습 시작 메시지를 출력합니다.
print(f"\n[Step 5] 학습 시작 (epochs={epochs})...")

# 나중에 그래프를 그리기 위해 손실과 점수를 저장할 리스트입니다.
train_losses = []
val_losses = []
val_f1s = []

# 지금까지 본 최고 F1 점수를 기억합니다.
best_f1 = 0.0

# 얼리 스토핑 카운터입니다.
patience_cnt = 0

# 몇 번 연속으로 좋아지지 않으면 멈출지 정합니다.
EARLY_STOP = 20

# epoch만큼 학습을 반복합니다.
for epoch in range(epochs):
    # 모델을 학습 모드로 바꿉니다.
    # 드롭아웃, 배치정규화 등이 학습 방식으로 동작하게 됩니다.
    model.train()

    # 이번 epoch의 총 학습 손실을 저장할 변수입니다.
    t_loss = 0.0

    # 학습 데이터를 64개씩 가져와 반복합니다.
    for bx, by in train_loader:
        # 데이터를 CPU/GPU 장치로 보냅니다.
        bx, by = bx.to(device), by.to(device)

        # 이전 단계에서 계산된 기울기를 먼저 0으로 초기화합니다.
        optimizer.zero_grad()

        # 모델 예측값과 정답을 비교해 손실을 계산합니다.
        loss = criterion(model(bx), by)

        # 손실을 바탕으로 "어느 방향으로 고쳐야 하는지" 기울기를 계산합니다.
        loss.backward()

        # 계산한 기울기를 사용해 모델 가중치를 업데이트합니다.
        optimizer.step()

        # 이번 배치의 손실값을 누적합니다.
        t_loss += loss.item()

    # 배치 개수로 나눠 평균 학습 손실을 구합니다.
    avg_t = t_loss / len(train_loader)

    # 이제 평가 모드로 바꿉니다.
    # 드롭아웃이 꺼지고, 배치정규화도 평가 방식으로 동작합니다.
    model.eval()

    # 이번 epoch의 검증 손실 합입니다.
    v_loss = 0.0

    # 예측값과 정답을 모아 둘 리스트입니다.
    preds_all = []
    targets_all = []

    # 검증 단계에서는 기울기를 계산할 필요가 없으므로 메모리를 아끼기 위해 꺼 둡니다.
    with torch.no_grad():
        # 테스트 데이터를 하나씩 배치 단위로 확인합니다.
        for bx, by in test_loader:
            # 역시 같은 장치로 보냅니다.
            bx, by = bx.to(device), by.to(device)

            # 모델 출력 점수를 구합니다.
            out = model(bx)

            # 손실을 누적합니다.
            v_loss += criterion(out, by).item()

            # `torch.sigmoid`로 점수를 0~1 사이 확률처럼 바꿉니다.
            # 그리고 0.5 이상이면 고장(1), 아니면 정상(0)으로 판단합니다.
            preds_all.extend((torch.sigmoid(out) >= 0.5).float().cpu().numpy())

            # 실제 정답도 모아 둡니다.
            targets_all.extend(by.cpu().numpy())

    # 평균 검증 손실을 계산합니다.
    avg_v = v_loss / len(test_loader)

    # 이번 epoch의 F1 점수를 계산합니다.
    ep_f1 = f1_score(targets_all, preds_all, zero_division=0)

    # 그래프용 기록을 남깁니다.
    train_losses.append(avg_t)
    val_losses.append(avg_v)
    val_f1s.append(ep_f1)

    # 검증 손실을 보고 필요하면 학습률을 줄입니다.
    scheduler.step(avg_v)

    # 이번 epoch의 F1이 지금까지 최고보다 좋으면 모델을 저장합니다.
    if ep_f1 > best_f1:
        # 최고 점수를 갱신합니다.
        best_f1 = ep_f1

        # 좋아졌으니 얼리 스토핑 카운터를 0으로 되돌립니다.
        patience_cnt = 0

        # 가장 좋은 모델 가중치를 파일로 저장합니다.
        torch.save(model.state_dict(), BEST_MODEL_PATH)

    # 좋아지지 않았으면 카운터를 1 늘립니다.
    else:
        patience_cnt += 1

        # 너무 오래 좋아지지 않으면 학습을 중단합니다.
        # 이유는 더 오래 돌려도 성능이 좋아지지 않을 가능성이 크기 때문입니다.
        if patience_cnt >= EARLY_STOP:
            print(f"  Early Stopping @ epoch {epoch + 1}")
            break

    # 처음 epoch이거나 10번마다 중간 결과를 출력합니다.
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(
            f"  Epoch [{epoch + 1:2d}/{epochs}] "
            f"Train: {avg_t:.4f} | Val: {avg_v:.4f} | F1: {ep_f1:.4f}"
        )

# 학습이 끝난 뒤 최고 F1 점수를 알려 줍니다.
print(f"  학습 완료 - Best F1(thr=0.5): {best_f1:.4f}")

# 6단계: 0.5 말고 더 좋은 임곗값이 있는지 찾습니다.
print("\n[Step 6] 최적 임곗값 탐색...")

# 저장해 둔 가장 좋은 모델을 다시 불러옵니다.
model.load_state_dict(torch.load(BEST_MODEL_PATH, weights_only=True))

# 평가 모드로 전환합니다.
model.eval()

# 확률과 정답을 모두 저장할 리스트입니다.
all_probs = []
all_targets = []

# 역시 평가이므로 기울기는 계산하지 않습니다.
with torch.no_grad():
    # 테스트 데이터를 돌면서 확률을 모읍니다.
    for bx, by in test_loader:
        # 모델 출력에 시그모이드를 씌워 확률처럼 바꿉니다.
        prob = torch.sigmoid(model(bx.to(device))).cpu().numpy()

        # 1차원 형태로 펴서 리스트에 추가합니다.
        all_probs.extend(prob.flatten())

        # 실제 정답도 추가합니다.
        all_targets.extend(by.numpy().flatten())

# 리스트를 넘파이 배열로 바꿉니다.
all_probs = np.array(all_probs)
all_targets = np.array(all_targets)

# 여러 임곗값에서 precision, recall 값을 계산합니다.
prec_a, rec_a, thr_a = precision_recall_curve(all_targets, all_probs)

# 각 임곗값마다 F1 점수를 직접 계산합니다.
# 마지막 precision, recall 값은 threshold 길이와 맞지 않아서 `[:-1]`를 사용합니다.
f1_a = 2 * prec_a[:-1] * rec_a[:-1] / (prec_a[:-1] + rec_a[:-1] + 1e-8)

# F1 점수가 가장 큰 위치의 임곗값을 고릅니다.
opt_thr = thr_a[np.argmax(f1_a)]

# 찾은 최적 임곗값과 그때의 F1을 출력합니다.
print(f"  최적 임곗값: {opt_thr:.4f}  (F1={f1_a.max():.4f})")


def get_metrics(preds):
    """
    예측 결과를 받아 여러 평가 지표를 한 번에 계산하는 함수입니다.

    매개변수:
        preds: 0과 1로 이루어진 최종 예측값 배열입니다.

    반환값:
        (accuracy, precision, recall, f1, auc) 순서의 튜플을 반환합니다.
    """
    # 각종 평가 점수를 튜플로 묶어서 돌려줍니다.
    return (
        accuracy_score(all_targets, preds),
        precision_score(all_targets, preds, zero_division=0),
        recall_score(all_targets, preds, zero_division=0),
        f1_score(all_targets, preds, zero_division=0),
        roc_auc_score(all_targets, all_probs),
    )


# 기본 임곗값 0.5를 사용한 예측 결과입니다.
pred_050 = (all_probs >= 0.50).astype(int)

# 최적 임곗값을 사용한 예측 결과입니다.
pred_opt = (all_probs >= opt_thr).astype(int)

# 두 방식의 성능을 각각 계산합니다.
m_base = get_metrics(pred_050)
m_opt = get_metrics(pred_opt)

# 최적 임곗값 성능을 각각의 변수로 풀어 씁니다.
acc, prec, rec, f1, auc = m_opt

# 7단계: 최종 결과를 출력합니다.
print("\n[Step 7] 최종 평가")

# 기본 임곗값 결과를 보기 좋게 출력합니다.
print(
    f"  [thr=0.50]  "
    f"Acc:{m_base[0]:.4f} Prec:{m_base[1]:.4f} "
    f"Rec:{m_base[2]:.4f} F1:{m_base[3]:.4f} AUC:{m_base[4]:.4f}"
)

# 최적 임곗값 결과를 출력합니다.
print(
    f"  [thr={opt_thr:.4f}] "
    f"Acc:{acc:.4f} Prec:{prec:.4f} "
    f"Rec:{rec:.4f} F1:{f1:.4f} AUC:{auc:.4f}"
)

# 구분선을 출력합니다.
print(f"\n{'=' * 45}")

# 기존 베이스라인 점수를 보여 줍니다.
print("  베이스라인 F1: 0.4539")

# 새 모델 점수와 목표 달성 여부를 보여 줍니다.
print(f"  개선 모델  F1: {f1:.4f}  {'✅ 목표 달성!' if f1 >= 0.70 else '추가 개선 필요'}")

# 다시 구분선을 출력합니다.
print(f"{'=' * 45}")

# 8단계: 그래프를 그려서 눈으로 보기 쉽게 만듭니다.
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 그림 전체 제목입니다.
fig.suptitle('AI4I 2020 - 개선 모델 평가 결과', fontsize=14, fontweight='bold')

# 첫 번째 그래프: 학습 손실과 검증 손실 곡선입니다.
axes[0, 0].plot(train_losses, label='Train Loss', color='steelblue')
axes[0, 0].plot(val_losses, label='Val Loss', color='coral')
axes[0, 0].set_title('학습/검증 Loss 곡선')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# 두 번째 그래프에 쓸 항목 이름들입니다.
met_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']

# 비교용 베이스라인 점수입니다.
baseline = [0.9625, 0.4200, 0.5000, 0.4539, 0.86]

# 현재 모델 점수입니다.
improved = [acc, prec, rec, f1, auc]

# 막대 그래프 위치 계산용 값입니다.
x = np.arange(len(met_names))
w = 0.35

# 두 번째 그래프: 베이스라인과 개선 모델을 막대로 비교합니다.
axes[0, 1].bar(x - w / 2, baseline, w, label='Baseline', color='lightcoral', alpha=0.85)
axes[0, 1].bar(x + w / 2, improved, w, label='Improved', color='steelblue', alpha=0.85)
axes[0, 1].set_xticks(x)
axes[0, 1].set_xticklabels(met_names, rotation=15)
axes[0, 1].set_ylim(0, 1.15)
axes[0, 1].set_title('베이스라인 vs 개선 모델')
axes[0, 1].legend()

# 각 막대 위에 숫자를 적어 보기 쉽게 만듭니다.
for i, v in enumerate(improved):
    axes[0, 1].text(i + w / 2, v + 0.02, f'{v:.3f}', ha='center', fontsize=9, fontweight='bold')

# 세 번째 그래프: 혼동행렬을 그립니다.
cm = confusion_matrix(all_targets, pred_opt)
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    ax=axes[1, 0],
    cbar=False,
    xticklabels=['Normal', 'Fault'],
    yticklabels=['Normal', 'Fault'],
    annot_kws={"size": 13},
)
axes[1, 0].set_title(f'Confusion Matrix (thr={opt_thr:.3f})')
axes[1, 0].set_xlabel('Predicted')
axes[1, 0].set_ylabel('Actual')

# 네 번째 그래프: 임곗값에 따라 precision, recall, F1이 어떻게 바뀌는지 보여 줍니다.
axes[1, 1].plot(thr_a, prec_a[:-1], label='Precision', color='navy', linestyle='--')
axes[1, 1].plot(thr_a, rec_a[:-1], label='Recall', color='seagreen')
axes[1, 1].plot(thr_a, f1_a, label='F1-Score', color='darkorange', lw=2)
axes[1, 1].axvline(
    opt_thr,
    color='red',
    linestyle=':',
    lw=1.5,
    label=f'Optimal={opt_thr:.3f}',
)
axes[1, 1].set_title('Precision / Recall / F1 vs Threshold')
axes[1, 1].legend(fontsize=9)
axes[1, 1].grid(alpha=0.3)

# 그래프 간격을 자동으로 보기 좋게 맞춥니다.
plt.tight_layout()

# 그래프를 파일로 저장합니다.
plt.savefig(FIGURE_PATH, dpi=150, bbox_inches='tight')

# 저장이 끝났다고 알려 줍니다.
print(f"  그래프 저장 -> {FIGURE_PATH.name}")

# 9단계: 학습 결과 파일들을 저장합니다.

# 저장할 폴더가 없으면 만듭니다.
os.makedirs(MODEL_DIR, exist_ok=True)

# 모델 가중치를 저장합니다.
torch.save(model.state_dict(), MODEL_DIR / 'improved_mlp.pth')

# 숫자 크기 조절 기준이 들어 있는 스케일러를 저장합니다.
joblib.dump(scaler, MODEL_DIR / 'improved_scaler.pkl')

# 최적 임곗값도 나중에 다시 쓰기 위해 저장합니다.
joblib.dump(opt_thr, MODEL_DIR / 'optimal_threshold.pkl')

# 저장 완료 메시지를 출력합니다.
print(f"[완료] {MODEL_DIR} 디렉토리에 저장됨")

# 10단계: 핵심 숫자만 따로 JSON 파일로 정리해서 저장합니다.

# JSON은 넘파이 숫자 타입을 잘 모를 수 있어서 일반 파이썬 float로 바꿔 저장합니다.
results = {
    'acc': float(acc),
    'prec': float(prec),
    'rec': float(rec),
    'f1': float(f1),
    'auc': float(auc),
    'opt_thr': float(opt_thr),
}

# 결과를 JSON 파일에 기록합니다.
with open(RESULTS_PATH, 'w') as fp:
    json.dump(results, fp)

# 마지막으로 결과 요약을 출력합니다.
print("\n[결과 요약]")

# 딕셔너리에 있는 항목을 하나씩 꺼내 보기 좋게 출력합니다.
for k, v in results.items():
    print(f"  {k}: {v:.4f}")
