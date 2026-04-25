"""
이 파일은 크로메이트 실습 전체를 한 파일에서 바로 실행해 보는 예제입니다.

전체 흐름:
1. 데이터 전처리 도구와 데이터셋을 준비합니다.
2. 학습 데이터와 테스트 데이터를 로더로 만듭니다.
3. 간단한 CNN 모델을 정의합니다.
4. 모델을 몇 번 학습합니다.
5. 마지막에 테스트 데이터 정확도를 계산합니다.
"""

# 딥러닝 계산을 위해 torch를 불러옵니다.
import torch

# 신경망 층을 만들기 위해 nn을 불러옵니다.
import torch.nn as nn

# 모델을 학습시키는 최적화 도구를 불러옵니다.
import torch.optim as optim

# 파일 경로를 다루기 위해 Path를 불러옵니다.
from pathlib import Path

# 데이터를 여러 장씩 묶기 위해 DataLoader를 불러옵니다.
from torch.utils.data import DataLoader

# torchvision 대신 쓰는 설명 버전 이미지 도구를 불러옵니다.
from z_explanation_simple_vision import Compose, Resize, SimpleImageFolder, ToTensor

# 이미지를 같은 크기로 만들고, 딥러닝용 숫자 텐서로 바꿉니다.
transform = Compose([Resize((256, 256)), ToTensor()])

# 현재 파일이 있는 폴더를 기준 경로로 저장합니다.
BASE_DIR = Path(__file__).resolve().parent

# 학습 데이터와 테스트 데이터 폴더 경로를 만듭니다.
train_dir = BASE_DIR / "data" / "학습"
test_dir = BASE_DIR / "data" / "테스트"

# 폴더 구조를 읽어서 학습용 데이터셋을 만듭니다.
train_dataset = SimpleImageFolder(root=train_dir, transform=transform)

# 같은 방식으로 테스트용 데이터셋도 만듭니다.
test_dataset = SimpleImageFolder(root=test_dir, transform=transform)

# 데이터를 16장씩 꺼내는 로더를 만듭니다.
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 데이터 개수를 먼저 확인해 봅니다.
print(f"학습 데이터 개수: {len(train_dataset)}")
print(f"테스트 데이터 개수: {len(test_dataset)}")


class SimpleCNN(nn.Module):
    """
    이미지를 보고 정상과 불량을 구분하는 간단한 CNN 모델입니다.

    매개변수:
        없음

    반환값:
        없음
    """

    def __init__(self):
        """
        CNN 안의 층들을 준비합니다.

        매개변수:
            없음

        반환값:
            없음
        """
        # 부모 클래스 초기화를 먼저 합니다.
        super(SimpleCNN, self).__init__()

        # 이미지에서 특징을 뽑는 합성곱 부분입니다.
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # 뽑아낸 특징을 바탕으로 최종 클래스를 결정하는 부분입니다.
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 64 * 64, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        """
        입력 이미지를 받아 예측 점수를 계산합니다.

        매개변수:
            x: 입력 이미지 텐서

        반환값:
            각 클래스의 예측 점수
        """
        # 합성곱 층으로 특징을 뽑습니다.
        x = self.conv_layers(x)

        # 완전연결층으로 최종 예측 점수를 만듭니다.
        x = self.fc_layers(x)

        # 예측 점수를 돌려줍니다.
        return x


# GPU가 있으면 GPU를 쓰고, 없으면 CPU를 씁니다.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델을 만들고 계산 장치로 보냅니다.
model = SimpleCNN().to(device)

# 분류 문제용 손실 함수를 준비합니다.
criterion = nn.CrossEntropyLoss()

# Adam 최적화 도구를 준비합니다.
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 실습용으로 3에폭만 학습합니다.
epochs = 3

# 학습을 시작합니다.
for epoch in range(epochs):
    # 학습 모드로 전환합니다.
    model.train()

    # 이번 에폭의 손실 합계를 담을 변수를 만듭니다.
    running_loss = 0.0

    # 학습 데이터 묶음을 하나씩 꺼내 학습합니다.
    for images, labels in train_loader:
        # 이미지와 라벨을 계산 장치로 보냅니다.
        images, labels = images.to(device), labels.to(device)

        # 예전 기울기를 먼저 지웁니다.
        optimizer.zero_grad()

        # 모델 예측 점수를 계산합니다.
        outputs = model(images)

        # 예측과 정답의 차이를 계산합니다.
        loss = criterion(outputs, labels)

        # 오차를 바탕으로 거꾸로 계산해 기울기를 구합니다.
        loss.backward()

        # 기울기로 가중치를 업데이트합니다.
        optimizer.step()

        # 손실 값을 더해 둡니다.
        running_loss += loss.item()

    # 한 에폭이 끝날 때 평균 손실을 출력합니다.
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

# 평가 모드로 바꿉니다.
model.eval()

# 맞춘 개수와 전체 개수를 셀 변수를 만듭니다.
correct = 0
total = 0

# 평가 때는 기울기를 계산하지 않습니다.
with torch.no_grad():
    # 테스트 데이터를 하나씩 확인합니다.
    for images, labels in test_loader:
        # 계산 장치로 보냅니다.
        images, labels = images.to(device), labels.to(device)

        # 모델 예측을 계산합니다.
        outputs = model(images)

        # 가장 큰 점수를 가진 클래스를 고릅니다.
        _, predicted = torch.max(outputs.data, 1)

        # 전체 개수를 더합니다.
        total += labels.size(0)

        # 맞춘 개수를 더합니다.
        correct += (predicted == labels).sum().item()

# 최종 정확도를 출력합니다.
print(f"테스트 데이터 정확도 (Accuracy): {100 * correct / total:.2f}%")

