"""
이 파일은 크로메이트 이미지 데이터를 이용해 간단한 CNN 분류 모델을 학습하는 단계입니다.

전체 흐름:
1. 딥러닝 학습에 필요한 라이브러리를 불러옵니다.
2. CNN 모델 구조를 클래스로 만듭니다.
3. 이미지 전처리와 데이터 로더를 준비합니다.
4. 손실 함수와 최적화 도구를 정합니다.
5. 여러 번 데이터를 보며 모델을 학습합니다.
6. 학습이 끝난 모델 가중치를 파일로 저장합니다.
"""

# 딥러닝 계산을 위해 torch를 불러옵니다.
import torch

# 신경망 층을 만들기 위해 torch.nn을 불러옵니다.
import torch.nn as nn

# 모델의 가중치를 업데이트하는 최적화 도구를 불러옵니다.
import torch.optim as optim

# 파일 경로를 다루기 위해 Path를 불러옵니다.
from pathlib import Path

# 데이터를 묶음 단위로 꺼내기 위해 DataLoader를 불러옵니다.
from torch.utils.data import DataLoader

# 이미지 전처리와 폴더 데이터셋 도구를 불러옵니다.
from z_explanation_simple_vision import Compose, Resize, SimpleImageFolder, ToTensor


class SimpleCNN(nn.Module):
    """
    정상과 불량을 구분하는 간단한 CNN 분류 모델입니다.

    매개변수:
        없음

    반환값:
        없음
    """

    def __init__(self):
        """
        CNN 안에 들어갈 층들을 준비합니다.

        매개변수:
            없음

        반환값:
            없음
        """
        # 부모 클래스의 준비 코드를 먼저 실행합니다.
        super(SimpleCNN, self).__init__()

        # 이미지를 보고 특징을 뽑아내는 합성곱 블록입니다.
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # 뽑아낸 특징을 보고 마지막에 2개 클래스 중 하나를 고르는 블록입니다.
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
            x: 모델에 넣을 이미지 배치 텐서

        반환값:
            각 클래스에 대한 예측 점수 텐서
        """
        # 먼저 합성곱 층으로 특징을 뽑습니다.
        x = self.conv_layers(x)

        # 다음으로 완전연결층에서 최종 점수를 만듭니다.
        x = self.fc_layers(x)

        # 계산된 점수를 반환합니다.
        return x


def train_model():
    """
    CNN 모델을 학습하고 결과를 파일로 저장합니다.

    매개변수:
        없음

    반환값:
        없음
    """
    # GPU가 있으면 GPU를 쓰고, 없으면 CPU를 사용합니다.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 어떤 장치에서 학습하는지 화면에 알려 줍니다.
    print(f"사용 디바이스: {device}")

    # 현재 파일이 있는 폴더를 기준 경로로 저장합니다.
    base_dir = Path(__file__).resolve().parent

    # 학습 전에 이미지 크기를 맞추고 텐서로 바꾸는 준비를 합니다.
    transform = Compose([Resize((256, 256)), ToTensor()])

    # 학습 데이터 폴더 경로를 만듭니다.
    train_dir = base_dir / "data" / "학습"

    # 학습용 데이터셋을 만듭니다.
    train_dataset = SimpleImageFolder(root=train_dir, transform=transform)

    # 데이터를 16장씩 섞어서 꺼내는 로더를 만듭니다.
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # CNN 모델을 만들고 학습 장치로 보냅니다.
    model = SimpleCNN().to(device)

    # 정답과 예측의 차이를 계산할 손실 함수를 정합니다.
    criterion = nn.CrossEntropyLoss()

    # Adam 최적화 도구로 모델 가중치를 조금씩 고칩니다.
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 실습용으로 3번만 전체 데이터를 반복해서 봅니다.
    epochs = 3

    # 학습 시작 안내를 출력합니다.
    print("=== 모델 학습 시작 ===")

    # 에폭 수만큼 전체 학습을 반복합니다.
    for epoch in range(epochs):
        # 학습 모드로 바꿉니다.
        model.train()

        # 이번 에폭의 손실 합계를 담을 변수를 준비합니다.
        running_loss = 0.0

        # 데이터 묶음을 하나씩 꺼내 학습합니다.
        for images, labels in train_loader:
            # 이미지와 라벨을 같은 장치로 보냅니다.
            images, labels = images.to(device), labels.to(device)

            # 이전 단계에서 계산된 기울기를 지웁니다.
            optimizer.zero_grad()

            # 모델에 이미지를 넣어 예측 점수를 얻습니다.
            outputs = model(images)

            # 예측 점수와 정답을 비교해 손실을 계산합니다.
            loss = criterion(outputs, labels)

            # 손실을 바탕으로 거꾸로 계산해 기울기를 구합니다.
            loss.backward()

            # 구한 기울기로 가중치를 업데이트합니다.
            optimizer.step()

            # 현재 손실 값을 더해 평균 손실을 나중에 계산합니다.
            running_loss += loss.item()

        # 한 에폭이 끝날 때 평균 손실을 보여 줍니다.
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")

    # 학습이 끝난 모델을 저장할 파일 경로를 만듭니다.
    model_path = results_dir / "cnn_model.pth"

    # 모델의 배운 가중치를 파일로 저장합니다.
    torch.save(model.state_dict(), model_path)

    # 저장이 끝났음을 알려 줍니다.
    print(f">>> 학습 완료! '{model_path.name}' 저장 완료.")


# 이 파일을 직접 실행했을 때만 학습을 시작합니다.
if __name__ == "__main__":
    train_model()

