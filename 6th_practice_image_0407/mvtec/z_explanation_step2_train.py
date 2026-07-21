"""
이 파일은 MVTec AD 이미지로 합성곱 오토인코더를 학습하는 단계입니다.

전체 흐름:
1. 오토인코더 모델 구조를 만듭니다.
2. 정상 학습 이미지만 모아 학습용 데이터셋을 준비합니다.
3. 모델이 입력 이미지를 다시 잘 복원하도록 학습합니다.
4. 학습이 끝난 모델 가중치를 파일로 저장합니다.
"""

# 파일 경로를 다루기 위해 Path를 불러옵니다.
from pathlib import Path

# 딥러닝 학습에 필요한 라이브러리를 불러옵니다.
import torch
import torch.nn as nn
import torch.optim as optim

# 데이터를 여러 장씩 묶어 읽기 위한 DataLoader를 불러옵니다.
from torch.utils.data import DataLoader

# 설명 버전 데이터셋과 전처리 도구를 불러옵니다.
from z_explanation_step1_data_eda import MVTecDataset
from z_explanation_simple_vision import Compose, Resize, ToTensor, resolve_data_dir


class ConvAutoencoder(nn.Module):
    """
    이미지를 압축했다가 다시 복원하는 합성곱 오토인코더 모델입니다.

    매개변수:
        없음

    반환값:
        없음
    """

    def __init__(self):
        """
        오토인코더의 인코더와 디코더 층을 준비합니다.

        매개변수:
            없음

        반환값:
            없음
        """
        # 부모 클래스 준비를 먼저 합니다.
        super(ConvAutoencoder, self).__init__()

        # 인코더는 이미지를 점점 작게 압축하며 중요한 특징만 남깁니다.
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
        )

        # 디코더는 압축된 특징을 다시 원래 이미지 크기로 복원합니다.
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
        입력 이미지를 압축했다가 다시 복원합니다.

        매개변수:
            x: 입력 이미지 텐서

        반환값:
            복원된 이미지 텐서
        """
        # 먼저 인코더로 압축합니다.
        encoded = self.encoder(x)

        # 압축된 특징을 다시 디코더로 복원합니다.
        return self.decoder(encoded)


# 이 파일을 직접 실행했을 때만 학습을 시작합니다.
if __name__ == "__main__":
    # 현재 파일이 있는 폴더를 기준 경로로 저장합니다.
    base_dir = Path(__file__).resolve().parent

    # 데이터 폴더를 찾습니다.
    ROOT_DIR = resolve_data_dir(base_dir)

    # 사용할 카테고리를 bottle로 정합니다.
    CATEGORY = "bottle"

    # 한 번에 읽을 이미지 수와 총 학습 횟수를 정합니다.
    BATCH_SIZE = 16
    NUM_EPOCHS = 50

    # 이미지 전처리를 준비합니다.
    transform = Compose([Resize((256, 256)), ToTensor()])

    # 학습용 정상 이미지 데이터셋을 만듭니다.
    train_dataset = MVTecDataset(ROOT_DIR, CATEGORY, is_train=True, transform=transform)

    # 데이터를 섞어서 여러 장씩 읽는 로더를 만듭니다.
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # GPU가 있으면 GPU를 쓰고, 없으면 CPU를 씁니다.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"학습 디바이스: {device}")

    # 오토인코더 모델을 만들고 계산 장치로 보냅니다.
    model = ConvAutoencoder().to(device)

    # 입력 이미지와 복원 이미지를 비교하기 위해 평균제곱오차를 사용합니다.
    criterion = nn.MSELoss()

    # Adam 최적화 도구를 준비합니다.
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 학습 시작 안내를 출력합니다.
    print("모델 학습 시작...")

    # 학습 모드로 전환합니다.
    model.train()

    # 여러 에폭 동안 같은 과정을 반복합니다.
    for epoch in range(NUM_EPOCHS):
        # 이번 에폭 손실 합계를 담을 변수를 만듭니다.
        epoch_loss = 0

        # 학습 데이터 묶음을 하나씩 꺼냅니다.
        for images, _, _ in train_loader:
            # 이미지들을 계산 장치로 보냅니다.
            images = images.to(device)

            # 모델이 이미지를 복원합니다.
            outputs = model(images)

            # 오토인코더는 자기 자신을 잘 복원하는 것이 목표입니다.
            loss = criterion(outputs, images)

            # 이전 기울기를 지웁니다.
            optimizer.zero_grad()

            # 역전파로 기울기를 계산합니다.
            loss.backward()

            # 계산된 기울기로 가중치를 업데이트합니다.
            optimizer.step()

            # 손실 값을 더해 둡니다.
            epoch_loss += loss.item()

        # 너무 자주 출력하면 복잡하니 10에폭마다 한 번씩만 보여 줍니다.
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {epoch_loss/len(train_loader):.4f}")

    # 저장할 모델 파일 경로를 만듭니다.
    SAVE_PATH = results_dir / "autoencoder_model.pth"

    # 학습된 가중치를 파일로 저장합니다.
    torch.save(model.state_dict(), SAVE_PATH)

    # 저장이 끝났음을 출력합니다.
    print(f"모델 저장 완료: {SAVE_PATH}")

