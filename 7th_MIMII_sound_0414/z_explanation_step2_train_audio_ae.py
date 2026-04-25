"""
프로그램 전체 흐름 설명:
1. 정상 밸브 소리 파일만 모읍니다.
2. 각 소리를 Mel-Spectrogram 이미지처럼 바꿔 PyTorch 데이터셋으로 만듭니다.
3. 오토인코더 모델을 만들어 정상 소리를 다시 복원하도록 학습합니다.
4. 복원 오차가 작아지도록 50번 반복 학습합니다.
5. 학습 곡선, 학습 기록, 모델 가중치를 저장합니다.
"""

import os  # 파일 경로를 만들 때 사용합니다.
import glob  # wav 파일 목록을 찾을 때 사용합니다.
import json  # 학습 요약을 json으로 저장할 때 사용합니다.
import numpy as np  # 배열 처리와 패딩에 사용합니다.
import pandas as pd  # 학습 기록을 CSV로 저장할 때 사용합니다.
import librosa  # 오디오를 읽고 Mel-Spectrogram으로 바꿀 때 사용합니다.
import matplotlib.pyplot as plt  # 학습 손실 그래프를 그릴 때 사용합니다.
import torch  # PyTorch 딥러닝 기본 도구입니다.
import torch.nn as nn  # 신경망 층을 만들 때 사용합니다.
import torch.optim as optim  # 모델을 학습시키는 최적화 도구입니다.
from torch.utils.data import Dataset, DataLoader  # 데이터셋과 미니배치 로더입니다.

device = torch.device(
    'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
)  # GPU, Mac MPS, CPU 중 사용 가능한 가장 좋은 장치를 고릅니다.
print(f"학습 장치(Device) 설정 완료: {device}")  # 어떤 장치를 쓰는지 출력합니다.

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 현재 코드가 있는 폴더입니다.
VIS_DIR = os.path.join(BASE_DIR, "visualizations")  # 그래프 저장 폴더입니다.
RESULTS_DIR = os.path.join(BASE_DIR, "results")  # 결과 저장 폴더입니다.
os.makedirs(VIS_DIR, exist_ok=True)  # 그래프 폴더가 없으면 만듭니다.
os.makedirs(RESULTS_DIR, exist_ok=True)  # 결과 폴더가 없으면 만듭니다.

normal_dir = os.path.join(BASE_DIR, '0_dB_valve', 'valve', 'id_02', 'normal')  # 정상 밸브 소리 폴더입니다.
normal_files = glob.glob(os.path.join(normal_dir, '*.wav'))  # 정상 wav 파일 목록입니다.

if not normal_files:  # 정상 파일이 없으면 학습할 수 없습니다.
    print(f"에러: '{normal_dir}' 경로에서 .wav 파일을 찾을 수 없습니다.")
    exit()  # 프로그램을 끝냅니다.

print(f"로드된 정상 학습 오디오 파일 개수: {len(normal_files)}개")  # 학습 파일 개수를 보여줍니다.


class MIMIIMelDataset(Dataset):
    """
    MIMII 정상 오디오 파일을 Mel-Spectrogram 텐서로 바꿔주는 데이터셋입니다.

    Args:
        file_paths: 학습에 사용할 wav 파일 경로 목록입니다.
        sr: 오디오를 읽을 샘플링 레이트입니다.
        n_mels: Mel-Spectrogram의 세로 칸 수입니다.
        max_frames: Mel-Spectrogram의 가로 칸 수를 고정하는 값입니다.

    Returns:
        Dataset: __getitem__에서 입력 텐서와 같은 정답 텐서를 반환합니다.
    """

    def __init__(self, file_paths, sr=16000, n_mels=128, max_frames=128):
        """
        데이터셋이 사용할 설정값을 저장합니다.

        Args:
            file_paths: wav 파일 경로 목록입니다.
            sr: 샘플링 레이트입니다.
            n_mels: Mel 주파수 칸 수입니다.
            max_frames: 시간 방향 길이를 맞출 칸 수입니다.

        Returns:
            None: 객체 안에 설정만 저장합니다.
        """

        self.file_paths = file_paths  # 파일 경로 목록을 저장합니다.
        self.sr = sr  # 모든 오디오를 같은 샘플링 레이트로 읽습니다.
        self.n_mels = n_mels  # Mel-Spectrogram 세로 크기입니다.
        self.max_frames = max_frames  # CNN 입력 크기를 128x128로 맞추기 위한 가로 크기입니다.

    def __len__(self):
        """
        데이터셋에 들어있는 파일 개수를 알려줍니다.

        Args:
            None

        Returns:
            int: wav 파일 개수입니다.
        """

        return len(self.file_paths)  # 파일 개수를 반환합니다.

    def __getitem__(self, idx):
        """
        idx번째 wav 파일을 읽어 학습용 텐서로 변환합니다.

        Args:
            idx: 가져올 파일의 번호입니다.

        Returns:
            tuple: 입력 Mel 텐서와 정답 Mel 텐서입니다. 오토인코더라 둘이 같습니다.
        """

        path = self.file_paths[idx]  # idx번째 파일 경로를 가져옵니다.
        y, _ = librosa.load(path, sr=self.sr)  # 오디오를 숫자 배열로 읽습니다.

        mel = librosa.feature.melspectrogram(y=y, sr=self.sr, n_mels=self.n_mels)  # 소리를 Mel-Spectrogram으로 바꿉니다.
        mel_db = librosa.power_to_db(mel, ref=np.max)  # dB 단위로 바꿉니다.

        mel_db = (mel_db + 80.0) / 80.0  # -80~0 dB 정도의 값을 0~1 범위로 맞춥니다.

        # CNN은 모든 입력 크기가 같아야 하므로 길면 자르고 짧으면 0으로 채웁니다.
        if mel_db.shape[1] > self.max_frames:  # 시간 방향이 너무 길면
            mel_db = mel_db[:, :self.max_frames]  # 앞부분 128프레임만 사용합니다.
        else:  # 시간 방향이 짧으면
            pad_width = self.max_frames - mel_db.shape[1]  # 부족한 칸 수를 계산합니다.
            mel_db = np.pad(mel_db, pad_width=((0, 0), (0, pad_width)), mode='constant')  # 오른쪽을 0으로 채웁니다.

        mel_tensor = torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0)  # (128, 128)을 (1, 128, 128)로 만듭니다.

        return mel_tensor, mel_tensor  # 오토인코더는 입력을 그대로 복원하는 법을 배웁니다.


train_dataset = MIMIIMelDataset(normal_files)  # 정상 소리 파일들로 데이터셋을 만듭니다.
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # 32개씩 섞어서 모델에 넣습니다.
print("학습용 DataLoader 구축 완료 (128x128 크기로 자동 전처리 적용)")  # 준비 완료를 출력합니다.


class AudioAutoencoder(nn.Module):
    """
    정상 소리의 Mel-Spectrogram을 압축했다가 다시 복원하는 CNN 오토인코더입니다.

    Args:
        None: 모델 구조는 코드 안에 고정되어 있습니다.

    Returns:
        nn.Module: 입력 이미지를 복원 이미지로 바꾸는 신경망입니다.
    """

    def __init__(self):
        """
        인코더와 디코더 층을 정의합니다.

        Args:
            None

        Returns:
            None: 모델 층들을 객체 안에 저장합니다.
        """

        super(AudioAutoencoder, self).__init__()  # PyTorch 부모 클래스 초기화를 실행합니다.

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # 1채널 이미지를 16채널로 줄이며 특징을 찾습니다.
            nn.ReLU(),  # 음수는 0으로 만들어 중요한 양수 신호를 남깁니다.
            nn.BatchNorm2d(16),  # 학습이 안정되도록 값의 분포를 맞춥니다.
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 더 깊은 특징을 찾고 크기를 줄입니다.
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 64채널 특징으로 압축합니다.
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 128채널 특징으로 압축합니다.
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 256채널의 작은 특징 지도로 만듭니다.
            nn.ReLU(),
            nn.Flatten(),  # 2D 특징 지도를 1줄 숫자로 펼칩니다.
            nn.Linear(256 * 4 * 4, 64)  # 많은 정보를 64개 숫자로 강하게 압축합니다.
        )  # 인코더는 큰 그림을 작은 비밀코드처럼 압축합니다.

        self.decoder_fc = nn.Sequential(
            nn.Linear(64, 256 * 4 * 4),  # 64개 숫자를 다시 작은 2D 특징 지도로 키웁니다.
            nn.ReLU()
        )  # 디코더 첫 부분입니다.
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # 이미지를 조금 키웁니다.
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # 더 크게 복원합니다.
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 원래 크기에 가까워집니다.
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # 더 세밀하게 복원합니다.
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # 최종 1채널 이미지로 만듭니다.
            nn.Sigmoid()  # 출력값을 0~1 사이로 제한합니다.
        )  # 디코더는 압축된 코드를 다시 그림으로 복원합니다.

    def forward(self, x):
        """
        입력 Mel-Spectrogram을 압축하고 다시 복원합니다.

        Args:
            x: 모양이 (배치, 1, 128, 128)인 입력 텐서입니다.

        Returns:
            torch.Tensor: 복원된 Mel-Spectrogram 텐서입니다.
        """

        encoded = self.encoder(x)  # 입력을 64개 숫자로 압축합니다.
        decoded = self.decoder_fc(encoded)  # 압축 숫자를 작은 2D 특징으로 키웁니다.
        decoded = decoded.view(-1, 256, 4, 4)  # 합성곱 디코더가 이해하는 모양으로 바꿉니다.
        reconstructed = self.decoder_conv(decoded)  # 원래 128x128 모양으로 복원합니다.
        return reconstructed  # 복원 결과를 반환합니다.


model = AudioAutoencoder().to(device)  # 모델을 만들고 계산 장치로 보냅니다.
print("\n[오토인코더 모델 준비 완료]")  # 모델 준비 완료를 알립니다.

criterion = nn.MSELoss()  # 원본과 복원본의 차이를 평균제곱오차로 계산합니다.
optimizer = optim.Adam(model.parameters(), lr=0.001)  # 모델 가중치를 조금씩 고치는 최적화 도구입니다.

epochs = 50  # 전체 데이터를 50번 반복해서 학습합니다.
loss_history = []  # 각 epoch의 평균 손실을 저장합니다.
print("\n[모델 학습 시작 - 실제 밸브의 정상 작동 소리 학습]")  # 학습 시작을 알립니다.
model.train()  # 모델을 학습 모드로 바꿉니다.

for epoch in range(epochs):  # 정해진 횟수만큼 반복합니다.
    epoch_loss = 0.0  # 이번 epoch의 손실 합계를 0으로 시작합니다.
    for batch_x, batch_target in train_loader:  # 미니배치를 하나씩 가져옵니다.
        batch_x = batch_x.to(device)  # 입력을 계산 장치로 보냅니다.
        batch_target = batch_target.to(device)  # 정답도 계산 장치로 보냅니다.

        outputs = model(batch_x)  # 모델이 입력을 복원합니다.
        loss = criterion(outputs, batch_target)  # 복원본과 원본의 차이를 계산합니다.

        optimizer.zero_grad()  # 이전 계산의 기울기를 지웁니다.
        loss.backward()  # 손실을 줄이려면 어떻게 고쳐야 하는지 계산합니다.
        optimizer.step()  # 계산한 방향으로 모델 가중치를 업데이트합니다.

        epoch_loss += loss.item()  # 이번 배치 손실을 누적합니다.

    avg_loss = epoch_loss / len(train_loader)  # epoch 평균 손실입니다.
    loss_history.append(avg_loss)  # 그래프를 그리기 위해 저장합니다.

    if (epoch + 1) % 2 == 0 or epoch == 0:  # 너무 자주 출력하지 않도록 2번마다 출력합니다.
        print(f"Epoch [{epoch + 1:2d}/{epochs}] | Reconstruction Loss (MSE): {avg_loss:.4f}")  # 학습 상태를 보여줍니다.

print("학습 완료!")  # 학습 종료를 알립니다.

plt.figure(figsize=(10, 5))  # 손실 그래프 그림판을 만듭니다.
plt.plot(range(1, epochs + 1), loss_history, marker='o')  # epoch별 손실을 선으로 그립니다.
plt.title('Autoencoder Training Loss')  # 그래프 제목입니다.
plt.xlabel('Epoch')  # x축은 학습 반복 번호입니다.
plt.ylabel('Reconstruction Loss (MSE)')  # y축은 복원 오차입니다.
plt.grid(True, alpha=0.3)  # 읽기 쉽게 격자를 표시합니다.
loss_plot_path = os.path.join(VIS_DIR, "step2_training_loss.png")  # 저장할 이미지 경로입니다.
plt.savefig(loss_plot_path, dpi=200, bbox_inches='tight')  # 손실 그래프를 저장합니다.
plt.close()  # 그림을 닫습니다.
print(f"학습 손실 곡선 저장 완료: {loss_plot_path}")  # 저장 위치를 출력합니다.
pd.DataFrame(
    {
        "epoch": list(range(1, epochs + 1)),  # epoch 번호입니다.
        "loss": loss_history,  # 각 epoch의 평균 손실입니다.
    }
).to_csv(os.path.join(RESULTS_DIR, "step2_training_history.csv"), index=False)  # 학습 기록을 CSV로 저장합니다.

audio_models_dir = os.path.join(BASE_DIR, 'audio_models')  # 모델 파일을 저장할 폴더입니다.
os.makedirs(audio_models_dir, exist_ok=True)  # 모델 폴더가 없으면 만듭니다.
model_path = os.path.join(audio_models_dir, 'audio_autoencoder_real.pth')  # 저장할 모델 파일 경로입니다.
torch.save(model.state_dict(), model_path)  # 모델의 배운 가중치를 저장합니다.
print(f"\n[저장 완료] 밸브 정상음 학습 모델이 '{model_path}'에 저장되었습니다.")  # 저장 완료를 출력합니다.
with open(os.path.join(RESULTS_DIR, "step2_train_summary.json"), "w", encoding="utf-8") as f:  # 학습 요약 파일을 엽니다.
    json.dump(
        {
            "normal_file_count": len(normal_files),  # 정상 학습 파일 개수입니다.
            "epochs": epochs,  # 학습 반복 횟수입니다.
            "device": str(device),  # 사용한 계산 장치입니다.
            "model_path": model_path,  # 저장된 모델 경로입니다.
        },
        f,
        ensure_ascii=False,
        indent=2,
    )  # 요약 정보를 json으로 저장합니다.
