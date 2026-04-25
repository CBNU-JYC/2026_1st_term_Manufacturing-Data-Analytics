"""
프로그램 전체 흐름 설명:
1. 학습된 오토인코더 모델을 불러옵니다.
2. 비정상 오디오 파일 하나를 골라 2초 창으로 잘라가며 분석합니다.
3. 각 2초 구간을 Mel-Spectrogram으로 바꾸고 복원 오차를 계산합니다.
4. 복원 오차가 임계값보다 크면 비정상, 작으면 정상으로 판단합니다.
5. 시간에 따른 이상 점수를 그래프로 그리고 CSV로 저장합니다.
"""

import os  # 파일과 폴더 경로를 만들 때 사용합니다.
import glob  # wav 파일 목록을 찾을 때 사용합니다.
import numpy as np  # 배열 처리와 패딩에 사용합니다.
import pandas as pd  # 추론 결과를 CSV로 저장할 때 사용합니다.
import librosa  # 오디오를 읽고 Mel-Spectrogram으로 바꿀 때 사용합니다.
import matplotlib.pyplot as plt  # 이상 점수 그래프를 그릴 때 사용합니다.
import torch  # PyTorch 모델을 실행할 때 사용합니다.
import torch.nn as nn  # 신경망 층과 MSELoss를 사용할 때 필요합니다.

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 현재 코드가 있는 폴더입니다.
VIS_DIR = os.path.join(BASE_DIR, "visualizations")  # 그림 저장 폴더입니다.
RESULTS_DIR = os.path.join(BASE_DIR, "results")  # 결과 저장 폴더입니다.
os.makedirs(VIS_DIR, exist_ok=True)  # 그림 폴더가 없으면 만듭니다.
os.makedirs(RESULTS_DIR, exist_ok=True)  # 결과 폴더가 없으면 만듭니다.


def save_current_figure(filename):
    """
    현재 그래프를 이미지 파일로 저장합니다.

    Args:
        filename: 저장할 이미지 파일 이름입니다.

    Returns:
        None: 이미지를 저장하고 저장 위치를 출력합니다.
    """

    save_path = os.path.join(VIS_DIR, filename)  # 저장할 전체 경로를 만듭니다.
    plt.savefig(save_path, dpi=200, bbox_inches='tight')  # 그래프를 선명하게 저장합니다.
    print(f"이미지 저장 완료: {save_path}")  # 저장 위치를 출력합니다.


class AudioAutoencoder(nn.Module):
    """
    학습 때 사용한 것과 같은 오토인코더 모델입니다.

    Args:
        None: 모델 구조는 코드 안에 고정되어 있습니다.

    Returns:
        nn.Module: 입력 Mel-Spectrogram을 복원하는 모델입니다.
    """

    def __init__(self):
        """
        인코더와 디코더 층을 만듭니다.

        Args:
            None

        Returns:
            None: 모델 층들을 객체에 저장합니다.
        """

        super(AudioAutoencoder, self).__init__()  # PyTorch 모델 초기화를 실행합니다.

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # 입력 이미지를 작게 만들며 특징을 찾습니다.
            nn.ReLU(),  # 음수를 0으로 바꿔 계산을 단순하게 합니다.
            nn.BatchNorm2d(16),  # 값 분포를 안정화합니다.
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 더 많은 특징을 찾습니다.
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 더 작고 깊은 특징으로 압축합니다.
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 정상 소리 패턴을 더 추상적으로 저장합니다.
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 4x4 크기의 작은 특징으로 줄입니다.
            nn.ReLU(),
            nn.Flatten(),  # 2D 특징을 한 줄 숫자로 펼칩니다.
            nn.Linear(256 * 4 * 4, 64)  # 64개 숫자로 강하게 압축합니다.
        )

        self.decoder_fc = nn.Sequential(
            nn.Linear(64, 256 * 4 * 4),  # 64개 숫자를 다시 작은 특징 지도 크기로 키웁니다.
            nn.ReLU()
        )
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # 크기를 2배씩 키웁니다.
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # 복원 과정을 계속합니다.
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 원본 크기에 가까워집니다.
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # 세부 패턴을 만듭니다.
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # 최종 1채널 이미지입니다.
            nn.Sigmoid()  # 출력값을 0~1 범위로 제한합니다.
        )

    def forward(self, x):
        """
        입력 Mel-Spectrogram을 압축하고 다시 복원합니다.

        Args:
            x: (배치, 1, 128, 128) 모양의 입력 텐서입니다.

        Returns:
            torch.Tensor: 복원된 Mel-Spectrogram 텐서입니다.
        """

        encoded = self.encoder(x)  # 입력을 64개 숫자로 압축합니다.
        decoded = self.decoder_fc(encoded)  # 압축 숫자를 작은 이미지 특징으로 키웁니다.
        decoded = decoded.view(-1, 256, 4, 4)  # 디코더가 처리할 수 있는 모양으로 바꿉니다.
        reconstructed = self.decoder_conv(decoded)  # 원래 크기의 이미지로 복원합니다.
        return reconstructed  # 복원 결과를 반환합니다.


device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')  # 사용할 계산 장치입니다.
model = AudioAutoencoder().to(device)  # 모델을 만들고 장치로 보냅니다.
model.load_state_dict(torch.load(os.path.join(BASE_DIR, 'audio_models', 'audio_autoencoder_real.pth'), map_location=device))  # 학습된 가중치를 불러옵니다.
model.eval()  # 추론 모드로 바꿉니다.

THRESHOLD = 0.0016  # 이 값보다 복원 오차가 크면 비정상으로 판단합니다.
print(f"모델 로드 완료. 설정된 결함 임계값: {THRESHOLD}")  # 사용 중인 임계값을 보여줍니다.

abnormal_files = glob.glob(os.path.join(BASE_DIR, '0_dB_valve', 'valve', 'id_02', 'abnormal', '*.wav'))  # 비정상 파일 목록입니다.
test_file = abnormal_files[0]  # 첫 번째 비정상 파일을 예시로 고릅니다.
print(f"추론 대상 파일: {test_file}")  # 어떤 파일을 분석하는지 출력합니다.

y, sr = librosa.load(test_file, sr=16000)  # 오디오 파일을 16,000Hz로 읽습니다.

window_size = sr * 2  # 2초 길이의 소리 조각을 분석합니다.
hop_length = sr * 1  # 1초씩 옆으로 이동하며 다음 조각을 봅니다.
n_mels = 128  # Mel-Spectrogram 세로 칸 수입니다.

scores = []  # 시간별 이상 점수를 저장합니다.
times = []  # 각 점수가 시작된 시간을 저장합니다.

print("\n실시간 모니터링 시뮬레이션 중...")  # 추론 시작을 알립니다.
for start in range(0, len(y) - window_size, hop_length):  # 0초부터 끝까지 1초씩 이동합니다.
    window = y[start:start + window_size]  # 현재 2초 구간을 잘라냅니다.

    mel = librosa.feature.melspectrogram(y=window, sr=sr, n_mels=n_mels)  # 2초 구간을 Mel-Spectrogram으로 바꿉니다.
    mel_db = librosa.power_to_db(mel, ref=np.max)  # dB 단위로 바꿉니다.

    max_frames = 128  # 모델 입력 시간 길이입니다.
    if mel_db.shape[1] > max_frames:  # 시간이 너무 길면
        mel_db = mel_db[:, :max_frames]  # 앞 128프레임만 사용합니다.
    else:  # 시간이 짧으면
        pad_width = max_frames - mel_db.shape[1]  # 부족한 프레임 수를 계산합니다.
        mel_db = np.pad(mel_db, pad_width=((0, 0), (0, pad_width)), mode='constant')  # 0으로 채웁니다.

    mel_db = (mel_db + 80.0) / 80.0  # 모델이 학습한 0~1 범위로 정규화합니다.

    input_tensor = torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)  # 모델 입력 모양으로 바꿉니다.

    with torch.no_grad():  # 추론에서는 기울기를 계산하지 않습니다.
        reconstructed = model(input_tensor)  # 모델이 현재 구간을 복원합니다.
        loss = nn.MSELoss()(reconstructed, input_tensor).item()  # 원본과 복원본의 차이를 이상 점수로 씁니다.

    scores.append(loss)  # 현재 구간의 이상 점수를 저장합니다.
    current_time = start / sr  # 현재 구간의 시작 시간을 초 단위로 계산합니다.
    times.append(current_time)  # 시간을 저장합니다.

    status = "[ABNORMAL]" if loss > THRESHOLD else "[NORMAL]"  # 임계값보다 크면 비정상입니다.
    print(f"Time: {current_time:4.1f}s | Anomaly Score: {loss:.4f} | Status: {status}")  # 시간별 판단을 출력합니다.

plt.figure(figsize=(12, 5))  # 시간 그래프 그림판을 만듭니다.
plt.plot(times, scores, marker='o', label='Anomaly Score')  # 시간별 이상 점수를 선으로 그립니다.
plt.axhline(y=THRESHOLD, color='red', linestyle='--', label='Detection Threshold')  # 임계값 선을 그립니다.
plt.title('Machine Status Monitoring over Time')  # 그래프 제목입니다.
plt.xlabel('Time (sec)')  # x축은 시간입니다.
plt.ylabel('Reconstruction Error (MSE)')  # y축은 복원 오차입니다.
plt.legend()  # 선 설명을 보여줍니다.
save_current_figure("step4_inference_scores.png")  # 그래프 이미지를 저장합니다.
plt.show()  # 그래프를 화면에 보여줍니다.

results_df = pd.DataFrame({
    "time_sec": times,  # 각 구간의 시작 시간입니다.
    "anomaly_score": scores,  # 각 구간의 복원 오차입니다.
    "threshold": [THRESHOLD] * len(times),  # 사용한 임계값입니다.
    "predicted_status": ["ABNORMAL" if score > THRESHOLD else "NORMAL" for score in scores],  # 최종 상태 판단입니다.
})  # 시간별 결과를 표로 만듭니다.
results_csv_path = os.path.join(RESULTS_DIR, "step4_inference_scores.csv")  # 저장할 CSV 경로입니다.
results_df.to_csv(results_csv_path, index=False)  # 추론 결과를 CSV로 저장합니다.
print(f"추론 결과 CSV 저장 완료: {results_csv_path}")  # 저장 위치를 출력합니다.
