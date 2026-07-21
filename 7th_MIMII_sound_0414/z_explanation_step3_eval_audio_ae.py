"""
프로그램 전체 흐름 설명:
1. Step 2에서 학습한 오토인코더 모델을 불러옵니다.
2. 정상 소리와 비정상 소리를 모델에 넣어 복원 오차를 계산합니다.
3. 정상은 오차가 작고, 비정상은 오차가 클 것이라는 생각으로 분포를 비교합니다.
4. ROC-AUC와 F1 점수를 이용해 좋은 임계값을 찾습니다.
5. 원본 Mel-Spectrogram과 모델이 복원한 그림을 비교해 모델이 어디서 어려워하는지 봅니다.
"""

import os  # 파일과 폴더 경로를 만들 때 사용합니다.
import glob  # wav 파일 목록을 찾을 때 사용합니다.
import json  # 평가 결과를 json으로 저장할 때 사용합니다.
import numpy as np  # 배열 계산과 오차 목록 처리에 사용합니다.
import pandas as pd  # 결과를 CSV 파일로 저장할 때 사용합니다.
import librosa  # 오디오를 읽고 Mel-Spectrogram으로 바꿀 때 사용합니다.
import librosa.display  # Mel-Spectrogram을 그림으로 보여줄 때 사용합니다.
import matplotlib.pyplot as plt  # 그래프를 그릴 때 사용합니다.
import seaborn as sns  # 오차 분포 히스토그램을 보기 좋게 그릴 때 사용합니다.
import torch  # PyTorch 모델을 실행할 때 사용합니다.
import torch.nn as nn  # 신경망 층과 손실 함수를 사용할 때 필요합니다.
from sklearn.metrics import roc_curve, auc  # ROC 곡선과 AUC 점수를 계산합니다.
from sklearn.metrics import precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay  # 임계값과 오차 행렬 도구입니다.

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 현재 코드가 있는 폴더입니다.
VIS_DIR = os.path.join(BASE_DIR, "0_result")  # 그림 저장 폴더입니다.
RESULTS_DIR = os.path.join(BASE_DIR, "0_result")  # 결과 저장 폴더입니다.
os.makedirs(VIS_DIR, exist_ok=True)  # 그림 폴더가 없으면 만듭니다.
os.makedirs(RESULTS_DIR, exist_ok=True)  # 결과 폴더가 없으면 만듭니다.


def save_current_figure(filename):
    """
    현재 그려진 그래프를 이미지 파일로 저장합니다.

    Args:
        filename: 저장할 이미지 파일 이름입니다.

    Returns:
        None: 이미지를 저장하고 저장 경로를 출력합니다.
    """

    save_path = os.path.join(VIS_DIR, filename)  # 저장할 전체 경로입니다.
    plt.savefig(save_path, dpi=200, bbox_inches='tight')  # 그래프를 선명하게 저장합니다.
    print(f"이미지 저장 완료: {save_path}")  # 저장 위치를 알려줍니다.


class AudioAutoencoder(nn.Module):
    """
    학습 때와 똑같은 CNN 오토인코더 구조입니다.

    Args:
        None: 모델 구조는 고정되어 있습니다.

    Returns:
        nn.Module: 입력 Mel-Spectrogram을 복원하는 모델입니다.
    """

    def __init__(self):
        """
        인코더와 디코더 층을 정의합니다.

        Args:
            None

        Returns:
            None: 모델 층을 객체에 저장합니다.
        """

        super(AudioAutoencoder, self).__init__()  # PyTorch 모델 초기화를 실행합니다.
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # 입력 이미지를 작게 줄이며 특징을 찾습니다.
            nn.ReLU(),  # 중요한 양수 신호를 남깁니다.
            nn.BatchNorm2d(16),  # 학습된 값의 분포를 안정적으로 맞춥니다.
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 더 많은 특징을 찾습니다.
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 이미지를 더 압축합니다.
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 고수준 특징을 만듭니다.
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 아주 작은 특징 지도로 만듭니다.
            nn.ReLU(),
            nn.Flatten(),  # 2D 정보를 1줄로 펼칩니다.
            nn.Linear(256 * 4 * 4, 64)  # 64개 숫자로 압축합니다.
        )
        self.decoder_fc = nn.Sequential(
            nn.Linear(64, 256 * 4 * 4),  # 압축 숫자를 다시 작은 특징 지도로 키웁니다.
            nn.ReLU()
        )
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # 이미지를 키웁니다.
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # 더 키웁니다.
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 원본 크기에 가까워집니다.
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # 세부 정보를 복원합니다.
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # 1채널 출력 이미지로 만듭니다.
            nn.Sigmoid()  # 출력값을 0~1 사이로 제한합니다.
        )

    def forward(self, x):
        """
        입력을 압축한 뒤 다시 복원합니다.

        Args:
            x: (배치, 1, 128, 128) 모양의 Mel-Spectrogram 텐서입니다.

        Returns:
            torch.Tensor: 복원된 Mel-Spectrogram 텐서입니다.
        """

        encoded = self.encoder(x)  # 입력을 작은 코드로 압축합니다.
        decoded = self.decoder_fc(encoded)  # 작은 코드를 2D 특징으로 키웁니다.
        decoded = decoded.view(-1, 256, 4, 4)  # 합성곱 디코더가 볼 수 있는 모양으로 바꿉니다.
        reconstructed = self.decoder_conv(decoded)  # 128x128 이미지로 복원합니다.
        return reconstructed  # 복원 결과를 반환합니다.


device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')  # 사용할 계산 장치입니다.
model = AudioAutoencoder().to(device)  # 모델을 만들고 장치로 보냅니다.
model.load_state_dict(torch.load(os.path.join(BASE_DIR, 'audio_models', 'audio_autoencoder_real.pth'), map_location=device))  # 저장된 가중치를 불러옵니다.
model.eval()  # 평가 모드로 바꿔 학습 때만 쓰는 동작을 끕니다.
print("실제 밸브 학습 모델 로드 완료")  # 로드 완료를 알립니다.

normal_test_dir = os.path.join(BASE_DIR, '0_dB_valve', 'valve', 'id_02', 'normal')  # 정상 평가 폴더입니다.
abnormal_test_dir = os.path.join(BASE_DIR, '0_dB_valve', 'valve', 'id_02', 'abnormal')  # 비정상 평가 폴더입니다.

normal_test_files = glob.glob(os.path.join(normal_test_dir, '*.wav'))[:50]  # 정상 파일 최대 50개를 사용합니다.
abnormal_test_files = glob.glob(os.path.join(abnormal_test_dir, '*.wav'))[:50]  # 비정상 파일 최대 50개를 사용합니다.


def compute_errors(file_list, sr=16000, n_mels=128, max_frames=128):
    """
    파일 목록의 각 오디오에 대해 오토인코더 복원 오차를 계산합니다.

    Args:
        file_list: 평가할 wav 파일 경로 목록입니다.
        sr: 오디오를 읽을 샘플링 레이트입니다.
        n_mels: Mel-Spectrogram 세로 칸 수입니다.
        max_frames: 시간 방향을 맞출 가로 칸 수입니다.

    Returns:
        np.ndarray: 파일별 MSE 복원 오차 배열입니다.
    """

    errors = []  # 파일별 오차를 담을 리스트입니다.
    criterion = nn.MSELoss()  # 원본과 복원본의 차이를 재는 도구입니다.

    with torch.no_grad():  # 평가할 때는 기울기 계산이 필요 없어서 꺼둡니다.
        for path in file_list:  # 파일을 하나씩 처리합니다.
            y, _ = librosa.load(path, sr=sr)  # 오디오를 읽습니다.
            mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)  # Mel-Spectrogram으로 바꿉니다.
            mel_db = librosa.power_to_db(mel, ref=np.max)  # dB 단위로 바꿉니다.

            mel_db = (mel_db + 80.0) / 80.0  # 모델 입력에 맞게 0~1 범위로 정규화합니다.

            # 모델은 128x128만 받으므로 길면 자르고 짧으면 0으로 채웁니다.
            mel_input = (
                mel_db[:, :max_frames]
                if mel_db.shape[1] >= max_frames
                else np.pad(mel_db, ((0, 0), (0, max_frames - mel_db.shape[1])))
            )  # 입력 크기를 고정합니다.

            input_tensor = torch.tensor(mel_input, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)  # 모델 입력 모양으로 바꿉니다.
            reconstructed = model(input_tensor)  # 모델이 입력을 복원합니다.

            loss = criterion(reconstructed, input_tensor)  # 복원 오차를 계산합니다.
            errors.append(loss.item())  # 숫자 오차를 리스트에 저장합니다.
    return np.array(errors)  # numpy 배열로 반환합니다.


print("정상 및 비정상 데이터의 복원 오차 계산 중...")  # 계산 시작을 알립니다.
normal_errors = compute_errors(normal_test_files)  # 정상 파일들의 복원 오차입니다.
abnormal_errors = compute_errors(abnormal_test_files)  # 비정상 파일들의 복원 오차입니다.

plt.figure(figsize=(10, 6))  # 오차 분포 그림판을 만듭니다.
sns.histplot(normal_errors, kde=True, color='blue', label='Normal (Valve)')  # 정상 오차 분포를 그립니다.
sns.histplot(abnormal_errors, kde=True, color='red', label='Abnormal (Valve)')  # 비정상 오차 분포를 그립니다.
plt.title('Reconstruction Error Distribution (MIMII Valve ID02)')  # 그래프 제목입니다.
plt.xlabel('Mean Squared Error (MSE)')  # x축은 복원 오차입니다.
plt.legend()  # 색깔 설명을 보여줍니다.
save_current_figure("step3_error_distribution.png")  # 분포 그림을 저장합니다.
plt.show()  # 그래프를 보여줍니다.

y_true = np.concatenate([np.zeros(len(normal_errors)), np.ones(len(abnormal_errors))])  # 정상은 0, 비정상은 1인 정답 배열입니다.
y_scores = np.concatenate([normal_errors, abnormal_errors])  # 복원 오차를 이상 점수로 사용합니다.

fpr, tpr, _ = roc_curve(y_true, y_scores)  # 임계값을 바꿨을 때의 성능 곡선을 계산합니다.
roc_auc = auc(fpr, tpr)  # ROC 곡선 아래 면적인 AUC를 계산합니다.

plt.figure(figsize=(8, 6))  # ROC 그림판을 만듭니다.
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')  # ROC 곡선을 그립니다.
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')  # 무작위 예측 기준선을 그립니다.
plt.title('Receiver Operating Characteristic (ROC)')  # 그래프 제목입니다.
plt.xlabel('False Positive Rate')  # x축은 정상인데 이상이라고 한 비율입니다.
plt.ylabel('True Positive Rate')  # y축은 이상을 이상이라고 잘 찾은 비율입니다.
plt.legend()  # 설명 상자를 보여줍니다.
save_current_figure("step3_roc_curve.png")  # ROC 그림을 저장합니다.
plt.show()  # 그래프를 보여줍니다.

precision, recall, thresholds = precision_recall_curve(y_true, y_scores)  # 임계값별 정밀도와 재현율을 계산합니다.
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)  # 0으로 나누지 않도록 작은 값을 더해 F1을 계산합니다.
optimal_idx = np.argmax(f1_scores)  # F1이 가장 큰 위치를 찾습니다.
optimal_threshold = thresholds[optimal_idx]  # 가장 좋은 임계값입니다.
optimal_f1 = f1_scores[optimal_idx]  # 그때의 F1 점수입니다.

print(f"\n최적의 이상치 탐지 임계값(Threshold): {optimal_threshold:.4f}")  # 최적 임계값을 출력합니다.
print(f"해당 임계값에서의 평가지표 - 최고 F1-Score: {optimal_f1:.4f}")  # 최고 F1을 출력합니다.
pd.DataFrame(
    {
        "threshold": thresholds,  # 후보 임계값들입니다.
        "precision": precision[:-1],  # 각 임계값의 정밀도입니다.
        "recall": recall[:-1],  # 각 임계값의 재현율입니다.
        "f1_score": f1_scores[:-1],  # 각 임계값의 F1 점수입니다.
    }
).to_csv(os.path.join(RESULTS_DIR, "step3_threshold_search.csv"), index=False)  # 임계값 탐색 결과를 저장합니다.

y_pred = (y_scores >= optimal_threshold).astype(int)  # 오차가 임계값 이상이면 비정상으로 예측합니다.

cm = confusion_matrix(y_true, y_pred)  # 실제와 예측을 비교한 오차 행렬입니다.
plt.figure(figsize=(6, 5))  # 오차 행렬 그림판을 만듭니다.
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Abnormal'])  # 오차 행렬 표시 도구입니다.
disp.plot(cmap='Blues', values_format='d', ax=plt.gca())  # 오차 행렬을 그림으로 그립니다.
plt.title(f'Confusion Matrix (Optimal Threshold: {optimal_threshold:.4f})')  # 제목에 임계값을 표시합니다.
save_current_figure("step3_confusion_matrix.png")  # 오차 행렬 이미지를 저장합니다.
plt.show()  # 그래프를 보여줍니다.
with open(os.path.join(RESULTS_DIR, "step3_eval_metrics.json"), "w", encoding="utf-8") as f:  # 평가 요약 json을 엽니다.
    json.dump(
        {
            "roc_auc": float(roc_auc),  # ROC-AUC 점수입니다.
            "optimal_threshold": float(optimal_threshold),  # 최적 임계값입니다.
            "optimal_f1": float(optimal_f1),  # 최적 F1 점수입니다.
            "confusion_matrix": cm.tolist(),  # 오차 행렬입니다.
        },
        f,
        ensure_ascii=False,
        indent=2,
    )  # 평가 요약을 저장합니다.
pd.DataFrame(
    {
        "y_true": y_true.tolist(),  # 실제 정답입니다.
        "y_pred": y_pred.tolist(),  # 예측 정답입니다.
        "y_score": y_scores.tolist(),  # 복원 오차 점수입니다.
    }
).to_csv(os.path.join(RESULTS_DIR, "step3_predictions.csv"), index=False)  # 파일별 예측 결과를 저장합니다.


def visualize_reconstruction(file_path, title):
    """
    오디오 하나의 원본 Mel-Spectrogram과 복원 Mel-Spectrogram을 나란히 보여줍니다.

    Args:
        file_path: 시각화할 wav 파일 경로입니다.
        title: 그래프 제목에 넣을 이름입니다.

    Returns:
        None: 그림을 저장하고 화면에 보여줍니다.
    """

    y, sr = librosa.load(file_path, sr=16000)  # 오디오 파일을 읽습니다.
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)  # Mel-Spectrogram으로 바꿉니다.
    mel_db = librosa.power_to_db(mel, ref=np.max)  # dB 단위로 변환합니다.

    mel_db_norm = (mel_db + 80.0) / 80.0  # 모델 입력 범위인 0~1로 바꿉니다.
    max_frames = 128  # 모델 입력 시간 길이입니다.

    if mel_db_norm.shape[1] >= max_frames:  # 길이가 충분하면
        mel_input = mel_db_norm[:, :max_frames]  # 앞 128프레임만 사용합니다.
    else:  # 길이가 짧으면
        mel_input = np.pad(mel_db_norm, ((0, 0), (0, max_frames - mel_db_norm.shape[1])))  # 부족한 부분을 0으로 채웁니다.

    input_tensor = torch.tensor(mel_input, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)  # 모델 입력 텐서로 바꿉니다.

    with torch.no_grad():  # 평가이므로 기울기 계산을 하지 않습니다.
        reconstructed_tensor = model(input_tensor)  # 모델이 Mel-Spectrogram을 복원합니다.

    original_img = input_tensor.cpu().squeeze().numpy()  # 원본 입력을 numpy 배열로 바꿉니다.
    reconstructed_img = reconstructed_tensor.cpu().squeeze().numpy()  # 복원 결과를 numpy 배열로 바꿉니다.

    original_db = original_img * 80.0 - 80.0  # 시각화를 위해 dB 값으로 되돌립니다.
    reconstructed_db = reconstructed_img * 80.0 - 80.0  # 복원본도 dB 값으로 되돌립니다.

    plt.figure(figsize=(12, 4))  # 좌우 비교 그림판을 만듭니다.

    plt.subplot(1, 2, 1)  # 왼쪽 칸을 선택합니다.
    librosa.display.specshow(original_db, sr=16000, x_axis='time', y_axis='mel')  # 원본 Mel 그림입니다.
    plt.title(f'{title} - Original (Input)')  # 원본 제목입니다.
    plt.colorbar(format='%+2.0f dB')  # 색상 값 설명입니다.

    plt.subplot(1, 2, 2)  # 오른쪽 칸을 선택합니다.
    librosa.display.specshow(reconstructed_db, sr=16000, x_axis='time', y_axis='mel')  # 복원 Mel 그림입니다.
    plt.title(f'{title} - Reconstructed (Output)')  # 복원 제목입니다.
    plt.colorbar(format='%+2.0f dB')  # 색상 값 설명입니다.

    plt.tight_layout()  # 그래프 간격을 정리합니다.
    safe_title = title.lower().replace(" ", "_").replace("(", "").replace(")", "")  # 파일 이름에 안전한 제목으로 바꿉니다.
    save_current_figure(f"step3_reconstruction_{safe_title}.png")  # 비교 그림을 저장합니다.
    plt.show()  # 그래프를 보여줍니다.


print("\n정상 및 비정상 오디오 스펙트로그램 복원 결과 시각화 중...")  # 시각화 시작을 알립니다.
visualize_reconstruction(normal_test_files[0], "Normal Valve Sound")  # 정상 파일의 복원 결과를 봅니다.
visualize_reconstruction(abnormal_test_files[0], "Abnormal Valve Sound")  # 비정상 파일의 복원 결과를 봅니다.
