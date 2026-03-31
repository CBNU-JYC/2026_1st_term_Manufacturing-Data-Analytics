import os  # 실행 환경 변수 설정을 위해 사용합니다.
from pathlib import Path  # 현재 스크립트 위치를 기준으로 경로를 다루기 위해 사용합니다.

BASE_DIR = Path(__file__).resolve().parent  # 이 평가 스크립트가 들어 있는 폴더 경로를 구합니다.
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".matplotlib"))  # matplotlib 설정/캐시 폴더를 현재 프로젝트 안쪽으로 지정합니다.
os.environ.setdefault("XDG_CACHE_HOME", str(BASE_DIR / ".cache"))  # 글꼴 등 캐시 폴더도 현재 프로젝트 안쪽으로 지정합니다.
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)  # matplotlib 캐시 폴더가 없으면 자동으로 만듭니다.
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)  # 일반 캐시 폴더도 없으면 자동으로 만듭니다.

import torch  # 모델과 텐서 연산을 위해 사용합니다.
import numpy as np  # 수치 계산과 배열 처리를 위해 사용합니다.
import matplotlib  # 그래프 출력 방식을 설정하기 위해 먼저 불러옵니다.
matplotlib.use("Agg")  # 화면에 띄우지 않고 이미지 파일로 저장하는 모드로 설정합니다.
import matplotlib.pyplot as plt  # 그래프를 그리기 위한 기본 라이브러리입니다.
import seaborn as sns  # 히스토그램과 히트맵을 더 보기 좋게 그리기 위해 사용합니다.
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score  # 평가 지표 계산 함수들을 불러옵니다.
import torch.nn as nn  # LSTM 모델 구조를 다시 정의하기 위해 사용합니다.


# 1. 모델 클래스 재정의
class LSTM_AE(nn.Module):  # 학습 때 사용한 것과 같은 모델 구조를 다시 정의합니다.
    def __init__(self, n_features, seq_len):
        super(LSTM_AE, self).__init__()  # 부모 클래스 초기화를 수행합니다.
        self.seq_len = seq_len  # 디코더에서 사용할 시퀀스 길이를 저장합니다.
        self.enc1 = nn.LSTM(input_size=n_features, hidden_size=64, batch_first=True)  # 첫 번째 인코더 LSTM입니다.
        self.enc2 = nn.LSTM(input_size=64, hidden_size=32, batch_first=True)  # 두 번째 인코더 LSTM입니다.
        self.dec1 = nn.LSTM(input_size=32, hidden_size=32, batch_first=True)  # 첫 번째 디코더 LSTM입니다.
        self.dec2 = nn.LSTM(input_size=32, hidden_size=64, batch_first=True)  # 두 번째 디코더 LSTM입니다.
        self.out = nn.Linear(64, n_features)  # 마지막 출력층은 64차원을 원래 특징 수로 바꿉니다.

    def forward(self, x):
        x, _ = self.enc1(x)  # 입력을 첫 번째 인코더에 통과시킵니다.
        x, (h_n, _) = self.enc2(x)  # 두 번째 인코더에 통과시키고 최종 은닉상태를 얻습니다.
        x = h_n.transpose(0, 1).repeat(1, self.seq_len, 1)  # 마지막 은닉상태를 시퀀스 길이만큼 반복해 디코더 입력으로 만듭니다.
        x, _ = self.dec1(x)  # 첫 번째 디코더를 통과합니다.
        x, _ = self.dec2(x)  # 두 번째 디코더를 통과합니다.
        x = self.out(x)  # 최종 출력층으로 각 시점의 특징 값을 만듭니다.
        return x  # 복원된 시퀀스를 반환합니다.


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # GPU 가능 여부에 따라 계산 장치를 선택합니다.

print("1. 데이터 및 모델 로딩 중...", flush=True)  # 평가 준비 시작 메시지를 출력합니다.
dataset = torch.load(BASE_DIR / 'processed_dataset.pt', weights_only=False)  # 전처리된 데이터 파일을 불러옵니다.
X_valid, Y_valid, Y_val_index = dataset['X_valid'], dataset['Y_valid'], dataset['Y_val_index']  # 전체 검증 데이터와 라벨을 꺼냅니다.
X_test, Y_test, Y_te_index = dataset['X_test'], dataset['Y_test'], dataset['Y_te_index']  # 전체 테스트 데이터와 라벨을 꺼냅니다.

model = LSTM_AE(n_features=5, seq_len=20).to(device)  # 학습 때와 같은 구조의 모델을 생성해 장치로 보냅니다.
model.load_state_dict(torch.load(BASE_DIR / 'best_lstm_ae.pth', weights_only=True))  # 가장 성능이 좋았던 저장 가중치를 불러옵니다.
model.eval()  # 추론 전용 평가 모드로 전환합니다.


def flatten_last_step(X):
    return X[:, -1, :].numpy()  # 각 시퀀스의 마지막 시점 값만 뽑아 넘파이 배열로 바꿉니다.


# ---------------------------------------------------------
# 2. 검증셋을 통한 최적 임곗값(Threshold) 탐색 및 시각화
# ---------------------------------------------------------
print("2. 최적 임곗값 탐색 중...", flush=True)  # 임곗값 탐색 단계 시작을 출력합니다.
with torch.no_grad():  # 평가 중에는 기울기 계산이 필요 없으므로 비활성화합니다.
    valid_predictions = model(X_valid.to(device)).cpu()  # 검증 입력을 모델에 넣어 예측 결과를 얻고 다시 CPU로 가져옵니다.

mse_valid = np.mean(np.power(flatten_last_step(Y_valid) - flatten_last_step(valid_predictions), 2), axis=1)  # 마지막 시점 기준 재구성 오차(MSE)를 샘플별로 계산합니다.

thresholds = np.linspace(0, 20, 100)  # 0부터 20까지 100개의 후보 임곗값을 만듭니다.
precisions, recalls = [], []  # 각 임곗값에서의 정밀도와 재현율을 기록할 리스트입니다.

for thr in thresholds:  # 임곗값 후보를 하나씩 검사합니다.
    pred_val = [1 if e > thr else 0 for e in mse_valid]  # 오차가 임곗값보다 크면 이상치 1, 아니면 정상 0으로 예측합니다.
    precisions.append(precision_score(Y_val_index, pred_val, zero_division=0))  # 해당 임곗값의 정밀도를 기록합니다.
    recalls.append(recall_score(Y_val_index, pred_val, zero_division=0))  # 해당 임곗값의 재현율을 기록합니다.

# 정밀도와 재현율 차이가 가장 작은 지점을 임곗값으로 사용
index_cnt = int(np.argmin(np.abs(np.array(precisions) - np.array(recalls))))  # 정밀도와 재현율 차이가 가장 작은 위치를 찾습니다.
optimal_threshold = thresholds[index_cnt]  # 그 위치의 임곗값을 최적 임곗값으로 사용합니다.
print(f"-> 설정된 최적 임곗값: {optimal_threshold:.4f}", flush=True)  # 선택된 임곗값을 출력합니다.

# [시각화 1] 정밀도-재현율 곡선
plt.figure(figsize=(8, 5))  # 첫 번째 그림의 크기를 설정합니다.
plt.title('Precision/Recall Curve for Threshold', fontsize=14)  # 그래프 제목을 지정합니다.
plt.plot(thresholds, precisions, label='Precision', color='navy', linestyle='--')  # 임곗값에 따른 정밀도 곡선을 그립니다.
plt.plot(thresholds, recalls, label='Recall', color='seagreen')  # 임곗값에 따른 재현율 곡선을 그립니다.
plt.plot(optimal_threshold, precisions[index_cnt], 'ro', label=f'Optimal Threshold ({optimal_threshold:.2f})')  # 선택된 최적 지점을 빨간 점으로 표시합니다.
plt.xlabel('Threshold')  # x축 이름을 지정합니다.
plt.ylabel('Score')  # y축 이름을 지정합니다.
plt.legend()  # 범례를 표시합니다.
plt.grid(True, alpha=0.3)  # 보기 편하도록 옅은 격자선을 추가합니다.
plt.tight_layout()  # 그래프 요소가 잘리지 않도록 여백을 자동 조정합니다.
plt.savefig(BASE_DIR / 'precision_recall_threshold.png', dpi=150)  # 첫 번째 그래프를 PNG 파일로 저장합니다.
plt.close()  # 현재 그래프 객체를 닫아 메모리를 정리합니다.

# ---------------------------------------------------------
# 3. 테스트셋 최종 평가 및 결과 시각화
# ---------------------------------------------------------
print("3. 테스트셋 모델 평가 중...", flush=True)  # 테스트 평가 단계 시작을 출력합니다.

np.random.seed(42)  # 셔플 결과를 항상 동일하게 재현할 수 있도록 난수 시드를 고정합니다.
shuffle_idx = np.random.permutation(len(Y_te_index))  # 테스트셋 전체 인덱스를 무작위 순서로 섞습니다.

X_test_shuffled = X_test[shuffle_idx]  # 섞인 순서대로 테스트 입력을 재정렬합니다.
Y_test_shuffled = Y_test[shuffle_idx]  # 섞인 순서대로 테스트 정답 시퀀스도 재정렬합니다.
Y_te_index_shuffled = Y_te_index[shuffle_idx]  # 섞인 순서대로 테스트 라벨도 재정렬합니다.

with torch.no_grad():  # 테스트 추론 중에도 기울기 계산은 필요 없습니다.
    test_predictions = model(X_test_shuffled.to(device)).cpu()  # 섞인 테스트 입력에 대해 모델 예측을 수행합니다.

mse_test = np.mean(np.power(flatten_last_step(Y_test_shuffled) - flatten_last_step(test_predictions), 2), axis=1)  # 테스트 샘플별 마지막 시점 MSE를 계산합니다.
pred_y = [1 if e > optimal_threshold else 0 for e in mse_test]  # 최적 임곗값을 기준으로 정상/이상 여부를 최종 판정합니다.

print(f"\n[최종 평가 결과]", flush=True)  # 최종 성능 요약 제목을 출력합니다.
print(f"정확도 (Accuracy): {accuracy_score(Y_te_index_shuffled, pred_y)*100:.2f}%", flush=True)  # 전체 정확도를 백분율로 출력합니다.
print(f"F1-Score: {f1_score(Y_te_index_shuffled, pred_y):.4f}", flush=True)  # 정밀도와 재현율 균형을 보는 F1 점수를 출력합니다.

# 정상 데이터와 이상 데이터의 오차 분리
mse_normal = mse_test[Y_te_index_shuffled == 0]  # 실제 정상 샘플들의 오차만 따로 추출합니다.
mse_anomaly = mse_test[Y_te_index_shuffled == 1]  # 실제 이상 샘플들의 오차만 따로 추출합니다.
idx_normal = np.where(Y_te_index_shuffled == 0)[0]  # 정상 샘플이 위치한 인덱스를 구합니다.
idx_anomaly = np.where(Y_te_index_shuffled == 1)[0]  # 이상 샘플이 위치한 인덱스를 구합니다.

# [시각화 2] 재구성 오차 산점도
plt.figure(figsize=(10, 5))  # 두 번째 그림의 크기를 지정합니다.
plt.title('Reconstruction Error by Class', fontsize=14)  # 그래프 제목을 지정합니다.
plt.scatter(idx_normal, mse_normal, label='Normal', color='coral', alpha=0.6, s=15)  # 정상 데이터의 오차를 산점도로 표시합니다.
plt.scatter(idx_anomaly, mse_anomaly, label='Anomaly', color='steelblue', alpha=0.6, s=15)  # 이상 데이터의 오차를 산점도로 표시합니다.
plt.axhline(optimal_threshold, color='red', linestyle='--', linewidth=2, label='Threshold')  # 기준 임곗값을 빨간 수평선으로 표시합니다.
plt.xlabel('Data Point Index')  # x축 이름을 지정합니다.
plt.ylabel('Reconstruction Error (MSE)')  # y축 이름을 지정합니다.
plt.legend()  # 범례를 표시합니다.
plt.grid(True, alpha=0.3)  # 옅은 격자선을 추가합니다.
plt.tight_layout()  # 레이아웃을 정리합니다.
plt.savefig(BASE_DIR / 'reconstruction_error_scatter.png', dpi=150)  # 산점도 그래프를 파일로 저장합니다.
plt.close()  # 그래프 객체를 닫습니다.

# [시각화 3] 재구성 오차 분포
plt.figure(figsize=(8, 5))  # 세 번째 그림의 크기를 지정합니다.
plt.title('Distribution of Reconstruction Errors', fontsize=14)  # 그래프 제목을 지정합니다.
sns.histplot(mse_normal, bins=50, color='coral', label='Normal', kde=True, stat="density")  # 정상 데이터 오차 분포를 히스토그램과 밀도곡선으로 표시합니다.
sns.histplot(mse_anomaly, bins=50, color='steelblue', label='Anomaly', kde=True, stat="density")  # 이상 데이터 오차 분포도 같은 방식으로 표시합니다.
plt.axvline(optimal_threshold, color='red', linestyle='--', linewidth=2, label='Threshold')  # 기준 임곗값을 세로선으로 표시합니다.
plt.xlabel('Reconstruction Error (MSE)')  # x축 이름을 지정합니다.
plt.ylabel('Density')  # y축 이름을 지정합니다.
plt.legend()  # 범례를 표시합니다.
plt.tight_layout()  # 레이아웃을 정리합니다.
plt.savefig(BASE_DIR / 'reconstruction_error_distribution.png', dpi=150)  # 분포 그래프를 파일로 저장합니다.
plt.close()  # 그래프 객체를 닫습니다.

# [시각화 4] 오차 행렬
cm = confusion_matrix(Y_te_index_shuffled, pred_y)  # 실제값과 예측값을 기반으로 혼동행렬을 계산합니다.
plt.figure(figsize=(6, 5))  # 네 번째 그림의 크기를 지정합니다.
sns.heatmap(cm, annot=True, cmap="Blues", fmt="g", cbar=False, annot_kws={"size": 14})  # 혼동행렬을 히트맵 형태로 보기 좋게 그립니다.
plt.xlabel('Predicted Labels', fontsize=12)  # x축 이름을 지정합니다.
plt.ylabel('Observed Labels', fontsize=12)  # y축 이름을 지정합니다.
plt.title('Confusion Matrix of LSTM-AE', fontsize=14)  # 그래프 제목을 지정합니다.
plt.xticks([0.5, 1.5], ['Normal (0)', 'Anomaly (1)'])  # x축 눈금 이름을 정상/이상으로 표시합니다.
plt.yticks([0.5, 1.5], ['Normal (0)', 'Anomaly (1)'])  # y축 눈금 이름도 정상/이상으로 표시합니다.
plt.tight_layout()  # 레이아웃을 정리합니다.
plt.savefig(BASE_DIR / 'confusion_matrix.png', dpi=150)  # 혼동행렬 이미지를 파일로 저장합니다.
plt.close()  # 그래프 객체를 닫습니다.

print("평가 완료! 결과 이미지 4개가 저장되었습니다.", flush=True)  # 평가 종료와 결과 저장 완료를 출력합니다.
