# 라이브러리 임포트 및 전처리 데이터 불러오기
import os  # 폴더 생성과 환경 변수 설정에 사용합니다.
from pathlib import Path  # 현재 스크립트 기준 경로를 만들기 위해 사용합니다.

BASE_DIR = Path(__file__).resolve().parent  # 이 파이썬 파일이 들어 있는 폴더 경로를 구합니다.
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".matplotlib"))  # matplotlib 캐시 폴더를 프로젝트 안으로 지정합니다.
os.environ.setdefault("XDG_CACHE_HOME", str(BASE_DIR / ".cache"))  # 글꼴 캐시 폴더를 프로젝트 안으로 지정합니다.
(BASE_DIR / ".matplotlib").mkdir(parents=True, exist_ok=True)  # matplotlib 캐시 폴더가 없으면 자동으로 만듭니다.
(BASE_DIR / ".cache").mkdir(parents=True, exist_ok=True)  # 일반 캐시 폴더가 없으면 자동으로 만듭니다.

import torch  # 딥러닝 모델과 텐서 계산을 위한 PyTorch입니다.
import torch.nn as nn  # 신경망 레이어와 손실 함수를 사용하기 위해 불러옵니다.
import torch.optim as optim  # Adam 같은 옵티마이저를 사용하기 위해 불러옵니다.
from torch.utils.tensorboard import SummaryWriter  # TensorBoard 로그를 남기기 위한 도구입니다.
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve  # 분류 성능 평가 지표 함수들입니다.
import matplotlib  # 그래프 저장 모드를 설정하기 위해 먼저 불러옵니다.
matplotlib.use("Agg")  # 화면에 띄우지 않고 이미지 파일로 저장하는 모드입니다.
import matplotlib.pyplot as plt  # 그래프를 그리기 위한 라이브러리입니다.
import seaborn as sns  # heatmap, barplot을 더 보기 좋게 그리기 위한 라이브러리입니다.

import joblib  # 스케일러 같은 파이썬 객체를 파일로 저장하는 데 사용합니다.


# 전처리 모듈 임포트
try:
    from step2_data_prep import train_loader, test_loader, X_train, scaler  # step2에서 만든 학습용 로더, 테스트용 로더, 입력 차원 정보, 스케일러를 그대로 가져옵니다.
    print("데이터 전처리 모듈(step2_data_prep) 로드 성공")  # step2 모듈 로드 성공 메시지입니다.
except ModuleNotFoundError:
    print("에러: 'step2_data_prep.py' 파일을 찾을 수 없습니다. 같은 폴더에 있는지 확인해주세요.")  # step2 파일을 못 찾으면 사용자에게 안내합니다.
    exit()  # 필요한 전처리 모듈이 없으므로 프로그램을 종료합니다.

# 학습 장치(Device) 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')  # CUDA GPU가 있으면 cuda, 애플 GPU가 있으면 mps, 없으면 cpu를 사용합니다.
print(f"학습 장치(Device) 설정 완료: {device}")  # 실제 사용할 장치를 출력합니다.

# 모델 아키텍처 정의 및 초기화
class FaultDiagnosisMLP(nn.Module):  # MLP 기반 고장 진단 모델을 정의합니다.
    def __init__(self, input_dim):
        super(FaultDiagnosisMLP, self).__init__()  # 부모 클래스 초기화를 수행합니다.
        self.network = nn.Sequential(  # 여러 레이어를 순서대로 쌓은 순차형 네트워크입니다.
            nn.Linear(input_dim, 32),  # 입력 특징 수에서 32차원 은닉층으로 변환합니다.
            nn.ReLU(),  # 비선형성을 위한 활성화 함수입니다.
            nn.BatchNorm1d(32),  # 학습 안정화를 위한 배치 정규화입니다.
            nn.Dropout(0.3),  # 과적합을 줄이기 위해 일부 노드를 무작위로 끕니다.

            nn.Linear(32, 16),  # 두 번째 은닉층으로 차원을 줄입니다.
            nn.ReLU(),  # 두 번째 활성화 함수입니다.
            nn.BatchNorm1d(16),  # 두 번째 배치 정규화입니다.
            nn.Dropout(0.3),  # 두 번째 드롭아웃입니다.

            nn.Linear(16, 1)  # 최종 출력은 고장 여부 판단용 로짓 1개입니다.
        )

    def forward(self, x):
        return self.network(x)  # 입력 x를 네트워크에 통과시켜 결과를 반환합니다.


input_dim = X_train.shape[1]  # 학습 데이터의 열 개수를 입력 차원으로 사용합니다.
model = FaultDiagnosisMLP(input_dim).to(device)  # 모델을 생성하고 선택한 장치로 이동시킵니다.

print(f"\n[모델 구조 확인]\n{model}")  # 모델 구조를 화면에 출력합니다.

# 손실 함수, 최적화 알고리즘 및 TensorBoard 설정
criterion = nn.BCEWithLogitsLoss()  # 이진 분류용 손실 함수입니다. 시그모이드가 내부에 포함된 형태입니다.
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam 옵티마이저를 사용해 가중치를 업데이트합니다.

# TensorBoard 기록을 위한 SummaryWriter 초기화 (저장 경로 지정)
log_dir = BASE_DIR / "runs" / "fault_diagnosis_experiment"  # TensorBoard 로그가 저장될 폴더 경로입니다.
writer = SummaryWriter(log_dir)  # 해당 경로에 로그를 기록하는 객체를 만듭니다.
print(f"TensorBoard 로그 디렉토리 설정: {log_dir}")  # TensorBoard 저장 경로를 출력합니다.

# 모델 그래프(구조)를 TensorBoard에 기록
# (더미 데이터를 하나 통과시켜서 그래프를 그립니다)
dummy_input = torch.randn(1, input_dim).to(device)  # 입력 차원과 같은 모양의 가짜 데이터 1개를 만듭니다.
writer.add_graph(model, dummy_input)  # 모델 구조를 TensorBoard에서 볼 수 있도록 기록합니다.

# 모델 학습 및 검증 루프 (Training & Validation Loop)
epochs = 30  # 전체 학습 반복 횟수를 30으로 설정합니다.
print("\n [모델 학습 시작]")  # 학습 시작 메시지를 출력합니다.

for epoch in range(epochs):  # 에폭 수만큼 학습과 검증을 반복합니다.
    # --- 1. Training Phase ---
    model.train()  # 학습 모드로 전환합니다.
    train_loss = 0.0  # 현재 에폭의 누적 학습 손실을 0으로 초기화합니다.

    for batch_X, batch_y in train_loader:  # 학습용 DataLoader에서 배치를 하나씩 꺼냅니다.
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)  # 입력과 라벨을 계산 장치로 보냅니다.

        optimizer.zero_grad()  # 이전 배치의 기울기를 초기화합니다.
        outputs = model(batch_X)  # 현재 배치에 대한 모델 출력을 계산합니다.
        loss = criterion(outputs, batch_y)  # 예측과 정답 사이의 손실을 계산합니다.
        loss.backward()  # 역전파로 기울기를 계산합니다.
        optimizer.step()  # 계산된 기울기를 이용해 가중치를 업데이트합니다.

        train_loss += loss.item()  # 현재 배치 손실을 누적합니다.

    avg_train_loss = train_loss / len(train_loader)  # 에폭 전체 평균 학습 손실을 구합니다.

    # --- 2. Validation(Test) Phase ---
    model.eval()  # 평가 모드로 전환합니다.
    val_loss = 0.0  # 현재 에폭의 누적 검증 손실을 0으로 초기화합니다.
    all_preds, all_targets = [], []  # 검증셋 예측값과 실제값을 저장할 리스트입니다.

    with torch.no_grad():  # 검증 중에는 기울기를 계산하지 않아 속도와 메모리를 절약합니다.
        for batch_X, batch_y in test_loader:  # 테스트 로더를 검증 용도로 순회합니다.
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)  # 데이터를 계산 장치로 이동시킵니다.

            outputs = model(batch_X)  # 모델 출력을 계산합니다.
            v_loss = criterion(outputs, batch_y)  # 검증 손실을 계산합니다.
            val_loss += v_loss.item()  # 검증 손실을 누적합니다.

            probs = torch.sigmoid(outputs)  # 출력 로짓을 0~1 확률값으로 변환합니다.
            preds = (probs >= 0.5).float()  # 0.5 이상이면 고장 1, 아니면 정상 0으로 판정합니다.

            all_preds.extend(preds.cpu().numpy())  # 예측 결과를 CPU 넘파이 배열로 옮겨 저장합니다.
            all_targets.extend(batch_y.cpu().numpy())  # 실제 라벨도 저장합니다.

    avg_val_loss = val_loss / len(test_loader)  # 평균 검증 손실을 계산합니다.
    val_f1 = f1_score(all_targets, all_preds, zero_division=0)  # 검증셋 기준 F1 점수를 계산합니다.

    # --- 3. TensorBoard에 지표 기록 ---
    writer.add_scalars('Loss', {'Train': avg_train_loss, 'Validation': avg_val_loss}, epoch)  # 에폭별 학습/검증 손실을 TensorBoard에 기록합니다.
    writer.add_scalar('Metrics/Validation_F1', val_f1, epoch)  # 검증 F1 점수도 TensorBoard에 기록합니다.

    # 진행 상황 출력
    if (epoch + 1) % 5 == 0 or epoch == 0:  # 첫 에폭과 이후 5에폭마다 진행 상황을 출력합니다.
        print(f"Epoch [{epoch+1:2d}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val F1: {val_f1:.4f}")  # 현재 에폭의 손실과 F1을 출력합니다.

writer.close()  # TensorBoard 기록 객체를 닫아 저장을 마무리합니다.
print("학습 및 TensorBoard 기록 완료!")  # 학습 종료 메시지를 출력합니다.

# 최종 모델 평가 (Evaluation)
model.eval()  # 최종 평가 전에 다시 평가 모드로 전환합니다.
all_preds, all_probs, all_targets = [], [], []  # 최종 예측 라벨, 확률값, 실제 라벨을 저장할 리스트입니다.

with torch.no_grad():  # 최종 평가에서도 기울기 계산은 필요 없습니다.
    for batch_X, batch_y in test_loader:  # 테스트 데이터 전체를 순회합니다.
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)  # 데이터를 계산 장치로 이동시킵니다.

        outputs = model(batch_X)  # 모델 출력을 계산합니다.
        probs = torch.sigmoid(outputs)  # 로짓을 확률값으로 변환합니다.
        preds = (probs >= 0.5).float()  # 0.5 기준으로 예측 라벨을 만듭니다.

        all_probs.extend(probs.cpu().numpy())  # 확률값을 저장합니다.
        all_preds.extend(preds.cpu().numpy())  # 예측 라벨을 저장합니다.
        all_targets.extend(batch_y.cpu().numpy())  # 실제 라벨을 저장합니다.

# 평가지표 계산
acc = accuracy_score(all_targets, all_preds)  # 정확도를 계산합니다.
prec = precision_score(all_targets, all_preds, zero_division=0)  # 정밀도를 계산합니다.
rec = recall_score(all_targets, all_preds, zero_division=0)  # 재현율을 계산합니다.
f1 = f1_score(all_targets, all_preds, zero_division=0)  # F1 점수를 계산합니다.
auc = roc_auc_score(all_targets, all_probs)  # ROC-AUC 점수를 계산합니다.

print("\n[최종 테스트 데이터셋 평가 결과]")  # 최종 평가 결과 제목을 출력합니다.
print(f"Accuracy (정확도):  {acc:.4f}")  # 정확도를 출력합니다.
print(f"Precision (정밀도): {prec:.4f}")  # 정밀도를 출력합니다.
print(f"Recall (재현율):    {rec:.4f}")  # 재현율을 출력합니다.
print(f"F1-Score:           {f1:.4f}")  # F1 점수를 출력합니다.
print(f"ROC-AUC:            {auc:.4f}")  # ROC-AUC를 출력합니다.

# 모델 가중치 및 스케일러 저장
# 디렉토리가 없으면 생성 (선택 사항)
os.makedirs(BASE_DIR / 'models', exist_ok=True)  # 모델 저장 폴더가 없으면 자동으로 생성합니다.
model_path = BASE_DIR / 'models' / 'fault_diagnosis_mlp.pth'  # 모델 가중치를 저장할 경로입니다.
scaler_path = BASE_DIR / 'models' / 'sensor_scaler.pkl'  # 스케일러를 저장할 경로입니다.

torch.save(model.state_dict(), model_path)  # 모델 가중치만 파일로 저장합니다.
joblib.dump(scaler, scaler_path)  # 스케일러 객체를 파일로 저장합니다.

print(f"\n[저장 완료] 모델 가중치('{model_path}')와 스케일러('{scaler_path}')가 저장되었습니다.")  # 저장 완료 메시지를 출력합니다.

# 모델 평가 결과 시각화 (Confusion Matrix 및 평가지표)
print("\n평가 결과 시각화 그래프를 생성합니다...")  # 시각화 시작 메시지를 출력합니다.

# 시각화 환경 및 레이아웃 설정 (1행 3열)
plt.style.use('seaborn-v0_8-whitegrid')  # 그래프 스타일을 지정합니다.
fig, axes = plt.subplots(1, 3, figsize=(18, 5))  # 1행 3열짜리 평가 그래프 캔버스를 만듭니다.

# --- 1. Confusion Matrix (혼동 행렬) ---
cm = confusion_matrix(all_targets, all_preds)  # 실제값과 예측값으로 혼동행렬을 계산합니다.
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Normal (0)', 'Fault (1)'],
            yticklabels=['Normal (0)', 'Fault (1)'],
            cbar=False, annot_kws={"size": 14})  # 혼동행렬을 히트맵으로 그립니다.
axes[0].set_title('Confusion Matrix', fontsize=14, pad=10)  # 첫 번째 그래프 제목을 설정합니다.
axes[0].set_ylabel('Actual Status', fontsize=12)  # y축 이름을 실제 상태로 지정합니다.
axes[0].set_xlabel('Predicted Status', fontsize=12)  # x축 이름을 예측 상태로 지정합니다.

# --- 2. 5가지 평가지표 요약 바 차트 ---
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']  # 막대그래프에 표시할 지표 이름 목록입니다.
metrics_values = [acc, prec, rec, f1, auc]  # 각 지표의 실제 값 목록입니다.

sns.barplot(x=metrics_names, y=metrics_values, hue=metrics_names, yerr=None, ax=axes[1], palette='viridis', legend=False)  # 다섯 지표를 막대그래프로 표시합니다.
axes[1].set_title('Evaluation Metrics Summary', fontsize=14, pad=10)  # 두 번째 그래프 제목입니다.
axes[1].set_ylim(0, 1.1)  # y축 범위를 0~1.1로 설정해 텍스트 표시 공간을 확보합니다.

# 막대 그래프 위에 정확한 수치 텍스트 표시
for i, v in enumerate(metrics_values):  # 각 막대의 위치와 값을 하나씩 가져옵니다.
    axes[1].text(i, v + 0.02, f'{v:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=11)  # 막대 위에 정확한 수치를 표시합니다.

# --- 3. ROC Curve (수신자 조작 특성 곡선) ---
fpr, tpr, thresholds = roc_curve(all_targets, all_probs)  # ROC Curve를 그리기 위한 FPR, TPR, 임곗값 배열을 계산합니다.
axes[2].plot(fpr, tpr, color='crimson', lw=2, label=f'ROC curve (AUC = {auc:.4f})')  # ROC 곡선을 그리고 범례에 AUC도 표시합니다.
axes[2].plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--', alpha=0.7)  # 무작위 분류 기준선을 점선으로 그립니다.
axes[2].set_xlim([-0.02, 1.0])  # x축 범위를 설정합니다.
axes[2].set_ylim([0.0, 1.05])  # y축 범위를 설정합니다.
axes[2].set_xlabel('False Positive Rate (FPR)', fontsize=12)  # x축 이름을 FPR로 설정합니다.
axes[2].set_ylabel('True Positive Rate (TPR)', fontsize=12)  # y축 이름을 TPR로 설정합니다.
axes[2].set_title('Receiver Operating Characteristic (ROC)', fontsize=14, pad=10)  # ROC 그래프 제목입니다.
axes[2].legend(loc="lower right", fontsize=11)  # 범례를 오른쪽 아래에 표시합니다.

# 그래프 간격 조절 및 출력
plt.tight_layout()  # 그래프 간격을 자동 정리합니다.
plt.savefig(BASE_DIR / 'training_evaluation_summary.png', dpi=150)  # 평가 요약 그래프를 PNG 파일로 저장합니다.
plt.close()  # 그래프 객체를 닫습니다.
print("평가 시각화 저장 완료! (training_evaluation_summary.png)")  # 저장 완료 메시지를 출력합니다.
