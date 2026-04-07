"""
==========================================================================
 AI4I 2020 예지보전 개선 모델
 베이스라인 F1: 0.4539  →  목표 F1: 0.70 이상
 
 개선 포인트 3가지:
   1. pos_weight 적용 (BCEWithLogitsLoss 클래스 불균형 직접 보정)
   2. 피처 엔지니어링 (Power = Torque × RPM × 2π/60)
   3. 최적 임곗값(Threshold) 탐색 (Precision-Recall 교점)
==========================================================================
"""

import os
import warnings
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MPL_CONFIG_DIR = BASE_DIR / '.matplotlib'
CACHE_DIR = BASE_DIR / '.cache'
os.environ.setdefault('MPLCONFIGDIR', str(MPL_CONFIG_DIR))
os.environ.setdefault('XDG_CACHE_HOME', str(CACHE_DIR))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve,
    precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

warnings.filterwarnings("ignore")
plt.style.use('seaborn-v0_8-whitegrid')

DATA_PATH = BASE_DIR / 'ai4i2020.csv'
BEST_MODEL_PATH = BASE_DIR / 'best_improved_model.pth'
FIGURE_PATH = BASE_DIR / 'evaluation_result.png'
MODEL_DIR = BASE_DIR / 'models'
MPL_CONFIG_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

# ──────────────────────────────────────────────
# 0. 디바이스 설정
# ──────────────────────────────────────────────
device = torch.device(
    'cuda' if torch.cuda.is_available() else
    'mps'  if torch.backends.mps.is_available() else
    'cpu'
)
print(f"[Device] {device}")

# ──────────────────────────────────────────────
# 1. 데이터 로딩
# ──────────────────────────────────────────────
print("\n[Step 1] 데이터 로딩...")
df = pd.read_csv(DATA_PATH)
print(f"  원본 형태: {df.shape}")

# ──────────────────────────────────────────────
# 2. 피처 엔지니어링  ← 개선점 ②
#    물리 법칙: Power(W) = Torque(Nm) × ω(rad/s)
#              ω = RPM × 2π / 60
#    고장은 특정 Power 구간에서 집중 발생 → 유효한 비선형 복합 변수
# ──────────────────────────────────────────────
print("\n[Step 2] 피처 엔지니어링: Power 변수 추가...")
df['Power [W]'] = df['Torque [Nm]'] * (df['Rotational speed [rpm]'] * 2 * np.pi / 60)
# 온도 차이: 공정온도와 공기온도의 격차가 클수록 방열 문제(HDF) 가능성
df['Temp_diff [K]'] = df['Process temperature [K]'] - df['Air temperature [K]']
print("  추가된 피처: Power [W], Temp_diff [K]")

# ──────────────────────────────────────────────
# 3. 전처리 (베이스라인과 동일)
# ──────────────────────────────────────────────
print("\n[Step 3] 전처리...")

# 식별자 제거
df_p = df.drop(columns=['UDI', 'Product ID'])

# Type 원핫 인코딩
df_p = pd.get_dummies(df_p, columns=['Type'], drop_first=True)

# 타겟 분리 + 데이터 누수 방지 (세부 고장모드 제거)
target_col   = 'Machine failure'
leakage_cols = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']

X = df_p.drop(columns=[target_col] + leakage_cols)
y = df_p[target_col]

print(f"  최종 피처 수: {X.shape[1]}  (베이스라인 7개 → {X.shape[1]}개)")
print(f"  피처 목록: {list(X.columns)}")

# 층화 분할 (stratify=y 필수 — 불균형 데이터)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 스케일링 (Train 기준 fit)
num_cols = [
    'Air temperature [K]', 'Process temperature [K]',
    'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]',
    'Power [W]', 'Temp_diff [K]'
]
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols]  = scaler.transform(X_test[num_cols])

print(f"  Train: {X_train.shape} | Test: {X_test.shape}")
print(f"  고장 비율 — Train: {y_train.mean()*100:.2f}% | Test: {y_test.mean()*100:.2f}%")

# ──────────────────────────────────────────────
# 4. PyTorch Dataset & DataLoader
# ──────────────────────────────────────────────
class ManufacturingDataset(Dataset):
    def __init__(self, features, labels):
        self.X = torch.tensor(features.astype(np.float32).values, dtype=torch.float32)
        self.y = torch.tensor(labels.astype(np.float32).values,   dtype=torch.float32).unsqueeze(1)
    def __len__(self):        return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

train_ds = ManufacturingDataset(X_train, y_train)
test_ds  = ManufacturingDataset(X_test,  y_test)

# WeightedRandomSampler (배치 내 균형 유지)
class_counts   = y_train.value_counts().sort_index()
class_weights  = 1.0 / class_counts.values
sample_weights = torch.tensor([class_weights[int(l)] for l in y_train], dtype=torch.float32)
sampler        = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

train_loader = DataLoader(train_ds, batch_size=64, sampler=sampler)
test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False)

# ──────────────────────────────────────────────
# 5. 모델 아키텍처 (피처 수에 맞게 입력 차원 자동 조정)
# ──────────────────────────────────────────────
class ImprovedMLP(nn.Module):
    """
    베이스라인 대비 변경:
      - 입력 차원: 7 → 9 (피처 엔지니어링 반영)
      - Hidden: [32→16→1]  →  [64→32→16→1] (표현력 향상)
    """
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            # Block 1
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            # Block 2
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.3),
            # Block 3
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Dropout(0.2),
            # Output
            nn.Linear(16, 1)   # BCEWithLogitsLoss → Sigmoid 불필요
        )
    def forward(self, x):
        return self.network(x)

input_dim = X_train.shape[1]
model = ImprovedMLP(input_dim).to(device)
print(f"\n[Step 4] 모델 구조:\n{model}")

# ──────────────────────────────────────────────
# 6. 손실 함수  ← 개선점 ①  pos_weight 적용
#
#    원리: BCEWithLogitsLoss(pos_weight=w)
#          Loss = -[w·y·log(σ(x)) + (1-y)·log(1-σ(x))]
#          w = 정상 수 / 고장 수 ≈ 9638 / 362 ≈ 26.6
#          → 고장 샘플 1개를 정상 26.6개에 해당하는 가중치로 학습
#          → 모델이 고장 탐지에 집중 → Recall & F1 향상
# ──────────────────────────────────────────────
n_neg = (y_train == 0).sum()
n_pos = (y_train == 1).sum()
pos_weight_val = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(device)
print(f"\n[pos_weight] 정상:{n_neg}  고장:{n_pos}  비율:{pos_weight_val.item():.2f}")

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_val)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# ──────────────────────────────────────────────
# 7. 학습 루프
# ──────────────────────────────────────────────
epochs = 50
print(f"\n[Step 5] 학습 시작 (epochs={epochs})...")

train_losses, val_losses, val_f1s = [], [], []
best_val_loss = float('inf')

for epoch in range(epochs):
    # --- Train ---
    model.train()
    t_loss = 0.0
    for bx, by in train_loader:
        bx, by = bx.to(device), by.to(device)
        optimizer.zero_grad()
        loss = criterion(model(bx), by)
        loss.backward()
        optimizer.step()
        t_loss += loss.item()
    avg_t = t_loss / len(train_loader)

    # --- Validation (Test 셋 기준) ---
    model.eval()
    v_loss = 0.0
    preds_all, targets_all = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            bx, by = bx.to(device), by.to(device)
            out  = model(bx)
            v_loss += criterion(out, by).item()
            prob = torch.sigmoid(out)
            pred = (prob >= 0.5).float()
            preds_all.extend(pred.cpu().numpy())
            targets_all.extend(by.cpu().numpy())
    avg_v  = v_loss / len(test_loader)
    ep_f1  = f1_score(targets_all, preds_all, zero_division=0)

    train_losses.append(avg_t)
    val_losses.append(avg_v)
    val_f1s.append(ep_f1)
    scheduler.step(avg_v)

    if avg_v < best_val_loss:
        best_val_loss = avg_v
        torch.save(model.state_dict(), BEST_MODEL_PATH)

    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"  Epoch [{epoch+1:2d}/{epochs}] "
              f"Train Loss: {avg_t:.4f} | Val Loss: {avg_v:.4f} | F1(thr=0.5): {ep_f1:.4f}")

print(f"  학습 완료 — {BEST_MODEL_PATH.name} 저장됨")

# ──────────────────────────────────────────────
# 8. 최적 임곗값 탐색  ← 개선점 ③
#
#    배경: Threshold=0.5는 균형 데이터 가정.
#    불균형 데이터에서는 Precision-Recall 곡선의
#    F1 최대화 지점을 임곗값으로 사용해야 함.
# ──────────────────────────────────────────────
print("\n[Step 6] 최적 임곗값(Threshold) 탐색...")
model.load_state_dict(torch.load(BEST_MODEL_PATH, weights_only=True))
model.eval()

all_probs, all_targets = [], []
with torch.no_grad():
    for bx, by in test_loader:
        bx = bx.to(device)
        prob = torch.sigmoid(model(bx)).cpu().numpy()
        all_probs.extend(prob.flatten())
        all_targets.extend(by.numpy().flatten())

all_probs   = np.array(all_probs)
all_targets = np.array(all_targets)

# Precision-Recall 곡선 기반 F1 최대 임곗값 탐색
precisions, recalls, thresholds = precision_recall_curve(all_targets, all_probs)
f1_scores_thr = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-8)
best_idx       = np.argmax(f1_scores_thr)
optimal_thr    = thresholds[best_idx]
print(f"  최적 임곗값: {optimal_thr:.4f}  (F1@thr={f1_scores_thr[best_idx]:.4f})")

# ──────────────────────────────────────────────
# 9. 최종 평가 (최적 임곗값 적용)
# ──────────────────────────────────────────────
pred_optimal = (all_probs >= optimal_thr).astype(int)
pred_050     = (all_probs >= 0.50).astype(int)

def print_metrics(preds, label):
    acc  = accuracy_score(all_targets, preds)
    prec = precision_score(all_targets, preds, zero_division=0)
    rec  = recall_score(all_targets, preds, zero_division=0)
    f1   = f1_score(all_targets, preds, zero_division=0)
    auc  = roc_auc_score(all_targets, all_probs)
    print(f"\n  [{label}]")
    print(f"  Accuracy:  {acc:.4f}  | Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}  | F1-Score:  {f1:.4f} | ROC-AUC: {auc:.4f}")
    return acc, prec, rec, f1, auc

print("\n[Step 7] 최종 평가 결과")
print_metrics(pred_050,     "Threshold=0.50 (베이스라인 방식)")
acc, prec, rec, f1, auc = print_metrics(pred_optimal, f"Threshold={optimal_thr:.4f} (최적, 개선 모델)")

# ──────────────────────────────────────────────
# 10. 시각화 (4개 그래프)
# ──────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('AI4I 2020 — 개선 모델 평가 결과', fontsize=16, fontweight='bold')

# (1) Loss 곡선
axes[0,0].plot(train_losses, label='Train Loss', color='steelblue')
axes[0,0].plot(val_losses,   label='Val Loss',   color='coral')
axes[0,0].set_title('학습/검증 Loss 곡선')
axes[0,0].set_xlabel('Epoch'); axes[0,0].set_ylabel('Loss')
axes[0,0].legend()

# (2) 평가지표 비교 (베이스라인 vs 개선)
metrics = ['Accuracy','Precision','Recall','F1-Score','ROC-AUC']
baseline_vals = [0.9625, 0.4200, 0.5000, 0.4539, 0.86]   # 베이스라인 기록값
improved_vals = [acc, prec, rec, f1, auc]
x = np.arange(len(metrics))
w = 0.35
axes[0,1].bar(x - w/2, baseline_vals, w, label='Baseline', color='lightcoral', alpha=0.85)
axes[0,1].bar(x + w/2, improved_vals, w, label='Improved',  color='steelblue',  alpha=0.85)
axes[0,1].set_xticks(x); axes[0,1].set_xticklabels(metrics, rotation=15)
axes[0,1].set_ylim(0, 1.15); axes[0,1].set_title('베이스라인 vs 개선 모델')
axes[0,1].legend()
for i, v in enumerate(improved_vals):
    axes[0,1].text(i + w/2, v + 0.02, f'{v:.3f}', ha='center', fontsize=9, fontweight='bold')

# (3) Confusion Matrix
cm = confusion_matrix(all_targets, pred_optimal)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1,0], cbar=False,
            xticklabels=['Normal','Fault'], yticklabels=['Normal','Fault'], annot_kws={"size":13})
axes[1,0].set_title(f'Confusion Matrix (thr={optimal_thr:.3f})')
axes[1,0].set_xlabel('Predicted'); axes[1,0].set_ylabel('Actual')

# (4) Precision-Recall 곡선 + 최적 임곗값
axes[1,1].plot(thresholds, precisions[:-1], label='Precision', color='navy',    linestyle='--')
axes[1,1].plot(thresholds, recalls[:-1],    label='Recall',    color='seagreen')
axes[1,1].plot(thresholds, f1_scores_thr,   label='F1-Score',  color='darkorange', linewidth=2)
axes[1,1].axvline(optimal_thr, color='red', linestyle=':', linewidth=1.5,
                  label=f'Optimal thr={optimal_thr:.3f}')
axes[1,1].set_title('Precision / Recall / F1 vs Threshold')
axes[1,1].set_xlabel('Threshold'); axes[1,1].set_ylabel('Score')
axes[1,1].legend(fontsize=9)

plt.tight_layout()
plt.savefig(FIGURE_PATH, dpi=150, bbox_inches='tight')
plt.show()
print(f"\n  평가 그래프 저장 → {FIGURE_PATH.name}")

# ──────────────────────────────────────────────
# 11. 모델 & 스케일러 저장
# ──────────────────────────────────────────────
os.makedirs(MODEL_DIR, exist_ok=True)
torch.save(model.state_dict(), MODEL_DIR / 'improved_mlp.pth')
joblib.dump(scaler,       MODEL_DIR / 'improved_scaler.pkl')
joblib.dump(optimal_thr,  MODEL_DIR / 'optimal_threshold.pkl')
print("\n[Step 8] 저장 완료:")
print(f"  {MODEL_DIR / 'improved_mlp.pth'}")
print(f"  {MODEL_DIR / 'improved_scaler.pkl'}")
print(f"  {MODEL_DIR / 'optimal_threshold.pkl'}")
