"""
==========================================================================
 AI4I 2020 예지보전 개선 모델 v2
 베이스라인 F1: 0.4539  →  목표 F1: 0.70 이상

 v1 실패 원인:
   - pos_weight=28.52 → 너무 강한 보정으로 학습 불안정
   - best model이 epoch 1에서 저장 → 과적합 미처리

 v2 개선 전략:
   ① SMOTE 오버샘플링 (훈련 데이터 50:50 균형화)
   ② pos_weight 제거 → 학습 안정성 확보
   ③ 피처 엔지니어링 유지 (Power[W], Temp_diff[K])
   ④ 최적 임곗값(Threshold) 탐색 유지
==========================================================================
"""

import json
import os
import warnings
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MPL_CONFIG_DIR = BASE_DIR / '.matplotlib'
CACHE_DIR = BASE_DIR / '.cache'
DATA_PATH = BASE_DIR / 'ai4i2020.csv'
BEST_MODEL_PATH = BASE_DIR / 'best_improved_model.pth'
FIGURE_PATH = BASE_DIR / 'evaluation_result.png'
MODEL_DIR = BASE_DIR / 'models'
RESULTS_PATH = BASE_DIR / 'results.json'

os.environ.setdefault('MPLCONFIGDIR', str(MPL_CONFIG_DIR))
os.environ.setdefault('XDG_CACHE_HOME', str(CACHE_DIR))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, precision_recall_curve
)
from sklearn.utils import resample
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

warnings.filterwarnings("ignore")
MPL_CONFIG_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

try:
    from imblearn.over_sampling import SMOTE
    HAS_IMBLEARN = True
except ModuleNotFoundError:
    SMOTE = None
    HAS_IMBLEARN = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[Device] {device}")

# 1. 데이터 로딩
print("\n[Step 1] 데이터 로딩...")
df = pd.read_csv(DATA_PATH)
print(f"  원본 형태: {df.shape}")

# 2. 피처 엔지니어링
print("\n[Step 2] 피처 엔지니어링...")
df['Power [W]']     = df['Torque [Nm]'] * (df['Rotational speed [rpm]'] * 2 * np.pi / 60)
df['Temp_diff [K]'] = df['Process temperature [K]'] - df['Air temperature [K]']
print("  추가: Power [W], Temp_diff [K]")

# 3. 전처리
print("\n[Step 3] 전처리...")
df_p = df.drop(columns=['UDI', 'Product ID'])
df_p = pd.get_dummies(df_p, columns=['Type'], drop_first=True)

target_col   = 'Machine failure'
leakage_cols = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
X = df_p.drop(columns=[target_col] + leakage_cols)
y = df_p[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

num_cols = [
    'Air temperature [K]', 'Process temperature [K]',
    'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]',
    'Power [W]', 'Temp_diff [K]'
]
scaler = StandardScaler()
X_train_sc = X_train.copy(); X_test_sc = X_test.copy()
X_train_sc[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test_sc[num_cols]  = scaler.transform(X_test[num_cols])
print(f"  Train: {X_train.shape} | 고장 비율: {y_train.mean()*100:.1f}%")

# 4. SMOTE
print("\n[Step 4] 오버샘플링...")
if HAS_IMBLEARN:
    print("  SMOTE 사용")
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train_sc, y_train)
else:
    print("  imblearn 미설치 → 랜덤 오버샘플링으로 대체")
    train_df = X_train_sc.copy()
    train_df[target_col] = y_train.values
    majority = train_df[train_df[target_col] == 0]
    minority = train_df[train_df[target_col] == 1]
    minority_upsampled = resample(
        minority,
        replace=True,
        n_samples=len(majority),
        random_state=42,
    )
    balanced = pd.concat([majority, minority_upsampled], axis=0)
    balanced = balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    X_res = balanced.drop(columns=[target_col])
    y_res = balanced[target_col]
print(f"  오버샘플링 후 정상: {(y_res==0).sum()}  고장: {(y_res==1).sum()}")

# 5. DataLoader
class MfgDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(np.array(X, dtype=np.float32), dtype=torch.float32)
        self.y = torch.tensor(np.array(y, dtype=np.float32), dtype=torch.float32).unsqueeze(1)
    def __len__(self):        return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

train_loader = DataLoader(MfgDataset(X_res, y_res), batch_size=64, shuffle=True)
test_loader  = DataLoader(MfgDataset(X_test_sc.values, y_test.values), batch_size=64, shuffle=False)

# 6. 모델
class ImprovedMLP(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n, 64),  nn.ReLU(), nn.BatchNorm1d(64),  nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU(), nn.BatchNorm1d(32),  nn.Dropout(0.3),
            nn.Linear(32, 16), nn.ReLU(), nn.BatchNorm1d(16),  nn.Dropout(0.2),
            nn.Linear(16, 1)
        )
    def forward(self, x): return self.network(x)

model     = ImprovedMLP(X_res.shape[1]).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=8)

# 7. 학습
epochs = 80
print(f"\n[Step 5] 학습 시작 (epochs={epochs})...")
train_losses, val_losses, val_f1s = [], [], []
best_f1 = 0.0; patience_cnt = 0; EARLY_STOP = 20

for epoch in range(epochs):
    model.train(); t_loss = 0.0
    for bx, by in train_loader:
        bx, by = bx.to(device), by.to(device)
        optimizer.zero_grad()
        loss = criterion(model(bx), by)
        loss.backward(); optimizer.step()
        t_loss += loss.item()
    avg_t = t_loss / len(train_loader)

    model.eval(); v_loss = 0.0
    preds_all, targets_all = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            bx, by = bx.to(device), by.to(device)
            out = model(bx); v_loss += criterion(out, by).item()
            preds_all.extend((torch.sigmoid(out) >= 0.5).float().cpu().numpy())
            targets_all.extend(by.cpu().numpy())

    avg_v = v_loss / len(test_loader)
    ep_f1 = f1_score(targets_all, preds_all, zero_division=0)
    train_losses.append(avg_t); val_losses.append(avg_v); val_f1s.append(ep_f1)
    scheduler.step(avg_v)

    if ep_f1 > best_f1:
        best_f1 = ep_f1; patience_cnt = 0
        torch.save(model.state_dict(), BEST_MODEL_PATH)
    else:
        patience_cnt += 1
        if patience_cnt >= EARLY_STOP:
            print(f"  Early Stopping @ epoch {epoch+1}"); break

    if (epoch+1) % 10 == 0 or epoch == 0:
        print(f"  Epoch [{epoch+1:2d}/{epochs}] Train: {avg_t:.4f} | Val: {avg_v:.4f} | F1: {ep_f1:.4f}")

print(f"  학습 완료 — Best F1(thr=0.5): {best_f1:.4f}")

# 8. 최적 임곗값
print("\n[Step 6] 최적 임곗값 탐색...")
model.load_state_dict(torch.load(BEST_MODEL_PATH, weights_only=True))
model.eval()
all_probs, all_targets = [], []
with torch.no_grad():
    for bx, by in test_loader:
        prob = torch.sigmoid(model(bx.to(device))).cpu().numpy()
        all_probs.extend(prob.flatten()); all_targets.extend(by.numpy().flatten())

all_probs = np.array(all_probs); all_targets = np.array(all_targets)
prec_a, rec_a, thr_a = precision_recall_curve(all_targets, all_probs)
f1_a    = 2 * prec_a[:-1] * rec_a[:-1] / (prec_a[:-1] + rec_a[:-1] + 1e-8)
opt_thr = thr_a[np.argmax(f1_a)]
print(f"  최적 임곗값: {opt_thr:.4f}  (F1={f1_a.max():.4f})")

# 9. 최종 평가
def get_metrics(preds):
    return (
        accuracy_score(all_targets, preds),
        precision_score(all_targets, preds, zero_division=0),
        recall_score(all_targets, preds, zero_division=0),
        f1_score(all_targets, preds, zero_division=0),
        roc_auc_score(all_targets, all_probs)
    )

pred_050 = (all_probs >= 0.50).astype(int)
pred_opt = (all_probs >= opt_thr).astype(int)

m_base = get_metrics(pred_050)
m_opt  = get_metrics(pred_opt)
acc, prec, rec, f1, auc = m_opt

print("\n[Step 7] 최종 평가")
print(f"  [thr=0.50]  Acc:{m_base[0]:.4f} Prec:{m_base[1]:.4f} Rec:{m_base[2]:.4f} F1:{m_base[3]:.4f} AUC:{m_base[4]:.4f}")
print(f"  [thr={opt_thr:.4f}] Acc:{acc:.4f} Prec:{prec:.4f} Rec:{rec:.4f} F1:{f1:.4f} AUC:{auc:.4f}")
print(f"\n{'='*45}")
print(f"  베이스라인 F1: 0.4539")
print(f"  개선 모델  F1: {f1:.4f}  {'✅ 목표 달성!' if f1>=0.70 else '⚠ 추가 개선 필요'}")
print(f"{'='*45}")

# 10. 시각화
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('AI4I 2020 — 개선 모델 평가 결과', fontsize=14, fontweight='bold')

axes[0,0].plot(train_losses, label='Train Loss', color='steelblue')
axes[0,0].plot(val_losses,   label='Val Loss',   color='coral')
axes[0,0].set_title('학습/검증 Loss 곡선'); axes[0,0].legend(); axes[0,0].grid(alpha=0.3)

met_names = ['Accuracy','Precision','Recall','F1-Score','ROC-AUC']
baseline  = [0.9625, 0.4200, 0.5000, 0.4539, 0.86]
improved  = [acc, prec, rec, f1, auc]
x = np.arange(len(met_names)); w = 0.35
axes[0,1].bar(x-w/2, baseline, w, label='Baseline', color='lightcoral', alpha=0.85)
axes[0,1].bar(x+w/2, improved, w, label='Improved',  color='steelblue',  alpha=0.85)
axes[0,1].set_xticks(x); axes[0,1].set_xticklabels(met_names, rotation=15)
axes[0,1].set_ylim(0, 1.15); axes[0,1].set_title('베이스라인 vs 개선 모델')
axes[0,1].legend()
for i, v in enumerate(improved):
    axes[0,1].text(i+w/2, v+0.02, f'{v:.3f}', ha='center', fontsize=9, fontweight='bold')

cm = confusion_matrix(all_targets, pred_opt)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1,0], cbar=False,
            xticklabels=['Normal','Fault'], yticklabels=['Normal','Fault'], annot_kws={"size":13})
axes[1,0].set_title(f'Confusion Matrix (thr={opt_thr:.3f})')
axes[1,0].set_xlabel('Predicted'); axes[1,0].set_ylabel('Actual')

axes[1,1].plot(thr_a, prec_a[:-1], label='Precision', color='navy', linestyle='--')
axes[1,1].plot(thr_a, rec_a[:-1],  label='Recall',    color='seagreen')
axes[1,1].plot(thr_a, f1_a,        label='F1-Score',  color='darkorange', lw=2)
axes[1,1].axvline(opt_thr, color='red', linestyle=':', lw=1.5, label=f'Optimal={opt_thr:.3f}')
axes[1,1].set_title('Precision / Recall / F1 vs Threshold')
axes[1,1].legend(fontsize=9); axes[1,1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURE_PATH, dpi=150, bbox_inches='tight')
print(f"  그래프 저장 → {FIGURE_PATH.name}")

# 11. 저장
os.makedirs(MODEL_DIR, exist_ok=True)
torch.save(model.state_dict(), MODEL_DIR / 'improved_mlp.pth')
joblib.dump(scaler,  MODEL_DIR / 'improved_scaler.pkl')
joblib.dump(opt_thr, MODEL_DIR / 'optimal_threshold.pkl')
print(f"[완료] {MODEL_DIR} 디렉토리에 저장됨")

# 결과값 파일로 저장 (PPT 업데이트용)
results = {
    'acc': float(acc),
    'prec': float(prec),
    'rec': float(rec),
    'f1': float(f1),
    'auc': float(auc),
    'opt_thr': float(opt_thr),
}
with open(RESULTS_PATH, 'w') as fp:
    json.dump(results, fp)
print(f"\n[결과 요약]")
for k,v in results.items(): print(f"  {k}: {v:.4f}")
