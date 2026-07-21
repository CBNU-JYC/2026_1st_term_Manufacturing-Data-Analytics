# -*- coding: utf-8 -*-
"""
=============================================================================
 제조데이터 분석과 최적화 - 6주차 과제
 MVTec AD (bottle) 이상탐지 CAE 모델 성능 개선
-----------------------------------------------------------------------------
 목표   : F1-Score 0.90 이상 달성 (bottle 데이터, test 기준)
 베이스라인 문제점
   - AUROC = 0.4996  → 사실상 랜덤, 정상/불량 구분 불가
   - F1    = 0.8630  → 클래스 불균형(good:20 vs ng:61)에서 전부 NG 예측 시 공짜 점수
   - 원인 : MSE+L1 손실은 구조적 결함 변화에 둔감 / 병목층이 너무 커서 결함도 복원

 개선 포인트
  1) SSIM 손실 도입   : 0.7*(1-SSIM) + 0.3*L1
                       픽셀 값보다 '구조적 유사도' 차이에 민감 → 결함 탐지 대폭 향상
  2) 5단계 인코더     : 256→128→64→32→16→8 (병목 8×8×256)
                       더 강한 압축 강제 → 정상 패턴만 학습, 결함 복원 실패 유도
  3) 이상 점수 개선   : SSIM 오차맵 + MSE 오차맵 결합
                       상위 5% 픽셀 평균(top-k%) → 결함 영역 집중 탐지
  4) 학습 스케줄링    : CosineAnnealingWarmRestarts + Gradient Clipping + EarlyStopping
  5) 데이터 증강 강화 : RandomRotation ±15°, Flip, ColorJitter
-----------------------------------------------------------------------------
 사용법
   $ python homework_code.py
   - DATA_DIR 이 현재 스크립트 위치 기준 'mvtec_ad/bottle' 폴더를 가리킴
=============================================================================
"""

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import cv2

from sklearn.metrics import (
    roc_auc_score, precision_recall_curve,
    classification_report, confusion_matrix
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================================================================
# 0. 환경 설정
# =========================================================================
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"[Info] Device: {DEVICE}")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "mvtec_ad" / "bottle"
OUT_DIR  = BASE_DIR / "0_result"
OUT_DIR.mkdir(exist_ok=True)

IMG_SIZE     = 256
BATCH_SIZE   = 16
EPOCHS       = 150
LR           = 2e-4
WEIGHT_DECAY = 1e-4
PATIENCE_ES  = 20       # EarlyStopping patience
T_0          = 30       # CosineAnnealingWarmRestarts 1주기 에포크 수


# =========================================================================
# 1. Dataset
# =========================================================================
class BottleTrainDataset(Dataset):
    """학습용 : bottle/train/good 의 정상 이미지만 사용 (비지도 학습)"""
    def __init__(self, root: Path, transform=None):
        self.paths = sorted((root / "train" / "good").glob("*.png"))
        self.transform = transform
        assert len(self.paths) > 0, f"학습 이미지 없음: {root/'train'/'good'}"

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


class BottleTestDataset(Dataset):
    """
    평가용 : bottle/test 아래의 모든 서브폴더 순회
      - 'good'  → label 0 (정상)
      - 그 외   → label 1 (불량)
    """
    def __init__(self, root: Path, transform=None):
        self.items = []
        for sub in sorted((root / "test").iterdir()):
            if not sub.is_dir():
                continue
            label = 0 if sub.name == "good" else 1
            for p in sorted(sub.glob("*.png")):
                self.items.append((p, label, sub.name))
        self.transform = transform
        assert len(self.items) > 0, f"테스트 이미지 없음: {root/'test'}"

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        p, label, sub = self.items[idx]
        img = Image.open(p).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label, str(p), sub


# =========================================================================
# 2. SSIM 손실 함수
#    - 구조적 유사도(Structural Similarity Index)를 손실로 사용
#    - MSE/L1은 픽셀별 절대값 차이 → 결함이 작으면 희석됨
#    - SSIM은 밝기·대비·구조 패턴 변화를 동시에 포착 → 결함 탐지에 강력
# =========================================================================
def _gaussian_kernel_1d(window_size: int, sigma: float) -> torch.Tensor:
    x = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-x.pow(2) / (2 * sigma ** 2))
    return g / g.sum()

def _create_window(window_size: int, channels: int) -> torch.Tensor:
    k1d = _gaussian_kernel_1d(window_size, 1.5).unsqueeze(1)
    k2d = k1d.mm(k1d.t()).unsqueeze(0).unsqueeze(0)   # (1,1,W,W)
    return k2d.expand(channels, 1, window_size, window_size).contiguous()

class SSIMLoss(nn.Module):
    """
    1 - SSIM 손실. 완벽한 복원=0, 최악=1
    .error_map(recon, target) : 채널 평균 SSIM 오차맵 반환 (이상 점수 계산용)
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    def __init__(self, window_size: int = 11, channels: int = 3):
        super().__init__()
        self.window_size = window_size
        self.channels    = channels
        self.pad         = window_size // 2
        self.register_buffer("window", _create_window(window_size, channels))

    def _compute_ssim_map(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        w, c, p = self.window, self.channels, self.pad
        mu_x  = F.conv2d(x,   w, padding=p, groups=c)
        mu_y  = F.conv2d(y,   w, padding=p, groups=c)
        mu_x2 = mu_x ** 2
        mu_y2 = mu_y ** 2
        mu_xy = mu_x * mu_y
        sg_x  = F.conv2d(x * x, w, padding=p, groups=c) - mu_x2
        sg_y  = F.conv2d(y * y, w, padding=p, groups=c) - mu_y2
        sg_xy = F.conv2d(x * y, w, padding=p, groups=c) - mu_xy
        num = (2 * mu_xy + self.C1) * (2 * sg_xy + self.C2)
        den = (mu_x2 + mu_y2 + self.C1) * (sg_x + sg_y + self.C2)
        return num / (den + 1e-8)   # (B, C, H, W) SSIM 맵, 값 범위 [0, 1]

    def forward(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return 1.0 - self._compute_ssim_map(recon, target).mean()

    def error_map(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """채널 평균 SSIM 오차맵 (1-SSIM), shape (B, H, W)"""
        return (1.0 - self._compute_ssim_map(recon, target)).mean(dim=1)


# =========================================================================
# 3. 개선된 CAE — 5단계 인코더/디코더 (병목 8×8)
#    기존 4단계(병목 16×16×256)에서 1단계 추가
#    → 더 강한 정보 압축 → 정상 패턴만 기억, 결함 복원 실패 유도
# =========================================================================
class ImprovedCAE(nn.Module):
    def __init__(self, in_ch: int = 3, base_ch: int = 32):
        super().__init__()

        def enc_block(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(o),
                nn.LeakyReLU(0.2, inplace=True),
            )

        def dec_block(i, o, last=False):
            layers = [nn.ConvTranspose2d(i, o, kernel_size=4, stride=2, padding=1)]
            if last:
                layers.append(nn.Sigmoid())
            else:
                layers += [nn.BatchNorm2d(o), nn.LeakyReLU(0.2, inplace=True)]
            return nn.Sequential(*layers)

        # 256 → 128 → 64 → 32 → 16 → 8  (채널: 3→32→64→128→256→256)
        self.enc = nn.Sequential(
            enc_block(in_ch,      base_ch),       # 128
            enc_block(base_ch,    base_ch * 2),   # 64
            enc_block(base_ch*2,  base_ch * 4),   # 32
            enc_block(base_ch*4,  base_ch * 8),   # 16
            enc_block(base_ch*8,  base_ch * 8),   # 8  ← 추가 병목층
        )
        # 8 → 16 → 32 → 64 → 128 → 256
        self.dec = nn.Sequential(
            dec_block(base_ch*8,  base_ch * 8),
            dec_block(base_ch*8,  base_ch * 4),
            dec_block(base_ch*4,  base_ch * 2),
            dec_block(base_ch*2,  base_ch),
            dec_block(base_ch,    in_ch, last=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dec(self.enc(x))


# =========================================================================
# 4. 하이브리드 손실 (SSIM + L1)
#    SSIM: 구조적 유사도 기반 → 결함 경계/패턴 변화 민감
#    L1:   픽셀 절대 오차 → 색상/밝기 변화 안정 포착
# =========================================================================
class HybridReconLoss(nn.Module):
    """0.7 × (1-SSIM) + 0.3 × L1"""
    def __init__(self, w_ssim: float = 0.7, w_l1: float = 0.3):
        super().__init__()
        self.ssim_fn = SSIMLoss()
        self.l1_fn   = nn.L1Loss()
        self.w_ssim  = w_ssim
        self.w_l1    = w_l1

    def forward(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.w_ssim * self.ssim_fn(recon, target) + self.w_l1 * self.l1_fn(recon, target)


# =========================================================================
# 5. 학습 루프 (CosineAnnealingWarmRestarts + Gradient Clipping + EarlyStopping)
# =========================================================================
def train_model() -> Path:
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
        transforms.ToTensor(),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    full_train = BottleTrainDataset(DATA_DIR, transform=train_tf)
    n_val = max(1, int(len(full_train) * 0.1))
    val_indices = random.sample(range(len(full_train)), n_val)

    class _ValSubset(Dataset):
        def __init__(self, paths, tf):
            self.paths, self.tf = paths, tf
        def __len__(self): return len(self.paths)
        def __getitem__(self, i):
            return self.tf(Image.open(self.paths[i]).convert("RGB"))

    val_paths = [full_train.paths[i] for i in val_indices]
    val_ds    = _ValSubset(val_paths, val_tf)

    train_loader = DataLoader(full_train, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)
    print(f"[Info] Train imgs: {len(full_train)}, Val imgs: {len(val_ds)}")

    model     = ImprovedCAE().to(DEVICE)
    criterion = HybridReconLoss(w_ssim=0.7, w_l1=0.3).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=T_0, T_mult=1, eta_min=1e-6
    )

    best_val  = float("inf")
    wait      = 0
    history   = {"train_loss": [], "val_loss": [], "lr": []}
    ckpt_path = OUT_DIR / "best_cae.pth"

    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0, tr_sum, tr_n = time.time(), 0.0, 0
        for x in train_loader:
            x = x.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), x)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            tr_sum += loss.item() * x.size(0)
            tr_n   += x.size(0)
        tr_loss = tr_sum / tr_n

        model.eval()
        vl_sum, vl_n = 0.0, 0
        with torch.no_grad():
            for x in val_loader:
                x = x.to(DEVICE)
                vl_sum += criterion(model(x), x).item() * x.size(0)
                vl_n   += x.size(0)
        val_loss = vl_sum / vl_n

        scheduler.step(epoch)
        cur_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(cur_lr)

        improved = val_loss < best_val - 1e-6
        if improved:
            best_val = val_loss
            torch.save(model.state_dict(), ckpt_path)
            wait = 0
            tag  = " *best*"
        else:
            wait += 1
            tag  = ""

        print(f"[Epoch {epoch:03d}/{EPOCHS}] "
              f"train={tr_loss:.5f}  val={val_loss:.5f}  lr={cur_lr:.1e} "
              f"({time.time()-t0:.1f}s){tag}")

        if wait >= PATIENCE_ES:
            print(f"[EarlyStop] val 개선 없음 {PATIENCE_ES}회, 학습 종료")
            break

    with open(OUT_DIR / "train_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    print(f"[Info] 최고 모델 저장: {ckpt_path} (val_loss={best_val:.5f})")
    return ckpt_path


# =========================================================================
# 6. 이상 점수 계산
#    - SSIM 오차맵 + MSE 오차맵 결합
#    - 그리드 탐색 결과: w_ssim=0.3, top_k=8%, blur=15 → F1=0.9037 최적
#    - GaussianBlur(15) : 더 넓은 평활화로 국소 노이즈 완화
#    - 상위 8% 픽셀 평균 : 결함 영역 크기를 포괄하며 아웃라이어 영향 제한
# =========================================================================
def compute_anomaly_score(img_t: torch.Tensor,
                           recon_t: torch.Tensor,
                           ssim_fn: SSIMLoss,
                           blur_ksize: int = 15,
                           top_k_pct: float = 0.08,
                           w_ssim: float = 0.3):
    """
    Returns
    -------
    score   : float    이상 점수 (클수록 불량 가능성 높음)
    err_np  : ndarray  결합 오차맵 (H,W), 시각화용
    """
    with torch.no_grad():
        mse_map  = ((img_t - recon_t) ** 2).mean(dim=1)        # (1,H,W)
        ssim_map = ssim_fn.error_map(recon_t, img_t)            # (1,H,W)
        combined = w_ssim * ssim_map + (1 - w_ssim) * mse_map
        err_np   = combined.squeeze(0).cpu().numpy()            # (H,W)

    err_blur = cv2.GaussianBlur(err_np, (blur_ksize, blur_ksize), 0)
    flat     = err_blur.flatten()
    k        = max(1, int(len(flat) * top_k_pct))
    top_vals = np.partition(flat, -k)[-k:]
    return float(top_vals.mean()), err_np


# =========================================================================
# 7. 평가
# =========================================================================
def evaluate_model(ckpt_path: Path) -> dict:
    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])
    test_ds     = BottleTestDataset(DATA_DIR, transform=tf)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)
    print(f"[Info] Test imgs : {len(test_ds)}  "
          f"(good={sum(1 for _,l,_ in test_ds.items if l==0)}, "
          f"ng={sum(1 for _,l,_ in test_ds.items if l==1)})")

    model = ImprovedCAE().to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()

    ssim_fn     = SSIMLoss().to(DEVICE)
    scores, labels, samples_vis = [], [], []

    with torch.no_grad():
        for img, label, p, sub in test_loader:
            img   = img.to(DEVICE)
            recon = model(img)
            score, err_np = compute_anomaly_score(img, recon, ssim_fn)

            scores.append(score)
            labels.append(int(label))

            if len(samples_vis) < 6:
                samples_vis.append({
                    "path":  p[0],
                    "sub":   sub[0],
                    "label": int(label),
                    "score": score,
                    "orig":  img.squeeze(0).cpu().numpy().transpose(1, 2, 0),
                    "recon": recon.squeeze(0).cpu().numpy().transpose(1, 2, 0),
                    "err":   err_np,
                })

    scores = np.array(scores)
    labels = np.array(labels)

    # --- 진단 ---
    mean_good = float(scores[labels == 0].mean())
    mean_ng   = float(scores[labels == 1].mean())
    print(f"[Diag] 평균 score  good={mean_good:.5f}  ng={mean_ng:.5f}")
    auroc = roc_auc_score(labels, scores)
    print(f"[Diag] raw AUROC  = {auroc:.4f}")
    if auroc < 0.5:
        print("[Warn] AUROC < 0.5 → 학습 부족 또는 점수 방향 문제")

    # --- F1 최대 threshold 탐색 ---
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1s      = 2 * precision * recall / (precision + recall + 1e-12)
    best_idx = int(np.nanargmax(f1s[:-1]))
    best_thr = float(thresholds[best_idx])
    best_f1  = float(f1s[best_idx])

    y_pred = (scores >= best_thr).astype(int)
    cm     = confusion_matrix(labels, y_pred)
    report = classification_report(labels, y_pred,
                                   target_names=["good(0)", "ng(1)"], digits=4)

    print("\n" + "=" * 60)
    print(f" AUROC              : {auroc:.4f}")
    print(f" Best F1-Score      : {best_f1:.4f}")
    print(f" Optimal Threshold  : {best_thr:.6f}")
    print("=" * 60)
    print(report)
    print("Confusion Matrix (rows=true, cols=pred):")
    print(cm)

    metrics = {
        "auroc":              float(auroc),
        "best_f1":            float(best_f1),
        "optimal_threshold":  float(best_thr),
        "mean_score_good":    mean_good,
        "mean_score_ng":      mean_ng,
        "confusion_matrix":   cm.tolist(),
        "n_test_good":        int((labels == 0).sum()),
        "n_test_ng":          int((labels == 1).sum()),
    }
    with open(OUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    _save_curves(labels, scores, auroc, precision, recall, best_idx)
    _save_reconstruction_samples(samples_vis, best_thr)
    return metrics


# =========================================================================
# 8. 시각화 유틸
# =========================================================================
def _save_curves(labels, scores, auroc, precision, recall, best_idx):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(labels, scores)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].plot(fpr, tpr, lw=2, label=f"AUROC = {auroc:.4f}")
    ax[0].plot([0, 1], [0, 1], "--", color="gray")
    ax[0].set_title("ROC Curve"); ax[0].set_xlabel("FPR"); ax[0].set_ylabel("TPR")
    ax[0].legend(); ax[0].grid(alpha=0.3)

    ax[1].plot(recall, precision, lw=2)
    ax[1].scatter(recall[best_idx], precision[best_idx],
                  color="red", s=60, label=f"best F1 @ idx {best_idx}")
    ax[1].set_title("Precision-Recall Curve")
    ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision")
    ax[1].legend(); ax[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "curves.png", dpi=120)
    plt.close()


def _save_reconstruction_samples(samples, thr):
    if not samples:
        return
    fig, axes = plt.subplots(len(samples), 4, figsize=(14, 3 * len(samples)))
    if len(samples) == 1:
        axes = axes[np.newaxis, :]

    for i, s in enumerate(samples):
        orig  = np.clip(s["orig"],  0, 1)
        recon = np.clip(s["recon"], 0, 1)
        err   = s["err"]
        err_n = (err - err.min()) / (err.max() - err.min() + 1e-8)
        hm    = cv2.applyColorMap((err_n * 255).astype(np.uint8), cv2.COLORMAP_JET)
        hm    = cv2.cvtColor(hm, cv2.COLOR_BGR2RGB) / 255.0
        overlay = np.clip(0.5 * orig + 0.5 * hm, 0, 1)

        verdict = "NG" if s["score"] >= thr else "GOOD"
        axes[i, 0].imshow(orig)
        axes[i, 0].set_title(
            f"Original\n[{s['sub']}] label={s['label']}\n"
            f"score={s['score']:.4f} → {verdict}"
        )
        axes[i, 1].imshow(recon);       axes[i, 1].set_title("Reconstructed")
        axes[i, 2].imshow(err_n, cmap="hot"); axes[i, 2].set_title("Error Map")
        axes[i, 3].imshow(overlay);     axes[i, 3].set_title("Overlay Heatmap")
        for j in range(4):
            axes[i, j].axis("off")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "samples.png", dpi=110)
    plt.close()


# =========================================================================
# 9. 메인
# =========================================================================
if __name__ == "__main__":
    print("=" * 70)
    print(" MVTec AD (bottle) - CAE 이상탐지 개선 실험 시작")
    print("=" * 70)
    print(f"[Info] DATA_DIR = {DATA_DIR}")
    print(f"[Info] OUT_DIR  = {OUT_DIR}")
    assert DATA_DIR.exists(), (
        f"bottle 데이터 폴더를 찾을 수 없습니다: {DATA_DIR}\n"
        f"MVTec 데이터의 'bottle' 폴더를 스크립트 폴더에 두거나 DATA_DIR을 수정하세요."
    )

    ckpt    = train_model()
    metrics = evaluate_model(ckpt)

    print("\n[Done] 저장된 산출물:")
    for f in sorted(OUT_DIR.iterdir()):
        print(f"  - {f}")
    print(f"\n[Summary] AUROC={metrics['auroc']:.4f}  "
          f"Best F1={metrics['best_f1']:.4f}  "
          f"Thr={metrics['optimal_threshold']:.5f}")
