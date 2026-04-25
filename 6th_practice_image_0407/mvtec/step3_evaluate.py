import os
import json
from pathlib import Path

base_dir = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(base_dir / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(base_dir / ".cache"))

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_curve
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader
from step1_data_eda import MVTecDataset
from step2_train import ConvAutoencoder
from simple_vision import Compose, Resize, ToTensor, resolve_data_dir

RESULTS_DIR = base_dir / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def blur_error_map(error_map, radius=4):
    normalized = error_map - error_map.min()
    normalized = normalized / (normalized.max() + 1e-8)
    blurred = Image.fromarray((normalized * 255).astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(radius=radius)
    )
    return np.asarray(blurred, dtype=np.float32) / 255.0

def evaluate_performance(model, test_loader, device):
    """테스트 데이터셋 전체를 평가하여 정량적 지표를 산출합니다."""
    model.eval()
    y_true = []
    y_scores = []
    
    print("전체 테스트 데이터셋 정량 평가를 진행합니다...")
    with torch.no_grad():
        for images, labels, _ in test_loader:
            images = images.to(device)
            outputs = model(images)
            
            # 1. 픽셀 단위 오차 계산 및 노이즈 제거 (가우시안 블러)
            error = torch.mean((images - outputs) ** 2, dim=1) 
            error_map = error.squeeze().cpu().numpy()
            error_map = blur_error_map(error_map)
            
            # 2. 이미지 레벨 이상치 점수(Anomaly Score) 산출
            # 제조품은 아주 작은 결함 하나만 있어도 전체가 불량입니다. 
            # 따라서 오차 맵에서 '가장 오차가 큰 픽셀의 값(Max)'을 해당 이미지의 대표 불량 점수로 사용합니다.
            anomaly_score = np.max(error_map)
            
            y_scores.append(anomaly_score)
            y_true.append(labels.item()) # 0: 정상, 1: 불량

    # 3. 정량적 지표 계산
    # AUROC: 임계값에 상관없이 모델의 전반적인 정상/불량 분류 능력을 평가
    auroc = roc_auc_score(y_true, y_scores)
    
    # Precision-Recall 기반 최적 임계값 및 F1-Score 탐색
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    
    # F1-Score 계산 (0으로 나누어지는 것 방지)
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_f1 = f1_scores[best_idx]
    best_threshold = thresholds[best_idx]
    
    print("-" * 40)
    print(f"[전체 평가 결과]")
    print(f"AUROC Score          : {auroc:.4f}")
    print(f"Best F1-Score        : {best_f1:.4f}")
    print(f"Optimal Threshold    : {best_threshold:.4f}")
    print("-" * 40)
    metrics = {
        "auroc": float(auroc),
        "best_f1": float(best_f1),
        "optimal_threshold": float(best_threshold),
    }
    metrics_path = RESULTS_DIR / "step3_evaluation_metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"평가 지표 저장 완료: {metrics_path}")
    
    return best_threshold

def visualize_anomaly(model, test_loader, device, threshold, num_samples=3):
    """결함 탐지 시각화 및 판정 결과를 출력합니다."""
    model.eval()
    samples_shown = 0
    saved_rows = []
    
    print(f"\n최적 임계값({threshold:.4f})을 적용하여 시각화를 시작합니다.")
    
    with torch.no_grad():
        for images, labels, image_paths in test_loader:
            if labels.item() == 0: 
                continue
                
            images = images.to(device)
            outputs = model(images)
            
            error = torch.mean((images - outputs) ** 2, dim=1) 
            error_map = error.squeeze().cpu().numpy()
            error_map = blur_error_map(error_map)
            
            anomaly_score = np.max(error_map)
            
            # 산출된 Threshold를 바탕으로 불량(NG) / 정상(OK) 판정
            prediction = "NG (Defect)" if anomaly_score >= threshold else "OK (Normal)"
            
            error_map_norm = error_map - error_map.min()
            error_map_norm = error_map_norm / (error_map_norm.max() + 1e-8)
            heatmap = plt.get_cmap('jet')(error_map_norm)[..., :3]
            heatmap = (heatmap * 255).astype(np.uint8)
            
            img_np = images.squeeze().cpu().permute(1, 2, 0).numpy()
            out_np = outputs.squeeze().cpu().permute(1, 2, 0).numpy()
            
            overlay = ((img_np * 255).astype(np.uint8) * 0.5 + heatmap * 0.5).clip(0, 255).astype(np.uint8)
            
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            # 타이틀에 예측 판정 결과 및 점수 표시
            axes[0].imshow(img_np); axes[0].set_title(f'Original\nScore: {anomaly_score:.4f} -> {prediction}')
            axes[1].imshow(out_np); axes[1].set_title('Reconstructed')
            axes[2].imshow(error_map, cmap='hot'); axes[2].set_title('Error Map')
            axes[3].imshow(overlay); axes[3].set_title('Overlay Heatmap')
            
            for ax in axes:
                ax.axis('off')
            save_path = RESULTS_DIR / f"step3_anomaly_sample_{samples_shown + 1}.png"
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            saved_rows.append(
                {
                    "image_path": image_paths[0],
                    "label": int(labels.item()),
                    "anomaly_score": float(anomaly_score),
                    "prediction": prediction,
                    "saved_figure": str(save_path),
                }
            )
            
            samples_shown += 1
            if samples_shown >= num_samples:
                break
    if saved_rows:
        csv_path = RESULTS_DIR / "step3_visualized_samples.csv"
        pd.DataFrame(saved_rows).to_csv(csv_path, index=False)
        print(f"시각화 샘플 목록 저장 완료: {csv_path}")

if __name__ == "__main__":
    ROOT_DIR = resolve_data_dir(base_dir)
    CATEGORY = 'bottle' 
    MODEL_PATH = base_dir / 'autoencoder_model.pth'

    transform = Compose([Resize((256, 256)), ToTensor()])

    test_dataset = MVTecDataset(ROOT_DIR, CATEGORY, is_train=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvAutoencoder().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    
    # 1. 정량 평가 수행 및 최적 임계값 도출
    optimal_thresh = evaluate_performance(model, test_loader, device)
    
    # 2. 도출된 임계값을 시각화 함수에 전달하여 실제 판정 시뮬레이션
    visualize_anomaly(model, test_loader, device, optimal_thresh, num_samples=3)
