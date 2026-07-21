import os
import json
from pathlib import Path

base_dir = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(base_dir / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(base_dir / ".cache"))

import torch

from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

# train.py에 정의된 모델 구조 임포트
from plot_utils import configure_korean_font
from step2_train import SimpleCNN 
from simple_vision import Compose, Resize, SimpleImageFolder, ToTensor

configure_korean_font()

RESULTS_DIR = base_dir / "0_result"
RESULTS_DIR.mkdir(exist_ok=True)

def evaluate_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = Compose([Resize((256, 256)), ToTensor()])

    test_dir = base_dir / 'data' / '테스트'
    test_dataset = SimpleImageFolder(root=test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    class_names = test_dataset.classes

    # 모델 불러오기
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(RESULTS_DIR / 'cnn_model.pth', map_location=device))
    model.eval()

    y_true, y_pred, y_scores = [], [], []

    print("=== 모델 평가 진행 중 ===")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            y_true.extend(labels.numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_scores.extend(probs[:, 1].cpu().numpy())

    # 결과 리포트 출력
    print("\n[Classification Report]")
    report_text = classification_report(y_true, y_pred, target_names=class_names)
    print(report_text)
    report_path = RESULTS_DIR / "step3_classification_report.txt"
    report_path.write_text(report_text, encoding="utf-8")
    print(f"분류 리포트 저장 완료: {report_path}")

    # Confusion Matrix 시각화
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('혼동 행렬 (Confusion Matrix)')
    plt.ylabel('실제 라벨')
    plt.xlabel('예측 라벨')
    cm_path = RESULTS_DIR / "step3_confusion_matrix.png"
    plt.tight_layout()
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"혼동 행렬 저장 완료: {cm_path}")

    # ROC Curve 시각화
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f'CNN (AUC = {roc_auc:.2f})', color='red')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    roc_path = RESULTS_DIR / "step3_roc_curve.png"
    plt.tight_layout()
    plt.savefig(roc_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"ROC 곡선 저장 완료: {roc_path}")
    metrics = {
        "roc_auc": float(roc_auc),
        "confusion_matrix": cm.tolist(),
    }
    metrics_path = RESULTS_DIR / "step3_evaluation_metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(
        {
            "y_true": y_true,
            "y_pred": y_pred,
            "y_score": y_scores,
        }
    ).to_csv(RESULTS_DIR / "step3_predictions.csv", index=False)
    print(f"평가 지표 저장 완료: {metrics_path}")

if __name__ == '__main__':
    evaluate_model()
