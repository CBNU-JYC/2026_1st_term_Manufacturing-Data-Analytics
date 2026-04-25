"""
이 파일은 학습된 CNN 모델이 테스트 데이터에서 얼마나 잘 맞히는지 평가하는 단계입니다.

전체 흐름:
1. 그래프와 모델 평가에 필요한 라이브러리를 불러옵니다.
2. 테스트 데이터를 같은 전처리 방식으로 준비합니다.
3. 저장된 CNN 모델을 불러옵니다.
4. 모델 예측 결과를 모아 분류 리포트를 출력합니다.
5. 혼동 행렬과 ROC 곡선을 그래프로 보여 줍니다.
"""

# 환경 변수를 다루기 위해 os를 불러옵니다.
import os

# 파일 경로를 쉽게 다루기 위해 Path를 불러옵니다.
from pathlib import Path

# 현재 설명 파일이 있는 폴더를 기준 경로로 저장합니다.
base_dir = Path(__file__).resolve().parent

# Matplotlib 설정 파일을 저장할 폴더를 프로젝트 안으로 잡습니다.
os.environ.setdefault("MPLCONFIGDIR", str(base_dir / ".mplconfig"))

# 글꼴 캐시 폴더도 프로젝트 안으로 잡아 경고를 줄입니다.
os.environ.setdefault("XDG_CACHE_HOME", str(base_dir / ".cache"))

# 딥러닝 계산을 위해 torch를 불러옵니다.
import torch

# 데이터를 묶음으로 읽기 위한 DataLoader를 불러옵니다.
from torch.utils.data import DataLoader

# 숫자 계산, 그래프, 시각화 도구를 불러옵니다.
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 평가 지표 계산 함수들을 불러옵니다.
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

# 설명 버전의 다른 모듈들을 불러옵니다.
from z_explanation_plot_utils import configure_korean_font
from z_explanation_step2_train import SimpleCNN
from z_explanation_simple_vision import Compose, Resize, SimpleImageFolder, ToTensor

# 그래프에서 한글이 잘 보이도록 설정합니다.
configure_korean_font()


def evaluate_model():
    """
    저장된 CNN 모델을 테스트 데이터로 평가합니다.

    매개변수:
        없음

    반환값:
        없음
    """
    # GPU가 있으면 GPU를 사용하고, 없으면 CPU를 사용합니다.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 평가할 때도 학습 때와 똑같은 전처리를 해야 공정합니다.
    transform = Compose([Resize((256, 256)), ToTensor()])

    # 테스트 데이터 폴더 경로를 만듭니다.
    test_dir = base_dir / "data" / "테스트"

    # 테스트 데이터셋을 읽습니다.
    test_dataset = SimpleImageFolder(root=test_dir, transform=transform)

    # 데이터를 16장씩 순서대로 읽는 로더를 만듭니다.
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # 클래스 이름 목록을 저장합니다.
    class_names = test_dataset.classes

    # 학습 때와 같은 구조의 모델을 만듭니다.
    model = SimpleCNN().to(device)

    # 저장된 가중치를 모델에 불러옵니다.
    model.load_state_dict(torch.load(base_dir / "cnn_model.pth", map_location=device))

    # 평가 모드로 바꿉니다.
    model.eval()

    # 정답, 예측, 확률 점수를 모아 둘 리스트를 만듭니다.
    y_true, y_pred, y_scores = [], [], []

    # 평가 시작 안내를 출력합니다.
    print("=== 모델 평가 진행 중 ===")

    # 평가 때는 기울기 계산이 필요 없어서 속도를 위해 꺼 둡니다.
    with torch.no_grad():
        # 테스트 데이터 묶음을 하나씩 처리합니다.
        for images, labels in test_loader:
            # 이미지를 계산 장치로 보냅니다.
            images = images.to(device)

            # 모델 예측 점수를 계산합니다.
            outputs = model(images)

            # softmax로 각 클래스의 확률처럼 바꿉니다.
            probs = torch.nn.functional.softmax(outputs, dim=1)

            # 가장 큰 점수를 가진 클래스를 최종 예측으로 고릅니다.
            _, predicted = torch.max(outputs, 1)

            # 정답 라벨을 리스트에 추가합니다.
            y_true.extend(labels.numpy())

            # 모델의 예측 라벨을 리스트에 추가합니다.
            y_pred.extend(predicted.cpu().numpy())

            # 두 번째 클래스의 확률을 점수로 저장합니다.
            y_scores.extend(probs[:, 1].cpu().numpy())

    # 글로 된 분류 리포트를 출력합니다.
    print("\n[Classification Report]")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # 혼동 행렬을 계산합니다.
    cm = confusion_matrix(y_true, y_pred)

    # 혼동 행렬을 그릴 그래프 창을 만듭니다.
    plt.figure(figsize=(6, 5))

    # heatmap으로 보기 쉽게 색칠해 보여 줍니다.
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )

    # 그래프 제목과 축 이름을 붙입니다.
    plt.title("혼동 행렬 (Confusion Matrix)")
    plt.ylabel("실제 라벨")
    plt.xlabel("예측 라벨")

    # 혼동 행렬 그래프를 화면에 보여 줍니다.
    plt.show()

    # ROC 곡선을 계산하기 위해 FPR, TPR, 임계값을 구합니다.
    fpr, tpr, _ = roc_curve(y_true, y_scores)

    # ROC 곡선 아래 면적(AUC)을 계산합니다.
    roc_auc = auc(fpr, tpr)

    # ROC 그래프 창을 만듭니다.
    plt.figure(figsize=(6, 6))

    # 모델의 ROC 곡선을 빨간색으로 그립니다.
    plt.plot(fpr, tpr, label=f"CNN (AUC = {roc_auc:.2f})", color="red")

    # 아무 정보가 없는 랜덤 기준선을 점선으로 그립니다.
    plt.plot([0, 1], [0, 1], "k--")

    # 그래프 제목과 축 이름을 붙입니다.
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()

    # ROC 곡선을 화면에 보여 줍니다.
    plt.show()


# 이 파일을 직접 실행했을 때만 평가를 시작합니다.
if __name__ == "__main__":
    evaluate_model()

