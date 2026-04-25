"""
이 파일은 학습된 오토인코더로 MVTec 테스트 데이터를 평가하는 단계입니다.

전체 흐름:
1. 테스트 이미지를 모델에 넣어 복원합니다.
2. 원본과 복원 이미지의 차이로 오차 맵을 만듭니다.
3. 오차 맵에서 가장 큰 값을 이미지의 이상 점수로 사용합니다.
4. AUROC, F1, 최적 임계값을 계산합니다.
5. 불량 샘플의 히트맵을 시각화해 모델이 어디를 이상하다고 봤는지 보여 줍니다.
"""

# 환경 변수와 경로 도구를 불러옵니다.
import os
from pathlib import Path

# 현재 설명 파일이 있는 폴더를 기준 경로로 저장합니다.
base_dir = Path(__file__).resolve().parent

# Matplotlib 설정 파일과 캐시 폴더를 프로젝트 안으로 정합니다.
os.environ.setdefault("MPLCONFIGDIR", str(base_dir / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(base_dir / ".cache"))

# 딥러닝, 숫자 계산, 그래프 도구를 불러옵니다.
import torch
import numpy as np
import matplotlib.pyplot as plt

# 평가 지표 계산 함수들을 불러옵니다.
from sklearn.metrics import roc_auc_score, precision_recall_curve

# 이미지 처리용 PIL을 불러옵니다.
from PIL import Image, ImageFilter

# 데이터를 묶음으로 읽기 위한 DataLoader를 불러옵니다.
from torch.utils.data import DataLoader

# 설명 버전의 다른 파일들을 불러옵니다.
from z_explanation_step1_data_eda import MVTecDataset
from z_explanation_step2_train import ConvAutoencoder
from z_explanation_simple_vision import Compose, Resize, ToTensor, resolve_data_dir


def blur_error_map(error_map, radius=4):
    """
    오차 맵의 작은 잡음을 줄이기 위해 부드럽게 흐리게 만듭니다.

    매개변수:
        error_map: 원본과 복원 이미지의 차이를 담은 2차원 배열
        radius: 얼마나 부드럽게 흐릴지 정하는 숫자

    반환값:
        0~1 범위로 정규화되고 블러 처리된 오차 맵 배열
    """
    # 최소값을 빼서 가장 작은 값을 0으로 맞춥니다.
    normalized = error_map - error_map.min()

    # 최대값으로 나누어 0~1 범위로 정규화합니다.
    normalized = normalized / (normalized.max() + 1e-8)

    # PIL 이미지로 바꾼 뒤 가우시안 블러를 적용합니다.
    blurred = Image.fromarray((normalized * 255).astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(radius=radius)
    )

    # 다시 NumPy 배열로 바꾸고 0~1 범위로 맞춰 돌려줍니다.
    return np.asarray(blurred, dtype=np.float32) / 255.0


def evaluate_performance(model, test_loader, device):
    """
    테스트 데이터 전체를 사용해 AUROC, F1, 임계값을 계산합니다.

    매개변수:
        model: 학습된 오토인코더 모델
        test_loader: 테스트 데이터 로더
        device: 계산 장치(cpu 또는 cuda)

    반환값:
        최적으로 계산된 임계값
    """
    # 평가 모드로 전환합니다.
    model.eval()

    # 정답 라벨과 이상 점수를 담을 리스트를 만듭니다.
    y_true = []
    y_scores = []

    # 평가 시작 안내를 출력합니다.
    print("전체 테스트 데이터셋 정량 평가를 진행합니다...")

    # 평가 때는 기울기 계산이 필요 없습니다.
    with torch.no_grad():
        # 테스트 이미지 한 장씩 확인합니다.
        for images, labels, _ in test_loader:
            # 이미지를 계산 장치로 보냅니다.
            images = images.to(device)

            # 모델이 이미지를 복원합니다.
            outputs = model(images)

            # 픽셀별 평균제곱오차를 계산해 오차 맵을 만듭니다.
            error = torch.mean((images - outputs) ** 2, dim=1)
            error_map = error.squeeze().cpu().numpy()

            # 작은 잡음을 줄이기 위해 오차 맵을 부드럽게 만듭니다.
            error_map = blur_error_map(error_map)

            # 가장 큰 오차를 이 이미지의 이상 점수로 사용합니다.
            anomaly_score = np.max(error_map)

            # 점수와 정답을 리스트에 저장합니다.
            y_scores.append(anomaly_score)
            y_true.append(labels.item())

    # AUROC는 전체적인 구분 능력을 보여 주는 점수입니다.
    auroc = roc_auc_score(y_true, y_scores)

    # 다양한 임계값에서 precision과 recall을 계산합니다.
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)

    # precision과 recall을 합쳐 F1 점수를 계산합니다.
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)

    # 가장 좋은 F1 점수 위치를 찾습니다.
    best_idx = np.argmax(f1_scores)
    best_f1 = f1_scores[best_idx]
    best_threshold = thresholds[best_idx]

    # 결과를 보기 좋게 출력합니다.
    print("-" * 40)
    print("[전체 평가 결과]")
    print(f"AUROC Score          : {auroc:.4f}")
    print(f"Best F1-Score        : {best_f1:.4f}")
    print(f"Optimal Threshold    : {best_threshold:.4f}")
    print("-" * 40)

    # 나중에 판정에 쓸 최적 임계값을 돌려줍니다.
    return best_threshold


def visualize_anomaly(model, test_loader, device, threshold, num_samples=3):
    """
    불량 샘플 몇 장을 골라 원본, 복원 결과, 오차 맵, 히트맵을 보여 줍니다.

    매개변수:
        model: 학습된 오토인코더 모델
        test_loader: 테스트 데이터 로더
        device: 계산 장치
        threshold: 정상/불량을 나눌 기준 점수
        num_samples: 몇 장을 보여 줄지 정하는 숫자

    반환값:
        없음
    """
    # 평가 모드로 전환합니다.
    model.eval()

    # 몇 장 보여 줬는지 셀 변수를 만듭니다.
    samples_shown = 0

    # 어떤 임계값을 쓰는지 안내합니다.
    print(f"\n최적 임계값({threshold:.4f})을 적용하여 시각화를 시작합니다.")

    # 평가 단계이므로 기울기를 계산하지 않습니다.
    with torch.no_grad():
        # 테스트 데이터를 한 장씩 확인합니다.
        for images, labels, _ in test_loader:
            # 정상 이미지는 건너뛰고, 불량 이미지 위주로 보여 줍니다.
            if labels.item() == 0:
                continue

            # 계산 장치로 보냅니다.
            images = images.to(device)

            # 모델이 이미지를 복원합니다.
            outputs = model(images)

            # 원본과 복원 이미지의 차이를 계산합니다.
            error = torch.mean((images - outputs) ** 2, dim=1)
            error_map = error.squeeze().cpu().numpy()
            error_map = blur_error_map(error_map)

            # 가장 큰 오차를 이상 점수로 씁니다.
            anomaly_score = np.max(error_map)

            # 임계값보다 크면 불량, 작으면 정상으로 판정합니다.
            prediction = "NG (Defect)" if anomaly_score >= threshold else "OK (Normal)"

            # 히트맵을 만들기 전에 0~1 범위로 다시 맞춥니다.
            error_map_norm = error_map - error_map.min()
            error_map_norm = error_map_norm / (error_map_norm.max() + 1e-8)

            # 컬러맵을 적용해 오차 맵을 컬러 히트맵으로 바꿉니다.
            heatmap = plt.get_cmap("jet")(error_map_norm)[..., :3]
            heatmap = (heatmap * 255).astype(np.uint8)

            # 원본 이미지와 복원 이미지를 NumPy 배열로 바꿉니다.
            img_np = images.squeeze().cpu().permute(1, 2, 0).numpy()
            out_np = outputs.squeeze().cpu().permute(1, 2, 0).numpy()

            # 원본 이미지와 히트맵을 반반 섞어 오버레이 이미지를 만듭니다.
            overlay = (
                (img_np * 255).astype(np.uint8) * 0.5 + heatmap * 0.5
            ).clip(0, 255).astype(np.uint8)

            # 4개의 그림을 나란히 보여 줄 그래프 판을 만듭니다.
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))

            # 원본 이미지를 보여 주고 점수와 판정도 제목에 적습니다.
            axes[0].imshow(img_np)
            axes[0].set_title(f"Original\nScore: {anomaly_score:.4f} -> {prediction}")

            # 복원 이미지를 보여 줍니다.
            axes[1].imshow(out_np)
            axes[1].set_title("Reconstructed")

            # 오차 맵을 뜨거운 색상으로 보여 줍니다.
            axes[2].imshow(error_map, cmap="hot")
            axes[2].set_title("Error Map")

            # 히트맵을 합친 결과를 보여 줍니다.
            axes[3].imshow(overlay)
            axes[3].set_title("Overlay Heatmap")

            # 눈금 숫자는 숨겨 그림만 보이게 합니다.
            for ax in axes:
                ax.axis("off")

            # 그래프를 화면에 보여 줍니다.
            plt.show()

            # 한 장을 보여 줬으니 개수를 1 증가시킵니다.
            samples_shown += 1

            # 원하는 개수만큼 보여 줬으면 멈춥니다.
            if samples_shown >= num_samples:
                break


# 이 파일을 직접 실행했을 때만 평가를 시작합니다.
if __name__ == "__main__":
    # 데이터 폴더를 찾습니다.
    ROOT_DIR = resolve_data_dir(base_dir)

    # 사용할 카테고리와 모델 파일 경로를 정합니다.
    CATEGORY = "bottle"
    MODEL_PATH = base_dir / "autoencoder_model.pth"

    # 이미지 전처리를 준비합니다.
    transform = Compose([Resize((256, 256)), ToTensor()])

    # 테스트 데이터셋과 데이터 로더를 만듭니다.
    test_dataset = MVTecDataset(ROOT_DIR, CATEGORY, is_train=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 계산 장치를 정하고 모델을 불러옵니다.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvAutoencoder().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    # 먼저 정량 평가를 수행해 최적 임계값을 구합니다.
    optimal_thresh = evaluate_performance(model, test_loader, device)

    # 그 임계값으로 실제 시각화를 진행합니다.
    visualize_anomaly(model, test_loader, device, optimal_thresh, num_samples=3)

