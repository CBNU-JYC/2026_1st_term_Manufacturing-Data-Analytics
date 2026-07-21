"""
이 파일은 학습된 CNN 모델로 한 장의 이미지를 예측하고,
Grad-CAM으로 모델이 어디를 보고 판단했는지 시각화하는 단계입니다.

전체 흐름:
1. 환경 설정과 필요한 라이브러리를 준비합니다.
2. Grad-CAM 클래스를 만들어 특징맵과 기울기를 저장합니다.
3. 한 장의 이미지를 전처리한 뒤 모델에 넣어 예측합니다.
4. 예측된 클래스를 기준으로 Grad-CAM 히트맵을 만듭니다.
5. 원본 이미지, 히트맵, 합성 이미지를 화면에 보여 줍니다.
"""

# 환경 변수를 다루기 위해 os를 불러옵니다.
import os

# 파일 경로를 쉽게 다루기 위해 Path를 불러옵니다.
from pathlib import Path

# 현재 설명 파일이 있는 폴더를 기준 경로로 저장합니다.
base_dir = Path(__file__).resolve().parent
RESULTS_DIR = base_dir / "0_result"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Matplotlib 설정 폴더를 프로젝트 안으로 정해 둡니다.
os.environ.setdefault("MPLCONFIGDIR", str(base_dir / ".mplconfig"))

# 글꼴 캐시 폴더도 프로젝트 안으로 정합니다.
os.environ.setdefault("XDG_CACHE_HOME", str(base_dir / ".cache"))

# 딥러닝 계산용 torch를 불러옵니다.
import torch

# softmax를 쓰기 위해 함수 모음을 불러옵니다.
import torch.nn.functional as F

# 이미지 읽기와 그래프 표시 도구를 불러옵니다.
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# 설명 버전 모듈들을 불러옵니다.
from z_explanation_plot_utils import configure_korean_font
from z_explanation_step2_train import SimpleCNN
from z_explanation_simple_vision import Compose, Resize, ToTensor

# 그래프에서 한글이 깨지지 않도록 설정합니다.
configure_korean_font()


class GradCAM:
    """
    모델이 어디를 보고 판단했는지 히트맵으로 보여 주는 도구입니다.

    매개변수:
        model: 이미 학습된 신경망 모델
        target_layer: 히트맵을 만들 때 기준이 될 합성곱 층

    반환값:
        없음
    """

    def __init__(self, model, target_layer):
        """
        Grad-CAM 계산에 필요한 저장 공간과 훅(hook)을 준비합니다.

        매개변수:
            model: 사용할 신경망 모델
            target_layer: 특징맵과 기울기를 저장할 대상 층

        반환값:
            없음
        """
        # 모델과 타깃 층을 저장합니다.
        self.model = model
        self.target_layer = target_layer

        # 나중에 저장할 기울기와 특징맵 자리를 미리 준비합니다.
        self.gradients = None
        self.activations = None

        # 순전파 때 특징맵을 저장하는 훅을 등록합니다.
        target_layer.register_forward_hook(self.save_activation)

        # 역전파 때 기울기를 저장하는 훅을 등록합니다.
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        """
        순전파 때 나온 특징맵을 저장합니다.

        매개변수:
            module: 훅이 걸린 층
            input: 그 층의 입력값
            output: 그 층의 출력값

        반환값:
            없음
        """
        # 나중에 히트맵을 만들기 위해 출력 특징맵을 저장합니다.
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        """
        역전파 때 나온 기울기를 저장합니다.

        매개변수:
            module: 훅이 걸린 층
            grad_input: 그 층 입력 쪽의 기울기
            grad_output: 그 층 출력 쪽의 기울기

        반환값:
            없음
        """
        # 출력 쪽 기울기를 저장합니다.
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx):
        """
        입력 이미지와 목표 클래스 번호를 받아 Grad-CAM 맵을 만듭니다.

        매개변수:
            x: 모델에 넣을 이미지 텐서
            class_idx: 설명하고 싶은 예측 클래스 번호

        반환값:
            0~1 범위로 정규화된 CAM 배열
        """
        # 모델을 평가 모드로 둡니다.
        self.model.eval()

        # 이미지를 모델에 넣어 예측 점수를 계산합니다.
        output = self.model(x)

        # 이전 계산의 기울기를 지워 새로 계산할 준비를 합니다.
        self.model.zero_grad()

        # 관심 있는 클래스의 점수만 뽑습니다.
        score = output[:, class_idx]

        # 그 점수를 기준으로 역전파를 실행합니다.
        score.backward(retain_graph=True)

        # 저장해 둔 기울기와 특징맵을 NumPy 배열로 꺼냅니다.
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]

        # 각 채널 기울기의 평균을 구해 채널별 중요도를 계산합니다.
        weights = np.mean(gradients, axis=(1, 2))

        # 특징맵들을 중요도만큼 더해 CAM을 만듭니다.
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # 음수 부분은 없애서 중요한 양의 영역만 남깁니다.
        cam = np.maximum(cam, 0)

        # 보기 좋게 0~1 범위로 맞춥니다.
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-8)

        # 완성된 CAM을 돌려줍니다.
        return cam


def predict_single_image_with_cam(image_path, model_path, class_names):
    """
    한 장의 이미지를 예측하고 Grad-CAM 히트맵을 보여 줍니다.

    매개변수:
        image_path: 예측할 이미지 파일 경로
        model_path: 학습된 모델 가중치 파일 경로
        class_names: 클래스 이름 목록

    반환값:
        없음
    """
    # GPU가 있으면 GPU를 사용하고, 없으면 CPU를 사용합니다.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 학습 때와 같은 방식으로 이미지 전처리를 준비합니다.
    transform = Compose([Resize((256, 256)), ToTensor()])

    # 이미지를 열어 RGB 색상 형식으로 맞춥니다.
    try:
        img_pil = Image.open(image_path).convert("RGB")
    except Exception as e:
        # 파일이 없거나 열 수 없으면 이유를 보여 주고 종료합니다.
        print(f"이미지 로드 실패: {e}")
        return

    # 모델이 받을 수 있도록 이미지에 전처리를 적용하고 배치 차원을 하나 추가합니다.
    img_tensor = transform(img_pil).unsqueeze(0).to(device)

    # 모델 구조를 만들고 저장된 가중치를 불러옵니다.
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 예측 단계에서는 기울기를 계산할 필요가 없어 속도를 위해 꺼 둡니다.
    with torch.no_grad():
        # 한 장의 이미지에 대한 예측 점수를 계산합니다.
        outputs = model(img_tensor)

        # softmax로 클래스별 확률처럼 바꿉니다.
        probs = F.softmax(outputs, dim=1)

        # 가장 높은 확률의 클래스와 그 값을 가져옵니다.
        prob_max, predicted = torch.max(probs, 1)

        # 텐서 값을 일반 숫자로 바꿉니다.
        pred_idx = predicted.item()
        pred_class = class_names[pred_idx]
        confidence = prob_max.item() * 100

    # 사람이 읽기 쉬운 예측 결과를 출력합니다.
    print(f">>> 분석 결과: [{pred_class}] (확신도: {confidence:.2f}%)")

    # 마지막 합성곱 층을 Grad-CAM의 기준 층으로 선택합니다.
    target_layer = model.conv_layers[3]

    # Grad-CAM 도구를 만듭니다.
    cam_extractor = GradCAM(model, target_layer)

    # 예측한 클래스 기준으로 히트맵을 계산합니다.
    cam = cam_extractor(img_tensor, pred_idx)

    # 원본 이미지를 256x256 크기로 바꿔 배열로 만듭니다.
    img_np = np.array(img_pil.resize((256, 256)))

    # CAM도 같은 크기로 맞춥니다.
    cam_resized = np.array(
        Image.fromarray((cam * 255).astype(np.uint8)).resize((256, 256))
    )

    # Matplotlib 색상표를 사용해 CAM을 컬러 히트맵으로 바꿉니다.
    heatmap = plt.get_cmap("jet")(cam_resized / 255.0)[..., :3]
    heatmap = (heatmap * 255).astype(np.uint8)

    # 원본 이미지와 히트맵을 섞어서 어디를 봤는지 더 쉽게 보이게 합니다.
    overlay = (img_np * 0.6 + heatmap * 0.4).clip(0, 255).astype(np.uint8)

    # 원본, 히트맵, 합성 결과를 나란히 그릴 그래프 판을 만듭니다.
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 첫 번째 칸에 원본 이미지를 보여 줍니다.
    axes[0].imshow(img_np)
    axes[0].set_title("원본 제품 이미지")
    axes[0].axis("off")

    # 두 번째 칸에 히트맵을 보여 줍니다.
    axes[1].imshow(heatmap)
    axes[1].set_title("판단 주요 영역 (히트맵)")
    axes[1].axis("off")

    # 세 번째 칸에 합성 결과를 보여 줍니다.
    axes[2].imshow(overlay)
    axes[2].set_title(f"합성 결과: {pred_class} ({confidence:.1f}%)")
    axes[2].axis("off")

    # 그림 간격을 자동으로 정리합니다.
    plt.tight_layout()

    # 그래프를 화면에 보여 줍니다.
    plt.show()


# 이 파일을 직접 실행했을 때만 아래 예시 추론을 수행합니다.
if __name__ == "__main__":
    # 기본으로 사용할 테스트 이미지 경로를 정합니다.
    TARGET_IMAGE = base_dir / "data" / "테스트" / "불량" / "KEMP_IMG_DATA_Error_55.png"

    # 불러올 모델 파일 경로를 정합니다.
    MODEL_PATH = RESULTS_DIR / "cnn_model.pth"

    # 클래스 이름 순서를 정합니다.
    CLASSES = ["불량", "정상"]

    # 한 장 이미지 예측과 Grad-CAM 시각화를 실행합니다.
    predict_single_image_with_cam(TARGET_IMAGE, MODEL_PATH, CLASSES)

