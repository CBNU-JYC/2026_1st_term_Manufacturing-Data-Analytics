import os
import json
from pathlib import Path

base_dir = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(base_dir / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(base_dir / ".cache"))

import torch
import torch.nn.functional as F

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# train.py에 정의된 모델 구조 임포트
from plot_utils import configure_korean_font
from step2_train import SimpleCNN 
from simple_vision import Compose, Resize, ToTensor

configure_korean_font()

RESULTS_DIR = base_dir / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# --- Grad-CAM 구현 클래스 ---
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # 모델의 순방향, 역방향 진행 시 특징 맵과 기울기를 저장하도록 Hook 등록
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx):
        self.model.eval()
        
        # 순방향 전파
        output = self.model(x)
        self.model.zero_grad()
        
        # 모델이 예측한 클래스의 점수에 대해 역전파 수행
        score = output[:, class_idx]
        score.backward(retain_graph=True)
        
        # 저장된 기울기와 특징 맵 가져오기
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        
        # 기울기의 평균을 구하여 각 채널의 가중치(Weight) 계산
        weights = np.mean(gradients, axis=(1, 2))
        
        # 가중치와 특징 맵을 곱하여 조합 (Linear Combination)
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        # 양의 영향력을 주는 부분만 남기기 위해 ReLU 적용
        cam = np.maximum(cam, 0)
        
        # 0 ~ 1 사이로 정규화
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-8)
        
        return cam
# -----------------------------

def predict_single_image_with_cam(image_path, model_path, class_names):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = Compose([Resize((256, 256)), ToTensor()])
    
    try:
        img_pil = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"이미지 로드 실패: {e}")
        return
    
    img_tensor = transform(img_pil).unsqueeze(0).to(device) 
    
    # 모델 불러오기
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() 
    
    # 1. 예측 수행
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)
        prob_max, predicted = torch.max(probs, 1)
        
        pred_idx = predicted.item()
        pred_class = class_names[pred_idx]
        confidence = prob_max.item() * 100
        
    print(f">>> 분석 결과: [{pred_class}] (확신도: {confidence:.2f}%)")

    # 2. Grad-CAM 추출
    # SimpleCNN의 마지막 Convolution 레이어를 타겟으로 설정합니다.
    # self.conv_layers 구조: 0:Conv2d, 1:ReLU, 2:MaxPool, 3:Conv2d, 4:ReLU, 5:MaxPool
    target_layer = model.conv_layers[3] 
    
    cam_extractor = GradCAM(model, target_layer)
    cam = cam_extractor(img_tensor, pred_idx)
    
    # 3. 원본 이미지와 Heatmap 합성 시각화
    img_np = np.array(img_pil.resize((256, 256))) # 원본 이미지를 256x256으로 변환
    cam_resized = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize((256, 256)))

    # Matplotlib의 컬러맵을 사용해 OpenCV 없이 히트맵 생성
    heatmap = plt.get_cmap('jet')(cam_resized / 255.0)[..., :3]
    heatmap = (heatmap * 255).astype(np.uint8)

    # 원본 이미지와 히트맵을 6:4 비율로 투명하게 겹치기
    overlay = (img_np * 0.6 + heatmap * 0.4).clip(0, 255).astype(np.uint8)
    
    # 화면에 띄우기
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img_np)
    axes[0].set_title("원본 제품 이미지")
    axes[0].axis('off')
    
    axes[1].imshow(heatmap)
    axes[1].set_title("판단 주요 영역 (히트맵)")
    axes[1].axis('off')
    
    axes[2].imshow(overlay)
    axes[2].set_title(f"합성 결과: {pred_class} ({confidence:.1f}%)")
    axes[2].axis('off')
    
    plt.tight_layout()
    save_path = RESULTS_DIR / "step4_inference_gradcam.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"추론 시각화 저장 완료: {save_path}")
    result = {
        "image_path": str(image_path),
        "predicted_class": pred_class,
        "confidence": round(float(confidence), 4),
        "saved_figure": str(save_path),
    }
    result_path = RESULTS_DIR / "step4_inference_result.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"추론 결과 저장 완료: {result_path}")

if __name__ == '__main__':
    # 실습 시 이 경로를 변경하도록 지도해 주세요.
    # TARGET_IMAGE = './data/평가/sample_ok.png' 
    TARGET_IMAGE = base_dir / 'data' / '테스트' / '불량' / 'KEMP_IMG_DATA_Error_55.png'

    MODEL_PATH = base_dir / 'cnn_model.pth'
    CLASSES = ['불량', '정상'] 
    
    predict_single_image_with_cam(TARGET_IMAGE, MODEL_PATH, CLASSES)
