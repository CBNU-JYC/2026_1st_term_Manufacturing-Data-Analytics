import os
import json
from pathlib import Path

base_dir = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(base_dir / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(base_dir / ".cache"))

import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from plot_utils import configure_korean_font
from simple_vision import Compose, Resize, SimpleImageFolder, ToTensor

configure_korean_font()

RESULTS_DIR = base_dir / "results"
RESULTS_DIR.mkdir(exist_ok=True)

def analyze_data(train_dir, test_dir):
    # 단순 시각화를 위해 텐서로만 변환
    transform = Compose([Resize((256, 256)), ToTensor()])

    # 데이터 로드
    train_dataset = SimpleImageFolder(root=train_dir, transform=transform)
    test_dataset = SimpleImageFolder(root=test_dir, transform=transform)

    print("=== 데이터 분석 결과 ===")
    print(f"클래스 매핑: {train_dataset.class_to_idx}")
    print(f"학습 데이터 개수: {len(train_dataset)}개")
    print(f"테스트 데이터 개수: {len(test_dataset)}개")
    summary = {
        "class_to_idx": train_dataset.class_to_idx,
        "train_count": len(train_dataset),
        "test_count": len(test_dataset),
    }
    summary_path = RESULTS_DIR / "step1_data_analysis_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"데이터 분석 요약 저장 완료: {summary_path}")

    # 샘플 이미지 시각화 (첫 번째 배치 로드)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    images, labels = next(iter(train_loader))
    class_names = train_dataset.classes

    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    for i in range(4):
        # PyTorch 텐서 (C, H, W)를 Matplotlib 형태 (H, W, C)로 변환
        img = images[i].permute(1, 2, 0).numpy()
        axes[i].imshow(img)
        axes[i].set_title(f"Label: {class_names[labels[i]]}")
        axes[i].axis('off')
        
    plt.suptitle("학습 데이터 샘플 확인")
    plt.tight_layout()
    save_path = RESULTS_DIR / "step1_train_samples.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"샘플 이미지 저장 완료: {save_path}")

if __name__ == '__main__':
    TRAIN_DIR = base_dir / 'data' / '학습'
    TEST_DIR = base_dir / 'data' / '테스트'
    analyze_data(TRAIN_DIR, TEST_DIR)
