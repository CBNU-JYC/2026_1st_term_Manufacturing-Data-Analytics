import os
import glob
import json
from pathlib import Path

base_dir = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(base_dir / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(base_dir / ".cache"))

import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from simple_vision import Compose, Resize, ToTensor, resolve_data_dir

RESULTS_DIR = base_dir / "results"
RESULTS_DIR.mkdir(exist_ok=True)

class MVTecDataset(Dataset):
    """MVTec AD 데이터셋 로더"""
    def __init__(self, root_dir, category, is_train=True, transform=None):
        self.transform = transform
        self.image_paths = []
        self.labels = [] # 0: Normal, 1: Anomaly
        
        if is_train:
            img_dir = os.path.join(root_dir, category, 'train', 'good')
            paths = glob.glob(os.path.join(img_dir, '*.png'))
            self.image_paths.extend(paths)
            self.labels.extend([0] * len(paths))
        else:
            test_dir = os.path.join(root_dir, category, 'test')
            for defect_type in os.listdir(test_dir):
                paths = glob.glob(os.path.join(test_dir, defect_type, '*.png'))
                self.image_paths.extend(paths)
                label = 0 if defect_type == 'good' else 1
                self.labels.extend([label] * len(paths))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label, img_path

def show_eda_images(dataset, num_images=4, save_path=None):
    """데이터 샘플 시각화 함수"""
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
    for i in range(num_images):
        img, label, _ = dataset[i]
        img_np = img.permute(1, 2, 0).numpy() # Tensor to Numpy
        title = "Normal" if label == 0 else "Anomaly"
        axes[i].imshow(img_np)
        axes[i].set_title(title)
        axes[i].axis('off')
    plt.suptitle("MVTec AD Dataset EDA", fontsize=16)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"EDA 샘플 이미지 저장 완료: {save_path}")
    plt.close(fig)

if __name__ == "__main__":
    # TODO: 학생들은 본인의 데이터셋 경로에 맞게 아래 변수를 수정하세요.
    ROOT_DIR = resolve_data_dir(base_dir)
    CATEGORY = 'bottle' 

    transform = Compose([Resize((256, 256)), ToTensor()])

    print("데이터셋 로딩 중...")
    test_dataset = MVTecDataset(ROOT_DIR, CATEGORY, is_train=False, transform=transform)
    print(f"테스트 데이터 개수: {len(test_dataset)}장")
    summary = {
        "category": CATEGORY,
        "test_image_count": len(test_dataset),
        "sample_image_paths": test_dataset.image_paths[:4],
    }
    summary_path = RESULTS_DIR / "step1_data_eda_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"EDA 요약 저장 완료: {summary_path}")
    
    # 샘플 이미지 확인
    show_eda_images(test_dataset, save_path=RESULTS_DIR / "step1_data_eda_samples.png")
