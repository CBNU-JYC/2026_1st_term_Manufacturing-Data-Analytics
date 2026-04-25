from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        for transform in self.transforms:
            image = transform(image)
        return image


class Resize:
    def __init__(self, size):
        self.size = size
        resampling = getattr(Image, "Resampling", Image)
        self.resample = resampling.BILINEAR

    def __call__(self, image):
        return image.resize(self.size, self.resample)


class ToTensor:
    def __call__(self, image):
        array = np.asarray(image, dtype=np.float32) / 255.0
        array = np.transpose(array, (2, 0, 1))
        return torch.from_numpy(array)


class SimpleImageFolder(Dataset):
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform
        if not self.root.exists():
            raise FileNotFoundError(f"데이터 폴더를 찾을 수 없습니다: {self.root}")

        self.classes = sorted(
            [path.name for path in self.root.iterdir() if path.is_dir()]
        )
        if not self.classes:
            raise ValueError(f"클래스 폴더가 없습니다: {self.root}")

        self.class_to_idx = {
            class_name: index for index, class_name in enumerate(self.classes)
        }
        self.samples = []

        for class_name in self.classes:
            class_dir = self.root / class_name
            for image_path in sorted(class_dir.rglob("*")):
                if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    self.samples.append((image_path, self.class_to_idx[class_name]))

        if not self.samples:
            raise ValueError(f"이미지 파일이 없습니다: {self.root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label
