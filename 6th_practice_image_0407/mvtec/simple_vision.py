from pathlib import Path

import numpy as np
import torch
from PIL import Image


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


def resolve_data_dir(base_dir, folder_name="mvtec_ad"):
    data_dir = Path(base_dir) / folder_name
    if not data_dir.exists():
        raise FileNotFoundError(f"데이터 폴더를 찾을 수 없습니다: {data_dir}")
    return data_dir
