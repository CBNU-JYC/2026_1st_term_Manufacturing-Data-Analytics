"""
이 파일은 torchvision 없이도 이미지를 읽고, 크기를 바꾸고, 텐서로 바꾸는 작은 도구 모음입니다.

전체 흐름:
1. 이미지 파일 경로를 다루는 도구를 준비합니다.
2. 이미지를 숫자 배열과 텐서로 바꾸는 도구를 준비합니다.
3. 여러 변환을 차례대로 적용하는 클래스를 만듭니다.
4. 폴더 안의 이미지를 찾아서 라벨과 함께 꺼내 주는 데이터셋 클래스를 만듭니다.
"""

# 파일 경로를 쉽게 다루기 위해 Path를 불러옵니다.
from pathlib import Path

# 숫자 배열 계산을 위해 NumPy를 불러옵니다.
import numpy as np

# 딥러닝에서 사용하는 텐서를 만들기 위해 torch를 불러옵니다.
import torch

# 이미지 파일을 열고 다루기 위해 PIL의 Image를 불러옵니다.
from PIL import Image

# PyTorch 데이터셋 클래스를 만들기 위해 Dataset을 불러옵니다.
from torch.utils.data import Dataset


# 어떤 확장자의 파일을 이미지로 볼지 미리 정해 둡니다.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class Compose:
    """
    여러 이미지 변환을 순서대로 실행하는 상자입니다.

    매개변수:
        transforms: 차례대로 적용할 변환 함수나 클래스 목록

    반환값:
        없음
    """

    def __init__(self, transforms):
        # 나중에 사용할 변환 목록을 저장합니다.
        self.transforms = transforms

    def __call__(self, image):
        """
        저장된 변환들을 이미지에 하나씩 적용합니다.

        매개변수:
            image: 변환할 원본 이미지

        반환값:
            변환이 모두 적용된 이미지
        """
        # 변환 목록을 앞에서부터 차례대로 실행합니다.
        for transform in self.transforms:
            image = transform(image)

        # 모두 바뀐 이미지를 돌려줍니다.
        return image


class Resize:
    """
    이미지를 원하는 크기로 바꾸는 클래스입니다.

    매개변수:
        size: 바꿀 목표 크기, 예를 들면 (256, 256)

    반환값:
        없음
    """

    def __init__(self, size):
        # 목표 크기를 저장합니다.
        self.size = size

        # Pillow 버전에 따라 다른 이름을 쓸 수 있어서 안전하게 처리합니다.
        resampling = getattr(Image, "Resampling", Image)

        # 이미지를 부드럽게 줄이거나 키우기 위한 방법을 저장합니다.
        self.resample = resampling.BILINEAR

    def __call__(self, image):
        """
        이미지를 저장된 크기로 바꿉니다.

        매개변수:
            image: 크기를 바꿀 이미지

        반환값:
            크기가 바뀐 이미지
        """
        # 이미지 크기를 바꿔서 돌려줍니다.
        return image.resize(self.size, self.resample)


class ToTensor:
    """
    PIL 이미지를 PyTorch 텐서로 바꾸는 클래스입니다.

    매개변수:
        없음

    반환값:
        없음
    """

    def __call__(self, image):
        """
        이미지를 0~1 범위의 텐서로 바꿉니다.

        매개변수:
            image: 텐서로 바꿀 이미지

        반환값:
            채널 순서가 (C, H, W)인 torch 텐서
        """
        # 이미지를 NumPy 배열로 바꾸고, 픽셀 값을 0~1로 맞춥니다.
        array = np.asarray(image, dtype=np.float32) / 255.0

        # 그림 배열은 보통 (높이, 너비, 색상)이지만,
        # PyTorch는 (색상, 높이, 너비)를 더 좋아해서 순서를 바꿉니다.
        array = np.transpose(array, (2, 0, 1))

        # NumPy 배열을 PyTorch 텐서로 바꿔서 돌려줍니다.
        return torch.from_numpy(array)


class SimpleImageFolder(Dataset):
    """
    폴더 구조를 읽어 이미지와 라벨을 꺼내 주는 간단한 데이터셋입니다.

    매개변수:
        root: 클래스 폴더들이 들어 있는 최상위 폴더 경로
        transform: 각 이미지에 적용할 전처리 도구, 기본값은 None

    반환값:
        없음
    """

    def __init__(self, root, transform=None):
        # 루트 경로를 Path 형태로 저장합니다.
        self.root = Path(root)

        # 이미지 변환 도구를 저장합니다.
        self.transform = transform

        # 루트 폴더가 없으면 더 진행할 수 없으므로 에러를 냅니다.
        if not self.root.exists():
            raise FileNotFoundError(f"데이터 폴더를 찾을 수 없습니다: {self.root}")

        # 루트 폴더 바로 아래에 있는 하위 폴더 이름들을 클래스 이름으로 사용합니다.
        self.classes = sorted(
            [path.name for path in self.root.iterdir() if path.is_dir()]
        )

        # 클래스 폴더가 하나도 없으면 데이터 구조가 잘못된 것이므로 에러를 냅니다.
        if not self.classes:
            raise ValueError(f"클래스 폴더가 없습니다: {self.root}")

        # 각 클래스 이름에 숫자 번호를 붙여 딥러닝 모델이 이해하기 쉽게 만듭니다.
        self.class_to_idx = {
            class_name: index for index, class_name in enumerate(self.classes)
        }

        # 이미지 파일 경로와 라벨을 담을 빈 리스트를 만듭니다.
        self.samples = []

        # 각 클래스 폴더 안을 돌면서 이미지 파일을 모두 찾습니다.
        for class_name in self.classes:
            class_dir = self.root / class_name
            for image_path in sorted(class_dir.rglob("*")):
                if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    self.samples.append((image_path, self.class_to_idx[class_name]))

        # 이미지가 하나도 없으면 학습할 것이 없으므로 에러를 냅니다.
        if not self.samples:
            raise ValueError(f"이미지 파일이 없습니다: {self.root}")

    def __len__(self):
        """
        데이터셋 안에 이미지가 몇 장 있는지 알려줍니다.

        매개변수:
            없음

        반환값:
            샘플 개수
        """
        # 저장된 샘플 개수를 반환합니다.
        return len(self.samples)

    def __getitem__(self, index):
        """
        원하는 번호의 이미지와 라벨을 꺼냅니다.

        매개변수:
            index: 가져올 샘플 번호

        반환값:
            (이미지, 라벨) 튜플
        """
        # 지정한 번호의 파일 경로와 라벨을 가져옵니다.
        image_path, label = self.samples[index]

        # 이미지를 열고 RGB 색상 형태로 맞춥니다.
        image = Image.open(image_path).convert("RGB")

        # 변환 도구가 있다면 이미지를 미리 가공합니다.
        if self.transform is not None:
            image = self.transform(image)

        # 모델이 사용할 이미지와 정답 라벨을 함께 돌려줍니다.
        return image, label

