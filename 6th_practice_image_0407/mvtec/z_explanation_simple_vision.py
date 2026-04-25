"""
이 파일은 mvtec 실습에서 torchvision 대신 쓰는 작은 이미지 도구 모음입니다.

전체 흐름:
1. 이미지를 다루는 데 필요한 라이브러리를 불러옵니다.
2. 여러 변환을 이어 붙이는 도구를 만듭니다.
3. 이미지 크기를 바꾸고 텐서로 만드는 도구를 만듭니다.
4. 데이터 폴더가 정말 있는지 확인하는 함수를 만듭니다.
"""

# 파일 경로를 쉽게 다루기 위해 Path를 불러옵니다.
from pathlib import Path

# 숫자 배열 계산을 위해 NumPy를 불러옵니다.
import numpy as np

# 딥러닝 텐서를 만들기 위해 torch를 불러옵니다.
import torch

# 이미지 파일을 읽기 위해 PIL의 Image를 불러옵니다.
from PIL import Image


class Compose:
    """
    여러 이미지 변환을 차례대로 실행하는 도구입니다.

    매개변수:
        transforms: 순서대로 적용할 변환 목록

    반환값:
        없음
    """

    def __init__(self, transforms):
        # 나중에 순서대로 실행할 변환들을 저장합니다.
        self.transforms = transforms

    def __call__(self, image):
        """
        변환 목록을 이미지에 차례대로 적용합니다.

        매개변수:
            image: 변환할 이미지

        반환값:
            모두 변환된 이미지
        """
        # 저장된 변환을 하나씩 실행합니다.
        for transform in self.transforms:
            image = transform(image)

        # 마지막까지 처리된 이미지를 돌려줍니다.
        return image


class Resize:
    """
    이미지를 원하는 크기로 바꾸는 도구입니다.

    매개변수:
        size: 목표 이미지 크기

    반환값:
        없음
    """

    def __init__(self, size):
        # 목표 크기를 저장합니다.
        self.size = size

        # Pillow 버전에 따라 이름이 다를 수 있어 안전하게 가져옵니다.
        resampling = getattr(Image, "Resampling", Image)

        # 이미지를 부드럽게 크기 변경할 방식입니다.
        self.resample = resampling.BILINEAR

    def __call__(self, image):
        """
        이미지를 저장된 크기로 바꿉니다.

        매개변수:
            image: 크기를 바꿀 이미지

        반환값:
            크기가 바뀐 이미지
        """
        # resize 함수를 사용해 새 크기로 바꿉니다.
        return image.resize(self.size, self.resample)


class ToTensor:
    """
    PIL 이미지를 PyTorch 텐서로 바꾸는 도구입니다.

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
            torch 텐서
        """
        # 이미지를 숫자 배열로 바꾸고 255로 나누어 0~1 범위로 맞춥니다.
        array = np.asarray(image, dtype=np.float32) / 255.0

        # PyTorch가 좋아하는 채널 순서로 바꿉니다.
        array = np.transpose(array, (2, 0, 1))

        # NumPy 배열을 torch 텐서로 바꿔 돌려줍니다.
        return torch.from_numpy(array)


def resolve_data_dir(base_dir, folder_name="mvtec_ad"):
    """
    데이터 폴더 경로를 만들고, 실제로 존재하는지 확인합니다.

    매개변수:
        base_dir: 현재 파일이 있는 기준 폴더
        folder_name: 찾고 싶은 데이터 폴더 이름

    반환값:
        존재하는 데이터 폴더의 Path 객체
    """
    # 기준 폴더와 데이터 폴더 이름을 합쳐 실제 경로를 만듭니다.
    data_dir = Path(base_dir) / folder_name

    # 폴더가 없으면 뒤 코드들이 모두 실패하므로 먼저 친절하게 멈춥니다.
    if not data_dir.exists():
        raise FileNotFoundError(f"데이터 폴더를 찾을 수 없습니다: {data_dir}")

    # 확인이 끝난 경로를 돌려줍니다.
    return data_dir

