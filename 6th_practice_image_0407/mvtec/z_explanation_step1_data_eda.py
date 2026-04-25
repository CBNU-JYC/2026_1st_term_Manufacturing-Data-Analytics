"""
이 파일은 MVTec AD 데이터셋을 먼저 살펴보는 EDA 단계입니다.

전체 흐름:
1. 파일 경로와 그래프 설정을 준비합니다.
2. MVTec 데이터셋을 읽는 클래스를 만듭니다.
3. 정상 이미지와 불량 이미지를 라벨과 함께 불러옵니다.
4. 샘플 이미지를 화면에 보여 주어 데이터가 잘 들어왔는지 확인합니다.
"""

# 환경 변수와 파일 목록 도구를 불러옵니다.
import os
import glob

# 파일 경로를 쉽게 다루기 위해 Path를 불러옵니다.
from pathlib import Path

# 현재 설명 파일이 있는 폴더를 기준 경로로 저장합니다.
base_dir = Path(__file__).resolve().parent

# Matplotlib 설정 폴더를 프로젝트 안으로 정합니다.
os.environ.setdefault("MPLCONFIGDIR", str(base_dir / ".mplconfig"))

# 캐시 폴더도 프로젝트 안으로 정합니다.
os.environ.setdefault("XDG_CACHE_HOME", str(base_dir / ".cache"))

# 그래프, 데이터셋, 이미지 도구를 불러옵니다.
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# 설명 버전 전처리 도구를 불러옵니다.
from z_explanation_simple_vision import Compose, Resize, ToTensor, resolve_data_dir


class MVTecDataset(Dataset):
    """
    MVTec AD 폴더 구조를 읽어 이미지, 라벨, 경로를 반환하는 데이터셋입니다.

    매개변수:
        root_dir: mvtec_ad 폴더 경로
        category: 사용할 카테고리 이름, 예를 들면 bottle
        is_train: True면 train/good만 읽고, False면 test 전체를 읽음
        transform: 이미지에 적용할 전처리 도구

    반환값:
        없음
    """

    def __init__(self, root_dir, category, is_train=True, transform=None):
        """
        데이터셋 안에 들어갈 이미지 경로와 라벨 목록을 준비합니다.

        매개변수:
            root_dir: 데이터셋 최상위 폴더
            category: 카테고리 이름
            is_train: 학습용인지 여부
            transform: 이미지 변환 도구

        반환값:
            없음
        """
        # 이미지 변환 도구를 저장합니다.
        self.transform = transform

        # 이미지 경로들을 담을 리스트를 만듭니다.
        self.image_paths = []

        # 라벨을 담을 리스트를 만듭니다.
        self.labels = []

        # 학습용이면 정상 이미지(good)만 읽습니다.
        if is_train:
            img_dir = os.path.join(root_dir, category, "train", "good")
            paths = glob.glob(os.path.join(img_dir, "*.png"))
            self.image_paths.extend(paths)
            self.labels.extend([0] * len(paths))
        else:
            # 테스트용이면 good과 모든 defect 폴더를 모두 읽습니다.
            test_dir = os.path.join(root_dir, category, "test")
            for defect_type in os.listdir(test_dir):
                paths = glob.glob(os.path.join(test_dir, defect_type, "*.png"))
                self.image_paths.extend(paths)

                # good이면 0, 나머지 결함은 1로 라벨을 붙입니다.
                label = 0 if defect_type == "good" else 1
                self.labels.extend([label] * len(paths))

    def __len__(self):
        """
        데이터셋 샘플 수를 반환합니다.

        매개변수:
            없음

        반환값:
            전체 샘플 개수
        """
        # 이미지 경로 개수를 그대로 돌려줍니다.
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        지정한 번호의 이미지, 라벨, 경로를 반환합니다.

        매개변수:
            idx: 가져올 샘플 번호

        반환값:
            (이미지, 라벨, 이미지 경로) 튜플
        """
        # 해당 번호의 이미지 경로를 꺼냅니다.
        img_path = self.image_paths[idx]

        # 이미지를 열고 RGB 형식으로 맞춥니다.
        image = Image.open(img_path).convert("RGB")

        # 라벨도 함께 가져옵니다.
        label = self.labels[idx]

        # 전처리 도구가 있으면 이미지를 가공합니다.
        if self.transform:
            image = self.transform(image)

        # 이미지, 라벨, 경로를 함께 돌려줍니다.
        return image, label, img_path


def show_eda_images(dataset, num_images=4):
    """
    데이터셋 샘플 이미지를 여러 장 화면에 보여 줍니다.

    매개변수:
        dataset: 보여 줄 데이터셋
        num_images: 몇 장을 보여 줄지 정하는 숫자

    반환값:
        없음
    """
    # 여러 장 이미지를 한 줄에 보여 줄 그래프 판을 만듭니다.
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))

    # 정한 개수만큼 이미지를 하나씩 꺼내 화면에 올립니다.
    for i in range(num_images):
        img, label, _ = dataset[i]

        # 텐서를 Matplotlib가 이해하기 쉬운 배열 모양으로 바꿉니다.
        img_np = img.permute(1, 2, 0).numpy()

        # 라벨 숫자를 사람이 읽기 쉬운 글자로 바꿉니다.
        title = "Normal" if label == 0 else "Anomaly"

        # 이미지를 그리고 제목을 붙입니다.
        axes[i].imshow(img_np)
        axes[i].set_title(title)
        axes[i].axis("off")

    # 전체 그림 제목을 붙입니다.
    plt.suptitle("MVTec AD Dataset EDA", fontsize=16)
    plt.tight_layout()
    plt.show()


# 이 파일을 직접 실행했을 때만 EDA를 시작합니다.
if __name__ == "__main__":
    # mvtec_ad 데이터 폴더를 찾습니다.
    ROOT_DIR = resolve_data_dir(base_dir)

    # bottle 카테고리를 사용합니다.
    CATEGORY = "bottle"

    # 이미지 전처리를 준비합니다.
    transform = Compose([Resize((256, 256)), ToTensor()])

    # 안내 문구를 출력합니다.
    print("데이터셋 로딩 중...")

    # 테스트용 데이터셋을 만듭니다.
    test_dataset = MVTecDataset(ROOT_DIR, CATEGORY, is_train=False, transform=transform)

    # 데이터 개수를 출력합니다.
    print(f"테스트 데이터 개수: {len(test_dataset)}장")

    # 샘플 이미지들을 화면에 보여 줍니다.
    show_eda_images(test_dataset)

