"""
이 파일은 크로메이트 CNN 실습에서 데이터를 먼저 살펴보는 단계입니다.

전체 흐름:
1. 그래프와 경로 설정에 필요한 도구를 준비합니다.
2. 한글이 깨지지 않도록 Matplotlib 환경을 설정합니다.
3. 학습 폴더와 테스트 폴더의 이미지를 읽어 데이터셋을 만듭니다.
4. 데이터 개수와 클래스 이름을 출력합니다.
5. 샘플 이미지를 화면에 보여 주어 데이터가 잘 들어왔는지 확인합니다.
"""

# 운영체제 환경 변수를 다루기 위해 os를 불러옵니다.
import os

# 파일 경로를 쉽게 다루기 위해 Path를 불러옵니다.
from pathlib import Path

# 현재 이 설명 파일이 들어 있는 폴더를 기준 경로로 잡습니다.
base_dir = Path(__file__).resolve().parent

# Matplotlib가 쓸 설정 폴더를 현재 프로젝트 안으로 정해 둡니다.
os.environ.setdefault("MPLCONFIGDIR", str(base_dir / ".mplconfig"))

# 글꼴 캐시 폴더도 현재 프로젝트 안으로 정해 둡니다.
os.environ.setdefault("XDG_CACHE_HOME", str(base_dir / ".cache"))

# 그래프를 그리기 위해 Matplotlib를 불러옵니다.
import matplotlib.pyplot as plt

# 데이터를 여러 장씩 묶어서 꺼내기 위해 DataLoader를 불러옵니다.
from torch.utils.data import DataLoader

# 한글 글꼴 설정 함수와 이미지 처리 도구를 불러옵니다.
from z_explanation_plot_utils import configure_korean_font
from z_explanation_simple_vision import Compose, Resize, SimpleImageFolder, ToTensor

# 그래프에서 한글이 잘 보이도록 설정합니다.
configure_korean_font()


def analyze_data(train_dir, test_dir):
    """
    학습 데이터와 테스트 데이터를 읽고 기본 정보를 보여 줍니다.

    매개변수:
        train_dir: 학습 이미지가 들어 있는 폴더 경로
        test_dir: 테스트 이미지가 들어 있는 폴더 경로

    반환값:
        없음
    """
    # 이미지를 같은 크기로 맞추고, 숫자 텐서로 바꾸는 전처리 묶음을 만듭니다.
    transform = Compose([Resize((256, 256)), ToTensor()])

    # 학습 폴더를 이미지 데이터셋 형태로 읽습니다.
    train_dataset = SimpleImageFolder(root=train_dir, transform=transform)

    # 테스트 폴더도 같은 방식으로 읽습니다.
    test_dataset = SimpleImageFolder(root=test_dir, transform=transform)

    # 데이터가 어떻게 구성되었는지 화면에 알려 줍니다.
    print("=== 데이터 분석 결과 ===")
    print(f"클래스 매핑: {train_dataset.class_to_idx}")
    print(f"학습 데이터 개수: {len(train_dataset)}개")
    print(f"테스트 데이터 개수: {len(test_dataset)}개")

    # 학습 데이터에서 4장씩 꺼내 보는 로더를 만듭니다.
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    # 첫 번째 묶음을 꺼내서 예시 이미지로 사용합니다.
    images, labels = next(iter(train_loader))

    # 숫자 라벨을 사람이 읽을 수 있는 클래스 이름으로 바꾸기 위해 이름 목록을 저장합니다.
    class_names = train_dataset.classes

    # 이미지 4장을 한 줄로 보여 줄 그래프 판을 만듭니다.
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))

    # 4장의 이미지를 차례대로 화면에 올립니다.
    for i in range(4):
        # PyTorch 텐서 모양인 (채널, 높이, 너비)를
        # Matplotlib가 좋아하는 (높이, 너비, 채널) 순서로 바꿉니다.
        img = images[i].permute(1, 2, 0).numpy()

        # 이미지를 그래프 칸에 그립니다.
        axes[i].imshow(img)

        # 이 이미지의 정답 이름을 제목으로 붙입니다.
        axes[i].set_title(f"Label: {class_names[labels[i]]}")

        # 축 숫자는 보기 복잡하니 숨깁니다.
        axes[i].axis("off")

    # 전체 그림의 큰 제목을 붙입니다.
    plt.suptitle("학습 데이터 샘플 확인")

    # 겹치지 않도록 간격을 자동으로 조절합니다.
    plt.tight_layout()

    # 그래프를 화면에 보여 줍니다.
    plt.show()


# 이 파일을 직접 실행했을 때만 아래 코드가 실행됩니다.
if __name__ == "__main__":
    # 학습 데이터 폴더 경로를 만듭니다.
    TRAIN_DIR = base_dir / "data" / "학습"

    # 테스트 데이터 폴더 경로를 만듭니다.
    TEST_DIR = base_dir / "data" / "테스트"

    # 위에서 만든 함수로 데이터 분석을 시작합니다.
    analyze_data(TRAIN_DIR, TEST_DIR)

