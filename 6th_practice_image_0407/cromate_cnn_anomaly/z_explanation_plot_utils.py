"""
이 파일은 그래프에서 한글이 깨지지 않도록 글꼴을 설정하는 도우미 파일입니다.

전체 흐름:
1. Matplotlib 그래프 도구를 불러옵니다.
2. 컴퓨터에 어떤 글꼴이 있는지 확인합니다.
3. 한글을 잘 보여줄 수 있는 글꼴이 있으면 그 글꼴을 사용합니다.
4. 그래프에서 마이너스 기호가 이상하게 보이지 않도록 설정합니다.
"""

# 그래프를 그릴 때 사용하는 Matplotlib를 불러옵니다.
import matplotlib.pyplot as plt

# 컴퓨터에 설치된 글꼴 목록을 확인하는 도구를 불러옵니다.
from matplotlib import font_manager


def configure_korean_font():
    """
    그래프에서 한글이 깨지지 않도록 글꼴을 설정합니다.

    매개변수:
        없음

    반환값:
        없음
    """
    # 설치된 글꼴 이름들을 빠르게 찾기 위해 집합(set)으로 모읍니다.
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}

    # 한글 표시가 잘 되는 글꼴 후보를 순서대로 확인합니다.
    for font_name in ("AppleGothic", "Malgun Gothic", "NanumGothic"):
        # 이 글꼴이 현재 컴퓨터에 있으면 사용합니다.
        if font_name in available_fonts:
            # 그래프 전체에서 사용할 기본 글꼴을 정합니다.
            plt.rcParams["font.family"] = font_name

            # 첫 번째로 찾은 좋은 글꼴을 쓰면 충분하므로 반복을 멈춥니다.
            break

    # 음수 기호(-)가 깨지지 않도록 추가 설정을 합니다.
    plt.rcParams["axes.unicode_minus"] = False

