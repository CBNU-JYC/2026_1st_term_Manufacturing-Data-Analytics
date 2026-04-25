import matplotlib.pyplot as plt
from matplotlib import font_manager


def configure_korean_font():
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in ("AppleGothic", "Malgun Gothic", "NanumGothic"):
        if font_name in available_fonts:
            plt.rcParams["font.family"] = font_name
            break

    plt.rcParams["axes.unicode_minus"] = False
