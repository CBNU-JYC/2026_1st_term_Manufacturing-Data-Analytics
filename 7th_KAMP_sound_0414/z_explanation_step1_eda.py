"""
프로그램 전체 흐름 설명:
1. 필요한 도구들을 불러오고, 오디오 분석 라이브러리가 준비되어 있는지 확인합니다.
2. 정상 팬 소리와 이상 팬 소리 파일을 찾습니다.
3. 소리의 파형, 주파수, MFCC 특징을 그림으로 확인합니다.
4. 몇 개의 파일에서 숫자 특징을 뽑아 표로 만들고, 특징끼리 얼마나 관련 있는지 히트맵으로 봅니다.
5. 결과 그림과 요약 파일을 저장해서 나중에 다시 볼 수 있게 합니다.
"""

import os  # 폴더 위치를 찾고 파일 경로를 만들 때 사용합니다.
import glob  # 특정 규칙에 맞는 파일 목록을 한 번에 찾을 때 사용합니다.
import sys  # 지금 실행 중인 Python 위치를 확인할 때 사용합니다.
import json  # 결과 요약을 json 파일로 저장할 때 사용합니다.
import numpy as np  # 숫자 계산, FFT 같은 수학 계산을 쉽게 해줍니다.
import pandas as pd  # 표 형태의 데이터를 만들고 저장할 때 사용합니다.
import matplotlib.pyplot as plt  # 그래프를 그릴 때 사용합니다.
import seaborn as sns  # 보기 좋은 히트맵을 그릴 때 사용합니다.
import warnings  # 경고 메시지를 숨기거나 관리할 때 사용합니다.
import importlib  # 라이브러리가 설치되어 있는지 직접 확인할 때 사용합니다.

print(f"현재 Python 실행 경로: {sys.executable}")  # 어떤 Python으로 실행 중인지 보여줍니다.

try:  # numba가 설치되어 있는지 확인해봅니다.
    numba = importlib.import_module("numba")  # numba 라이브러리를 이름으로 불러옵니다.
except ModuleNotFoundError as exc:  # numba가 없으면 친절한 설치 안내를 보여줍니다.
    raise ModuleNotFoundError(
        "numba가 설치되어 있지 않습니다. "
        f"현재 Python({sys.executable}) 기준으로 다음 명령을 실행해주세요: "
        f"'{sys.executable}' -m pip install -r 7th_KAMP_sound_0414/requirements.txt"
    ) from exc


def _disable_numba_cache(decorator):
    """
    numba가 만든 임시 저장 기능(cache)을 끄는 함수입니다.

    Args:
        decorator: numba.jit처럼 함수를 빠르게 만들기 위해 감싸는 도구입니다.

    Returns:
        function: cache=False 옵션을 자동으로 넣어주는 새 포장 함수입니다.
    """

    def wrapped(*args, **kwargs):
        """
        실제 numba 데코레이터를 대신 호출합니다.

        Args:
            *args: 원래 데코레이터에 전달되는 위치 인자들입니다.
            **kwargs: 원래 데코레이터에 전달되는 이름 인자들입니다.

        Returns:
            object: cache 옵션이 꺼진 numba 데코레이터 결과입니다.
        """

        kwargs["cache"] = False  # 캐시를 끄면 권한 문제나 임시파일 문제를 줄일 수 있습니다.
        return decorator(*args, **kwargs)  # 원래 기능은 그대로 실행합니다.

    return wrapped  # 새로 만든 포장 함수를 돌려줍니다.


numba.jit = _disable_numba_cache(numba.jit)  # jit를 사용할 때 캐시를 끄도록 바꿉니다.
numba.vectorize = _disable_numba_cache(numba.vectorize)  # vectorize도 같은 방식으로 바꿉니다.
numba.guvectorize = _disable_numba_cache(numba.guvectorize)  # guvectorize도 같은 방식으로 바꿉니다.

try:  # librosa가 설치되어 있는지 확인합니다.
    import librosa  # 소리 파일을 읽고 특징을 뽑는 대표 라이브러리입니다.
    import librosa.display  # librosa로 만든 소리 데이터를 그림으로 보여줍니다.
except ModuleNotFoundError as exc:  # librosa가 없으면 설치 방법을 알려줍니다.
    raise ModuleNotFoundError(
        "librosa가 설치되어 있지 않습니다. "
        f"현재 Python({sys.executable}) 기준으로 다음 명령을 실행해주세요: "
        f"'{sys.executable}' -m pip install -r 7th_KAMP_sound_0414/requirements.txt"
    ) from exc

warnings.filterwarnings('ignore')  # 실습 화면을 깔끔하게 보기 위해 경고를 숨깁니다.
plt.style.use('ggplot')  # 그래프 스타일을 보기 좋게 바꿉니다.
plt.rcParams['font.family'] = 'Malgun Gothic'  # 한글 글자가 깨지지 않도록 글꼴을 정합니다.
plt.rcParams['axes.unicode_minus'] = False  # 그래프의 마이너스 기호가 깨지지 않게 합니다.

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 이 코드 파일이 들어있는 폴더입니다.
VIS_DIR = os.path.join(BASE_DIR, "0_result")  # 그림을 저장할 폴더 경로입니다.
RESULTS_DIR = os.path.join(BASE_DIR, "0_result")  # 표와 요약 결과를 저장할 폴더 경로입니다.
os.makedirs(VIS_DIR, exist_ok=True)  # 그림 저장 폴더가 없으면 새로 만듭니다.
os.makedirs(RESULTS_DIR, exist_ok=True)  # 결과 저장 폴더가 없으면 새로 만듭니다.


def save_current_figure(filename):
    """
    지금 화면에 그려진 그래프를 이미지 파일로 저장합니다.

    Args:
        filename: 저장할 이미지 파일 이름입니다. 예: "step1_waveform.png"

    Returns:
        None: 파일을 저장하고 저장 위치를 출력만 합니다.
    """

    save_path = os.path.join(VIS_DIR, filename)  # 저장 폴더와 파일 이름을 합칩니다.
    plt.savefig(save_path, dpi=200, bbox_inches='tight')  # 그래프를 선명하게 저장합니다.
    print(f"이미지 저장 완료: {save_path}")  # 저장된 위치를 알려줍니다.


ok_path = os.path.join(BASE_DIR, 'FAN_sound_OK', '*')  # 정상 팬 소리 파일들을 찾는 규칙입니다.
err_path = os.path.join(BASE_DIR, 'FAN_sound_error', '*')  # 이상 팬 소리 파일들을 찾는 규칙입니다.

ok_files = glob.glob(ok_path)  # 정상 파일 목록을 가져옵니다.
err_files = glob.glob(err_path)  # 이상 파일 목록을 가져옵니다.

print(f"정상(OK) 데이터 개수: {len(ok_files)}")  # 정상 파일이 몇 개인지 보여줍니다.
print(f"이상(Error) 데이터 개수: {len(err_files)}")  # 이상 파일이 몇 개인지 보여줍니다.

if not ok_files or not err_files:  # 둘 중 하나라도 비어 있으면 분석할 수 없습니다.
    raise FileNotFoundError(
        "오디오 파일을 찾지 못했습니다. "
        "스크립트와 같은 폴더 아래의 FAN_sound_OK, FAN_sound_error 폴더를 확인해주세요."
    )

sample_ok, sr_ok = librosa.load(ok_files[0])  # 첫 번째 정상 소리 파일을 읽습니다.
sample_err, sr_err = librosa.load(err_files[0])  # 첫 번째 이상 소리 파일을 읽습니다.

plt.figure(figsize=(15, 4))  # 가로로 긴 그림판을 만듭니다.
plt.subplot(1, 2, 1)  # 그림판을 1행 2열로 나눈 뒤 첫 번째 칸을 선택합니다.
librosa.display.waveshow(sample_ok, sr=sr_ok)  # 정상 소리의 파형을 그립니다.
plt.title('정상(OK) 사운드 파형')  # 첫 번째 그래프 제목입니다.
plt.xlabel('Time')  # x축은 시간입니다.
plt.ylabel('Amplitude')  # y축은 소리의 흔들림 크기입니다.

plt.subplot(1, 2, 2)  # 두 번째 칸을 선택합니다.
librosa.display.waveshow(sample_err, sr=sr_err)  # 이상 소리의 파형을 그립니다.
plt.title('이상(Error) 사운드 파형')  # 두 번째 그래프 제목입니다.
plt.xlabel('Time')  # x축은 시간입니다.
plt.ylabel('Amplitude')  # y축은 소리의 흔들림 크기입니다.
plt.tight_layout()  # 제목과 그래프가 겹치지 않도록 간격을 정리합니다.
save_current_figure("step1_waveform.png")  # 파형 그림을 저장합니다.
plt.show()  # 그래프를 화면에 보여줍니다.


def plot_half_spectrum(y, sr, title):
    """
    소리 신호를 주파수 그래프로 바꾸고 절반만 그립니다.

    Args:
        y: librosa가 읽어온 소리의 숫자 배열입니다.
        sr: 1초에 몇 번 소리를 측정했는지 나타내는 샘플링 레이트입니다.
        title: 그래프 위에 보여줄 제목입니다.

    Returns:
        None: 그래프를 화면에 그리기만 합니다.
    """

    fft = np.fft.fft(y)  # 소리를 주파수 성분으로 나눕니다.
    magnitude = np.abs(fft)  # 각 주파수 성분의 크기만 가져옵니다.
    fre = np.linspace(0, sr, len(magnitude))  # 0부터 sr까지 주파수 눈금을 만듭니다.

    # FFT 결과는 좌우가 비슷하게 반복되므로 절반만 보아도 핵심 정보를 볼 수 있습니다.
    haf_spectrum = magnitude[:int(len(magnitude) / 2)]  # 주파수 크기 중 앞 절반만 사용합니다.
    haf_fre = fre[:int(len(magnitude) / 2)]  # 주파수 눈금도 앞 절반만 사용합니다.

    plt.plot(haf_fre, haf_spectrum)  # 주파수와 크기를 선 그래프로 그립니다.
    plt.title(title)  # 그래프 제목을 붙입니다.
    plt.xlabel('Frequency')  # x축은 주파수입니다.
    plt.ylabel('Magnitude')  # y축은 주파수 성분의 크기입니다.


plt.figure(figsize=(15, 4))  # 주파수 그래프를 담을 그림판을 만듭니다.
plt.subplot(1, 2, 1)  # 첫 번째 칸을 선택합니다.
plot_half_spectrum(sample_ok, sr_ok, '정상(OK) Half Spectrum')  # 정상 소리 주파수를 그립니다.

plt.subplot(1, 2, 2)  # 두 번째 칸을 선택합니다.
plot_half_spectrum(sample_err, sr_err, '이상(Error) Half Spectrum')  # 이상 소리 주파수를 그립니다.
plt.tight_layout()  # 그래프 간격을 정리합니다.
save_current_figure("step1_half_spectrum.png")  # 주파수 그림을 저장합니다.
plt.show()  # 화면에 보여줍니다.

hop_length = 512  # MFCC를 만들 때 다음 조각으로 얼마나 이동할지 정합니다.
n_fft = 2048  # 한 번에 분석할 소리 조각의 길이입니다.

mfcc_ok = librosa.feature.mfcc(
    y=sample_ok, sr=sr_ok, n_fft=n_fft, hop_length=hop_length, n_mfcc=13
)  # 정상 소리에서 MFCC 특징 13개를 뽑습니다.
mfcc_err = librosa.feature.mfcc(
    y=sample_err, sr=sr_err, n_fft=n_fft, hop_length=hop_length, n_mfcc=13
)  # 이상 소리에서 MFCC 특징 13개를 뽑습니다.

plt.figure(figsize=(15, 6))  # MFCC 그림을 담을 그림판을 만듭니다.
plt.subplot(1, 2, 1)  # 첫 번째 칸을 선택합니다.
librosa.display.specshow(mfcc_ok, sr=sr_ok, hop_length=hop_length, x_axis='time')  # 정상 MFCC를 색으로 표시합니다.
plt.colorbar()  # 색이 어떤 값을 뜻하는지 막대를 보여줍니다.
plt.title('정상(OK) MFCCs')  # 제목을 붙입니다.
plt.ylabel('MFCC coefficients')  # y축은 MFCC 번호입니다.

plt.subplot(1, 2, 2)  # 두 번째 칸을 선택합니다.
librosa.display.specshow(mfcc_err, sr=sr_err, hop_length=hop_length, x_axis='time')  # 이상 MFCC를 색으로 표시합니다.
plt.colorbar()  # 색상 설명 막대를 보여줍니다.
plt.title('이상(Error) MFCCs')  # 제목을 붙입니다.
plt.ylabel('MFCC coefficients')  # y축 이름을 붙입니다.
plt.tight_layout()  # 그래프 간격을 정리합니다.
save_current_figure("step1_mfcc.png")  # MFCC 그림을 저장합니다.
plt.show()  # 화면에 보여줍니다.


def get_features(y, sr):
    """
    소리 하나에서 모델이 이해하기 쉬운 숫자 특징 4개를 뽑습니다.

    Args:
        y: 소리를 숫자로 바꾼 배열입니다.
        sr: 샘플링 레이트입니다.

    Returns:
        tuple: 스펙트럼 최솟값, 스펙트럼 최댓값, MFCC 최솟값, MFCC 최댓값입니다.
    """

    fft = np.fft.fft(y)  # 소리를 주파수 정보로 바꿉니다.
    magnitude = np.abs(fft)  # 주파수별 크기를 구합니다.
    haf_spectrum = magnitude[:int(len(magnitude) / 2)]  # 의미가 중복되는 뒤 절반은 빼고 앞 절반만 씁니다.

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_fft=2048, hop_length=512, n_mfcc=13)  # 소리의 음색 특징을 뽑습니다.

    return np.min(haf_spectrum), np.max(haf_spectrum), np.min(mfcc), np.max(mfcc)  # 네 가지 대표 숫자를 돌려줍니다.


spec_mins, spec_maxs, mfcc_mins, mfcc_maxs, labels = [], [], [], [], []  # 특징과 정답을 담을 빈 바구니입니다.

for path in ok_files[:10]:  # 정상 파일 중 앞 10개만 빠르게 실습합니다.
    y, sr = librosa.load(path, sr=100)  # 가이드북 기준에 맞춰 100Hz로 소리를 읽습니다.
    s_min, s_max, m_min, m_max = get_features(y, sr)  # 정상 소리의 특징 4개를 뽑습니다.
    spec_mins.append(s_min)  # 스펙트럼 최솟값을 저장합니다.
    spec_maxs.append(s_max)  # 스펙트럼 최댓값을 저장합니다.
    mfcc_mins.append(m_min)  # MFCC 최솟값을 저장합니다.
    mfcc_maxs.append(m_max)  # MFCC 최댓값을 저장합니다.
    labels.append(0)  # 정상은 0이라고 표시합니다.

for path in err_files[:10]:  # 이상 파일 중 앞 10개만 빠르게 실습합니다.
    y, sr = librosa.load(path, sr=100)  # 같은 기준으로 소리를 읽어야 공정하게 비교할 수 있습니다.
    s_min, s_max, m_min, m_max = get_features(y, sr)  # 이상 소리의 특징 4개를 뽑습니다.
    spec_mins.append(s_min)  # 스펙트럼 최솟값을 저장합니다.
    spec_maxs.append(s_max)  # 스펙트럼 최댓값을 저장합니다.
    mfcc_mins.append(m_min)  # MFCC 최솟값을 저장합니다.
    mfcc_maxs.append(m_max)  # MFCC 최댓값을 저장합니다.
    labels.append(1)  # 이상은 1이라고 표시합니다.

df_features = pd.DataFrame({
    'mfcc_min': mfcc_mins,  # MFCC 최솟값 열입니다.
    'mfcc_max': mfcc_maxs,  # MFCC 최댓값 열입니다.
    'spectrum_min': spec_mins,  # 스펙트럼 최솟값 열입니다.
    'spectrum_max': spec_maxs,  # 스펙트럼 최댓값 열입니다.
    'NG': labels  # 정상/이상 정답 열입니다.
})  # 여러 리스트를 하나의 표로 묶습니다.

plt.figure(figsize=(8, 6))  # 히트맵을 그릴 그림판을 만듭니다.
corr = df_features.iloc[:, :-1].corr()  # 정답 열을 제외하고 특징끼리의 상관관계를 계산합니다.
sns.heatmap(corr, annot=True, cmap='Greens', annot_kws={'size': 15})  # 상관관계를 색과 숫자로 보여줍니다.
plt.title('Feature Correlation Heatmap')  # 그래프 제목을 붙입니다.
save_current_figure("step1_feature_correlation_heatmap.png")  # 히트맵 그림을 저장합니다.
plt.show()  # 화면에 보여줍니다.

summary = {
    "ok_file_count": len(ok_files),  # 정상 파일 개수입니다.
    "error_file_count": len(err_files),  # 이상 파일 개수입니다.
    "feature_sample_count": len(df_features),  # 특징을 뽑은 샘플 개수입니다.
    "feature_columns": df_features.columns.tolist(),  # 표의 열 이름들입니다.
}  # 나중에 확인하기 쉽게 요약 정보를 딕셔너리로 만듭니다.
with open(os.path.join(RESULTS_DIR, "step1_eda_summary.json"), "w", encoding="utf-8") as f:  # 요약 파일을 쓰기 모드로 엽니다.
    json.dump(summary, f, ensure_ascii=False, indent=2)  # 한글이 깨지지 않게 json으로 저장합니다.
df_features.to_csv(os.path.join(RESULTS_DIR, "step1_feature_samples.csv"), index=False)  # 특징 표를 CSV 파일로 저장합니다.
