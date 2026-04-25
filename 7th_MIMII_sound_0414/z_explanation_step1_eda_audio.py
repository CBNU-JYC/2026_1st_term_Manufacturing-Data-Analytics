"""
프로그램 전체 흐름 설명:
1. 오디오 분석에 필요한 라이브러리를 불러오고 준비 상태를 확인합니다.
2. MIMII 밸브 데이터에서 정상 소리와 비정상 소리 파일을 하나씩 찾습니다.
3. 두 소리를 파형, 스펙트로그램, Mel-Spectrogram 그림으로 비교합니다.
4. AI 모델 입력으로 쓰기 좋은 2D Mel-Spectrogram 모양을 확인합니다.
5. 그림과 요약 정보를 파일로 저장합니다.
"""

import os  # 폴더와 파일 경로를 만들 때 사용합니다.
import glob  # 조건에 맞는 wav 파일 목록을 찾을 때 사용합니다.
import sys  # 현재 Python 실행 위치를 출력할 때 사용합니다.
import importlib  # 라이브러리 설치 여부를 확인할 때 사용합니다.
import json  # 요약 정보를 json 파일로 저장할 때 사용합니다.
import numpy as np  # 숫자 배열과 스펙트로그램 계산에 사용합니다.
import matplotlib.pyplot as plt  # 그래프를 그릴 때 사용합니다.

print(f"현재 Python 실행 경로: {sys.executable}")  # 어떤 Python으로 실행 중인지 보여줍니다.

try:  # numba가 설치되어 있는지 확인합니다.
    numba = importlib.import_module("numba")  # numba를 불러옵니다.
except ModuleNotFoundError as exc:  # numba가 없으면 설치 명령을 알려줍니다.
    raise ModuleNotFoundError(
        "numba가 설치되어 있지 않습니다. "
        f"현재 Python({sys.executable}) 기준으로 다음 명령을 실행해주세요: "
        f"'{sys.executable}' -m pip install librosa numba sounddevice"
    ) from exc


def _disable_numba_cache(decorator):
    """
    numba가 임시 캐시 파일을 만들지 않도록 cache=False를 자동으로 넣습니다.

    Args:
        decorator: numba.jit 같은 numba 데코레이터입니다.

    Returns:
        function: cache 옵션을 끈 새 포장 함수입니다.
    """

    def wrapped(*args, **kwargs):
        """
        원래 numba 데코레이터를 cache=False와 함께 실행합니다.

        Args:
            *args: 원래 데코레이터에 전달되는 위치 인자입니다.
            **kwargs: 원래 데코레이터에 전달되는 이름 인자입니다.

        Returns:
            object: 원래 데코레이터의 실행 결과입니다.
        """

        kwargs["cache"] = False  # 권한 문제를 줄이기 위해 캐시를 끕니다.
        return decorator(*args, **kwargs)  # 원래 기능은 그대로 사용합니다.

    return wrapped  # 포장 함수를 반환합니다.


numba.jit = _disable_numba_cache(numba.jit)  # jit 캐시를 끕니다.
numba.vectorize = _disable_numba_cache(numba.vectorize)  # vectorize 캐시를 끕니다.
numba.guvectorize = _disable_numba_cache(numba.guvectorize)  # guvectorize 캐시를 끕니다.

try:  # librosa 설치 여부를 확인합니다.
    import librosa  # 오디오 파일을 읽고 특징을 만들 때 사용합니다.
    import librosa.display  # 오디오 특징을 그림으로 보여줄 때 사용합니다.
except ModuleNotFoundError as exc:  # librosa가 없으면 설치 방법을 알려줍니다.
    raise ModuleNotFoundError(
        "librosa가 설치되어 있지 않습니다. "
        f"현재 Python({sys.executable}) 기준으로 다음 명령을 실행해주세요: "
        f"'{sys.executable}' -m pip install librosa numba sounddevice"
    ) from exc

try:  # sounddevice가 있으면 소리를 직접 재생할 수 있습니다.
    import sounddevice as sd  # 컴퓨터 스피커로 오디오를 재생하는 라이브러리입니다.
    SOUNDDEVICE_AVAILABLE = True  # 재생 가능 표시입니다.
except ModuleNotFoundError:  # sounddevice가 없어도 그림 분석은 가능합니다.
    sd = None  # 재생 도구가 없다는 뜻입니다.
    SOUNDDEVICE_AVAILABLE = False  # 재생 불가능 표시입니다.

plt.style.use('seaborn-v0_8-whitegrid')  # 그래프 배경을 보기 좋게 설정합니다.
plt.rcParams['figure.figsize'] = (12, 6)  # 기본 그림 크기를 정합니다.
print("라이브러리 로드 완료")  # 준비 완료를 알립니다.

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 현재 코드가 들어있는 폴더입니다.
VIS_DIR = os.path.join(BASE_DIR, "visualizations")  # 그림 저장 폴더입니다.
RESULTS_DIR = os.path.join(BASE_DIR, "results")  # 결과 저장 폴더입니다.
os.makedirs(VIS_DIR, exist_ok=True)  # 그림 저장 폴더가 없으면 만듭니다.
os.makedirs(RESULTS_DIR, exist_ok=True)  # 결과 저장 폴더가 없으면 만듭니다.


def save_current_figure(filename):
    """
    현재 그려진 그래프를 visualizations 폴더에 저장합니다.

    Args:
        filename: 저장할 이미지 파일 이름입니다.

    Returns:
        None: 이미지 파일을 저장하고 경로를 출력합니다.
    """

    save_path = os.path.join(VIS_DIR, filename)  # 저장할 전체 경로를 만듭니다.
    plt.savefig(save_path, dpi=200, bbox_inches='tight')  # 그래프를 선명한 이미지로 저장합니다.
    print(f"이미지 저장 완료: {save_path}")  # 저장 위치를 출력합니다.


normal_dir = os.path.join(BASE_DIR, '0_dB_valve', 'valve', 'id_02', 'normal')  # 정상 밸브 소리 폴더입니다.
abnormal_dir = os.path.join(BASE_DIR, '0_dB_valve', 'valve', 'id_02', 'abnormal')  # 비정상 밸브 소리 폴더입니다.

normal_files = glob.glob(os.path.join(normal_dir, '*.wav'))  # 정상 wav 파일 목록입니다.
abnormal_files = glob.glob(os.path.join(abnormal_dir, '*.wav'))  # 비정상 wav 파일 목록입니다.

if not normal_files or not abnormal_files:  # 한쪽이라도 파일이 없으면 비교할 수 없습니다.
    print("에러: 지정된 경로에서 .wav 파일을 찾을 수 없습니다. 경로 구조를 확인해주세요.")
    print(f" - 확인된 정상 경로: {normal_dir}")  # 정상 폴더 경로를 보여줍니다.
    print(f" - 확인된 비정상 경로: {abnormal_dir}")  # 비정상 폴더 경로를 보여줍니다.
else:
    normal_audio_path = normal_files[0]  # 정상 파일 중 첫 번째를 예시로 선택합니다.
    abnormal_audio_path = abnormal_files[0]  # 비정상 파일 중 첫 번째를 예시로 선택합니다.

    print(f"로드된 정상 데이터: {normal_audio_path}")  # 선택된 정상 파일을 출력합니다.
    print(f"로드된 비정상 데이터: {abnormal_audio_path}")  # 선택된 비정상 파일을 출력합니다.

    sr_target = 16000  # 모든 소리를 16,000Hz 기준으로 맞춥니다.
    y_normal, sr = librosa.load(normal_audio_path, sr=sr_target)  # 정상 소리를 숫자 배열로 읽습니다.
    y_anomaly, _ = librosa.load(abnormal_audio_path, sr=sr_target)  # 비정상 소리를 숫자 배열로 읽습니다.

    print(f"\n오디오 샘플링 레이트(sr): {sr} Hz")  # 실제 사용한 샘플링 레이트를 보여줍니다.
    print(f"정상 데이터 길이: {len(y_normal)} 샘플 ({len(y_normal) / sr:.2f}초)")  # 정상 소리 길이를 보여줍니다.

fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(12, 6))  # 위아래 두 칸짜리 그림판을 만듭니다.

librosa.display.waveshow(y_normal, sr=sr, ax=ax[0], color='steelblue')  # 정상 소리 파형을 그립니다.
ax[0].set_title('Normal Valve Sound - Waveform')  # 정상 파형 제목입니다.

librosa.display.waveshow(y_anomaly, sr=sr, ax=ax[1], color='crimson')  # 비정상 소리 파형을 그립니다.
ax[1].set_title('Abnormal Valve Sound - Waveform')  # 비정상 파형 제목입니다.

plt.tight_layout()  # 그래프가 겹치지 않도록 정리합니다.
save_current_figure("step1_waveform.png")  # 파형 이미지를 저장합니다.
plt.show()  # 그래프를 화면에 보여줍니다.

if SOUNDDEVICE_AVAILABLE:  # 소리 재생 라이브러리가 있으면 직접 들어봅니다.
    print("정상 소리 재생 중...")  # 정상 소리 재생 안내입니다.
    sd.play(y_normal, sr)  # 정상 소리를 재생합니다.
    sd.wait()  # 재생이 끝날 때까지 기다립니다.

    print("비정상 소리 재생 중...")  # 비정상 소리 재생 안내입니다.
    sd.play(y_anomaly, sr)  # 비정상 소리를 재생합니다.
    sd.wait()  # 재생이 끝날 때까지 기다립니다.
else:  # 라이브러리가 없으면 재생만 건너뜁니다.
    print(
        "sounddevice가 설치되어 있지 않아 오디오 재생은 건너뜁니다. "
        f"필요하면 '{sys.executable}' -m pip install sounddevice 로 설치해주세요."
    )

D_normal = librosa.stft(y_normal)  # 정상 소리를 시간별 주파수 정보로 바꿉니다.
S_db_normal = librosa.amplitude_to_db(np.abs(D_normal), ref=np.max)  # 사람이 보기 좋은 dB 단위로 바꿉니다.

D_anomaly = librosa.stft(y_anomaly)  # 비정상 소리를 시간별 주파수 정보로 바꿉니다.
S_db_anomaly = librosa.amplitude_to_db(np.abs(D_anomaly), ref=np.max)  # dB 단위로 바꿉니다.

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))  # 좌우 두 칸짜리 그림판을 만듭니다.
img1 = librosa.display.specshow(S_db_normal, sr=sr, x_axis='time', y_axis='hz', ax=ax[0], cmap='magma')  # 정상 스펙트로그램입니다.
ax[0].set_title('Spectrogram - Normal Valve')  # 정상 스펙트로그램 제목입니다.

img2 = librosa.display.specshow(S_db_anomaly, sr=sr, x_axis='time', y_axis='hz', ax=ax[1], cmap='magma')  # 비정상 스펙트로그램입니다.
ax[1].set_title('Spectrogram - Abnormal Valve')  # 비정상 스펙트로그램 제목입니다.

fig.colorbar(img1, ax=ax, format="%+2.0f dB")  # 색깔이 뜻하는 dB 값을 보여줍니다.
save_current_figure("step1_spectrogram.png")  # 스펙트로그램 이미지를 저장합니다.
plt.show()  # 그래프를 화면에 보여줍니다.

n_mels = 128  # 주파수 영역을 128칸의 Mel 눈금으로 압축합니다.

mel_normal = librosa.feature.melspectrogram(y=y_normal, sr=sr, n_mels=n_mels)  # 정상 소리의 Mel-Spectrogram입니다.
mel_db_normal = librosa.power_to_db(mel_normal, ref=np.max)  # 보기 좋은 dB 단위로 바꿉니다.

mel_anomaly = librosa.feature.melspectrogram(y=y_anomaly, sr=sr, n_mels=n_mels)  # 비정상 소리의 Mel-Spectrogram입니다.
mel_db_anomaly = librosa.power_to_db(mel_anomaly, ref=np.max)  # 보기 좋은 dB 단위로 바꿉니다.

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))  # Mel 그림용 좌우 두 칸을 만듭니다.
img1 = librosa.display.specshow(mel_db_normal, sr=sr, x_axis='time', y_axis='mel', ax=ax[0], cmap='viridis')  # 정상 Mel 그림입니다.
ax[0].set_title('Mel-Spectrogram - Normal Valve')  # 정상 Mel 제목입니다.

img2 = librosa.display.specshow(mel_db_anomaly, sr=sr, x_axis='time', y_axis='mel', ax=ax[1], cmap='viridis')  # 비정상 Mel 그림입니다.
ax[1].set_title('Mel-Spectrogram - Abnormal Valve')  # 비정상 Mel 제목입니다.

fig.colorbar(img1, ax=ax, format="%+2.0f dB")  # 색상 값 설명 막대를 붙입니다.
save_current_figure("step1_mel_spectrogram.png")  # Mel-Spectrogram 이미지를 저장합니다.
plt.show()  # 그래프를 화면에 보여줍니다.

print(f"변환된 2D Mel-Spectrogram 텐서 형태: {mel_db_normal.shape}")  # AI 입력 이미지의 세로/가로 크기를 보여줍니다.
with open(os.path.join(RESULTS_DIR, "step1_eda_audio_summary.json"), "w", encoding="utf-8") as f:  # 요약 json 파일을 엽니다.
    json.dump(
        {
            "normal_audio_path": normal_audio_path,  # 정상 예시 파일 경로입니다.
            "abnormal_audio_path": abnormal_audio_path,  # 비정상 예시 파일 경로입니다.
            "sample_rate": int(sr),  # 샘플링 레이트입니다.
            "normal_num_samples": int(len(y_normal)),  # 정상 소리 샘플 개수입니다.
            "mel_shape": list(mel_db_normal.shape),  # Mel-Spectrogram 모양입니다.
        },
        f,
        ensure_ascii=False,
        indent=2,
    )  # 요약 정보를 저장합니다.
