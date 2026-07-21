"""
프로그램 전체 흐름 설명:
1. 정상/이상 팬 소리 파일을 모두 찾습니다.
2. 각 소리에서 주파수 특징과 MFCC 특징을 숫자로 뽑습니다.
3. 표를 만들고 학습용 데이터와 테스트용 데이터로 나눕니다.
4. 의사결정나무 모델을 학습시킵니다.
5. 학습된 모델과 테스트 데이터를 파일로 저장합니다.
"""

import os  # 폴더와 파일 경로를 다루기 위해 사용합니다.
import glob  # 여러 오디오 파일을 한 번에 찾기 위해 사용합니다.
import sys  # 현재 Python 실행 파일 위치를 출력하기 위해 사용합니다.
import json  # 학습 요약을 json 파일로 저장하기 위해 사용합니다.
import numpy as np  # FFT와 최솟값/최댓값 계산에 사용합니다.
import pandas as pd  # 특징을 표 형태로 저장하기 위해 사용합니다.
import importlib  # 라이브러리 설치 여부를 확인하기 위해 사용합니다.
from sklearn.model_selection import train_test_split  # 데이터를 학습용과 테스트용으로 나눕니다.
from sklearn.tree import DecisionTreeClassifier  # 의사결정나무 분류 모델입니다.
import joblib  # 학습된 모델을 파일로 저장하고 다시 불러올 때 사용합니다.

print(f"현재 Python 실행 경로: {sys.executable}")  # 실행 중인 Python 위치를 알려줍니다.

try:  # numba 라이브러리가 있는지 확인합니다.
    numba = importlib.import_module("numba")  # numba를 불러옵니다.
except ModuleNotFoundError as exc:  # numba가 없으면 설치 안내를 보여줍니다.
    raise ModuleNotFoundError(
        "numba가 설치되어 있지 않습니다. "
        f"현재 Python({sys.executable}) 기준으로 다음 명령을 실행해주세요: "
        f"'{sys.executable}' -m pip install -r 7th_KAMP_sound_0414/requirements.txt"
    ) from exc


def _disable_numba_cache(decorator):
    """
    numba의 캐시 저장 기능을 끄는 포장 함수를 만듭니다.

    Args:
        decorator: numba.jit, numba.vectorize 같은 numba 데코레이터입니다.

    Returns:
        function: cache=False 옵션을 자동으로 넣는 새 함수입니다.
    """

    def wrapped(*args, **kwargs):
        """
        원래 numba 데코레이터를 cache=False 옵션과 함께 실행합니다.

        Args:
            *args: 원래 함수에 전달될 위치 인자들입니다.
            **kwargs: 원래 함수에 전달될 이름 인자들입니다.

        Returns:
            object: numba 데코레이터가 만든 결과입니다.
        """

        kwargs["cache"] = False  # 캐시 파일 때문에 생길 수 있는 문제를 피합니다.
        return decorator(*args, **kwargs)  # 원래 데코레이터 기능을 실행합니다.

    return wrapped  # 포장 함수를 돌려줍니다.


numba.jit = _disable_numba_cache(numba.jit)  # jit 캐시를 끕니다.
numba.vectorize = _disable_numba_cache(numba.vectorize)  # vectorize 캐시를 끕니다.
numba.guvectorize = _disable_numba_cache(numba.guvectorize)  # guvectorize 캐시를 끕니다.

try:  # librosa가 있는지 확인합니다.
    import librosa  # 오디오 파일을 읽고 MFCC를 만들 때 사용합니다.
except ModuleNotFoundError as exc:  # librosa가 없으면 설치 안내를 보여줍니다.
    raise ModuleNotFoundError(
        "librosa가 설치되어 있지 않습니다. "
        f"현재 Python({sys.executable}) 기준으로 다음 명령을 실행해주세요: "
        f"'{sys.executable}' -m pip install -r 7th_KAMP_sound_0414/requirements.txt"
    ) from exc


def mk_Frequency(y, sr):
    """
    소리 신호를 주파수 정보로 바꾸고 앞 절반만 반환합니다.

    Args:
        y: 오디오를 숫자로 바꾼 배열입니다.
        sr: 샘플링 레이트입니다.

    Returns:
        tuple: 절반 스펙트럼 값 배열과 그에 대응하는 주파수 배열입니다.
    """

    fft = np.fft.fft(y)  # 시간 순서의 소리를 주파수 성분으로 바꿉니다.
    magnitude = np.abs(fft)  # 복소수 결과에서 크기만 뽑습니다.
    fre = np.linspace(0, sr, len(magnitude))  # 주파수 눈금을 만듭니다.

    # FFT는 대칭 성질이 있어 절반만 사용하면 계산과 저장을 줄일 수 있습니다.
    haf_spectrum = magnitude[:int(len(magnitude) / 2)]  # 스펙트럼 앞 절반만 가져옵니다.
    haf_fre = fre[:int(len(magnitude) / 2)]  # 주파수 눈금도 앞 절반만 가져옵니다.
    return haf_spectrum, haf_fre  # 특징값과 주파수를 함께 돌려줍니다.


BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 현재 코드 파일의 폴더입니다.
RESULTS_DIR = os.path.join(BASE_DIR, "0_result")  # 결과 파일을 모아둘 폴더입니다.
os.makedirs(RESULTS_DIR, exist_ok=True)  # 결과 폴더가 없으면 만듭니다.

ok_path = os.path.join(BASE_DIR, 'FAN_sound_OK', '*')  # 정상 소리 파일을 찾을 경로 규칙입니다.
err_path = os.path.join(BASE_DIR, 'FAN_sound_error', '*')  # 이상 소리 파일을 찾을 경로 규칙입니다.

ok_files = glob.glob(ok_path)  # 정상 파일 목록입니다.
err_files = glob.glob(err_path)  # 이상 파일 목록입니다.

if not ok_files or not err_files:  # 데이터가 없으면 학습할 수 없습니다.
    raise FileNotFoundError(
        "오디오 파일을 찾지 못했습니다. "
        "스크립트와 같은 폴더 아래의 FAN_sound_OK, FAN_sound_error 폴더를 확인해주세요."
    )

spectrum_mins, spectrum_maxs, mfcc_mins, mfcc_maxs, labels = [], [], [], [], []  # 특징과 정답을 담을 리스트입니다.

print("데이터 특성 추출을 시작합니다...")  # 작업 시작을 알려줍니다.
for path in ok_files:  # 모든 정상 소리 파일을 하나씩 처리합니다.
    y, sr = librosa.load(path, sr=100)  # 같은 기준으로 비교하려고 100Hz로 읽습니다.
    haf_spectrum, _ = mk_Frequency(y, sr)  # 주파수 특징을 뽑습니다.
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_fft=2048, hop_length=512, n_mfcc=13)  # MFCC 특징을 뽑습니다.
    spectrum_mins.append(np.min(haf_spectrum))  # 스펙트럼 최솟값을 저장합니다.
    spectrum_maxs.append(np.max(haf_spectrum))  # 스펙트럼 최댓값을 저장합니다.
    mfcc_mins.append(np.min(mfcc))  # MFCC 최솟값을 저장합니다.
    mfcc_maxs.append(np.max(mfcc))  # MFCC 최댓값을 저장합니다.
    labels.append(0)  # 정상은 0으로 표시합니다.

for path in err_files:  # 모든 이상 소리 파일을 하나씩 처리합니다.
    y, sr = librosa.load(path, sr=100)  # 정상과 같은 방식으로 읽습니다.
    haf_spectrum, _ = mk_Frequency(y, sr)  # 주파수 특징을 뽑습니다.
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_fft=2048, hop_length=512, n_mfcc=13)  # MFCC 특징을 뽑습니다.
    spectrum_mins.append(np.min(haf_spectrum))  # 스펙트럼 최솟값을 저장합니다.
    spectrum_maxs.append(np.max(haf_spectrum))  # 스펙트럼 최댓값을 저장합니다.
    mfcc_mins.append(np.min(mfcc))  # MFCC 최솟값을 저장합니다.
    mfcc_maxs.append(np.max(mfcc))  # MFCC 최댓값을 저장합니다.
    labels.append(1)  # 이상은 1로 표시합니다.

df_sound = pd.DataFrame({
    'mfcc_min': mfcc_mins,  # MFCC 최솟값 열입니다.
    'mfcc_max': mfcc_maxs,  # MFCC 최댓값 열입니다.
    'spectrum_min': spectrum_mins,  # 스펙트럼 최솟값 열입니다.
    'spectrum_max': spectrum_maxs,  # 스펙트럼 최댓값 열입니다.
    'NG': labels,  # 정답 열입니다. 0은 정상, 1은 이상입니다.
    'filepath': ok_files + err_files  # 나중에 어떤 파일인지 확인하기 위한 경로입니다.
})  # 여러 특징 리스트를 하나의 표로 합칩니다.
df_sound.to_csv(os.path.join(RESULTS_DIR, 'step2_extracted_features.csv'), index=False)  # 추출한 특징 표를 저장합니다.

data = df_sound[['mfcc_min', 'mfcc_max', 'spectrum_min', 'filepath']]  # 모델 입력으로 쓸 열을 고릅니다.
target = df_sound['NG']  # 모델이 맞혀야 하는 정답 열입니다.

X_train, X_test, y_train, y_test = train_test_split(
    data, target, test_size=0.3, shuffle=True, stratify=target, random_state=34
)  # 데이터를 학습 70%, 테스트 30%로 나누되 정상/이상 비율을 비슷하게 유지합니다.

test_filepaths = X_test['filepath']  # 테스트 파일 경로는 따로 보관합니다.
X_train = X_train.drop(columns=['filepath'])  # 파일 경로는 숫자 특징이 아니므로 학습에서 뺍니다.
X_test = X_test.drop(columns=['filepath'])  # 테스트 입력에서도 파일 경로를 뺍니다.

X_test.to_csv(os.path.join(RESULTS_DIR, 'X_test.csv'), index=False)  # 평가 단계에서 쓸 테스트 입력을 저장합니다.
y_test.to_csv(os.path.join(RESULTS_DIR, 'y_test.csv'), index=False)  # 평가 단계에서 쓸 테스트 정답을 저장합니다.
test_filepaths.to_csv(os.path.join(RESULTS_DIR, 'test_filepaths.csv'), index=False)  # 오분류 분석에 쓸 파일 경로를 저장합니다.
print("테스트 데이터(X_test.csv, y_test.csv, test_filepaths.csv)를 성공적으로 저장했습니다.")  # 저장 완료를 알려줍니다.

print("의사결정나무 모델 학습을 시작합니다...")  # 모델 학습 시작을 알려줍니다.
Dtc = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)  # 질문을 최대 3단계만 하는 단순한 나무 모델을 만듭니다.
Dtc.fit(X_train, y_train)  # 학습 데이터로 정상과 이상을 구분하는 규칙을 배웁니다.

joblib.dump(Dtc, os.path.join(RESULTS_DIR, 'dtc_sound_model.pkl'))  # 학습된 모델을 파일로 저장합니다.
print("모델이 'dtc_sound_model.pkl' 이름으로 성공적으로 저장되었습니다!")  # 저장 완료를 알려줍니다.
with open(os.path.join(RESULTS_DIR, "step2_train_summary.json"), "w", encoding="utf-8") as f:  # 학습 요약 파일을 엽니다.
    json.dump(
        {
            "total_samples": len(df_sound),  # 전체 샘플 개수입니다.
            "train_samples": int(len(X_train)),  # 학습에 쓴 샘플 개수입니다.
            "test_samples": int(len(X_test)),  # 테스트에 남겨둔 샘플 개수입니다.
            "feature_columns": X_train.columns.tolist(),  # 모델이 실제로 본 특징 이름입니다.
        },
        f,
        ensure_ascii=False,
        indent=2,
    )  # 요약 정보를 json으로 저장합니다.
