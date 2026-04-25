"""
프로그램 전체 흐름 설명:
1. 학습된 팬 소리 의사결정나무 모델을 불러옵니다.
2. 새 오디오 파일을 읽고 학습 때와 같은 방식으로 특징을 뽑습니다.
3. 모델에 특징을 넣어 정상인지 이상인지 예측합니다.
4. 예측 결과와 신뢰도를 화면에 보여주고 CSV 파일로 저장합니다.
"""

import os  # 파일 경로를 만들 때 사용합니다.
import sys  # 현재 Python 실행 경로를 확인할 때 사용합니다.
import importlib  # 라이브러리 설치 여부를 확인할 때 사용합니다.
import warnings  # 경고 메시지를 숨길 때 사용합니다.
import numpy as np  # FFT와 숫자 계산에 사용합니다.
import pandas as pd  # 예측 결과를 표와 CSV로 저장할 때 사용합니다.
import joblib  # 저장된 모델을 불러올 때 사용합니다.

print(f"현재 Python 실행 경로: {sys.executable}")  # 실행 중인 Python 위치를 출력합니다.

try:  # numba가 설치되어 있는지 확인합니다.
    numba = importlib.import_module("numba")  # numba를 불러옵니다.
except ModuleNotFoundError as exc:  # numba가 없으면 설치 안내를 보여줍니다.
    raise ModuleNotFoundError(
        "numba가 설치되어 있지 않습니다. "
        f"현재 Python({sys.executable}) 기준으로 다음 명령을 실행해주세요: "
        f"'{sys.executable}' -m pip install -r 7th_KAMP_sound_0414/requirements.txt"
    ) from exc


def _disable_numba_cache(decorator):
    """
    numba의 캐시 기능을 끈 상태로 데코레이터를 실행하게 만듭니다.

    Args:
        decorator: numba.jit 같은 데코레이터 함수입니다.

    Returns:
        function: cache=False를 자동으로 넣는 포장 함수입니다.
    """

    def wrapped(*args, **kwargs):
        """
        numba 데코레이터에 cache=False 옵션을 추가합니다.

        Args:
            *args: 데코레이터에 전달되는 위치 인자입니다.
            **kwargs: 데코레이터에 전달되는 이름 인자입니다.

        Returns:
            object: 원래 데코레이터의 실행 결과입니다.
        """

        kwargs["cache"] = False  # 캐시 파일 생성 문제를 피하기 위해 끕니다.
        return decorator(*args, **kwargs)  # 원래 기능을 실행합니다.

    return wrapped  # 포장 함수를 돌려줍니다.


numba.jit = _disable_numba_cache(numba.jit)  # jit 캐시를 끕니다.
numba.vectorize = _disable_numba_cache(numba.vectorize)  # vectorize 캐시를 끕니다.
numba.guvectorize = _disable_numba_cache(numba.guvectorize)  # guvectorize 캐시를 끕니다.

try:  # librosa가 있는지 확인합니다.
    import librosa  # 오디오 파일을 읽고 MFCC를 뽑는 도구입니다.
except ModuleNotFoundError as exc:  # librosa가 없으면 설치 안내를 보여줍니다.
    raise ModuleNotFoundError(
        "librosa가 설치되어 있지 않습니다. "
        f"현재 Python({sys.executable}) 기준으로 다음 명령을 실행해주세요: "
        f"'{sys.executable}' -m pip install -r 7th_KAMP_sound_0414/requirements.txt"
    ) from exc

warnings.filterwarnings('ignore')  # 실습 중 불필요한 경고를 숨깁니다.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 현재 코드가 있는 폴더입니다.
RESULTS_DIR = os.path.join(BASE_DIR, "results")  # 추론 결과를 저장할 폴더입니다.
os.makedirs(RESULTS_DIR, exist_ok=True)  # 결과 폴더가 없으면 만듭니다.


def mk_Frequency(y, sr):
    """
    소리 배열을 주파수 스펙트럼으로 바꾸고 앞 절반만 반환합니다.

    Args:
        y: 오디오를 숫자로 바꾼 배열입니다.
        sr: 샘플링 레이트입니다.

    Returns:
        tuple: 절반 스펙트럼 배열과 절반 주파수 배열입니다.
    """

    fft = np.fft.fft(y)  # 소리를 주파수 성분으로 분해합니다.
    magnitude = np.abs(fft)  # 주파수 성분의 크기만 가져옵니다.
    fre = np.linspace(0, sr, len(magnitude))  # 주파수 눈금을 만듭니다.

    # FFT 결과는 뒤쪽 절반이 앞쪽 정보와 겹치므로 앞 절반만 사용합니다.
    haf_spectrum = magnitude[:int(len(magnitude) / 2)]  # 스펙트럼 앞 절반입니다.
    haf_fre = fre[:int(len(magnitude) / 2)]  # 주파수 눈금 앞 절반입니다.
    return haf_spectrum, haf_fre  # 두 배열을 반환합니다.


def predict_fan_status(audio_path, model_path=None):
    """
    새 팬 소리 파일 하나를 정상 또는 이상으로 예측합니다.

    Args:
        audio_path: 분석할 오디오 파일 경로입니다. 상대 경로면 현재 폴더 기준으로 찾습니다.
        model_path: 사용할 모델 파일 경로입니다. 없으면 기본 모델 파일을 사용합니다.

    Returns:
        dict | None: 예측 결과와 특징값이 담긴 딕셔너리입니다. 모델이 없으면 None을 반환합니다.
    """

    print(f"[{audio_path}] 파일 분석을 시작합니다...")  # 어떤 파일을 분석하는지 알려줍니다.

    if model_path is None:  # 모델 경로를 따로 주지 않았다면 기본 경로를 씁니다.
        model_path = os.path.join(BASE_DIR, 'dtc_sound_model.pkl')  # Step 2에서 저장한 모델 경로입니다.

    if not os.path.isabs(audio_path):  # 오디오 경로가 절대 경로가 아니면 현재 폴더 기준으로 바꿉니다.
        audio_path = os.path.join(BASE_DIR, audio_path)  # 상대 경로를 절대 경로처럼 사용할 수 있게 합칩니다.

    try:  # 모델 파일을 불러옵니다.
        model = joblib.load(model_path)  # 저장된 의사결정나무 모델입니다.
    except FileNotFoundError:  # 모델 파일이 없으면 학습을 먼저 해야 합니다.
        print("오류: 모델 파일을 찾을 수 없습니다. 학습 코드를 먼저 실행하여 모델을 저장해주세요.")
        return None  # 예측 결과가 없음을 알려줍니다.

    y, sr = librosa.load(audio_path, sr=100)  # 학습 때와 같은 100Hz 기준으로 오디오를 읽습니다.

    haf_spectrum, _ = mk_Frequency(y, sr)  # 주파수 특징을 만듭니다.
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_fft=2048, hop_length=512, n_mfcc=13)  # MFCC 특징을 만듭니다.

    s_min = np.min(haf_spectrum)  # 스펙트럼 최솟값입니다.
    s_max = np.max(haf_spectrum)  # 스펙트럼 최댓값입니다. 여기서는 결과 기록용으로만 둡니다.
    m_min = np.min(mfcc)  # MFCC 최솟값입니다.
    m_max = np.max(mfcc)  # MFCC 최댓값입니다.

    input_data = pd.DataFrame({
        'mfcc_min': [m_min],  # 학습 때 사용한 첫 번째 특징입니다.
        'mfcc_max': [m_max],  # 학습 때 사용한 두 번째 특징입니다.
        'spectrum_min': [s_min]  # 학습 때 사용한 세 번째 특징입니다.
    })  # 모델 입력은 학습 때 열 이름과 순서가 맞아야 안전합니다.

    prediction = model.predict(input_data)[0]  # 모델이 0 또는 1로 예측합니다.

    probabilities = model.predict_proba(input_data)[0]  # 각 클래스일 확률을 가져옵니다.
    confidence = probabilities[prediction] * 100  # 예측한 답에 대한 확률을 퍼센트로 바꿉니다.

    status = "이상(Error)" if prediction == 1 else "정상(OK)"  # 숫자 결과를 사람이 읽는 글자로 바꿉니다.

    print("\n==================================")  # 결과 박스 윗줄입니다.
    print("         [ AI 진단 결과 ]         ")  # 결과 제목입니다.
    print("==================================")  # 구분선입니다.
    print(f"진단 상태 : {status}")  # 정상/이상 결과입니다.
    print(f"신 뢰 도  : {confidence:.2f}%")  # 모델의 예측 확률입니다.
    print("==================================\n")  # 결과 박스 아랫줄입니다.

    result = {
        "audio_path": audio_path,  # 분석한 오디오 파일 경로입니다.
        "prediction": int(prediction),  # 0 또는 1 예측값입니다.
        "status": status,  # 글자로 된 예측 결과입니다.
        "confidence": round(float(confidence), 2),  # 소수 둘째 자리까지의 신뢰도입니다.
        "mfcc_min": float(m_min),  # 예측에 사용한 MFCC 최솟값입니다.
        "mfcc_max": float(m_max),  # 예측에 사용한 MFCC 최댓값입니다.
        "spectrum_min": float(s_min),  # 예측에 사용한 스펙트럼 최솟값입니다.
        "spectrum_max_record_only": float(s_max),  # 참고용으로 남긴 스펙트럼 최댓값입니다.
    }  # 결과를 나중에 CSV로 저장하기 좋은 딕셔너리로 만듭니다.
    return result  # 예측 결과를 돌려줍니다.


if __name__ == "__main__":  # 이 파일을 직접 실행했을 때만 아래 예시가 실행됩니다.
    test_audio_ok = 'FAN_sound_OK/FAN_sound_01.wav'  # 정상 파일 예시 경로입니다.
    results = []  # 여러 예측 결과를 담을 리스트입니다.
    results.append(predict_fan_status(test_audio_ok))  # 정상 예시 파일을 예측하고 결과를 저장합니다.

    test_audio_err = 'FAN_sound_error/FAN_sound_error_01.wav'  # 이상 파일 예시 경로입니다.
    results.append(predict_fan_status(test_audio_err))  # 이상 예시 파일을 예측하고 결과를 저장합니다.

    results_df = pd.DataFrame(results)  # 결과 리스트를 표로 바꿉니다.
    results_csv_path = os.path.join(RESULTS_DIR, "step4_inference_results.csv")  # 저장할 CSV 경로입니다.
    results_df.to_csv(results_csv_path, index=False)  # 추론 결과를 CSV로 저장합니다.
    print(f"추론 결과 CSV 저장 완료: {results_csv_path}")  # 저장 위치를 알려줍니다.

    print("팁: 위 코드의 주석을 풀고 실제 음원 파일 경로를 넣어 AI의 진단 결과를 확인해보세요")  # 사용 팁을 출력합니다.
