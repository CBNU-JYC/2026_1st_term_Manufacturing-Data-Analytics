import numpy as np
import pandas as pd
import joblib
import warnings
import os
import sys
import importlib

print(f"현재 Python 실행 경로: {sys.executable}")

try:
    numba = importlib.import_module("numba")
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "numba가 설치되어 있지 않습니다. "
        f"현재 Python({sys.executable}) 기준으로 다음 명령을 실행해주세요: "
        f"'{sys.executable}' -m pip install -r 7th_KAMP_sound_0414/requirements.txt"
    ) from exc

def _disable_numba_cache(decorator):
    def wrapped(*args, **kwargs):
        kwargs["cache"] = False
        return decorator(*args, **kwargs)
    return wrapped

numba.jit = _disable_numba_cache(numba.jit)
numba.vectorize = _disable_numba_cache(numba.vectorize)
numba.guvectorize = _disable_numba_cache(numba.guvectorize)

try:
    import librosa
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "librosa가 설치되어 있지 않습니다. "
        f"현재 Python({sys.executable}) 기준으로 다음 명령을 실행해주세요: "
        f"'{sys.executable}' -m pip install -r 7th_KAMP_sound_0414/requirements.txt"
    ) from exc

warnings.filterwarnings('ignore')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# 1. 주파수 스펙트럼 추출 함수 정의 (학습 단계와 동일)
def mk_Frequency(y, sr):
    fft = np.fft.fft(y)
    magnitude = np.abs(fft)
    fre = np.linspace(0, sr, len(magnitude))
    
    haf_spectrum = magnitude[:int(len(magnitude)/2)]
    haf_fre = fre[:int(len(magnitude)/2)]
    return haf_spectrum, haf_fre

# 2. 새로운 사운드 파일에 대한 추론 함수 정의
def predict_fan_status(audio_path, model_path=None):
    print(f"[{audio_path}] 파일 분석을 시작합니다...")

    if model_path is None:
        model_path = os.path.join(BASE_DIR, 'dtc_sound_model.pkl')

    if not os.path.isabs(audio_path):
        audio_path = os.path.join(BASE_DIR, audio_path)

    try:
        # 1) 학습된 모델 불러오기 (Load)
        model = joblib.load(model_path)
    except FileNotFoundError:
        print("오류: 모델 파일을 찾을 수 없습니다. 학습 코드를 먼저 실행하여 모델을 저장해주세요.")
        return
    
    # 2) 사운드 데이터 로드
    # 가이드북 기준에 따라 샘플링 레이트(sr)를 100으로 설정합니다
    y, sr = librosa.load(audio_path, sr=100) 
    
    # 3) 특징(Feature) 추출
    # 주파수 스펙트럼 변환 및 MFCC 추출 (n_mfcc=13, n_fft=2048, hop_length=512)
    haf_spectrum, _ = mk_Frequency(y, sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_fft=2048, hop_length=512, n_mfcc=13) 
    
    # 학습 시 사용했던 특성 추출
    s_min = np.min(haf_spectrum)
    s_max = np.max(haf_spectrum)
    m_min = np.min(mfcc)
    m_max = np.max(mfcc)
    
    # 4) 모델 입력용 데이터프레임 생성 (학습 데이터의 컬럼명과 일치해야 함)
    input_data = pd.DataFrame({
        'mfcc_min': [m_min],
        'mfcc_max': [m_max],
        'spectrum_min': [s_min]
    })
    
    # 5) 이상 여부 추론 (Prediction)
    prediction = model.predict(input_data)[0] # 0: 정상, 1: 이상
    
    # 의사결정나무의 예측 확률값(Confidence)도 함께 확인 가능합니다.
    probabilities = model.predict_proba(input_data)[0]
    confidence = probabilities[prediction] * 100
    
    # 6) 결과 출력
    status = "이상(Error)" if prediction == 1 else "정상(OK)"

    print("\n==================================")
    print("         [ AI 진단 결과 ]         ")
    print("==================================")
    print(f"진단 상태 : {status}")
    print(f"신 뢰 도  : {confidence:.2f}%")
    print("==================================\n")

    result = {
        "audio_path": audio_path,
        "prediction": int(prediction),
        "status": status,
        "confidence": round(float(confidence), 2),
        "mfcc_min": float(m_min),
        "mfcc_max": float(m_max),
        "spectrum_min": float(s_min),
    }
    return result


# 3. 실제 추론 테스트 실행
if __name__ == "__main__":
    # 테스트해보고 싶은 사운드 파일의 경로를 입력하세요.
    # 예시로 폴더 내 파일 1개를 지정합니다. 실제 파일명에 맞게 수정하여 사용하세요.
    
    # 1. 정상 파일 테스트 예시
    test_audio_ok = 'FAN_sound_OK/FAN_sound_01.wav' # 실제 경로로 수정 필요
    results = []
    results.append(predict_fan_status(test_audio_ok))
    
    # 2. 이상 파일 테스트 예시
    test_audio_err = 'FAN_sound_error/FAN_sound_error_01.wav' # 실제 경로로 수정 필요
    results.append(predict_fan_status(test_audio_err))

    results_df = pd.DataFrame(results)
    results_csv_path = os.path.join(RESULTS_DIR, "step4_inference_results.csv")
    results_df.to_csv(results_csv_path, index=False)
    print(f"추론 결과 CSV 저장 완료: {results_csv_path}")
    
    print("팁: 위 코드의 주석을 풀고 실제 음원 파일 경로를 넣어 AI의 진단 결과를 확인해보세요")
