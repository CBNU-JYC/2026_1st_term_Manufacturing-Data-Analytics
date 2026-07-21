import os
import glob
import sys
import json
import numpy as np
import pandas as pd
import importlib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib # 모델 저장을 위한 라이브러리 추가

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

# 1. 주파수 스펙트럼 추출 함수 정의
def mk_Frequency(y, sr):
    fft = np.fft.fft(y)
    magnitude = np.abs(fft)
    fre = np.linspace(0, sr, len(magnitude))
    
    haf_spectrum = magnitude[:int(len(magnitude)/2)]
    haf_fre = fre[:int(len(magnitude)/2)]
    return haf_spectrum, haf_fre

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "0_result")
os.makedirs(RESULTS_DIR, exist_ok=True)

# 2. 데이터 경로 설정 및 특성 추출 (EDA 단계와 동일)
ok_path = os.path.join(BASE_DIR, 'FAN_sound_OK', '*')
err_path = os.path.join(BASE_DIR, 'FAN_sound_error', '*')

ok_files = glob.glob(ok_path)
err_files = glob.glob(err_path)

if not ok_files or not err_files:
    raise FileNotFoundError(
        "오디오 파일을 찾지 못했습니다. "
        "스크립트와 같은 폴더 아래의 FAN_sound_OK, FAN_sound_error 폴더를 확인해주세요."
    )

spectrum_mins, spectrum_maxs, mfcc_mins, mfcc_maxs, labels = [], [], [], [], []

print("데이터 특성 추출을 시작합니다...")
# (학생들에게는 이전에 작성한 반복문 코드가 이 자리에 들어간다고 설명해주시면 됩니다)
for path in ok_files:
    y, sr = librosa.load(path, sr=100)
    haf_spectrum, _ = mk_Frequency(y, sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_fft=2048, hop_length=512, n_mfcc=13)
    spectrum_mins.append(np.min(haf_spectrum)); spectrum_maxs.append(np.max(haf_spectrum))
    mfcc_mins.append(np.min(mfcc)); mfcc_maxs.append(np.max(mfcc))
    labels.append(0)

for path in err_files:
    y, sr = librosa.load(path, sr=100)
    haf_spectrum, _ = mk_Frequency(y, sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_fft=2048, hop_length=512, n_mfcc=13)
    spectrum_mins.append(np.min(haf_spectrum)); spectrum_maxs.append(np.max(haf_spectrum))
    mfcc_mins.append(np.min(mfcc)); mfcc_maxs.append(np.max(mfcc))
    labels.append(1)

df_sound = pd.DataFrame({
    'mfcc_min': mfcc_mins, 'mfcc_max': mfcc_maxs,
    'spectrum_min': spectrum_mins, 'spectrum_max': spectrum_maxs,
    'NG': labels,
    'filepath': ok_files + err_files
})
df_sound.to_csv(os.path.join(RESULTS_DIR, 'step2_extracted_features.csv'), index=False)

# 3. 데이터 분리 및 저장
data = df_sound[['mfcc_min', 'mfcc_max', 'spectrum_min', 'filepath']] 
target = df_sound['NG']

X_train, X_test, y_train, y_test = train_test_split(
    data, target, test_size=0.3, shuffle=True, stratify=target, random_state=34
)

# 파일 경로는 학습에 사용되지 않으므로 따로 분리합니다.
test_filepaths = X_test['filepath']
X_train = X_train.drop(columns=['filepath'])
X_test = X_test.drop(columns=['filepath'])

# 차후 평가 파일에서 사용하기 위해 테스트 데이터를 CSV로 저장합니다.
X_test.to_csv(os.path.join(RESULTS_DIR, 'X_test.csv'), index=False)
y_test.to_csv(os.path.join(RESULTS_DIR, 'y_test.csv'), index=False)
test_filepaths.to_csv(os.path.join(RESULTS_DIR, 'test_filepaths.csv'), index=False)
print("테스트 데이터(X_test.csv, y_test.csv, test_filepaths.csv)를 성공적으로 저장했습니다.")

# 4. 모델링 및 학습
print("의사결정나무 모델 학습을 시작합니다...")
Dtc = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
Dtc.fit(X_train, y_train)

# 5. 학습된 모델 저장 (.pkl 형식)
joblib.dump(Dtc, os.path.join(RESULTS_DIR, 'dtc_sound_model.pkl'))
print("모델이 'dtc_sound_model.pkl' 이름으로 성공적으로 저장되었습니다!")
with open(os.path.join(RESULTS_DIR, "step2_train_summary.json"), "w", encoding="utf-8") as f:
    json.dump(
        {
            "total_samples": len(df_sound),
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "feature_columns": X_train.columns.tolist(),
        },
        f,
        ensure_ascii=False,
        indent=2,
    )
