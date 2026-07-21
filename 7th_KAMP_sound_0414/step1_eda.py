import os
import glob
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
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
    import librosa.display
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "librosa가 설치되어 있지 않습니다. "
        f"현재 Python({sys.executable}) 기준으로 다음 명령을 실행해주세요: "
        f"'{sys.executable}' -m pip install -r 7th_KAMP_sound_0414/requirements.txt"
    ) from exc

# 경고 무시 및 시각화 설정
warnings.filterwarnings('ignore')
plt.style.use('ggplot')
plt.rcParams['font.family'] = 'Malgun Gothic' # 한글 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIS_DIR = os.path.join(BASE_DIR, "0_result")
RESULTS_DIR = os.path.join(BASE_DIR, "0_result")
os.makedirs(VIS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def save_current_figure(filename):
    save_path = os.path.join(VIS_DIR, filename)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"이미지 저장 완료: {save_path}")

# 2. 데이터 경로 설정
ok_path = os.path.join(BASE_DIR, 'FAN_sound_OK', '*')
err_path = os.path.join(BASE_DIR, 'FAN_sound_error', '*')

ok_files = glob.glob(ok_path)
err_files = glob.glob(err_path)

print(f"정상(OK) 데이터 개수: {len(ok_files)}")
print(f"이상(Error) 데이터 개수: {len(err_files)}")

if not ok_files or not err_files:
    raise FileNotFoundError(
        "오디오 파일을 찾지 못했습니다. "
        "스크립트와 같은 폴더 아래의 FAN_sound_OK, FAN_sound_error 폴더를 확인해주세요."
    )

# 3. Raw 사운드 데이터 시각화 (Sample)
# 첫 번째 정상 데이터와 이상 데이터를 로드하여 파형을 확인합니다.
sample_ok, sr_ok = librosa.load(ok_files[0]) 
sample_err, sr_err = librosa.load(err_files[0])

plt.figure(figsize=(15, 4))
plt.subplot(1, 2, 1)
librosa.display.waveshow(sample_ok, sr=sr_ok)
plt.title('정상(OK) 사운드 파형')
plt.xlabel('Time')
plt.ylabel('Amplitude')

plt.subplot(1, 2, 2)
librosa.display.waveshow(sample_err, sr=sr_err)
plt.title('이상(Error) 사운드 파형')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.tight_layout()
save_current_figure("step1_waveform.png")
plt.show()

# 4. 주파수 스펙트럼 (FFT) 분석 및 시각화
# 신호를 주파수 대역으로 분해하여 특징을 확인합니다. 대칭 구조이므로 절반(Half Spectrum)만 사용합니다
def plot_half_spectrum(y, sr, title):
    fft = np.fft.fft(y) 
    magnitude = np.abs(fft) 
    fre = np.linspace(0, sr, len(magnitude))
    
    haf_spectrum = magnitude[:int(len(magnitude)/2)] 
    haf_fre = fre[:int(len(magnitude)/2)]
    
    plt.plot(haf_fre, haf_spectrum)
    plt.title(title)
    plt.xlabel('Frequency')
    plt.ylabel('Magnitude')

plt.figure(figsize=(15, 4))
plt.subplot(1, 2, 1)
plot_half_spectrum(sample_ok, sr_ok, '정상(OK) Half Spectrum')

plt.subplot(1, 2, 2)
plot_half_spectrum(sample_err, sr_err, '이상(Error) Half Spectrum')
plt.tight_layout()
save_current_figure("step1_half_spectrum.png")
plt.show()

# 5. MFCC (Mel Frequency Cepstral Coefficient) 추출 및 시각화
# 소리의 고유한 특징을 추출합니다. 가이드북 기준 n_mfcc=13을 사용합니다
hop_length = 512 # 전체 frame 수 
n_fft = 2048 # frame 하나당 sample 수 

mfcc_ok = librosa.feature.mfcc(y=sample_ok, sr=sr_ok, n_fft=n_fft, hop_length=hop_length, n_mfcc=13)
mfcc_err = librosa.feature.mfcc(y=sample_err, sr=sr_err, n_fft=n_fft, hop_length=hop_length, n_mfcc=13)

plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
librosa.display.specshow(mfcc_ok, sr=sr_ok, hop_length=hop_length, x_axis='time')
plt.colorbar()
plt.title('정상(OK) MFCCs')
plt.ylabel('MFCC coefficients')

plt.subplot(1, 2, 2)
librosa.display.specshow(mfcc_err, sr=sr_err, hop_length=hop_length, x_axis='time')
plt.colorbar()
plt.title('이상(Error) MFCCs')
plt.ylabel('MFCC coefficients')
plt.tight_layout()
save_current_figure("step1_mfcc.png")
plt.show()

# 6. 상관관계 분석을 위한 데이터 프레임 구축 및 히트맵
# 가이드북에 따라 mfcc_min, mfcc_max, spectrum_min, spectrum_max 특성을 추출합니다
# (수업 진행 시 전체 데이터에 대해 반복문을 돌리는 부분의 축소판입니다.)

def get_features(y, sr):
    fft = np.fft.fft(y)
    magnitude = np.abs(fft)
    haf_spectrum = magnitude[:int(len(magnitude)/2)]
    
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_fft=2048, hop_length=512, n_mfcc=13)
    
    return np.min(haf_spectrum), np.max(haf_spectrum), np.min(mfcc), np.max(mfcc)

# 예시용 데이터 수집 리스트
spec_mins, spec_maxs, mfcc_mins, mfcc_maxs, labels = [], [], [], [], []

# 정상 데이터 Feature 추출 (샘플 10개만)
for path in ok_files[:10]:
    y, sr = librosa.load(path, sr=100) # 가이드북 명시에 따라 sr=100 적용
    s_min, s_max, m_min, m_max = get_features(y, sr)
    spec_mins.append(s_min); spec_maxs.append(s_max)
    mfcc_mins.append(m_min); mfcc_maxs.append(m_max)
    labels.append(0) # 정상 0

# 이상 데이터 Feature 추출 (샘플 10개만)
for path in err_files[:10]:
    y, sr = librosa.load(path, sr=100) # 가이드북 명시에 따라 sr=100 적용
    s_min, s_max, m_min, m_max = get_features(y, sr)
    spec_mins.append(s_min); spec_maxs.append(s_max)
    mfcc_mins.append(m_min); mfcc_maxs.append(m_max)
    labels.append(1) # 이상 1

# DataFrame 생성
df_features = pd.DataFrame({
    'mfcc_min': mfcc_mins,
    'mfcc_max': mfcc_maxs,
    'spectrum_min': spec_mins,
    'spectrum_max': spec_maxs,
    'NG': labels
})

# 상관계수 히트맵 시각화
plt.figure(figsize=(8, 6))
# spectrum_max는 상관관계가 낮아 후진 소거법을 통해 제외를 고려할 수 있습니다
corr = df_features.iloc[:, :-1].corr() 
sns.heatmap(corr, annot=True, cmap='Greens', annot_kws={'size':15}) 
plt.title('Feature Correlation Heatmap')
save_current_figure("step1_feature_correlation_heatmap.png")
plt.show()

summary = {
    "ok_file_count": len(ok_files),
    "error_file_count": len(err_files),
    "feature_sample_count": len(df_features),
    "feature_columns": df_features.columns.tolist(),
}
with open(os.path.join(RESULTS_DIR, "step1_eda_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
df_features.to_csv(os.path.join(RESULTS_DIR, "step1_feature_samples.csv"), index=False)
