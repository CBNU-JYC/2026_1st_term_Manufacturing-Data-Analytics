import os  # 환경 변수와 폴더 경로 설정을 위해 사용합니다.
import pandas as pd  # CSV 파일을 표 형태 데이터프레임으로 읽기 위해 사용합니다.
import numpy as np  # 수치 계산용 라이브러리입니다. 이 파일에서는 보조적으로 함께 불러옵니다.
from pathlib import Path  # 현재 스크립트 위치를 기준으로 안정적인 경로를 만들기 위해 사용합니다.

BASE_DIR = Path(__file__).resolve().parent  # 이 파이썬 파일이 들어 있는 실제 폴더 경로를 구합니다.
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".matplotlib"))  # matplotlib 캐시 폴더를 현재 프로젝트 폴더 안으로 지정합니다.
os.environ.setdefault("XDG_CACHE_HOME", str(BASE_DIR / ".cache"))  # 글꼴 캐시 등도 현재 프로젝트 폴더 안으로 지정합니다.
(BASE_DIR / ".matplotlib").mkdir(parents=True, exist_ok=True)  # matplotlib 캐시 폴더가 없으면 자동으로 만듭니다.
(BASE_DIR / ".cache").mkdir(parents=True, exist_ok=True)  # 일반 캐시 폴더가 없으면 자동으로 만듭니다.
import matplotlib  # 그래프 출력 방식을 설정하기 위해 먼저 불러옵니다.
matplotlib.use("Agg")  # 화면에 띄우지 않고 PNG 파일로 저장하는 모드로 설정합니다.
import matplotlib.pyplot as plt  # 그래프를 그리는 기본 도구입니다.
import seaborn as sns  # 히스토그램, 히트맵, countplot을 더 보기 좋게 그리기 위한 라이브러리입니다.

# 시각화 설정
plt.style.use('seaborn-v0_8-whitegrid')  # 흰 배경과 격자가 있는 시각화 스타일을 적용합니다.
plt.rcParams['figure.figsize'] = (10, 6)  # 기본 그래프 크기를 가로 10, 세로 6으로 설정합니다.

# 데이터 로드
df = pd.read_csv(BASE_DIR / 'ai4i2020.csv')  # 현재 폴더 안의 AI4I 2020 CSV 데이터를 읽습니다.

print(f"데이터셋 형태: {df.shape}")  # 데이터의 행 개수와 열 개수를 출력합니다.
print(df.head())  # 앞의 5개 행을 보여주어 데이터 형태를 빠르게 확인합니다.

# 결측치 및 데이터 타입 확인
df.info()  # 각 열의 자료형, 결측치 개수, 메모리 사용량 등을 출력합니다.

# 주요 센서 및 공정 변수
sensor_cols = [  # 분포와 상관관계를 볼 핵심 센서 변수 이름 목록입니다.
    'Air temperature [K]', 'Process temperature [K]',
    'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]'
]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))  # 2행 3열짜리 서브플롯 캔버스를 생성합니다.
axes = axes.flatten()  # 2차원 배열 형태의 축 정보를 1차원으로 펴서 다루기 쉽게 만듭니다.

for i, col in enumerate(sensor_cols):  # 센서 변수들을 순서대로 꺼내면서 번호도 함께 받습니다.
    sns.histplot(df[col], kde=True, ax=axes[i], color='steelblue')  # 각 센서값 분포를 히스토그램과 KDE 곡선으로 그립니다.
    axes[i].set_title(f'Distribution of {col}')  # 각 그래프의 제목에 센서 이름을 표시합니다.

# 남는 그래프 공간 숨기기
axes[5].set_visible(False)  # 6칸 중 마지막 한 칸은 사용하지 않으므로 보이지 않게 합니다.
plt.tight_layout()  # 그래프 간 간격을 자동으로 정리합니다.
plt.savefig(BASE_DIR / 'eda_sensor_distributions.png', dpi=150)  # 센서 분포 그래프를 PNG 파일로 저장합니다.
plt.close()  # 현재 그래프 객체를 닫아 메모리를 정리합니다.

# 센서 변수들 간의 피어슨 상관계수 계산
corr_matrix = df[sensor_cols].corr()  # 센서 변수들끼리의 선형 상관계수를 계산합니다.

plt.figure(figsize=(8, 6))  # 상관계수 히트맵용 새 그림 크기를 지정합니다.
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f")  # 상관계수를 색상과 숫자로 같이 보여주는 히트맵을 그립니다.
plt.title('Correlation Matrix of Sensor Variables')  # 히트맵 제목을 지정합니다.
plt.tight_layout()  # 여백을 정리합니다.
plt.savefig(BASE_DIR / 'eda_correlation_matrix.png', dpi=150)  # 상관계수 히트맵을 PNG로 저장합니다.
plt.close()  # 그래프 객체를 닫습니다.

# Machine failure (0: 정상, 1: 고장) 분포 확인
failure_counts = df['Machine failure'].value_counts()  # 정상과 고장 클래스의 개수를 셉니다.
failure_ratio = failure_counts / len(df) * 100  # 전체 데이터 대비 각 클래스 비율을 백분율로 계산합니다.

print(f"정상 데이터 비율: {failure_ratio[0]:.2f}%")  # 정상 클래스 비율을 출력합니다.
print(f"고장 데이터 비율: {failure_ratio[1]:.2f}%\n")  # 고장 클래스 비율을 출력합니다.

plt.figure(figsize=(6, 4))  # 타깃 분포 막대그래프용 그림 크기를 지정합니다.
sns.countplot(data=df, x='Machine failure', hue='Machine failure', palette='Set2', legend=False)  # 정상과 고장 개수를 막대그래프로 그립니다.
plt.title('Distribution of Target Variable (Machine failure)')  # 그래프 제목을 지정합니다.
plt.tight_layout()  # 여백을 정리합니다.
plt.savefig(BASE_DIR / 'eda_target_distribution.png', dpi=150)  # 타깃 분포 그래프를 PNG 파일로 저장합니다.
plt.close()  # 그래프 객체를 닫습니다.

# 세부 고장 유형 파악 (TWF, HDF, PWF, OSF, RNF)
failure_modes = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']  # 세부 고장 모드 열 이름을 리스트로 정리합니다.
print("세부 고장 유형별 발생 건수:")  # 세부 고장 모드 집계 시작 안내 문구입니다.
print(df[failure_modes].sum())  # 각 고장 모드 열의 1 개수를 더해 발생 건수를 출력합니다.
print("\nEDA 완료! 이미지 3개가 저장되었습니다.")  # EDA 결과가 이미지 파일로 저장되었음을 알려줍니다.
