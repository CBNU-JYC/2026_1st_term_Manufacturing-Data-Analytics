import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / 'uci-secom.csv'
OUTPUT_PATH = BASE_DIR / 'cleaned_secom_data.csv'

# 1. 데이터 로드
df = pd.read_csv(INPUT_PATH)
print(f"원본 데이터 크기: {df.shape}")

# [지표 1: 유일성 Uniqueness] 중복 데이터 제거
# 시간과 센서 값이 모두 동일한 중복 행이 있는지 확인 후 제거
before_rows = len(df)
df = df.drop_duplicates()
after_rows = len(df)
print(f"1. 유일성 확보: 중복 행 {before_rows - after_rows}건 제거")

# [지표 2: 완전성 Completeness] 결측치 처리
# 가이드라인: 결측치 50% 이상인 컬럼 삭제, 나머지는 중앙값(Median)으로 보간
nan_threshold = 0.5
null_counts = df.isnull().mean()
cols_to_drop = null_counts[null_counts > nan_threshold].index
df = df.drop(columns=cols_to_drop)

# 남은 결측치는 중앙값으로 보간 (제조 데이터의 이상치 영향을 최소화하기 위함)
df = df.fillna(df.median(numeric_only=True))
print(f"2. 완전성 확보: 결측치 과다 컬럼 {len(cols_to_drop)}개 삭제 및 나머지 중앙값 보간")

# [지표 3: 유효성 Validity] 상수 컬럼(분산 0) 제거
# 모든 값이 동일한 센서는 분석 가치가 없으므로 제거
constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
df = df.drop(columns=constant_cols)
print(f"3. 유효성 확보: 변동성 없는 상수 센서 {len(constant_cols)}개 제거")

# [지표 4: 일관성 Consistency] 라벨링 표준화 및 타입 통일
# Pass/Fail: -1 -> 0 (양품), 1 -> 1 (불량)으로 변환하여 이진 분류 표준 형식 준수
df['Pass/Fail'] = df['Pass/Fail'].replace(-1, 0).astype(int)
print("4. 일관성 확보: 라벨 포맷 변경 (-1, 1) -> (0, 1)")

# [지표 5: 정확성 Accuracy] 시간 형식 변환 및 이상치 범위 제한
# 'Time' 컬럼을 datetime 객체로 변환하여 시계열 분석 가능하게 처리
df['Time'] = pd.to_datetime(df['Time'])

# (선택) 센서 데이터의 이상치를 IQR 기준으로 클리핑(Clipping)하여 정확성 보정
def handle_outliers(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return series.clip(lower=lower_bound, upper=upper_bound)

# 수치형 센서 컬럼만 추출하여 이상치 처리 (Time, Pass/Fail 제외)
sensor_cols = df.select_dtypes(include=[np.number]).columns.drop('Pass/Fail')
df[sensor_cols] = df[sensor_cols].apply(handle_outliers)
print("5. 정확성 확보: 시간 형식 변환 및 IQR 기반 센서 이상치 보정 완료")

# 3. 최종 결과 확인
print("\n" + "="*30)
print(f"최종 정제 데이터 크기: {df.shape}")
print(f"결측치 총합: {df.isnull().sum().sum()}")
print("="*30)

# 정제된 데이터 저장
df.to_csv(OUTPUT_PATH, index=False)
print(f"정제된 파일 저장 완료: {OUTPUT_PATH}")
