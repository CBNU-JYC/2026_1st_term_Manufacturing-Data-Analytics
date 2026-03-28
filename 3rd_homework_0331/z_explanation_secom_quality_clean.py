# =============================================================================
# 파일명  : z_explanation_secom_quality_clean.py
# 설명용  : secom_quality_clean.py의 각 구문 의미를 쉽게 이해할 수 있도록 주석을 강화한 버전
# =============================================================================

import pandas as pd  # 표 형태 데이터를 읽고 다루기 위한 라이브러리입니다.
import numpy as np  # 수치 계산과 배열 연산을 위한 라이브러리입니다.
import warnings  # 실행 중 뜨는 경고 메시지를 제어할 때 사용합니다.
from pathlib import Path  # 현재 파일 기준 경로를 안정적으로 만들기 위해 사용합니다.

warnings.filterwarnings('ignore')  # 학습용 출력이 복잡해지지 않도록 경고를 숨깁니다.

BASE_DIR = Path(__file__).resolve().parent  # 이 파이썬 파일이 들어 있는 폴더 경로를 구합니다.
INPUT_PATH = BASE_DIR / 'uci-secom.csv'  # 입력 데이터 파일 경로입니다.
OUTPUT_PATH = BASE_DIR / 'cleaned_secom_data.csv'  # 정제 후 저장할 결과 파일 경로입니다.


# =============================================================================
# 데이터 로드
# =============================================================================
# uci-secom.csv 컬럼 구조:
#   Time      : 공정 시간 정보
#   0 ~ 589   : 센서 데이터 590개
#   Pass/Fail : 레이블 (-1=양품, 1=불량)

print("=" * 65)  # 화면에 구분선을 출력합니다.
print("   SECOM 반도체 공정 데이터셋 품질 확보 파이프라인")  # 프로그램 제목을 출력합니다.
print("   제조 데이터 분석과 최적화 | 충북대학교 산업인공지능학과")  # 수업 정보를 출력합니다.
print("=" * 65)  # 다시 구분선을 출력합니다.

df = pd.read_csv(INPUT_PATH)  # CSV 파일을 읽어 데이터프레임으로 불러옵니다.

sensor_cols = [c for c in df.columns if c not in ['Time', 'Pass/Fail']]  # 센서 컬럼만 따로 골라 목록으로 만듭니다.

print(f"\n[원본 데이터 로드 완료]")  # 데이터 로드가 끝났음을 알려줍니다.
print(f"  전체 크기    : {df.shape[0]}행 x {df.shape[1]}열")  # 전체 행/열 수를 출력합니다.
print(f"  센서 수      : {len(sensor_cols)}개 (컬럼명: '0' ~ '589')")  # 센서 컬럼 개수를 출력합니다.
print(f"  타임스탬프   : {df['Time'].iloc[0]} ~ {df['Time'].iloc[-1]}")  # 시작 시각과 끝 시각을 보여줍니다.
print(f"  레이블 분포  : {df['Pass/Fail'].value_counts().to_dict()}")  # 양품/불량 개수를 출력합니다.
print(f"               (-1=양품/Pass, 1=불량/Fail)")  # 레이블 뜻을 설명합니다.
print(f"  전체 결측률  : {df[sensor_cols].isnull().mean().mean()*100:.1f}%")  # 센서 전체 평균 결측률을 출력합니다.
print()  # 보기 좋게 한 줄 띄웁니다.

df_clean = df.copy()  # 원본 보존을 위해 복사본을 만들어 정제 작업에 사용합니다.
N_original = len(df_clean)  # 원본 샘플 수를 저장합니다.
N_cols_original = len(sensor_cols)  # 원본 센서 수를 저장합니다.


# =============================================================================
# MDQI 점수 계산 함수
# =============================================================================

def calc_qi(N, E, name):
    """단일 품질 지표 Qi를 계산하고 출력합니다."""
    qi = ((N - E) / N) * 100 if N > 0 else 0.0  # 오류 개수를 제외한 비율을 점수로 환산합니다.
    print(f"   {name:12s}  Qi = (({N} - {E}) / {N}) x 100 = {qi:.2f}점")  # 계산식을 함께 보여줍니다.
    return qi  # 계산된 점수를 반환합니다.


# =============================================================================
# 1. 유일성 (Uniqueness) 검증
# =============================================================================

print("=" * 65)  # 구분선을 출력합니다.
print("[1. 유일성 (Uniqueness) 검증]")  # 1단계 제목입니다.
print("   기준 : 590개 센서값이 모두 동일한 행 -> 중복 로깅 오류")  # 유일성 판정 기준을 설명합니다.
print("-" * 65)  # 단계 구분선을 출력합니다.

sensor_cols = [c for c in df_clean.columns if c not in ['Time', 'Pass/Fail']]  # 현재 남아 있는 센서 컬럼을 다시 계산합니다.

duplicates = df_clean.duplicated(subset=sensor_cols, keep=False)  # 센서값이 완전히 같은 중복 행을 찾습니다.
E_uni = int(duplicates.sum())  # 중복으로 판단된 행 수를 오류 개수로 저장합니다.
N_uni = N_original  # 유일성 평가 기준 전체 수는 원본 샘플 수입니다.

print(f"   중복 기록 발견 : {E_uni}건")  # 중복 개수를 출력합니다.

df_clean = df_clean.drop_duplicates(subset=sensor_cols, keep='first')  # 첫 번째만 남기고 중복 행을 제거합니다.
print(f"   중복 제거 후 남은 샘플 : {len(df_clean)}건")  # 제거 후 남은 샘플 수를 출력합니다.

Q_uni = calc_qi(N_uni, E_uni, "유일성")  # 유일성 점수를 계산합니다.
print("   유일성 검증 완료\n")  # 단계 종료 문구입니다.


# =============================================================================
# 2. 완전성 (Completeness) 검증
# =============================================================================

print("=" * 65)  # 구분선을 출력합니다.
print("[2. 완전성 (Completeness) 검증]")  # 2단계 제목입니다.
print("   기준 A : 센서 결측률 > 50%  -> 해당 센서 컬럼 제거")  # 컬럼 결측 기준입니다.
print("   기준 B : 샘플 결측률 > 50%  -> 해당 행 제거")  # 행 결측 기준입니다.
print("   기준 C : 나머지 결측치       -> 컬럼 중앙값으로 보간")  # 나머지 결측치 처리 기준입니다.
print("-" * 65)  # 단계 구분선을 출력합니다.

sensor_cols = [c for c in df_clean.columns if c not in ['Time', 'Pass/Fail']]  # 현재 센서 컬럼 목록을 다시 구합니다.

col_missing_rate = df_clean[sensor_cols].isnull().mean()  # 각 센서 컬럼의 결측 비율을 계산합니다.
drop_cols = col_missing_rate[col_missing_rate > 0.50].index.tolist()  # 결측률 50% 초과 센서를 제거 대상으로 고릅니다.
E_com_col = len(drop_cols)  # 제거할 센서 개수를 기록합니다.

df_clean = df_clean.drop(columns=drop_cols)  # 결측률이 너무 높은 센서를 제거합니다.
sensor_cols = [c for c in df_clean.columns if c not in ['Time', 'Pass/Fail']]  # 제거 후 센서 목록을 다시 만듭니다.
print(f"   [A] 결측률 > 50% 센서 제거: {E_com_col}개 | 남은 센서: {len(sensor_cols)}개")  # 제거 결과를 출력합니다.

row_missing_rate = df_clean[sensor_cols].isnull().mean(axis=1)  # 각 행의 결측 비율을 계산합니다.
drop_rows_mask = row_missing_rate > 0.50  # 결측률 50% 초과 행을 True로 표시합니다.
E_com_row = int(drop_rows_mask.sum())  # 제거할 행 수를 기록합니다.

df_clean = df_clean[~drop_rows_mask]  # 결측률이 너무 높은 행을 제거합니다.
print(f"   [B] 결측률 > 50% 샘플 제거: {E_com_row}건 | 남은 샘플: {len(df_clean)}건")  # 제거 결과를 출력합니다.

sensor_cols = [c for c in df_clean.columns if c not in ['Time', 'Pass/Fail']]  # 현재 센서 목록을 다시 구합니다.
remaining_missing = int(df_clean[sensor_cols].isnull().sum().sum())  # 남아 있는 결측치 총개수를 계산합니다.
print(f"   [C] 남은 결측치: {remaining_missing}개 -> 컬럼 중앙값으로 보간")  # 남은 결측치 수를 출력합니다.

col_medians = df_clean[sensor_cols].median()  # 각 센서 컬럼의 중앙값을 구합니다.
df_clean[sensor_cols] = df_clean[sensor_cols].fillna(col_medians)  # 남은 결측치를 중앙값으로 채웁니다.

N_com = N_original * N_cols_original  # 완전성 점수 계산의 전체 기준 수입니다.
E_com = (E_com_col * N_original) + (E_com_row * len(sensor_cols)) + remaining_missing  # 완전성 오류량을 계산합니다.
Q_com = calc_qi(N_com, E_com, "완전성")  # 완전성 점수를 계산합니다.
print("   완전성 검증 완료\n")  # 단계 종료 문구입니다.


# =============================================================================
# 3. 유효성 (Validity) 검증
# =============================================================================

print("=" * 65)  # 구분선을 출력합니다.
print("[3. 유효성 (Validity) 검증]")  # 3단계 제목입니다.
print("   기준 A : ±Inf 값           -> NaN 치환 후 중앙값 보간")  # Inf 처리 기준을 설명합니다.
print("   기준 B : 분산 = 0 인 센서  -> 정보 없는 고장 센서 제거")  # 상수 센서 처리 기준을 설명합니다.
print("-" * 65)  # 단계 구분선을 출력합니다.

sensor_cols = [c for c in df_clean.columns if c not in ['Time', 'Pass/Fail']]  # 현재 센서 컬럼을 다시 구합니다.

inf_mask = np.isinf(df_clean[sensor_cols].values)  # 센서 데이터에 Inf 또는 -Inf가 있는지 검사합니다.
E_val_inf = int(inf_mask.sum())  # 발견된 Inf 개수를 셉니다.
print(f"   [A] Inf/-Inf 값 발견: {E_val_inf}개", end="")  # Inf 개수를 출력합니다.
if E_val_inf > 0:  # Inf가 있으면 처리합니다.
    df_clean[sensor_cols] = df_clean[sensor_cols].replace([np.inf, -np.inf], np.nan)  # Inf를 결측치로 바꿉니다.
    col_medians = df_clean[sensor_cols].median()  # 중앙값을 다시 계산합니다.
    df_clean[sensor_cols] = df_clean[sensor_cols].fillna(col_medians)  # 결측치를 중앙값으로 채웁니다.
    print(" -> NaN 치환 후 보간 완료")  # 처리 완료 문구입니다.
else:
    print(" (처리 불필요)")  # 처리할 것이 없음을 출력합니다.

col_var = df_clean[sensor_cols].var()  # 각 센서의 분산을 계산합니다.
zero_var_cols = col_var[col_var == 0].index.tolist()  # 분산이 0인 센서를 찾습니다.
E_val_zerovar = len(zero_var_cols)  # 분산 0 센서 개수를 셉니다.
print(f"   [B] 분산=0 센서 발견: {E_val_zerovar}개 -> 해당 센서 제거")  # 상수 센서 수를 출력합니다.

df_clean = df_clean.drop(columns=zero_var_cols)  # 분산 0 센서를 제거합니다.
sensor_cols = [c for c in df_clean.columns if c not in ['Time', 'Pass/Fail']]  # 제거 후 센서 목록을 다시 계산합니다.
print(f"       제거 후 남은 센서: {len(sensor_cols)}개")  # 남은 센서 수를 출력합니다.

N_val = N_original * N_cols_original  # 유효성 점수 계산의 전체 기준 수입니다.
E_val = E_val_inf + (E_val_zerovar * N_original)  # 유효성 오류량을 계산합니다.
Q_val = calc_qi(N_val, E_val, "유효성")  # 유효성 점수를 계산합니다.
print("   유효성 검증 완료\n")  # 단계 종료 문구입니다.


# =============================================================================
# 4. 일관성 (Consistency) 검증
# =============================================================================

print("=" * 65)  # 구분선을 출력합니다.
print("[4. 일관성 (Consistency) 검증]")  # 4단계 제목입니다.
print("   기준 : 센서별 IQR 기반 극단 이상치 -> 클리핑(Clipping) 처리")  # 이상치 처리 기준입니다.
print("   범위 : Q1 - 3*IQR  ~  Q3 + 3*IQR")  # 이상치 경계 공식을 설명합니다.
print("-" * 65)  # 단계 구분선을 출력합니다.

sensor_cols = [c for c in df_clean.columns if c not in ['Time', 'Pass/Fail']]  # 현재 센서 목록을 구합니다.

Q1 = df_clean[sensor_cols].quantile(0.25)  # 각 센서의 1사분위수를 계산합니다.
Q3 = df_clean[sensor_cols].quantile(0.75)  # 각 센서의 3사분위수를 계산합니다.
IQR = Q3 - Q1  # 사분위 범위(IQR)를 계산합니다.

lower_bound = Q1 - 3 * IQR  # 하한 경계를 계산합니다.
upper_bound = Q3 + 3 * IQR  # 상한 경계를 계산합니다.

outlier_mask = (df_clean[sensor_cols] < lower_bound) | (df_clean[sensor_cols] > upper_bound)  # 경계를 벗어난 값을 이상치로 표시합니다.
E_con = int(outlier_mask.values.sum())  # 이상치 셀 개수를 계산합니다.
N_con = df_clean[sensor_cols].size  # 전체 센서 셀 개수를 계산합니다.

print(f"   이상치 셀 발견: {E_con}개 (전체 {N_con}셀 중 {E_con/N_con*100:.2f}%)")  # 이상치 비율을 출력합니다.

df_clean[sensor_cols] = df_clean[sensor_cols].clip(lower=lower_bound, upper=upper_bound, axis=1)  # 이상치를 경계값으로 잘라냅니다.
print(f"   이상치 클리핑 완료 (경계값으로 대체)")  # 처리 완료 문구입니다.

Q_con = calc_qi(N_con, E_con, "일관성")  # 일관성 점수를 계산합니다.
print("   일관성 검증 완료\n")  # 단계 종료 문구입니다.


# =============================================================================
# 5. 정확성 (Accuracy) 검증
# =============================================================================

print("=" * 65)  # 구분선을 출력합니다.
print("[5. 정확성 (Accuracy) 검증]")  # 5단계 제목입니다.
print("   기준 : 유효 레이블 = {-1, 1} 외의 값 -> 기록 오류, 제거")  # 레이블 검증 기준을 설명합니다.
print("   SECOM 레이블: -1=양품(Pass), 1=불량(Fail)")  # 유효한 레이블 뜻을 설명합니다.
print("-" * 65)  # 단계 구분선을 출력합니다.

label_counts_before = df_clean['Pass/Fail'].value_counts()  # 현재 레이블 분포를 계산합니다.
print(f"   레이블 분포: {label_counts_before.to_dict()}")  # 레이블 분포를 출력합니다.

valid_labels = {-1, 1}  # 유효한 레이블 집합을 정의합니다.
invalid_mask = ~df_clean['Pass/Fail'].isin(valid_labels) | df_clean['Pass/Fail'].isnull()  # 잘못된 레이블 또는 결측 레이블을 찾습니다.
E_acc = int(invalid_mask.sum())  # 잘못된 레이블 샘플 수를 계산합니다.
N_acc = N_original  # 정확성 점수 계산의 전체 기준 수입니다.
print(f"   유효하지 않은 레이블 샘플: {E_acc}건", end="")  # 오류 레이블 수를 출력합니다.

if E_acc > 0:  # 잘못된 레이블이 있으면 제거합니다.
    df_clean = df_clean[~invalid_mask].copy()  # 오류 레이블 샘플을 제거합니다.
    print(" -> 제거 완료")  # 처리 완료 문구입니다.
else:
    print(" (처리 불필요)")  # 처리할 것이 없음을 출력합니다.

print(f"   정확성 처리 후 남은 샘플: {len(df_clean)}건")  # 처리 후 남은 샘플 수를 출력합니다.
print(f"   최종 레이블 분포: {df_clean['Pass/Fail'].value_counts().to_dict()}")  # 최종 레이블 분포를 출력합니다.

Q_acc = calc_qi(N_acc, E_acc, "정확성")  # 정확성 점수를 계산합니다.
print("   정확성 검증 완료\n")  # 단계 종료 문구입니다.


# =============================================================================
# MDQI 종합 품질 점수 계산
# =============================================================================

MDQI = (Q_uni + Q_com + Q_val + Q_con + Q_acc) / 5  # 다섯 품질 점수의 평균으로 MDQI를 계산합니다.

def get_grade(score):
    """MDQI 점수를 등급 문자열로 변환합니다."""
    if score >= 99:  # 99점 이상이면 최우수입니다.
        return "S (최우수) – Golden Data | AI 즉시 투입 가능"
    elif score >= 95:  # 95점 이상이면 우수입니다.
        return "A (우수)   – Good Data   | 경미한 전처리 필요"
    elif score >= 80:  # 80점 이상이면 보통입니다.
        return "B (보통)   – Fair Data   | 집중 정제 필요"
    else:  # 그보다 낮으면 미흡입니다.
        return "C (미흡)   – Poor Data   | 현장 인프라 점검 요망"

print("=" * 65)  # 구분선을 출력합니다.
print("   [MDQI 종합 품질 점수 계산]")  # 종합 점수 섹션 제목입니다.
print("-" * 65)  # 구분선을 출력합니다.
print(f"   유일성  (Q_uni) : {Q_uni:.2f}점")  # 유일성 점수를 출력합니다.
print(f"   완전성  (Q_com) : {Q_com:.2f}점")  # 완전성 점수를 출력합니다.
print(f"   유효성  (Q_val) : {Q_val:.2f}점")  # 유효성 점수를 출력합니다.
print(f"   일관성  (Q_con) : {Q_con:.2f}점")  # 일관성 점수를 출력합니다.
print(f"   정확성  (Q_acc) : {Q_acc:.2f}점")  # 정확성 점수를 출력합니다.
print("-" * 65)  # 구분선을 출력합니다.
print(f"   MDQI = ({Q_uni:.2f}+{Q_com:.2f}+{Q_val:.2f}+{Q_con:.2f}+{Q_acc:.2f}) / 5")  # 계산식을 보여줍니다.
print(f"        = {MDQI:.2f}점")  # 최종 점수를 출력합니다.
print(f"   등급   : {get_grade(MDQI)}")  # 점수에 대응하는 등급을 출력합니다.
print("=" * 65)  # 구분선을 출력합니다.


# =============================================================================
# 최종 결과 요약 및 저장
# =============================================================================

sensor_cols_final = [c for c in df_clean.columns if c not in ['Time', 'Pass/Fail']]  # 최종 남은 센서 컬럼 목록을 계산합니다.
removed_samples = N_original - len(df_clean)  # 제거된 샘플 수를 계산합니다.
removed_sensors = N_cols_original - len(sensor_cols_final)  # 제거된 센서 수를 계산합니다.

print()  # 보기 좋게 한 줄 띄웁니다.
print("=" * 65)  # 구분선을 출력합니다.
print("   [최종 확보된 고품질 데이터셋 요약]")  # 최종 요약 제목입니다.
print("-" * 65)  # 구분선을 출력합니다.
print(f"   원본 데이터셋     : {N_original}행 x {N_cols_original}개 센서")  # 원본 크기를 출력합니다.
print(f"   제거된 샘플       : {removed_samples}행 ({removed_samples/N_original*100:.1f}% 제거)")  # 제거된 샘플 수와 비율을 출력합니다.
print(f"   제거된 센서       : {removed_sensors}개 ({removed_sensors/N_cols_original*100:.1f}% 제거)")  # 제거된 센서 수와 비율을 출력합니다.
print(f"   최종 데이터셋     : {len(df_clean)}행 x {len(sensor_cols_final)}개 센서")  # 최종 데이터 크기를 출력합니다.
print(f"   최종 레이블 분포  : {df_clean['Pass/Fail'].value_counts().to_dict()}")  # 최종 레이블 분포를 출력합니다.
print(f"   MDQI 점수         : {MDQI:.2f}점  [{get_grade(MDQI).split('|')[0].strip()}]")  # 최종 점수와 등급 요약을 출력합니다.
print("-" * 65)  # 구분선을 출력합니다.
print("   [적용된 정제 기준 요약]")  # 적용 기준 제목입니다.
print("   유일성 : 완전 중복 행 탐지 및 제거")  # 유일성 처리 요약입니다.
print("   완전성 : 결측 과다 센서/샘플 제거 + 중앙값 보간")  # 완전성 처리 요약입니다.
print("   유효성 : Inf 탐지 제거 + 분산=0 고장 센서 제거")  # 유효성 처리 요약입니다.
print("   일관성 : IQR 기반 이상치 클리핑 (3배 IQR 경계)")  # 일관성 처리 요약입니다.
print("   정확성 : 유효 레이블 외 이상 샘플 제거")  # 정확성 처리 요약입니다.
print("=" * 65)  # 구분선을 출력합니다.

df_clean.to_csv(OUTPUT_PATH, index=False)  # 정제된 데이터를 CSV 파일로 저장합니다.
print(f"\n파일이 '{OUTPUT_PATH}'로 저장되었습니다.")  # 저장 위치를 출력합니다.
print(f"  크기: {len(df_clean)}행 x {len(df_clean.columns)}열")  # 저장된 데이터 크기를 출력합니다.
