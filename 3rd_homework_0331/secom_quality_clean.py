# =============================================================================
# 파일명  : secom_quality_clean.py
# 과제명  : 제조 데이터셋 품질 확보 과제 – SECOM 반도체 공정 데이터
# 수업    : 제조 데이터 분석과 최적화 | 충북대학교 산업인공지능학과
# 데이터  : uci-secom.csv (Kaggle: paresh2047/uci-semcom)
# =============================================================================
#
# ┌─────────────────────────────────────────────────────────────────┐
# │      [나만의 SECOM 데이터 품질 정제 가이드라인 v1.0]            │
# │                                                                 │
# │  SECOM은 반도체 제조 공정에서 수집된 실제 센서 데이터입니다.   │
# │  1,567개 공정 사이클 × 590개 센서로 구성되며,                  │
# │  현장 특유의 문제(결측, 불변 센서, 이상값)가 포함됩니다.       │
# │                                                                 │
# │  [실제 데이터 구조 확인 결과]                                   │
# │  - 컬럼명 : Time | 0 ~ 589 (센서) | Pass/Fail                  │
# │  - 레이블 : -1 = 양품(Pass) | 1 = 불량(Fail)                   │
# │  - 결측률 : 전체 4.5% (센서별 최대 100% 존재)                  │
# │  - 분산=0 센서 : 116개 (고장/상수 출력 센서)                   │
# │  - Inf 값 : 없음                                               │
# │  - 완전 중복 행 : 없음                                          │
# │                                                                 │
# │  [설계 철학]                                                    │
# │  "데이터를 무조건 많이 남기는 것이 아니라,                     │
# │   AI가 신뢰할 수 있는 데이터만 남긴다."                        │
# └─────────────────────────────────────────────────────────────────┘
#
# ─── 5대 품질 지표별 나만의 정제 기준 ────────────────────────────
#
# [1] 유일성 (Uniqueness)
#     -> 590개 센서값이 완전히 동일한 행 = 중복 로깅 오류
#     -> 기준: 완전 중복 행 탐지 후 제거 (keep='first')
#     -> 근거: 반도체 공정은 매 사이클마다 고유한 패턴을 가져야 함
#     -> [실제 결과]: 중복 없음 → Qi = 100점
#
# [2] 완전성 (Completeness)
#     -> 기준 A: 센서(컬럼) 결측률 > 50%  -> 정보 없는 센서, 제거
#     -> 기준 B: 샘플(행)   결측률 > 50%  -> 측정 실패 샘플, 제거
#     -> 기준 C: 나머지 결측치             -> 컬럼별 중앙값(Median) 보간
#     -> 근거: 중앙값은 이상치에 덜 민감 (공정 데이터는 skew 분포)
#     -> [실제 결과]: 결측 과다 센서 28개 제거, 전체 결측률 4.5%
#
# [3] 유효성 (Validity)
#     -> 기준 A: ±Inf(무한값) -> 측정 오류, NaN 치환 후 보간
#     -> 기준 B: 분산 = 0 인 센서 -> 고장/상수 출력 센서, 제거
#     -> 근거: 분산 0 피처는 AI 모델에 정보를 제공하지 못함
#     -> [실제 결과]: Inf 없음, 분산=0 센서 116개 제거
#
# [4] 일관성 (Consistency)
#     -> 기준: 센서별 IQR(사분위수) 기반 극단 이상치 탐지
#     -> 이상치 범위: Q1 - 3*IQR ~ Q3 + 3*IQR 초과 -> 클리핑
#     -> 근거: 불량(1) 샘플이 6.6%로 매우 희귀 -> 제거 대신 클리핑으로 보존
#
# [5] 정확성 (Accuracy)
#     -> 기준: Pass/Fail 레이블 값 범위 검증
#     -> SECOM 레이블: -1(양품/Pass), 1(불량/Fail)
#     -> 이 두 값 외의 이상 레이블 샘플 제거
#     -> 근거: 정의된 범위 밖의 레이블은 기록 오류로 판단
#
# ─────────────────────────────────────────────────────────────────
#
# [MDQI 품질 점수 공식] (강의 슬라이드 기반)
#   Qi = ((N - E) / N) x 100   (N: 전체, E: 오류)
#   MDQI = (Q_uni + Q_com + Q_val + Q_con + Q_acc) / 5
#
# =============================================================================

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / 'uci-secom.csv'
OUTPUT_PATH = BASE_DIR / 'cleaned_secom_data.csv'

# =============================================================================
# 데이터 로드
# =============================================================================
# uci-secom.csv 컬럼 구조:
#   Time      : 공정 타임스탬프 (문자열, ex: "2008-07-19 11:55:00")
#   0 ~ 589   : 센서 데이터 (590개, float)
#   Pass/Fail : 레이블 (-1=양품, 1=불량)

print("=" * 65)
print("   SECOM 반도체 공정 데이터셋 품질 확보 파이프라인")
print("   제조 데이터 분석과 최적화 | 충북대학교 산업인공지능학과")
print("=" * 65)

df = pd.read_csv(INPUT_PATH)

# 센서 컬럼 목록 (Time, Pass/Fail 제외)
sensor_cols = [c for c in df.columns if c not in ['Time', 'Pass/Fail']]

print(f"\n[원본 데이터 로드 완료]")
print(f"  전체 크기    : {df.shape[0]}행 x {df.shape[1]}열")
print(f"  센서 수      : {len(sensor_cols)}개 (컬럼명: '0' ~ '589')")
print(f"  타임스탬프   : {df['Time'].iloc[0]} ~ {df['Time'].iloc[-1]}")
print(f"  레이블 분포  : {df['Pass/Fail'].value_counts().to_dict()}")
print(f"               (-1=양품/Pass, 1=불량/Fail)")
print(f"  전체 결측률  : {df[sensor_cols].isnull().mean().mean()*100:.1f}%")
print()

# 원본 크기 보존 (MDQI 계산 기준)
df_clean = df.copy()
N_original      = len(df_clean)        # 원본 샘플 수
N_cols_original = len(sensor_cols)     # 원본 센서 수


# =============================================================================
# MDQI 점수 계산 함수
# =============================================================================
# 강의 슬라이드 수식: Qi = ((N - E) / N) x 100

def calc_qi(N, E, name):
    """단일 품질 지표 Qi를 계산하고 출력합니다."""
    qi = ((N - E) / N) * 100 if N > 0 else 0.0
    print(f"   {name:12s}  Qi = (({N} - {E}) / {N}) x 100 = {qi:.2f}점")
    return qi


# =============================================================================
# 1. 유일성 (Uniqueness) 검증
# =============================================================================
# [나의 기준]
#   모든 센서값이 완전히 동일한 행 = 중복 로깅 오류
#   keep='first': 첫 번째 기록만 유지하고 이후 중복 제거

print("=" * 65)
print("[1. 유일성 (Uniqueness) 검증]")
print("   기준 : 590개 센서값이 모두 동일한 행 -> 중복 로깅 오류")
print("-" * 65)

sensor_cols = [c for c in df_clean.columns if c not in ['Time', 'Pass/Fail']]

duplicates = df_clean.duplicated(subset=sensor_cols, keep=False)
E_uni = int(duplicates.sum())
N_uni = N_original

print(f"   중복 기록 발견 : {E_uni}건")

df_clean = df_clean.drop_duplicates(subset=sensor_cols, keep='first')
print(f"   중복 제거 후 남은 샘플 : {len(df_clean)}건")

Q_uni = calc_qi(N_uni, E_uni, "유일성")
print("   유일성 검증 완료\n")


# =============================================================================
# 2. 완전성 (Completeness) 검증
# =============================================================================
# [나의 기준]
#   Step A : 센서(컬럼) 결측률 > 50%  -> 정보가 절반 이상 없는 센서 -> 제거
#   Step B : 샘플(행)   결측률 > 50%  -> 측정이 절반 이상 안 된 샘플 -> 제거
#   Step C : 나머지 결측치             -> 컬럼별 중앙값으로 보간

print("=" * 65)
print("[2. 완전성 (Completeness) 검증]")
print("   기준 A : 센서 결측률 > 50%  -> 해당 센서 컬럼 제거")
print("   기준 B : 샘플 결측률 > 50%  -> 해당 행 제거")
print("   기준 C : 나머지 결측치       -> 컬럼 중앙값으로 보간")
print("-" * 65)

sensor_cols = [c for c in df_clean.columns if c not in ['Time', 'Pass/Fail']]

# ── Step A: 결측률 > 50% 센서 컬럼 제거 ───────────────────────
col_missing_rate = df_clean[sensor_cols].isnull().mean()
drop_cols = col_missing_rate[col_missing_rate > 0.50].index.tolist()
E_com_col = len(drop_cols)

df_clean = df_clean.drop(columns=drop_cols)
sensor_cols = [c for c in df_clean.columns if c not in ['Time', 'Pass/Fail']]
print(f"   [A] 결측률 > 50% 센서 제거: {E_com_col}개 | 남은 센서: {len(sensor_cols)}개")

# ── Step B: 샘플 결측률 > 50% 행 제거 ─────────────────────────
row_missing_rate = df_clean[sensor_cols].isnull().mean(axis=1)
drop_rows_mask = row_missing_rate > 0.50
E_com_row = int(drop_rows_mask.sum())

df_clean = df_clean[~drop_rows_mask]
print(f"   [B] 결측률 > 50% 샘플 제거: {E_com_row}건 | 남은 샘플: {len(df_clean)}건")

# ── Step C: 나머지 결측치 -> 중앙값 보간 ─────────────────────
sensor_cols = [c for c in df_clean.columns if c not in ['Time', 'Pass/Fail']]
remaining_missing = int(df_clean[sensor_cols].isnull().sum().sum())
print(f"   [C] 남은 결측치: {remaining_missing}개 -> 컬럼 중앙값으로 보간")

col_medians = df_clean[sensor_cols].median()
df_clean[sensor_cols] = df_clean[sensor_cols].fillna(col_medians)

# MDQI 완전성 점수
N_com = N_original * N_cols_original
E_com = (E_com_col * N_original) + (E_com_row * len(sensor_cols)) + remaining_missing
Q_com = calc_qi(N_com, E_com, "완전성")
print("   완전성 검증 완료\n")


# =============================================================================
# 3. 유효성 (Validity) 검증
# =============================================================================
# [나의 기준]
#   Step A: ±Inf 값 -> 측정 장비 오류 -> NaN 치환 후 중앙값 보간
#   Step B: 분산 = 0 인 센서 -> 항상 동일값 출력 (고장 센서) -> 제거
#   [실제 확인]: Inf 없음 / 분산=0 센서 116개 존재

print("=" * 65)
print("[3. 유효성 (Validity) 검증]")
print("   기준 A : ±Inf 값           -> NaN 치환 후 중앙값 보간")
print("   기준 B : 분산 = 0 인 센서  -> 정보 없는 고장 센서 제거")
print("-" * 65)

sensor_cols = [c for c in df_clean.columns if c not in ['Time', 'Pass/Fail']]

# ── Step A: Inf 처리 ──────────────────────────────────────────
inf_mask = np.isinf(df_clean[sensor_cols].values)
E_val_inf = int(inf_mask.sum())
print(f"   [A] Inf/-Inf 값 발견: {E_val_inf}개", end="")
if E_val_inf > 0:
    df_clean[sensor_cols] = df_clean[sensor_cols].replace([np.inf, -np.inf], np.nan)
    col_medians = df_clean[sensor_cols].median()
    df_clean[sensor_cols] = df_clean[sensor_cols].fillna(col_medians)
    print(" -> NaN 치환 후 보간 완료")
else:
    print(" (처리 불필요)")

# ── Step B: 분산 = 0 인 센서 제거 ─────────────────────────────
col_var = df_clean[sensor_cols].var()
zero_var_cols = col_var[col_var == 0].index.tolist()
E_val_zerovar = len(zero_var_cols)
print(f"   [B] 분산=0 센서 발견: {E_val_zerovar}개 -> 해당 센서 제거")

df_clean = df_clean.drop(columns=zero_var_cols)
sensor_cols = [c for c in df_clean.columns if c not in ['Time', 'Pass/Fail']]
print(f"       제거 후 남은 센서: {len(sensor_cols)}개")

N_val = N_original * N_cols_original
E_val = E_val_inf + (E_val_zerovar * N_original)
Q_val = calc_qi(N_val, E_val, "유효성")
print("   유효성 검증 완료\n")


# =============================================================================
# 4. 일관성 (Consistency) 검증
# =============================================================================
# [나의 기준]
#   IQR(사분위 범위) 기반 극단 이상치 탐지 후 클리핑(Clipping)
#   범위: Q1 - 3*IQR  ~  Q3 + 3*IQR
#   클리핑 선택 이유: 불량 샘플(1)이 전체의 6.6%로 매우 희귀
#   -> 제거 시 불량 정보 손실 위험 -> 경계값으로 대체

print("=" * 65)
print("[4. 일관성 (Consistency) 검증]")
print("   기준 : 센서별 IQR 기반 극단 이상치 -> 클리핑(Clipping) 처리")
print("   범위 : Q1 - 3*IQR  ~  Q3 + 3*IQR")
print("-" * 65)

sensor_cols = [c for c in df_clean.columns if c not in ['Time', 'Pass/Fail']]

Q1  = df_clean[sensor_cols].quantile(0.25)
Q3  = df_clean[sensor_cols].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 3 * IQR
upper_bound = Q3 + 3 * IQR

outlier_mask = (df_clean[sensor_cols] < lower_bound) | (df_clean[sensor_cols] > upper_bound)
E_con = int(outlier_mask.values.sum())
N_con = df_clean[sensor_cols].size

print(f"   이상치 셀 발견: {E_con}개 (전체 {N_con}셀 중 {E_con/N_con*100:.2f}%)")

df_clean[sensor_cols] = df_clean[sensor_cols].clip(
    lower=lower_bound, upper=upper_bound, axis=1
)
print(f"   이상치 클리핑 완료 (경계값으로 대체)")

Q_con = calc_qi(N_con, E_con, "일관성")
print("   일관성 검증 완료\n")


# =============================================================================
# 5. 정확성 (Accuracy) 검증
# =============================================================================
# [나의 기준]
#   SECOM 레이블 구조: -1(양품/Pass), 1(불량/Fail)
#   이 두 값 외의 레이블은 기록 오류로 판단 -> 제거
#   추가로 레이블이 NaN인 샘플도 제거 (지도학습 불가)

print("=" * 65)
print("[5. 정확성 (Accuracy) 검증]")
print("   기준 : 유효 레이블 = {-1, 1} 외의 값 -> 기록 오류, 제거")
print("   SECOM 레이블: -1=양품(Pass), 1=불량(Fail)")
print("-" * 65)

label_counts_before = df_clean['Pass/Fail'].value_counts()
print(f"   레이블 분포: {label_counts_before.to_dict()}")

# -1, 1 외의 값 또는 NaN 탐지
valid_labels = {-1, 1}
invalid_mask = ~df_clean['Pass/Fail'].isin(valid_labels) | df_clean['Pass/Fail'].isnull()
E_acc = int(invalid_mask.sum())
N_acc = N_original
print(f"   유효하지 않은 레이블 샘플: {E_acc}건", end="")

if E_acc > 0:
    df_clean = df_clean[~invalid_mask].copy()
    print(" -> 제거 완료")
else:
    print(" (처리 불필요)")

print(f"   정확성 처리 후 남은 샘플: {len(df_clean)}건")
print(f"   최종 레이블 분포: {df_clean['Pass/Fail'].value_counts().to_dict()}")

Q_acc = calc_qi(N_acc, E_acc, "정확성")
print("   정확성 검증 완료\n")


# =============================================================================
# MDQI 종합 품질 점수 계산
# =============================================================================

MDQI = (Q_uni + Q_com + Q_val + Q_con + Q_acc) / 5

def get_grade(score):
    """MDQI 점수를 강의 슬라이드 기준 등급으로 변환합니다."""
    if score >= 99:
        return "S (최우수) – Golden Data | AI 즉시 투입 가능"
    elif score >= 95:
        return "A (우수)   – Good Data   | 경미한 전처리 필요"
    elif score >= 80:
        return "B (보통)   – Fair Data   | 집중 정제 필요"
    else:
        return "C (미흡)   – Poor Data   | 현장 인프라 점검 요망"

print("=" * 65)
print("   [MDQI 종합 품질 점수 계산]")
print("-" * 65)
print(f"   유일성  (Q_uni) : {Q_uni:.2f}점")
print(f"   완전성  (Q_com) : {Q_com:.2f}점")
print(f"   유효성  (Q_val) : {Q_val:.2f}점")
print(f"   일관성  (Q_con) : {Q_con:.2f}점")
print(f"   정확성  (Q_acc) : {Q_acc:.2f}점")
print("-" * 65)
print(f"   MDQI = ({Q_uni:.2f}+{Q_com:.2f}+{Q_val:.2f}+{Q_con:.2f}+{Q_acc:.2f}) / 5")
print(f"        = {MDQI:.2f}점")
print(f"   등급   : {get_grade(MDQI)}")
print("=" * 65)


# =============================================================================
# 최종 결과 요약 및 저장
# =============================================================================

sensor_cols_final = [c for c in df_clean.columns if c not in ['Time', 'Pass/Fail']]
removed_samples   = N_original - len(df_clean)
removed_sensors   = N_cols_original - len(sensor_cols_final)

print()
print("=" * 65)
print("   [최종 확보된 고품질 데이터셋 요약]")
print("-" * 65)
print(f"   원본 데이터셋     : {N_original}행 x {N_cols_original}개 센서")
print(f"   제거된 샘플       : {removed_samples}행 ({removed_samples/N_original*100:.1f}% 제거)")
print(f"   제거된 센서       : {removed_sensors}개 ({removed_sensors/N_cols_original*100:.1f}% 제거)")
print(f"   최종 데이터셋     : {len(df_clean)}행 x {len(sensor_cols_final)}개 센서")
print(f"   최종 레이블 분포  : {df_clean['Pass/Fail'].value_counts().to_dict()}")
print(f"   MDQI 점수         : {MDQI:.2f}점  [{get_grade(MDQI).split('|')[0].strip()}]")
print("-" * 65)
print("   [적용된 정제 기준 요약]")
print("   유일성 : 완전 중복 행 탐지 및 제거")
print("   완전성 : 결측 과다 센서/샘플 제거 + 중앙값 보간")
print("   유효성 : Inf 탐지 제거 + 분산=0 고장 센서 제거")
print("   일관성 : IQR 기반 이상치 클리핑 (3배 IQR 경계)")
print("   정확성 : 유효 레이블 외 이상 샘플 제거")
print("=" * 65)

df_clean.to_csv(OUTPUT_PATH, index=False)
print(f"\n파일이 '{OUTPUT_PATH}'로 저장되었습니다.")
print(f"  크기: {len(df_clean)}행 x {len(df_clean.columns)}열")
