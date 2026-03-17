import pandas as pd

# 1. 실제 데이터 로드
df = pd.read_csv('labeled_data.csv')
print(f"[최초 데이터 로드] 총 레코드 수: {len(df)}건\n")

df_clean = df.copy()

# ==========================================
# 1. 유일성 (Uniqueness) 검증
# ==========================================
# 실습 포인트: 사출성형기의 특성상 좌/우 파츠(예: RH, LH)가 동시에 생산될 수 있으므로
# 동일 시간(TimeStamp), 동일 설비(EQUIP_CD), 동일 파츠(PART_NAME)가 완전히 같은 기록은 중복 로깅된 오류입니다.
print("[1. 유일성 평가]")
duplicates = df_clean.duplicated(subset=['TimeStamp', 'EQUIP_CD', 'PART_NAME'], keep=False)
print(f" - 중복 기록 데이터 수: {duplicates.sum()}건 발견")

# 첫 번째 기록만 남기고 제거
df_clean = df_clean.drop_duplicates(subset=['TimeStamp', 'EQUIP_CD', 'PART_NAME'], keep='first')
print("중복 데이터 제거 완료\n")


# ==========================================
# 2. 완전성 (Completeness) 검증
# ==========================================
# 실습 포인트: 공정의 핵심이 되는 센서값(사출시간, 사이클시간, 배럴온도 등)에 
# 결측치(NaN/Null)가 있는지 확인합니다.
print("[2. 완전성 평가]")
critical_cols = ['Injection_Time', 'Cycle_Time', 'Barrel_Temperature_1']
missing_count = df_clean[critical_cols].isnull().any(axis=1).sum()
print(f" - 핵심 센서 결측치(NaN) 수: {missing_count}건 발견")

# 결측치가 포함된 행 제거
df_clean = df_clean.dropna(subset=critical_cols)
print("결측치 포함 사이클 제거 완료\n")


# ==========================================
# 3. 유효성 (Validity) 검증
# ==========================================
# 실습 포인트: 사출 시간 및 사이클 시간은 물리적으로 0이나 음수가 될 수 없습니다.
print("[3. 유효성 평가]")
invalid_time = (df_clean['Cycle_Time'] <= 0) | (df_clean['Injection_Time'] <= 0)
print(f" - 유효하지 않은 시간(<=0) 데이터 수: {invalid_time.sum()}건 발견")

df_clean = df_clean[~invalid_time]
print("유효성 위배 데이터 제거 완료\n")


# ==========================================
# 4. 일관성 (Consistency) 검증
# ==========================================
# 실습 포인트 1: 양불 판정(PassOrFail)이 'Y(양품)'인데, 불량 사유(Reason)가 적혀있으면 모순입니다.
# 실습 포인트 2: 사출 시간(Injection_Time)이 전체 사이클 시간(Cycle_Time)보다 길면 모순입니다.
print("[4. 일관성 평가]")

# 판정 모순
inconsistent_label = (df_clean['PassOrFail'] == 'Y') & (df_clean['Reason'].notna() & (df_clean['Reason'] != 'None'))
# 시간 모순
inconsistent_time = df_clean['Injection_Time'] >= df_clean['Cycle_Time']

print(f" - 양불/사유 논리적 모순 수: {inconsistent_label.sum()}건 발견")
print(f" - 공정 시간 논리적 모순(사출>=사이클) 수: {inconsistent_time.sum()}건 발견")

df_clean = df_clean[~inconsistent_label]
df_clean = df_clean[~inconsistent_time]
print("일관성 위배 데이터 제거 완료\n")


# ==========================================
# 5. 정확성 (Accuracy) 검증
# ==========================================
# 실습 포인트: 본 데이터셋의 배럴 온도(Barrel_Temperature)는 통상 200~300도 사이에서 운용됩니다.
# 센서 단선이나 오작동으로 인해 50도 미만이거나 400도 초과인 상식 밖의 데이터(물리적 오류)를 걸러냅니다.
print("[5. 정확성 평가]")
inaccurate_temp = (df_clean['Barrel_Temperature_1'] < 50) | (df_clean['Barrel_Temperature_1'] > 400)
print(f" - 물리적으로 부정확한 배럴 온도 수: {inaccurate_temp.sum()}건 발견")

df_clean = df_clean[~inaccurate_temp]
print("정확성 위배 데이터 제거 완료\n")


# ==========================================
# [최종 결과] 정제된 고품질 데이터셋 저장
# ==========================================
print("=" * 60)
print("[최종 확보된 고품질 데이터셋 (Golden Dataset)]")
print(f" - 최초 원본 데이터: {len(df)}건")
print(f" - 최종 확보 데이터: {len(df_clean)}건")
print("=" * 60)

# 정제된 데이터를 새로운 CSV 파일로 저장
df_clean.to_csv('labeled_data_clean.csv', index=False)
print("파일이 'labeled_data_clean.csv'로 성공적으로 저장되었습니다.")