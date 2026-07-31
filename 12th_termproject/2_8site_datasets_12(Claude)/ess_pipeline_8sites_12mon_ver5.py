# ═══════════════════════════════════════════════════════════════════════════
#  ESS 최적 용량 산정 파이프라인 ver5
#  「제조 데이터 분석과 최적화」 충북대학교 산업인공지능학과
#
#  ★ ver5 변경사항 (ver4 대비 — 문서 불일치 3건 수정):
#    Fix 1 : cumulative_sr 추가 (STEP 4b)
#            월 단위 누적 자립률 컬럼 생성 → ZEB 등급 추이 추적 가능
#            근거: 문서 §VI "[ZEB 추적 — 발표 슬라이드용]" 명시 요구사항
#
#    Fix 2 : S1 전용 IQR_K = 5.0 적용 (STEP 2a)
#            S1 MAIN 계량기 20kWh 펄스 분해능 한계 → k=3.0 적용 시
#            정상 냉난방 피크(60kWh/h)가 이상치로 오탐됨
#            근거: 문서 §III "S1 사이트의 특수한 분해능 문제" 권고사항
#            구현: SITE_CONFIGS에 per-site 'iqr_k' 파라미터 추가
#
#    Fix 3 : 중기 보간 요일+시간대 중위값 적용 (STEP 2a)
#            기존: groupby(hour).median()  → 요일 무관 시간대만 사용
#            수정: groupby([dayofweek, hour]).median() → 요일 일치 우선
#            근거: 문서 §III "중기(6~24h): 동일 요일·동일 시간대 중위값"
#
#  ver4에서 유지되는 내용:
#    - STEP 2a: IQR 이상치 탐지 + 계층적 보간
#    - STEP 2b: 물리적 배율 타당성 검증
#    - STEP 2c: 데이터 몰림(Catch-up) 재분배 (S3 불필요, S8 등 7개소 필요)
#    - STEP 3 : ZEB 배율 보정 (설명용 — ESS 최적화에는 미적용)
#    - STEP 4b: 파생 변수 생성 (is_redistributed 포함)
#    - STEP 4c: 계절별 대표 패턴 추출
#    - STEP 5 : 10kWh 단위 점진적 ESS 분석
#    - STEP 6 : Optuna Bayesian 최적화
#    - STEP 7 : 시각화 (영문 출력)
#    - STEP 8~9: 일반화 프레임워크 + 최종 보고서
#
#  작성자: 정용철 (2026254005)
# ═══════════════════════════════════════════════════════════════════════════

import os, csv, warnings, time
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from collections import defaultdict
from tabulate import tabulate

warnings.filterwarnings('ignore')

# 이미지 출력은 영문 전용 — DejaVu Sans로 고정
plt.rcParams['font.family']     = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


# ═══════════════════════════════════════════════════════════════════════════
# STEP 0 : 설정 상수
# ═══════════════════════════════════════════════════════════════════════════

DATA_ROOT = (
    '/Users/jeong-yongcheol/Desktop/00_CBNU_AI/My_project/'
    'ManDA_Lecture/12th_termproject/2_8site_datasets_12(Claude)'
)
OUTPUT_DIR = (
    '/Users/jeong-yongcheol/Desktop/00_CBNU_AI/My_project/'
    'ManDA_Lecture/12th_termproject/0_results_ess_pipeline_8sites_12mon_ver5'
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

ANALYSIS_START = '2024-12-01'
ANALYSIS_END   = '2025-12-31'

# ── ZEB 등급 기준 (전기 자립률 기준 적용) ────────────────────────────────
# 교수님 교육 목적: ZEB 공식 등급(전기+가스 1차에너지)을 전기 자립률로 매핑
ELEC_ZEB_GRADES = {
    5: (0.00, 0.40),  # 0% ~ 40% 미만 → ZEB 5등급 수준
    4: (0.40, 0.60),  # 40% ~ 60% 미만 → ZEB 4등급 수준
    3: (0.60, 0.80),  # 60% ~ 80% 미만 → ZEB 3등급 수준
    2: (0.80, 0.95),  # 80% ~ 95% 미만 → ZEB 2등급 수준
    1: (0.95, 1.01),  # 95% 이상        → ZEB 1등급 수준
}
GRADE_COLORS = {1: '#1a7a2e', 2: '#3498db', 3: '#9b59b6',
                4: '#e67e22', 5: '#e74c3c'}
GRADE_THRESHOLDS = [0.40, 0.60, 0.80, 0.95]  # 4→3, 3→2, 2→1 전환점
ZEB_TARGET       = GRADE_THRESHOLDS[0]         # 5등급 → 4등급 (40%)

# ── ★ ver3 핵심: ESS 탐색 범위 (무보정 실측 기준, 현실적) ─────────────────
ESS_UNIT_KWH = 10.0     # 단위 ESS (kWh)
ESS_MAX_KWH  = 200.0    # 최대 탐색 범위 (200 kWh — 참조표 상한 이내)
ESS_N_MAX    = int(ESS_MAX_KWH / ESS_UNIT_KWH)  # = 20단계

# ── ESS 효율 파라미터 ─────────────────────────────────────────────────────
ETA_C, ETA_D = 0.95, 0.95
SOC_MIN_R    = 0.10
SOC_MAX_R    = 0.90

# ── 한전 계시별 요금 (원/kWh) ─────────────────────────────────────────────
TARIFF = {
    **{h: 56.1  for h in list(range(0, 9)) + [23]},
    **{h: 93.5  for h in [9, 12, 17, 18, 19, 20, 21, 22]},
    **{h: 129.5 for h in [10, 11, 13, 14, 15, 16]},
}

# ── UTRON 보고서 기준 데이터 몰림 재분배 ──────────────────────────────────
CATCHUP_REDISTRIBUTE_SITES = {
    'S1': True, 'S2': True,
    'S3': False,   # UTRON 보고서: 몰림 없음
    'S4': True, 'S5': True, 'S6': True, 'S7': True,
    'S8': True,    # UTRON 보고서: 전처리 후에도 여전 → 재분배 필요
}
CATCHUP_ZERO_RUN_MIN = 3
CATCHUP_SPIKE_RATIO  = 2.5
IQR_K          = 3.0
SHORT_GAP_H    = 6
MID_GAP_H      = 24
MAX_HOURLY_KWH = 2_000

# ── ★ 건물 유형별 ESS 참조 범위 (이미지 표 기준) ─────────────────────────
# 출처: 공공기관 ESS 설치 의무화 제도 계약전력 5% 기준
ESS_REF_TABLE = {
    '소형(500~1000㎡)':  (50,  150,  '소형 행정복지센터·육아지원센터'),
    '중형(1000~3000㎡)': (150, 300,  '일반 행정복지센터·초등학교'),
    '대형(3000~5000㎡)': (300, 500,  '대형 행정복지센터·복합 시설'),
}
# 건물 유형→참조 크기 매핑
SITE_SIZE_CLASS = {
    'S1': '소형(500~1000㎡)', 'S2': '중형(1000~3000㎡)',
    'S3': '소형(500~1000㎡)', 'S4': '소형(500~1000㎡)',
    'S5': '소형(500~1000㎡)', 'S6': '중형(1000~3000㎡)',
    'S7': '소형(500~1000㎡)', 'S8': '중형(1000~3000㎡)',
}

# ── 계절 정의 (영문 — 이미지 출력 기준) ──────────────────────────────────
SEASONS = {'Winter':[12,1,2],'Spring':[3,4,5],'Summer':[6,7,8],'Autumn':[9,10,11]}
SEASON_COLORS = {'Winter':'#4475B4','Spring':'#5DB85D','Summer':'#D94E4E','Autumn':'#E98A30'}

# ── 이미지 출력용 영문 변환 매핑 ─────────────────────────────────────────
TYPE_EN = {
    '문화시설': 'Cultural',  '복지시설': 'Welfare',
    '행정시설': 'Admin.',    '교육시설': 'Education',
}
OP_EN = {
    '정기': 'Regular', '비정기': 'Irregular',
    '정기+방학': 'Regular+Vacation',
}

OPTUNA_TRIALS = 200

PHYSICAL_LOAD_RANGE = {
    '문화시설':(5_000, 500_000),'복지시설':(10_000, 800_000),
    '행정시설':(8_000, 400_000),'교육시설':(15_000, 600_000),
}

SITE_CONFIGS = [
    # ── Fix 2: S1은 20kWh 펄스 분해능 한계 → IQR_K = 5.0 적용 (k=3.0 시 60kWh 정상피크 오탐)
    # 근거: 문서 §III "S1 사이트의 특수한 분해능 문제" — IQR이 0이 되어 상한=20kWh 고정됨
    {'id':'S1','name':'강동_청소년문화센터','display':'강동 청소년 문화센터',
     'type':'문화시설','sub':'청소년시설','operation':'비정기','region':'경북 구미',
     'main_names':['MAIN'],'main_mult_doc':200,
     'solar_names':['태양광 1'],'solar_mult_doc':1,'has_gas':True,'has_bidir':True,
     'iqr_k': 5.0},   # ★ Fix 2: 20kWh 펄스 계량기 분해능 보정
    {'id':'S2','name':'구미강동_꿈나무문화나눔터','display':'구미 강동 꿈나무문화나눔터',
     'type':'문화시설','sub':'도서관','operation':'정기','region':'경북 구미',
     'main_names':['메인','MAIN'],'main_mult_doc':320,
     'solar_names':['지하전기실MCCB 태양광1','지하전기실MCCB 태양광2','태양광','태양광 1'],
     'solar_mult_doc':10,'has_gas':True,'has_bidir':False},
    {'id':'S3','name':'구미_육아지원센터','display':'구미 육아지원센터',
     'type':'복지시설','sub':'복지관','operation':'정기','region':'경북 구미',
     'main_names':['MAIN'],'main_mult_doc':1,
     'solar_names':['태양광','태양광 1'],'solar_mult_doc':1,'has_gas':True,'has_bidir':False},
    {'id':'S4','name':'김제금구면_행정복지센터','display':'김제 금구면 행정복지센터',
     'type':'행정시설','sub':'행정복지센터','operation':'정기','region':'전북 김제',
     'main_names':['행정복지센터 MAIN','MAIN'],            # ★ 실제 CSV dev_name 우선
     'main_mult_doc':60,
     'solar_names':['1층_EPS_태양광','태양광','복지관태양광','태양광 1','1층 EPS 태양광'],
     'solar_mult_doc':40,'has_gas':False,'has_bidir':True},
    {'id':'S5','name':'밀양_의열체험관','display':'밀양 의열체험관',
     'type':'문화시설','sub':'체험관','operation':'비정기','region':'경남 밀양',
     'main_names':['메인','MAIN'],                        # ★ 실제 CSV dev_name 우선
     'main_mult_doc':50,
     'solar_names':['태양광','옥상 태양광','태양광 1'],   # ★ 옥상 태양광 추가
     'solar_mult_doc':1,'has_gas':True,'has_bidir':True},
    {'id':'S6','name':'숭미초등학교','display':'숭미초등학교',
     'type':'교육시설','sub':'초등학교','operation':'정기+방학','region':'서울 도봉',
     'main_names':['MAIN'],'main_mult_doc':50,
     'solar_names':['신관태양광','본관태양광','태양광','태양광 1'],
     'solar_mult_doc':1,'has_gas':True,'has_bidir':True},
    {'id':'S7','name':'아주_청소년문화의집','display':'아주 청소년문화의집',
     'type':'문화시설','sub':'청소년시설','operation':'비정기','region':'경남 거제',
     'main_names':['MAIN'],'main_mult_doc':50,
     'solar_names':['태양광','태양광 1'],'solar_mult_doc':1,'has_gas':False,'has_bidir':True},
    {'id':'S8','name':'완주용봉초등학교','display':'완주용봉초등학교',
     'type':'교육시설','sub':'초등학교','operation':'정기+방학','region':'전북 완주',
     'main_names':['메인_1층EPS','메인','MAIN'],'main_mult_doc':60,
     'solar_names':['태양광_1층EPS','태양광','태양광 1'],
     'solar_mult_doc':24,'has_gas':True,'has_bidir':True},
]


# ═══════════════════════════════════════════════════════════════════════════
# ── 공통 유틸리티 (ver2와 동일)
# ═══════════════════════════════════════════════════════════════════════════

def find_csv_files(root):
    import re
    paths = []
    for dp, _, fns in os.walk(root):
        top = os.path.relpath(dp, root).split(os.sep)[0]
        if not re.match(r'^[1-8]_', top):   # 사이트 폴더(1_~8_)만 허용
            continue
        for fn in fns:
            if fn.lower().endswith('.csv'):
                paths.append(os.path.join(dp, fn))
    return sorted(paths)

def load_csv_to_raw(fp):
    raw = defaultdict(list)
    try:
        with open(fp, encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                try:
                    raw[row['dev_name']].append(
                        (pd.to_datetime(row['updated']), float(row['power_value'])))
                except: pass
    except: pass
    return raw

def merge_raws(rl):
    m = defaultdict(list)
    for r in rl:
        for k, v in r.items(): m[k].extend(v)
    return m

def to_hourly_raw(records, mult=1):
    if not records: return pd.Series(dtype=float)
    df = pd.DataFrame(records, columns=['t','v']).set_index('t').sort_index()
    df = df[~df.index.duplicated(keep='first')]
    return (df['v'] * mult).resample('1h').last().ffill().bfill()

def diff_series(s): return s.diff().clip(lower=0).fillna(0)

def get_cfg(sid):
    for c in SITE_CONFIGS:
        if c['id'] == sid: return c
    return {}

def match_site(ns, cfg):
    return (any(m in ns for m in cfg['main_names']) and
            any(s in ns for s in cfg['solar_names']))


# ═══════════════════════════════════════════════════════════════════════════
# STEP 1 : 데이터 로드
# ═══════════════════════════════════════════════════════════════════════════

def step1_load(root):
    print("\n[STEP 1] 데이터 로드")
    csvs = find_csv_files(root)
    print(f"  CSV {len(csvs)}개 발견")
    if not csvs: return {}
    folder_map = defaultdict(list)
    for fp in csvs:
        top = os.path.relpath(fp, root).split(os.sep)[0] \
              if os.sep in os.path.relpath(fp, root) else '__flat__'
        folder_map[top].append(fp)
    if len(folder_map) <= 1:
        file_raws = {fp: load_csv_to_raw(fp) for fp in csvs}
        site_fps  = defaultdict(list)
        for fp, raw in file_raws.items():
            ns = set(raw.keys())
            for cfg in SITE_CONFIGS:
                if match_site(ns, cfg):
                    site_fps[cfg['id']].append(fp); break
        sr = {sid: merge_raws([file_raws[fp] for fp in fps])
              for sid, fps in site_fps.items()}
    else:
        sr = {}
        for i, (folder, fps) in enumerate(sorted(folder_map.items())):
            cfg = next((c for c in SITE_CONFIGS
                        if any(p in folder for p in c['name'].split('_') + [c['id']])),
                       SITE_CONFIGS[i] if i < len(SITE_CONFIGS) else None)
            if cfg:
                sr[cfg['id']] = merge_raws([load_csv_to_raw(fp) for fp in fps])
    print(f"  매칭 사이트: {len(sr)}개 → {list(sr.keys())}")
    return sr


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2a : 이상치 탐지 및 계층적 보간 (ver2와 동일)
# ═══════════════════════════════════════════════════════════════════════════

def step2a_clean_outliers(diff_s, iqr_k=IQR_K, short_h=SHORT_GAP_H,
                            mid_h=MID_GAP_H, hard_max=MAX_HOURLY_KWH, label=''):
    s = diff_s.copy(); original_sum = s.sum()
    nonzero = s[s > 0]
    q1 = nonzero.quantile(0.25) if len(nonzero) > 10 else 0
    q3 = nonzero.quantile(0.75) if len(nonzero) > 10 else s.quantile(0.75)
    iqr = q3 - q1; upper_iqr = q3 + iqr_k * iqr
    outlier_mask = (s > upper_iqr) | (s < 0) | (s > hard_max)
    n_out = int(outlier_mask.sum()); s[outlier_mask] = np.nan
    is_na = s.isna()
    gap_id = is_na.ne(is_na.shift()).cumsum()
    gap_lengths = is_na.groupby(gap_id).transform('sum')
    short = is_na & (gap_lengths <= short_h)
    if short.any():
        tmp = s.copy(); tmp[~short] = np.nan
        s.update(tmp.interpolate('linear').dropna())
    mid = s.isna() & (gap_lengths > short_h) & (gap_lengths <= mid_h)
    if mid.any():
        # ── Fix 3: 동일 요일 + 동일 시간대 중위값 보간 (문서 §III 요구사항)
        # 기존(ver4): groupby(hour).median()          → 요일 구분 없이 시간대만 사용
        # 수정(ver5): groupby([dayofweek, hour]).median() → 동일 요일 우선 매칭
        # 근거: 주중/주말 소비 패턴 차이가 큰 사이트(S2 도서관, S5 체험관 등)에서
        #       보간 정확도 개선 목적
        valid_s = s[s > 0]
        # 1차: 요일+시간대 중위값 테이블
        hm_dow = valid_s.groupby(
            [valid_s.index.dayofweek, valid_s.index.hour]
        ).median()
        # 2차 폴백: 시간대만 (요일 데이터 부족 시)
        hm_hour = valid_s.groupby(valid_s.index.hour).median()

        for idx in s[mid].index:
            # 1차 시도: 동일 요일 + 동일 시간대
            key_dow = (idx.dayofweek, idx.hour)
            v = hm_dow.get(key_dow, np.nan)
            # 2차 폴백: 동일 시간대 (요일 조합 없을 때)
            if pd.isna(v):
                v = hm_hour.get(idx.hour, valid_s.median() if len(valid_s) > 0 else np.nan)
            if pd.notna(v):
                s.at[idx] = v
    lng = s.isna()
    if lng.any():
        vd = s.dropna()
        if len(vd) > 100:
            sh = vd.groupby([
                vd.index.month.map(lambda m:
                    '겨울' if m in [12,1,2] else '봄' if m in [3,4,5]
                    else '여름' if m in [6,7,8] else '가을'),
                vd.index.hour]).median()
            for idx in s[lng].index:
                m = idx.month
                sn = ('겨울' if m in [12,1,2] else '봄' if m in [3,4,5]
                      else '여름' if m in [6,7,8] else '가을')
                try:
                    fv = sh.get((sn, idx.hour), np.nan)
                    if pd.notna(fv): s.at[idx] = fv
                except: pass
    n_rem = int(s.isna().sum()); s = s.fillna(0).clip(lower=0)
    q = {'n_outliers': n_out, 'outlier_rate': n_out/max(len(s),1),
         'upper_iqr': upper_iqr, 'sum_change_pct':
         (s.sum()-original_sum)/max(original_sum,1)*100,
         'outlier_idx': diff_s.index[outlier_mask].tolist()}
    if n_out > 0 and label:
        print(f"    [{label}] 이상치 {n_out}건 (상한={upper_iqr:.1f}kWh), "
              f"합계변화 {q['sum_change_pct']:+.1f}%")
    return s, q


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2c : 데이터 몰림 재분배 (ver2와 동일)
# ═══════════════════════════════════════════════════════════════════════════

def step2c_redistribute_catchup(diff_s, zero_run_min=CATCHUP_ZERO_RUN_MIN,
                                  spike_ratio=CATCHUP_SPIKE_RATIO, label=''):
    s = diff_s.copy().values.astype(float); idx = diff_s.index; n = len(s)
    redis_mask = np.zeros(n, dtype=bool); n_events = 0; total_h = 0
    nonzero = s[s > 0]
    typical = float(np.median(nonzero)) if len(nonzero) > 5 else 1.0
    i = 0
    while i < n:
        if s[i] <= 0.0:
            zs = i
            while i < n and s[i] <= 0.0: i += 1
            zl = i - zs
            if zl >= zero_run_min and i < n:
                sv = s[i]
                if sv > typical * spike_ratio:
                    slots = zl + 1; unit = sv / slots
                    for j in range(zs, zs + zl + 1):
                        if j < n: s[j] = unit; redis_mask[j] = True
                    n_events += 1; total_h += slots
        else:
            i += 1
    result = pd.Series(s, index=idx); mask_s = pd.Series(redis_mask, index=idx)
    sb = float(diff_s.sum()); sa = float(result.sum())
    q = {'n_catchup_events': n_events, 'redistributed_hours': total_h,
         'energy_preserved': abs(sa - sb) < sb * 0.01}
    if n_events > 0 and label:
        print(f"    [{label}] 재분배 {n_events}건/{total_h}h, "
              f"에너지보존: {'✓' if q['energy_preserved'] else '⚠'}")
    elif label:
        print(f"    [{label}] 재분배 대상 없음")
    return result, mask_s, q


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2b : 물리적 배율 타당성 검증 (ver2와 동일)
# ═══════════════════════════════════════════════════════════════════════════

def step2b_validate_mult(load_h, solar_h, cfg):
    al = float(load_h.sum()); as_ = float(solar_h.sum())
    mh = float(load_h.max())
    sr = as_ / al if al > 0 else 0
    lo, hi = PHYSICAL_LOAD_RANGE.get(cfg.get('type',''), (1000, 5_000_000))
    return {'annual_load_kWh': al, 'annual_solar_kWh': as_,
            'max_hourly_kWh': mh, 'annual_sr': sr,
            'V1_load_range': lo <= al <= hi,
            'V2_max_hourly': mh < MAX_HOURLY_KWH,
            'V3_sr_range':   0.10 <= sr <= 0.35,
            'overall_valid': sum([lo<=al<=hi, mh<MAX_HOURLY_KWH, 0.10<=sr<=0.35])>=2}


# ═══════════════════════════════════════════════════════════════════════════
# STEP 3 : ZEB 배율 보정 (★ 설명용만 — ESS 최적화에는 미적용)
# ═══════════════════════════════════════════════════════════════════════════

def step3_zeb_context(load_h, solar_h, cfg):
    """
    ZEB 5등급 맥락 설명용 역산 — ESS 최적화에 사용하지 않음.
    전기만 계측된 데이터에서 가스 미계측 비중 역산.
    """
    al = float(load_h.sum()); as_ = float(solar_h.sum())
    sr_elec = min(as_ / al, 5.0) if al > 0 else 0
    # ZEB 20% 역산: primary_total = solar / 0.20
    primary_total = as_ / 0.20 if as_ > 0 else 0
    elec_primary  = al * 2.75
    gas_primary   = max(primary_total - elec_primary, 0)
    gas_ratio     = gas_primary / max(primary_total, 1)
    note = (f'전기SR={sr_elec*100:.1f}% | '
            f'ZEB 20% 성립: 가스 {gas_ratio*100:.0f}% 의존 | '
            f'ESS는 실측 전기 데이터({al/1000:.1f}MWh/년) 기준으로 최적화')
    return {'sr_elec': sr_elec, 'gas_ratio': gas_ratio,
            'annual_load': al, 'annual_solar': as_, 'note': note}


# ═══════════════════════════════════════════════════════════════════════════
# STEP 4b : 파생 변수 생성 (is_redistributed + cumulative_sr 포함 — ver5)
# ═══════════════════════════════════════════════════════════════════════════

def step4b_derive(df):
    d = df.copy()
    d['hour']        = d.index.hour
    d['dayofweek']   = d.index.dayofweek
    d['month']       = d.index.month
    d['is_weekend']  = (d['dayofweek'] >= 5).astype(int)
    d['is_business'] = ((d['dayofweek'] < 5) & d['hour'].between(8,18)).astype(int)
    d['season_name'] = d['month'].map({
        12:'Winter',1:'Winter',2:'Winter',
        3:'Spring',4:'Spring',5:'Spring',
        6:'Summer',7:'Summer',8:'Summer',
        9:'Autumn',10:'Autumn',11:'Autumn'})
    d['P_surplus']  = (d['P_solar'] - d['P_load']).clip(lower=0)
    d['P_deficit']  = (d['P_load']  - d['P_solar']).clip(lower=0)
    d['hourly_SR']  = (d['P_solar'] / d['P_load'].replace(0,np.nan)).clip(upper=5).fillna(0)
    d['tariff']     = d['hour'].map(TARIFF)
    d['solar_active'] = (d['P_solar'] > d['P_solar'].quantile(0.15)).astype(int)
    d['is_peak']    = (d['P_load'] > d['P_load'].quantile(0.90)).astype(int)
    if 'is_redistributed' not in d.columns:
        d['is_redistributed'] = 0

    # ── Fix 1: cumulative_sr — 월 누적 자립률 (ZEB 등급 실시간 추적)
    # 문서 §VI "[ZEB 추적 — 발표 슬라이드용]" 요구사항
    # 계산: 월 첫 시간부터 현재까지의 누적 태양광 / 누적 부하 비율
    # 용도: 월별 ZEB 등급 달성 여부를 시계열로 시각화
    try:
        monthly_groups = d.groupby(d.index.to_period('M'))
        sr_parts = []
        for period, grp in monthly_groups:
            load_cum  = grp['P_load'].cumsum()
            solar_cum = grp['P_solar'].cumsum()
            sr_cum = (solar_cum / load_cum.replace(0, np.nan)).clip(upper=5).fillna(0)
            sr_parts.append(sr_cum)
        if sr_parts:
            d['cumulative_sr'] = pd.concat(sr_parts).sort_index()
        else:
            d['cumulative_sr'] = 0.0
    except Exception:
        d['cumulative_sr'] = 0.0

    return d


# ═══════════════════════════════════════════════════════════════════════════
# STEP 4c : 계절별 대표 패턴 (ver2와 동일)
# ═══════════════════════════════════════════════════════════════════════════

def step4c_seasonal(df):
    patterns = {}
    for sn, months in SEASONS.items():
        grp = df[df.index.month.isin(months)]
        if len(grp) < 24: continue
        hg = grp.groupby(grp.index.hour)
        load_24  = hg['P_load'].median().values
        solar_24 = hg['P_solar'].median().values
        surplus  = np.maximum(solar_24 - load_24, 0)
        deficit  = np.maximum(load_24 - solar_24, 0)
        patterns[sn] = {
            'P_load_24h': load_24, 'P_solar_24h': solar_24,
            'P_surplus_24h': surplus, 'P_deficit_24h': deficit,
            'deficit_hours': int((solar_24 < load_24).sum()),
            'daily_deficit_kWh': float(deficit.sum()),
            'daily_surplus_kWh': float(surplus.sum()),
            'n_days': len(grp) // 24,
        }
    return patterns


# ═══════════════════════════════════════════════════════════════════════════
# ── Greedy 시뮬레이션 (공통)
# ═══════════════════════════════════════════════════════════════════════════

def greedy_sim(Ps, Pl, C_kwh):
    if C_kwh <= 0:
        grid = float(np.maximum(Pl - Ps, 0).sum())
        return {'grid': grid, 'SR': 1.0 - grid/max(float(Pl.sum()),1e-9)}
    T, SOC, grid = len(Ps), C_kwh*0.5, 0.0
    for t in range(T):
        s = float(Ps[t]) - float(Pl[t])
        if s > 0:
            ch = min(s, C_kwh/2, (C_kwh*SOC_MAX_R - SOC)/max(ETA_C,1e-9))
            SOC += max(ch,0)*ETA_C
        else:
            dis = min(-s, C_kwh/2, (SOC - C_kwh*SOC_MIN_R)*ETA_D)
            SOC -= max(dis,0)/ETA_D; grid += max(-s - max(dis,0), 0)
    return {'grid': grid, 'SR': max(0, 1.0 - grid/max(float(Pl.sum()),1e-9))}


# ═══════════════════════════════════════════════════════════════════════════
# STEP 5 : ★ 10kWh 단위 점진적 ESS 분석 (ver3 핵심 신규)
# ═══════════════════════════════════════════════════════════════════════════

def get_zeb_grade(sr):
    """전기 자립률 → ZEB 등급 (교육 목적 매핑)"""
    for grade, (lo, hi) in sorted(ELEC_ZEB_GRADES.items()):
        if lo <= sr < hi: return grade
    return 1

def step5_progressive_ess(df, sid):
    """
    ★ ver3 핵심: 10kWh 단위 점진적 ESS 용량 추가 → 등급 향상 분석

    ① 실측 전기 데이터(ZEB 보정 없이)로 Greedy 시뮬레이션
    ② 0 → 200kWh (10kWh 단위, 20단계)
    ③ 각 단계별: SR, ZEB등급, SR상승폭(한계효용), 절감비용
    ④ 등급 전환점(grade_transitions) 자동 탐지
    ⑤ 한계효용 최적점: 단위당 SR 상승폭이 급감하는 지점
    """
    Pl = df['P_load'].values
    Ps = df['P_solar'].values

    baseline    = greedy_sim(Ps, Pl, 0)
    sr_base     = baseline['SR']
    grade_base  = get_zeb_grade(sr_base)
    total_load  = float(Pl.sum())

    caps = [n * ESS_UNIT_KWH for n in range(ESS_N_MAX + 1)]
    results_by_cap = {}
    prev_sr = sr_base

    print(f"    기준 SR: {sr_base*100:.1f}% (ZEB {grade_base}등급) | "
          f"연간부족: {baseline['grid']/1000:.1f}MWh")

    grade_transitions = {}   # {달성등급: 최소ESS kWh}
    marginal_benefits = []   # 10kWh당 SR 상승폭

    for C in caps:
        res  = greedy_sim(Ps, Pl, C)
        sr   = res['SR']
        gr   = get_zeb_grade(sr)
        delta = sr - prev_sr if C > 0 else 0
        grid_kwh = res['grid']
        # 절감 비용 (계통 구매 감소 × 평균 요금)
        saved_kwh  = baseline['grid'] - grid_kwh
        saved_cost = saved_kwh * 93.5  # 원 (중간 요금 기준)

        results_by_cap[C] = {
            'SR': sr, 'grade': gr, 'delta_sr': delta,
            'grid_kWh': grid_kwh, 'saved_kWh': saved_kwh,
            'saved_cost_won': saved_cost,
        }

        # 등급 전환점 기록
        if gr < grade_base and gr not in grade_transitions:
            grade_transitions[gr] = C
            print(f"    → ★ ZEB {gr}등급 달성: {C:.0f}kWh "
                  f"(SR={sr*100:.1f}%, +{delta*100:.1f}pp)")
        elif C > 0 and delta > 0:
            print(f"    {C:4.0f}kWh: SR={sr*100:.1f}% "
                  f"(ZEB {gr}등급, +{delta*100:.1f}pp)")

        if C > 0:
            marginal_benefits.append({'C_kWh': C, 'delta_sr': delta, 'SR': sr})
        prev_sr = sr

        # 조기 종료: 100% 달성
        if sr >= 0.999:
            break

    # 최적점: 한계효용이 최대인 지점 (첫 번째 10kWh 추가 기준)
    if marginal_benefits:
        best_mb = max(marginal_benefits, key=lambda x: x['delta_sr'])
        optimal_C = best_mb['C_kWh']
    else:
        optimal_C = ESS_MAX_KWH

    # 목표 ZEB 4등급 달성 ESS
    target_C = grade_transitions.get(4, grade_transitions.get(3,
               grade_transitions.get(2, grade_transitions.get(1, ESS_MAX_KWH))))

    # 참조 범위 검증
    size_class = SITE_SIZE_CLASS.get(sid, '소형(500~1000㎡)')
    ref_lo, ref_hi, _ = ESS_REF_TABLE.get(size_class, (50, 150, '-'))
    in_ref_range = ref_lo <= target_C <= ref_hi

    return {
        'sr_base':          sr_base,
        'grade_base':       grade_base,
        'results_by_cap':   results_by_cap,
        'grade_transitions': grade_transitions,
        'marginal_benefits': marginal_benefits,
        'optimal_C':        optimal_C,
        'target_C':         target_C,    # ZEB 4등급 달성 최소 ESS
        'size_class':       size_class,
        'ref_range':        (ref_lo, ref_hi),
        'in_ref_range':     in_ref_range,
        'total_load_kWh':   total_load,
    }


# ═══════════════════════════════════════════════════════════════════════════
# STEP 6 : Optuna Bayesian 최적화 (실측 기준)
# ═══════════════════════════════════════════════════════════════════════════

def step6_optuna(df, target_sr=ZEB_TARGET, n_trials=OPTUNA_TRIALS):
    import optuna; optuna.logging.set_verbosity(optuna.logging.WARNING)
    Pl = df['P_load'].values; Ps = df['P_solar'].values
    log = []
    def obj(trial):
        n = trial.suggest_int('n', 0, ESS_N_MAX)
        C = n * ESS_UNIT_KWH
        r = greedy_sim(Ps, Pl, C)
        penalty = max(0, target_sr - r['SR']) * 500
        score = -r['SR'] + penalty + n * 0.002
        log.append({'n': n, 'SR': r['SR']})
        return score
    t0 = time.time()
    st = optuna.create_study(direction='minimize',
         sampler=optuna.samplers.TPESampler(seed=42))
    st.optimize(obj, n_trials=n_trials)
    bn = st.best_params['n']; bC = bn * ESS_UNIT_KWH
    br = greedy_sim(Ps, Pl, bC)
    return {**br, 'C_kWh': bC, 'n_units': bn,
            'trial_log': log, 'elapsed_s': time.time()-t0,
            'grade': get_zeb_grade(br['SR'])}


# ═══════════════════════════════════════════════════════════════════════════
# STEP 7 : ★ Progressive ESS Analysis Visualization (ver3 — English output)
# ═══════════════════════════════════════════════════════════════════════════

def step7_progressive_plot(sid, prog, opt_res, patterns, cfg, out_dir):
    """
    6-panel progressive ESS analysis visualization (all text in English)
    Panel 1: ESS capacity vs Electrical Self-Sufficiency Rate (ZEB grade bands)
    Panel 2: Marginal benefit per 10 kWh
    Panel 3: Annual cost savings vs ESS capacity
    Panel 4: ZEB grade transition roadmap
    Panel 5: Worst-season 24h charge/discharge strategy
    """
    fig = plt.figure(figsize=(18, 14))
    gs  = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.35)
    fig.suptitle(
        f"{sid} — Progressive ESS Sizing (10 kWh steps, ver3)\n"
        f"[Based on actual electrical data — ZEB calibration NOT applied to ESS]",
        fontsize=12, fontweight='bold')

    rbycap = prog['results_by_cap']
    caps   = sorted(rbycap.keys())
    srs    = [rbycap[c]['SR'] * 100 for c in caps]

    # ── Panel 1: ESS vs SR  (ZEB grade band background) ─────────────────
    ax1 = fig.add_subplot(gs[0, :])

    grade_bg = {5:(1.0, 0.92, 0.92), 4:(1.0, 0.96, 0.86),
                3:(0.93, 1.0, 0.93), 2:(0.86, 0.94, 1.0), 1:(0.88, 1.0, 0.95)}
    grade_labels = {5:'ZEB Grade 5', 4:'ZEB Grade 4', 3:'ZEB Grade 3',
                    2:'ZEB Grade 2', 1:'ZEB Grade 1'}

    for g, (lo, hi) in sorted(ELEC_ZEB_GRADES.items(), reverse=True):
        ax1.axhspan(lo*100, hi*100, alpha=0.35, color=grade_bg.get(g,(0.95,0.95,0.95)))
        ax1.text(ESS_MAX_KWH * 1.01, (lo+hi)/2*100,
                 grade_labels[g], fontsize=8, va='center',
                 color=GRADE_COLORS.get(g, 'gray'))

    for thr in GRADE_THRESHOLDS:
        ax1.axhline(thr*100, color='#666', ls=':', lw=0.8, alpha=0.6)

    ax1.plot(caps, srs, 'o-', color='steelblue', lw=2.5, ms=6,
             label='Electrical Self-Sufficiency Rate (%)', zorder=3)

    for g, c in sorted(prog['grade_transitions'].items()):
        sr_at = rbycap[c]['SR'] * 100
        ax1.scatter([c], [sr_at], s=120, c=GRADE_COLORS.get(g,'gray'),
                    zorder=5, edgecolors='white', lw=1.5)
        ax1.annotate(f"Grade {g}\n{c:.0f} kWh",
                     (c, sr_at), xytext=(c+3, sr_at+3),
                     fontsize=8, color=GRADE_COLORS.get(g,'gray'), fontweight='bold')

    tc = prog['target_C']
    ax1.axvline(tc, color='crimson', ls='--', lw=1.8,
                label=f'Min. ESS for ZEB Grade 4: {tc:.0f} kWh')
    ax1.axvline(opt_res['C_kWh'], color='purple', ls=':', lw=1.5,
                label=f'Optuna Optimal: {opt_res["C_kWh"]:.0f} kWh')
    rl, rh = prog['ref_range']
    ax1.axvspan(rl, rh, alpha=0.12, color='green',
                label=f'Building Reference Range: {rl}~{rh} kWh')

    ax1.set_xlabel('ESS Capacity (kWh)')
    ax1.set_ylabel('Electrical Self-Sufficiency Rate (%)')
    ax1.set_title(
        f'ZEB Grade Improvement Curve — 10 kWh Incremental ESS Addition\n'
        f'Baseline SR: {prog["sr_base"]*100:.1f}%  (ZEB Grade {prog["grade_base"]})')
    ax1.set_xlim(-5, ESS_MAX_KWH + 30); ax1.set_ylim(0, 115)
    ax1.set_xticks(caps)
    ax1.tick_params(axis='x', rotation=45, labelsize=8)
    ax1.legend(fontsize=8, loc='lower right')

    # ── Panel 2: Marginal benefit (SR gain per 10 kWh) ──────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    mb = prog['marginal_benefits']
    if mb:
        mb_c  = [m['C_kWh'] for m in mb if m['delta_sr'] >= 0]
        mb_d  = [m['delta_sr']*100 for m in mb if m['delta_sr'] >= 0]
        colors_mb = [GRADE_COLORS.get(rbycap.get(c,{}).get('grade',5),'gray')
                     for c in mb_c]
        ax2.bar(mb_c, mb_d, width=8, color=colors_mb, alpha=0.8, edgecolor='white')
        ax2.axvline(prog['optimal_C'], color='orange', ls='--', lw=1.5,
                    label=f'Peak Marginal Benefit: {prog["optimal_C"]:.0f} kWh')
        ax2.axvline(tc, color='crimson', ls='--', lw=1.5,
                    label=f'ZEB Grade 4 target: {tc:.0f} kWh')
    ax2.set_xlabel('ESS Capacity (kWh)')
    ax2.set_ylabel('SR Gain (%p / 10 kWh)')
    ax2.set_title('Marginal Benefit Analysis\n'
                  '(Diminishing returns zone = low cost-effectiveness)')
    ax2.legend(fontsize=8)

    # ── Panel 3: Annual cost savings ────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    saved = [rbycap[c]['saved_cost_won'] / 10000 for c in caps]
    ax3.plot(caps, saved, 's-', color='mediumseagreen', lw=2, ms=5)
    ax3.axvline(tc, color='crimson', ls='--', lw=1.5,
                label=f'ZEB Grade 4: {tc:.0f} kWh')
    ax3.fill_between(caps, saved, alpha=0.15, color='mediumseagreen')
    ax3.set_xlabel('ESS Capacity (kWh)')
    ax3.set_ylabel('Annual Grid Cost Savings (10,000 KRW)')
    ax3.set_title('Annual Grid Purchase Savings vs ESS Capacity\n'
                  '(Time-of-Use tariff applied)')
    ax3.legend(fontsize=8)

    # ── Panel 4: ZEB Grade Transition Roadmap ───────────────────────────
    ax4 = fig.add_subplot(gs[2, 0])
    trans = prog['grade_transitions']
    if trans:
        all_grades = sorted(trans.keys())
        all_caps   = [trans[g] for g in all_grades]
        all_srs    = [rbycap[trans[g]]['SR']*100 for g in all_grades]
        bar_colors = [GRADE_COLORS.get(g,'gray') for g in all_grades]

        ax4.bar([f'Current\nGrade {prog["grade_base"]}\n(No ESS)'],
                [prog['sr_base']*100],
                color=GRADE_COLORS.get(prog['grade_base'],'gray'),
                alpha=0.6, width=0.5)
        ax4.bar([f'Grade {g}\n{c:.0f} kWh' for g, c in zip(all_grades, all_caps)],
                all_srs, color=bar_colors, alpha=0.85, width=0.5)
        for i, (g, c, sr) in enumerate(zip(all_grades, all_caps, all_srs)):
            ax4.text(i+1, sr+1, f'{sr:.1f}%', ha='center', fontsize=9)
        for thr in GRADE_THRESHOLDS:
            ax4.axhline(thr*100, color='#999', ls=':', lw=0.8)
    ax4.set_ylabel('SR (%)')
    ax4.set_ylim(0, 115)
    ax4.set_title('ZEB Grade Transition Roadmap\n'
                  '(Minimum ESS to achieve each grade)')
    ax4.tick_params(axis='x', labelsize=8)

    # ── Panel 5: Worst-season 24h charge/discharge strategy ─────────────
    ax5 = fig.add_subplot(gs[2, 1])
    if patterns:
        hours = range(24)
        worst_season = max(patterns,
                           key=lambda s: patterns[s]['daily_deficit_kWh'],
                           default=None)
        if worst_season:
            pat = patterns[worst_season]
            ax5.fill_between(hours, pat['P_solar_24h'], alpha=0.4, color='gold',
                             label=f'{worst_season} Solar Gen.')
            ax5.fill_between(hours, pat['P_load_24h'],  alpha=0.5, color='steelblue',
                             label=f'{worst_season} Load')
            surp = pat['P_surplus_24h']
            defi = pat['P_deficit_24h']
            ax5.fill_between(hours, pat['P_solar_24h'], pat['P_load_24h'],
                             where=surp>0, alpha=0.3, color='gold',
                             label='ESS Charge Zone')
            ax5.fill_between(hours, pat['P_solar_24h'], pat['P_load_24h'],
                             where=defi>0, alpha=0.35, color='red',
                             label='ESS Discharge Zone')
            ax5.set_xlabel('Hour of Day')
            ax5.set_ylabel('kWh/h')
            ax5.set_title(
                f'Bottleneck Season ({worst_season}) — 24h Charge/Discharge Strategy\n'
                f'Daily deficit: {pat["daily_deficit_kWh"]:.1f} kWh  '
                f'→ Recommended ESS ≥ {pat["daily_deficit_kWh"]:.0f} kWh')
            ax5.legend(fontsize=8)
            ax5.set_xticks(range(0, 24, 3))

    plt.tight_layout()
    outpath = os.path.join(out_dir, f'{sid}_progressive_v3.png')
    plt.savefig(outpath, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"    -> {os.path.basename(outpath)} saved")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 8 : 일반화 프레임워크 (실측 기준)
# ═══════════════════════════════════════════════════════════════════════════

def step8_generalization(all_results, out_dir):
    rows = []
    for r in all_results:
        g  = r.get('grade_transitions', {})
        rl, rh = r.get('ref_range', (0, 0))
        tc = r.get('target_C', 0)
        rows.append({
            'Site':          r['id'],
            'BuildingType':  TYPE_EN.get(r['type'], r['type']),
            'Operation':     OP_EN.get(r['operation'], r['operation']),
            'CurrentSR(%)':  round(r['sr_base']*100, 1),
            'CurrentGrade':  r['grade_base'],
            'ZEB4ESS(kWh)':  g.get(4, tc),
            'ZEB3ESS(kWh)':  g.get(3, '-'),
            'ZEB2ESS(kWh)':  g.get(2, '-'),
            'ZEB1ESS(kWh)':  g.get(1, '-'),
            'RefRange(kWh)': f"{rl}~{rh}",
            'InRefRange':    'Yes' if r.get('in_ref_range') else 'No',
            'Optuna(kWh)':   r.get('opt_res', {}).get('C_kWh', '-'),
        })

    df = pd.DataFrame(rows)
    type_sum = df.groupby('BuildingType').agg(
        {'ZEB4ESS(kWh)': lambda x: pd.to_numeric(x, errors='coerce').mean(),
         'CurrentSR(%)': 'mean'}).round(1).reset_index()

    # 회귀식: ESS = f(연간부족)
    deficit = []
    for r in all_results:
        al  = r.get('annual_load', 0)
        sr  = r.get('sr_base', 0)
        def_ = al * max(0, ZEB_TARGET - sr)
        deficit.append(def_)
    ess_vals = [r.get('target_C', 0) for r in all_results]
    x = np.array(deficit); y = np.array(ess_vals)
    mask = (x > 0) & (y > 0)
    if mask.sum() >= 2:
        coef = np.polyfit(x[mask], y[mask], 1)
        reg_str = (f"ESS(kWh) = {coef[0]:.5f} x Annual_Deficit(kWh)"
                   f" + {coef[1]:.1f}")
    else:
        coef = None; reg_str = "Insufficient sites"

    # ── Visualization (English) ──────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        'ZEB Grade 5 → Grade 4  ESS Generalization Framework (ver3 — Actual Data)',
        fontsize=12, fontweight='bold')

    ax1 = axes[0]
    # building type name → English
    type_map_en = df.copy()
    tg = type_map_en.groupby('BuildingType')['ZEB4ESS(kWh)'].apply(
        lambda x: pd.to_numeric(x, errors='coerce').mean()).sort_values()
    clrs = ['steelblue','tomato','mediumseagreen','orchid'][:len(tg)]
    tg.plot(kind='bar', ax=ax1, color=clrs, alpha=0.85)
    for i, v in enumerate(tg.values):
        if pd.notna(v): ax1.text(i, v+1, f'{v:.0f} kWh', ha='center', fontsize=9)
    ax1.set_ylabel('Avg. ESS Capacity (kWh)')
    ax1.set_title('Avg. Min. ESS for ZEB Grade 4\n(By Building Type)')
    ax1.tick_params(axis='x', rotation=20)

    ax2 = axes[1]
    ids   = [r['id'] for r in all_results]
    sr_b  = [r['sr_base']*100 for r in all_results]
    gt4_ess = [r.get('grade_transitions',{}).get(4, 0) for r in all_results]
    x_arr = np.arange(len(ids)); w = 0.35
    ax2.bar(x_arr, sr_b, w, label='Current SR (%)', color='lightcoral', alpha=0.85)
    ax2b = ax2.twinx()
    ax2b.bar(x_arr+w, gt4_ess, w, label='Grade 4 ESS (kWh)',
             color='royalblue', alpha=0.7)
    ax2.axhline(40, color='crimson', ls='--', lw=1, label='ZEB Grade 4  (40%)')
    ax2.set_xticks(x_arr+w/2); ax2.set_xticklabels(ids, rotation=30, fontsize=8)
    ax2.set_ylabel('SR (%)'); ax2b.set_ylabel('ESS (kWh)')
    ax2.set_title('Current SR & Min. ESS for ZEB Grade 4\n(Per Site)')
    ax2.legend(fontsize=8, loc='upper left')
    ax2b.legend(fontsize=8, loc='upper right')

    ax3 = axes[2]
    ax3.scatter(deficit, ess_vals, s=80, c='steelblue', zorder=3)
    for i, r in enumerate(all_results):
        ax3.annotate(r['id'], (deficit[i], ess_vals[i]),
                     fontsize=8, xytext=(3,3), textcoords='offset points')
    if coef is not None:
        xl = np.linspace(min(deficit)*0.9, max(deficit)*1.1, 50)
        ax3.plot(xl, np.polyval(coef, xl), 'r--', lw=1.5, label=reg_str)
        ax3.legend(fontsize=8)
    ax3.set_xlabel('Annual Deficit to Grade 4 (kWh)')
    ax3.set_ylabel('Min. ESS for Grade 4 (kWh)')
    ax3.set_title('Generalization Regression\n(ESS Prediction for New Buildings)')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '8_generalization_v3.png'),
                dpi=130, bbox_inches='tight')
    plt.close()

    return df, type_sum, reg_str


# ═══════════════════════════════════════════════════════════════════════════
# STEP 9 : 최종 보고서
# ═══════════════════════════════════════════════════════════════════════════

def step9_final_report(all_results, fw_df, type_sum, reg_str, out_dir):
    print("\n" + "═"*72)
    print("  최종 보고서 — ZEB 5→4등급 ESS 최적화 (ver3 실측 기준)")
    print("═"*72)

    # ① 데이터 몰림 재분배 현황
    print("\n■ [데이터 몰림(Catch-up) 재분배 현황]")
    rd_rows = []
    for r in all_results:
        needs = CATCHUP_REDISTRIBUTE_SITES.get(r['id'], False)
        ev = r.get('n_catchup_events', 0); rh = r.get('redis_hours', 0)
        if r['id'] == 'S3':   s = '불필요 (몰림 없음)'
        elif r['id'] == 'S8': s = f'★ 전처리 후에도 몰림 여전→재분배 {ev}건'
        elif needs and ev > 0: s = f'완료 {ev}건/{rh}h'
        else:                  s = '불필요/미탐지'
        rd_rows.append([r['id'], r['display'][:14], '필요' if needs else '-', ev, rh, s])
    print(tabulate(rd_rows,
        headers=['ID','사이트','필요','이벤트','처리h','결과'], tablefmt='grid'))
    print("  ※ S3 불필요, S8 등 7개소 재분배 필요 (UTRON 보고서 명시)")

    # ② 10kWh 단위 등급 향상 결과
    print("\n■ [10kWh 단위 ZEB 등급 향상 — 실측 전기 기준]")
    g_rows = []
    for r in all_results:
        gt = r.get('grade_transitions', {})
        g_rows.append([
            r['id'], r['display'][:12],
            f"{r['sr_base']*100:.1f}%", r['grade_base'],
            f"{gt.get(4, '-')}",
            f"{gt.get(3, '-')}",
            f"{gt.get(2, '-')}",
            f"{gt.get(1, '-')}",
            f"{r['ref_range'][0]}~{r['ref_range'][1]}",
            '○' if r.get('in_ref_range') else '✗',
        ])
    print(tabulate(g_rows,
        headers=['ID','사이트','현재SR','현재등급',
                 'ZEB4ESS','ZEB3ESS','ZEB2ESS','ZEB1ESS',
                 '참조범위','범위內'], tablefmt='grid'))

    # ③ 알고리즘 비교
    print("\n■ [알고리즘 비교 — ZEB 4등급 달성 ESS]")
    a_rows = []
    for r in all_results:
        gt   = r.get('grade_transitions', {})
        oc   = r.get('optimal_C', '-')
        optc = r.get('opt_res', {}).get('C_kWh', '-')
        a_rows.append([r['id'], r['display'][:12],
                       f"{gt.get(4, '-')}", f"{oc}", f"{optc}"])
    print(tabulate(a_rows,
        headers=['ID','사이트','Greedy ZEB4','최적점(한계효용)','Optuna'],
        tablefmt='grid'))

    # ④ 건물유형 가이드라인
    print("\n■ [건물 유형별 ESS 설계 가이드라인]")
    print(tabulate(type_sum.values.tolist(),
                   headers=type_sum.columns.tolist(),
                   tablefmt='grid', floatfmt='.1f'))
    print(f"\n  일반화 회귀식: {reg_str}")

    # ⑤ 최종 비교 시각화 (English labels)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(
        '8-Site ZEB ESS Optimization Summary (ver3 — Actual Electrical Data)',
        fontsize=12, fontweight='bold')

    ax1 = axes[0]
    ids  = [r['id'] for r in all_results]
    sr_b = [r['sr_base']*100 for r in all_results]
    sr_a = []
    for r in all_results:
        tc_g4 = r.get('grade_transitions',{}).get(4, None)
        if tc_g4 is not None:
            rc = r.get('results_by_cap', {})
            sr_a.append(rc.get(tc_g4, {}).get('SR', r['sr_base']) * 100)
        else:
            sr_a.append(r['sr_base'] * 100)
    x = np.arange(len(ids)); w = 0.35
    ax1.bar(x-w/2, sr_b, w, label='Current SR',
            color='lightcoral', alpha=0.85)
    ax1.bar(x+w/2, sr_a, w, label='SR after ZEB Grade 4 ESS',
            color='royalblue', alpha=0.85)
    ax1.axhline(40, color='crimson', ls='--', lw=1.5, label='ZEB Grade 4 (40%)')
    ax1.axhline(60, color='purple',  ls='--', lw=1,   label='ZEB Grade 3 (60%)')
    ax1.set_xticks(x); ax1.set_xticklabels(ids)
    ax1.set_ylabel('Self-Sufficiency Rate (%)')
    ax1.set_title('SR Improvement by Site\n(Current vs After ESS Installation)')
    ax1.legend(fontsize=8)

    ax2 = axes[1]
    ess_v = [r.get('grade_transitions',{}).get(4, ESS_MAX_KWH) for r in all_results]
    type_color_map = {
        '문화시설': 'gold',  '복지시설': 'tomato',
        '행정시설': 'steelblue', '교육시설': 'mediumseagreen',
    }
    type_label_map = {
        '문화시설': 'Cultural', '복지시설': 'Welfare',
        '행정시설': 'Admin.',   '교육시설': 'Education',
    }
    tc_colors = [type_color_map.get(r['type'], 'gray') for r in all_results]
    b = ax2.bar(ids, ess_v, color=tc_colors, alpha=0.85, width=0.6)
    for bar, v in zip(b, ess_v):
        ax2.text(bar.get_x()+bar.get_width()/2, v+0.5,
                 f'{v:.0f} kWh', ha='center', fontsize=9)
    from matplotlib.patches import Patch
    ax2.legend(
        handles=[Patch(fc=v, label=type_label_map.get(k, k))
                 for k, v in type_color_map.items()
                 if k in {r['type'] for r in all_results}],
        fontsize=8)
    ax2.set_ylabel('Min. ESS Capacity (kWh)')
    ax2.set_title('Min. ESS for ZEB Grade 4\n(Colored by Building Type)')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '9_final_summary_v3.png'),
                dpi=130, bbox_inches='tight')
    plt.close()

    fw_df.to_csv(os.path.join(out_dir, 'ess_results_v3.csv'),
                 index=False, encoding='utf-8-sig')
    type_sum.to_csv(os.path.join(out_dir, 'ess_type_guide_v3.csv'),
                    index=False, encoding='utf-8-sig')
    print(f"\n  ✔ 결과 저장 완료: {out_dir}")


# ═══════════════════════════════════════════════════════════════════════════
# ── MAIN PIPELINE (ver3)
# ═══════════════════════════════════════════════════════════════════════════

def run_pipeline_v3():
    print("═"*72)
    print("  ESS 최적화 파이프라인 ver3 — 10kWh 단위 점진적 등급 향상")
    print("  실측 전기 데이터 기준 (ZEB 보정 ESS에 미적용)")
    print(f"  분석 기간: {ANALYSIS_START} ~ {ANALYSIS_END}")
    print("═"*72)

    site_raws = step1_load(DATA_ROOT)
    if not site_raws:
        print("!! 데이터 없음. DATA_ROOT 경로 확인하세요.")
        return

    all_results = []

    for cfg in SITE_CONFIGS:
        sid = cfg['id']
        if sid not in site_raws:
            print(f"\n  [{sid}] 데이터 없음 — 건너뜀"); continue

        raw = site_raws[sid]
        print(f"\n{'─'*60}")
        print(f"  [{sid}] {cfg['display']}")
        print(f"{'─'*60}")

        main_name   = next((m for m in cfg['main_names']  if m in raw), None)
        solar_avail = [s for s in cfg['solar_names'] if s in raw]
        if not main_name or not solar_avail:
            print("    !! 필수 계량기 없음 — 건너뜀"); continue

        # 누적 → 차분
        load_cum  = to_hourly_raw(raw[main_name], cfg['main_mult_doc'])
        solar_cum = to_hourly_raw(raw[solar_avail[0]], cfg['solar_mult_doc'])
        for ex in solar_avail[1:]:
            solar_cum = solar_cum.add(to_hourly_raw(raw[ex], cfg['solar_mult_doc']),
                                      fill_value=0)
        load_d  = diff_series(load_cum).loc[ANALYSIS_START:ANALYSIS_END]
        solar_d = diff_series(solar_cum).loc[ANALYSIS_START:ANALYSIS_END]
        if len(load_d) < 168:
            print(f"    !! 데이터 부족 ({len(load_d)}h)"); continue

        ds = str(load_d.index[0].date()); de = str(load_d.index[-1].date())
        print(f"    기간: {ds} ~ {de}  ({len(load_d)}h)")

        # STEP 2a: IQR 이상치 탐지
        # ── Fix 2: 사이트별 IQR_K 읽어 적용 (S1은 5.0, 나머지 기본 3.0)
        site_iqr_k = cfg.get('iqr_k', IQR_K)
        print(f"  [STEP 2a] IQR 이상치 탐지 + 계층적 보간 "
              f"(IQR_K={site_iqr_k}"
              f"{' ← Fix2: S1 펄스 분해능 보정' if site_iqr_k != IQR_K else ''})")
        load_c,  ql = step2a_clean_outliers(load_d,  iqr_k=site_iqr_k,
                                             label=f'{sid}/MAIN')
        solar_c, qs = step2a_clean_outliers(solar_d, iqr_k=site_iqr_k,
                                             label=f'{sid}/태양광')

        # STEP 2c: 데이터 몰림 재분배
        needs_r = CATCHUP_REDISTRIBUTE_SITES.get(sid, False)
        print(f"  [STEP 2c] 데이터 몰림 재분배"
              f" ({'적용' if needs_r else '불필요(S3)'})")
        if needs_r:
            load_c,  lmask, qrl = step2c_redistribute_catchup(
                load_c,  label=f'{sid}/MAIN')
            solar_c, smask, _   = step2c_redistribute_catchup(
                solar_c, label=f'{sid}/태양광')
            if sid == 'S8' and qrl['n_catchup_events'] > 0:
                print(f"    ★ S8 완주용봉초: 전처리 후에도 몰림 여전 — "
                      f"재분배 {qrl['n_catchup_events']}건/{qrl['redistributed_hours']}h")
        else:
            lmask = pd.Series(False, index=load_c.index)
            qrl   = {'n_catchup_events': 0, 'redistributed_hours': 0}

        # STEP 2b: 물리적 검증
        print("  [STEP 2b] 물리적 배율 타당성 검증")
        val = step2b_validate_mult(load_c, solar_c, cfg)
        print(f"    V1={val['V1_load_range']} V2={val['V2_max_hourly']} "
              f"V3={val['V3_sr_range']} → "
              f"{'통과' if val['overall_valid'] else '주의'}")

        # STEP 3: ZEB 맥락 설명 (ESS에 미적용)
        print("  [STEP 3] ZEB 맥락 역산 (ESS 미적용)")
        zeb_ctx = step3_zeb_context(load_c, solar_c, cfg)
        print(f"    {zeb_ctx['note'][:70]}...")

        # DataFrame (★ 실측 무보정 — ZEB 보정 없이)
        df = pd.DataFrame({
            'P_load':           load_c,      # ★ 보정 없는 실측
            'P_solar':          solar_c,
            'is_redistributed': lmask.astype(int),
        }).dropna()
        df = df[df.index >= ANALYSIS_START]
        df = df[df.index <= ANALYSIS_END]

        # STEP 4b: 파생 변수
        print("  [STEP 4b] 파생 변수 생성")
        df = step4b_derive(df)

        # STEP 4c: 계절 패턴
        print("  [STEP 4c] 계절별 대표 패턴 추출")
        patterns = step4c_seasonal(df)
        for sn, p in patterns.items():
            print(f"    [{sn}] 일부족={p['daily_deficit_kWh']:.1f}kWh | "
                  f"일잉여={p['daily_surplus_kWh']:.1f}kWh")

        # STEP 5: ★ 10kWh 단위 점진적 분석
        print("  [STEP 5] 10kWh 단위 점진적 ESS 등급 향상 분석")
        prog = step5_progressive_ess(df, sid)
        print(f"    ZEB 4등급 달성: {prog['target_C']:.0f}kWh | "
              f"참조범위 {prog['ref_range'][0]}~{prog['ref_range'][1]}kWh | "
              f"범위內: {'○' if prog['in_ref_range'] else '✗'}")

        # STEP 6: Optuna
        print(f"  [STEP 6] Optuna ({OPTUNA_TRIALS} trials)")
        opt_res = step6_optuna(df)
        print(f"    최적: {opt_res['n_units']}대/{opt_res['C_kWh']:.0f}kWh | "
              f"SR={opt_res['SR']*100:.1f}% (ZEB {opt_res['grade']}등급) | "
              f"{opt_res['elapsed_s']:.1f}s")

        # STEP 7: 점진적 시각화
        print("  [STEP 7] 점진적 ESS 분석 시각화")
        step7_progressive_plot(sid, prog, opt_res, patterns, cfg, OUTPUT_DIR)

        all_results.append({
            **cfg,
            'data_start':         ds, 'data_end': de,
            'n_hours':            len(df),
            'n_catchup_events':   qrl['n_catchup_events'],
            'redis_hours':        qrl['redistributed_hours'],
            'needs_redis':        needs_r,
            'annual_load':        df['P_load'].sum(),
            'annual_solar':       df['P_solar'].sum(),
            **prog,
            'opt_res':            opt_res,
            'patterns':           patterns,
        })

    if not all_results:
        print("\n!! 분석 완료 사이트 없음"); return

    # STEP 8: 일반화
    print(f"\n{'─'*60}")
    print("  [STEP 8] 일반화 프레임워크")
    fw_df, type_sum, reg_str = step8_generalization(all_results, OUTPUT_DIR)

    # STEP 9: 최종 보고서
    step9_final_report(all_results, fw_df, type_sum, reg_str, OUTPUT_DIR)

    print("\n" + "═"*72)
    print("  ✔ ver3 파이프라인 완료")
    print(f"  ✔ 결과 경로: {OUTPUT_DIR}")
    print("═"*72)


if __name__ == '__main__':
    run_pipeline_v3()
