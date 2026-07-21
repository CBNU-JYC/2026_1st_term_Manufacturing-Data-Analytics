# -*- coding: utf-8 -*-
"""
=============================================================
  Step 1: ILP 절대 최적해 (Absolute Optimal Solution)
=============================================================
JSSP를 수학적으로 정식화(MILP)하여 "증명된 최적해"를 구합니다.
이 결과는 이후 모든 기법의 평가 기준(Ground Truth)이 됩니다.

핵심 개념:
  - 결정 변수(Decision Variable): 각 작업의 시작 시간
  - 제약 조건(Constraints): 선행공정, 설비 중복 방지
  - 목적 함수(Objective): Makespan(전체 완료 시간) 최소화

실행: conda activate mfg_data && python step1_ilp_optimal.py
=============================================================
"""

import pulp
import pandas as pd
import time
from utils import MACHINES, JOB_DATA, print_problem, draw_gantt


def solve_jssp_ilp(job_data, machines, verbose=True):
    """
    JSSP를 Mixed Integer Linear Programming(MILP)으로 풀어 최적해를 구합니다.

    수학적 정식화:
    ┌────────────────────────────────────────────────────────────┐
    │ Minimize  C_max (Makespan)                                │
    │                                                           │
    │ Subject to:                                               │
    │  ① 선행 제약: S[j,next_m] ≥ S[j,curr_m] + p[j,curr_m]    │
    │  ② 이접 제약: 같은 설비에 두 작업이 동시에 배정되지 않음     │
    │  ③ Makespan:  C_max ≥ S[j,last_m] + p[j,last_m]  ∀j     │
    └────────────────────────────────────────────────────────────┘

    Args:
        job_data: 작업 데이터 {job_name: [(machine, processing_time), ...]}
        machines: 설비 이름 리스트
        verbose: 진행 상황 출력 여부

    Returns:
        (DataFrame, makespan): 스케줄 결과와 최적 Makespan
    """
    if verbose:
        print("  🔧 ILP 솔버가 최적화 계산을 시작합니다...")

    t0 = time.time()

    # ── 1. Big-M (충분히 큰 수): 모든 가공시간의 합 ──
    big_m = sum(t for ops in job_data.values() for _, t in ops)

    # ── 2. 최적화 모델 생성 (목표: Makespan 최소화) ──
    prob = pulp.LpProblem("JSSP_Optimal", pulp.LpMinimize)

    # ── 3. 결정 변수 선언 ──
    # S[job, machine]: 시작 시간 (연속 변수, 0 이상)
    S = pulp.LpVariable.dicts("Start",
        ((job, m) for job in job_data for m, _ in job_data[job]),
        lowBound=0, cat='Continuous')

    # C_max: 전체 완료 시간 (우리가 최소화할 목표)
    C_max = pulp.LpVariable("Makespan", lowBound=0, cat='Continuous')

    # X[j1, j2, m]: "설비 m에서 j1이 j2보다 먼저인가?" (이진 변수, 0 또는 1)
    job_names = list(job_data.keys())
    X = pulp.LpVariable.dicts("Seq",
        ((j1, j2, m) for j1 in job_names for j2 in job_names
         if j1 < j2 for m in machines),
        cat='Binary')

    # ── 4. 제약 조건 추가 ──

    # ① 선행 제약 (Precedence): 한 Job 내에서 공정 순서를 지켜야 함
    #    예: Engine_Block은 반드시 Milling → Lathe → Drill 순서
    for job, ops in job_data.items():
        for i in range(len(ops) - 1):
            curr_m, curr_time = ops[i]
            next_m, _ = ops[i + 1]
            prob += S[job, next_m] >= S[job, curr_m] + curr_time, \
                     f"Prec_{job}_{i}"

    # ② 이접 제약 (Disjunctive): 같은 설비에 두 작업이 겹치면 안 됨
    #    Big-M 기법으로 "둘 중 하나가 먼저" 조건을 수학적으로 표현
    for m in machines:
        for j1 in job_names:
            for j2 in job_names:
                if j1 < j2:
                    t1 = next((t for mach, t in job_data[j1] if mach == m), None)
                    t2 = next((t for mach, t in job_data[j2] if mach == m), None)
                    if t1 is not None and t2 is not None:
                        prob += S[j2, m] >= S[j1, m] + t1 - big_m * (1 - X[j1, j2, m])
                        prob += S[j1, m] >= S[j2, m] + t2 - big_m * X[j1, j2, m]

    # ③ Makespan 제약: 모든 작업의 마지막 공정 완료 이후여야 함
    for job, ops in job_data.items():
        last_m, last_time = ops[-1]
        prob += C_max >= S[job, last_m] + last_time

    # ── 5. 목적 함수: Makespan 최소화 ──
    prob += C_max

    # ── 6. 풀이 ──
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    elapsed = time.time() - t0

    # ── 7. 결과 처리 ──
    if pulp.LpStatus[prob.status] != 'Optimal':
        print("  ❌ 최적해를 찾지 못했습니다.")
        return pd.DataFrame(), None, elapsed

    optimal_makespan = pulp.value(C_max)

    if verbose:
        print(f"  ✅ 최적해 발견! Makespan = {optimal_makespan:.0f}분 ({elapsed:.3f}초)")
        print()

        # 스케줄 상세 출력
        print("  ┌────────────────┬──────────┬────────┬────────┬──────────┐")
        print("  │    작업(Job)    │  설비     │  시작  │  종료  │ 가공시간  │")
        print("  ├────────────────┼──────────┼────────┼────────┼──────────┤")

    results = []
    for job, ops in job_data.items():
        for m, p_time in ops:
            start_t = pulp.value(S[job, m])
            end_t = start_t + p_time
            results.append({
                'Job': job, 'Machine': m,
                'Start_Time': start_t, 'End_Time': end_t,
                'Processing_Time': p_time
            })
            if verbose:
                print(f"  │ {job:<14} │ {m:<8} │ {start_t:>5.0f}  │ {end_t:>5.0f}  │ {p_time:>7}분 │")

    if verbose:
        print("  └────────────────┴──────────┴────────┴────────┴──────────┘")

    result_df = pd.DataFrame(results).sort_values(
        by=['Start_Time', 'Machine']).reset_index(drop=True)

    return result_df, optimal_makespan, elapsed


# ══════════════════════════════════════════════════════════════
#  메인 실행
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print_problem()

    print("=" * 60)
    print("  Step 1: ILP 절대 최적해 (Mathematical Optimization)")
    print("=" * 60)
    print()

    result_df, makespan, elapsed = solve_jssp_ilp(JOB_DATA, MACHINES)

    print()
    print(f"  📌 이 Makespan = {makespan:.0f}분이 '수학적으로 증명된 최적해'입니다.")
    print(f"     → 어떤 방법을 써도 이보다 짧은 스케줄은 불가능합니다.")
    print(f"     → 이 값을 기준으로 휴리스틱/DRL의 성능을 평가합니다.")
    print()

    draw_gantt(result_df, f"ILP Optimal Schedule (Makespan: {makespan:.0f})")
