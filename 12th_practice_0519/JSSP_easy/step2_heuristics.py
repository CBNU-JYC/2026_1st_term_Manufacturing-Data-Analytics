# -*- coding: utf-8 -*-
"""
=============================================================
  Step 2: 휴리스틱 스케줄링 (Dispatching Rules)
=============================================================
현장에서 가장 많이 쓰이는 "경험 기반 규칙"으로 스케줄링합니다.
수학적 최적해를 보장하지 않지만, 즉시 결과를 얻을 수 있습니다.

학습할 규칙:
  - FIFO: 먼저 온 작업을 먼저 처리
  - SPT:  가공시간이 짧은 작업을 먼저 처리
  - LPT:  가공시간이 긴 작업을 먼저 처리

실행: conda activate mfg_data && python step2_heuristics.py
=============================================================
"""

import simpy
import pandas as pd
import time
from utils import MACHINES, JOB_DATA, print_problem, draw_gantt, draw_gantt_console


# ══════════════════════════════════════════════════════════════
#  1. 우선순위 규칙(Dispatching Rules) 정의
# ══════════════════════════════════════════════════════════════
#
# simpy.PriorityResource는 priority 값이 "작을수록" 먼저 처리합니다.

def rule_fifo(job_name, op_idx, m_name, p_time):
    """
    FIFO (First In First Out): 선입선출
    → 먼저 도착한 작업을 먼저 처리 (동일 우선순위 = 도착순)
    """
    return 0  # 모두 같은 우선순위 → 자연스럽게 도착 순서대로


def rule_spt(job_name, op_idx, m_name, p_time):
    """
    SPT (Shortest Processing Time): 최단 가공시간 우선
    → 빨리 끝나는 작업부터 처리하여 평균 대기시간을 줄임
    """
    return p_time  # 가공시간이 짧을수록(숫자가 작을수록) 높은 우선순위


def rule_lpt(job_name, op_idx, m_name, p_time):
    """
    LPT (Longest Processing Time): 최장 가공시간 우선
    → 오래 걸리는 작업부터 처리하여 설비 유휴시간을 줄임
    """
    return -p_time  # 가공시간이 길수록(음수가 작을수록) 높은 우선순위


# ══════════════════════════════════════════════════════════════
#  2. SimPy 기반 JSSP 환경
# ══════════════════════════════════════════════════════════════

class JSSPSimEnvironment:
    """SimPy PriorityResource를 활용한 JSSP 시뮬레이션 환경"""

    def __init__(self, env, machine_names):
        self.env = env
        self.machines = {
            m: simpy.PriorityResource(env, capacity=1)
            for m in machine_names
        }
        self.log = []

    def process_job(self, job_name, operations, priority_rule=None):
        """하나의 작업이 공정 순서대로 설비를 거치는 과정"""
        for op_idx, (m_name, p_time) in enumerate(operations):
            machine = self.machines[m_name]

            # 우선순위 결정
            priority = 0
            if priority_rule is not None:
                priority = priority_rule(job_name, op_idx, m_name, p_time)

            # 설비 요청 (우선순위 기반)
            req = machine.request(priority=priority)
            yield req

            start_time = self.env.now
            yield self.env.timeout(p_time)
            end_time = self.env.now

            machine.release(req)

            self.log.append({
                'Job': job_name,
                'Machine': m_name,
                'Start_Time': start_time,
                'End_Time': end_time,
                'Processing_Time': p_time,
            })

    def get_result_df(self):
        if not self.log:
            return pd.DataFrame()
        return pd.DataFrame(self.log).sort_values(
            by=['Start_Time', 'Machine']).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════
#  3. 시뮬레이션 실행 함수
# ══════════════════════════════════════════════════════════════

def run_dispatching(job_data, machines, rule_name='FIFO', verbose=True):
    """
    선택된 규칙으로 SimPy 시뮬레이션을 실행합니다.

    Args:
        rule_name: 'FIFO', 'SPT', 'LPT'

    Returns:
        (DataFrame, makespan, elapsed_time)
    """
    rules_map = {'FIFO': rule_fifo, 'SPT': rule_spt, 'LPT': rule_lpt}
    selected_rule = rules_map.get(rule_name.upper(), rule_fifo)

    t0 = time.time()

    env = simpy.Environment()
    job_shop = JSSPSimEnvironment(env, machines)

    for job_name, operations in job_data.items():
        env.process(job_shop.process_job(job_name, operations,
                                          priority_rule=selected_rule))
    env.run()
    elapsed = time.time() - t0

    result_df = job_shop.get_result_df()
    makespan = result_df['End_Time'].max() if not result_df.empty else 0

    if verbose:
        print(f"  [{rule_name}] Makespan = {makespan:.0f}분 ({elapsed:.4f}초)")

    return result_df, makespan, elapsed


# ══════════════════════════════════════════════════════════════
#  메인 실행
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print_problem()

    print("=" * 60)
    print("  Step 2: 휴리스틱 스케줄링 (Dispatching Rules)")
    print("=" * 60)
    print()
    print("  ★ 규칙별 핵심 전략:")
    print("    FIFO: '먼저 온 순서대로' (가장 단순)")
    print("    SPT:  '빨리 끝나는 것부터' (평균 대기시간 최소화)")
    print("    LPT:  '오래 걸리는 것부터' (설비 유휴시간 최소화)")
    print()

    results = []
    for rule in ['FIFO', 'SPT', 'LPT']:
        df, ms, et = run_dispatching(JOB_DATA, MACHINES, rule)
        results.append({'name': f'휴리스틱 ({rule})', 'makespan': ms, 'time': et, 'df': df})
        draw_gantt_console(df, f"Heuristic: {rule} (Makespan: {ms:.0f})")

    # ILP 최적해와 비교
    print()
    print("  ┌──────────┬────────────┬───────────────────────────────┐")
    print("  │   규칙    │ Makespan   │ 코멘트                        │")
    print("  ├──────────┼────────────┼───────────────────────────────┤")
    for r in results:
        name = r['name'].replace('휴리스틱 ', '')
        print(f"  │ {name:<8} │ {r['makespan']:>8.0f}분 │", end="")
        if r['makespan'] <= 27:
            print(f" 최적해와 동일! ✅                │")
        else:
            gap = r['makespan'] - 27
            print(f" 최적해 대비 +{gap:.0f}분 ({gap/27*100:.1f}%) ❌      │")
    print("  └──────────┴────────────┴───────────────────────────────┘")
    print()
    print("  📌 결론: 휴리스틱은 빠르지만, 반드시 최적해를 찾지는 못합니다.")
    print("     → 어떤 규칙이 좋을지는 문제마다 다릅니다!")
    print("     → 더 좋은 결과를 원한다면? → Step 3 (강화학습)으로!")
