# -*- coding: utf-8 -*-
"""
=============================================================
  통합 비교: ILP vs 휴리스틱 vs DRL
=============================================================
모든 스케줄링 기법을 순차적으로 실행하고,
Makespan과 실행 시간을 한눈에 비교합니다.

실행: conda activate mfg_data && python main_compare.py
=============================================================
"""

import time
from utils import (
    MACHINES, JOB_DATA, print_problem,
    draw_gantt, print_comparison_table, draw_comparison_charts,
    draw_utilization_chart, draw_learning_curve
)


def main():
    print_problem()

    print("╔══════════════════════════════════════════════════════════╗")
    print("║  🏭 JSSP 스케줄링 기법 통합 비교                        ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    all_results = []

    # ══════════════════════════════════════════════════════════
    #  1. ILP 절대 최적해
    # ══════════════════════════════════════════════════════════
    print("  ── [1/4] ILP 절대 최적해 ──")
    from step1_ilp_optimal import solve_jssp_ilp
    ilp_df, ilp_ms, ilp_time = solve_jssp_ilp(JOB_DATA, MACHINES, verbose=False)
    print(f"  ✅ ILP: Makespan = {ilp_ms:.0f}분 ({ilp_time:.4f}초)")
    all_results.append({
        'name': 'ILP (절대 최적해)', 'makespan': ilp_ms,
        'time': ilp_time, 'df': ilp_df
    })

    # ══════════════════════════════════════════════════════════
    #  2. 휴리스틱 (3가지 규칙)
    # ══════════════════════════════════════════════════════════
    print()
    print("  ── [2/4] 휴리스틱 스케줄링 ──")
    from step2_heuristics import run_dispatching

    for rule in ['FIFO', 'SPT', 'LPT']:
        h_df, h_ms, h_time = run_dispatching(
            JOB_DATA, MACHINES, rule, verbose=True)
        all_results.append({
            'name': f'휴리스틱 ({rule})', 'makespan': h_ms,
            'time': h_time, 'df': h_df
        })

    # ══════════════════════════════════════════════════════════
    #  3. 기초 DRL (Random Agent)
    # ══════════════════════════════════════════════════════════
    print()
    print("  ── [3/4] 기초 DRL (Random Agent) ──")
    from step3_drl_basic import run_random_agent

    rand_df, rand_ms, rand_time = run_random_agent(
        JOB_DATA, MACHINES, n_episodes=100, verbose=True)
    all_results.append({
        'name': 'DRL (Random Agent)', 'makespan': rand_ms,
        'time': rand_time, 'df': rand_df
    })

    # ══════════════════════════════════════════════════════════
    #  4. 고도화 DRL (Maskable PPO)
    # ══════════════════════════════════════════════════════════
    print()
    print("  ── [4/4] 고도화 DRL (Maskable PPO) ──")
    from step4_drl_advanced import run_maskable_ppo

    drl_df, drl_ms, drl_time = run_maskable_ppo(
        JOB_DATA, MACHINES, total_timesteps=100000, verbose=True)
    all_results.append({
        'name': 'DRL (Maskable PPO 100k)', 'makespan': drl_ms,
        'time': drl_time, 'df': drl_df
    })

    # ══════════════════════════════════════════════════════════
    #  5. 고도화 DRL 학습 곡선 실험
    # ══════════════════════════════════════════════════════════
    print()
    print("  ── [5/5] 고도화 DRL 학습 곡선 실험 ──")
    from step4_drl_advanced import training_curve_experiment
    training_results = training_curve_experiment(JOB_DATA, MACHINES, verbose=True)

    # ══════════════════════════════════════════════════════════
    #  비교 테이블 출력
    # ══════════════════════════════════════════════════════════
    print_comparison_table(all_results)

    # 간트 차트 (최적해 vs 최선 DRL) 및 전체 비교 시각화
    print("  ── 결과 시각화 및 분석 자료 생성 ──")
    draw_comparison_charts(all_results, save_path="10th_comparison_summary.png")
    draw_utilization_chart(all_results, save_path="10th_utilization_analysis.png")
    draw_learning_curve(training_results, save_path="10th_drl_learning_curve.png", optimal_ms=ilp_ms)
    
    draw_gantt(ilp_df, f"ILP Optimal (Makespan: {ilp_ms:.0f})", save_path="10th_gantt_ilp.png")
    if drl_df is not None and drl_ms < 999:
        draw_gantt(drl_df, f"Maskable PPO (Makespan: {drl_ms:.0f})", save_path="10th_gantt_drl.png")

    # ══════════════════════════════════════════════════════════
    #  교육 요약
    # ══════════════════════════════════════════════════════════
    print("=" * 60)
    print("  📚 10주차 핵심 정리")
    print("=" * 60)
    print()
    print("  ┌──────────────┬────────────┬────────────┬──────────────┐")
    print("  │     기법      │ 최적 보장?  │   속도     │   적용 범위   │")
    print("  ├──────────────┼────────────┼────────────┼──────────────┤")
    print("  │ ILP          │  ✅ 보장   │ 작은문제OK  │ 수학적 정식화 │")
    print("  │ 휴리스틱      │  ❌ 미보장  │ ⚡ 즉시    │ 현장 적용 용이│")
    print("  │ DRL          │  ❌ 확률적  │ 🔶 학습필요 │ 복잡한 문제   │")
    print("  └──────────────┴────────────┴────────────┴──────────────┘")
    print()
    print("  💡 실무에서는:")
    print("     • 작은 문제 → ILP (정확한 최적해)")
    print("     • 실시간 의사결정 → 휴리스틱 (SPT/LPT)")
    print("     • 대규모/동적 환경 → DRL (학습 기반 적응)")
    print()


if __name__ == '__main__':
    main()
