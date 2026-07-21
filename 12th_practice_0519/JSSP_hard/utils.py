# -*- coding: utf-8 -*-
"""
=============================================================
  공통 유틸리티 — JSSP 문제 정의 & 시각화
=============================================================
모든 Step에서 공유하는:
  1. JSSP 문제 데이터 (Job, Machine, 가공시간)
  2. 간트 차트 시각화 (matplotlib + 콘솔 ASCII 백업)
  3. 결과 비교 테이블 출력
=============================================================
"""

import pandas as pd
import numpy as np
import time

# ══════════════════════════════════════════════════════════════
#  1. JSSP 문제 정의 (자동차 부품 공장)
# ══════════════════════════════════════════════════════════════

# 사용 가능한 설비 목록
MACHINES = ['Milling', 'Lathe', 'Drill', 'CNC', 'Grinding']

# 작업(Job) 데이터: 각 작업이 설비를 거치는 순서와 가공시간 (5x5 시나리오)
JOB_DATA = {
    'Engine_Block': [('Milling', 10), ('Lathe', 8), ('Drill', 4), ('CNC', 5), ('Grinding', 3)],
    'Crankshaft':   [('Lathe', 6), ('CNC', 7), ('Milling', 5), ('Grinding', 4), ('Drill', 8)],
    'Piston':       [('Drill', 3), ('Lathe', 7), ('Grinding', 5), ('CNC', 4), ('Milling', 6)],
    'Camshaft':     [('CNC', 5), ('Grinding', 6), ('Milling', 8), ('Lathe', 4), ('Drill', 5)],
    'Flywheel':     [('Grinding', 4), ('Drill', 5), ('Lathe', 6), ('CNC', 8), ('Milling', 7)],
}


def print_problem():
    """JSSP 문제를 콘솔에 출력합니다."""
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║    🏭 Job Shop Scheduling Problem (JSSP) 정의          ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()
    print("  [문제] 3개의 작업(Job)을 3대의 설비(Machine)에 배정하여")
    print("         전체 완료 시간(Makespan)을 최소화하세요.")
    print()
    print(f"  설비: {', '.join(MACHINES)}")
    print()
    print("  ┌──────────────┬────────────────────────────────────────┐")
    print("  │   작업(Job)   │  공정 순서 (설비, 가공시간)             │")
    print("  ├──────────────┼────────────────────────────────────────┤")
    for job, ops in JOB_DATA.items():
        ops_str = " → ".join([f"{m}({t}분)" for m, t in ops])
        print(f"  │ {job:<12} │ {ops_str:<38} │")
    print("  └──────────────┴────────────────────────────────────────┘")
    print()
    total = sum(t for ops in JOB_DATA.values() for _, t in ops)
    print(f"  총 가공시간 합계: {total}분")
    print(f"  이론적 하한(Lower Bound): {max(sum(t for _, t in ops) for ops in JOB_DATA.values())}분")
    print()


# ══════════════════════════════════════════════════════════════
#  2. 간트 차트 시각화
# ══════════════════════════════════════════════════════════════

# matplotlib 확인
try:
    import os
    import matplotlib
    # 환경변수 MPLBACKEND가 설정되어 있으면 그것을 사용, 아니면 TkAgg
    if 'MPLBACKEND' not in os.environ:
        matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# 작업별 색상 팔레트
JOB_COLORS = {
    'Engine_Block': '#42A5F5',
    'Crankshaft':   '#66BB6A',
    'Piston':       '#FFA726',
    'Camshaft':     '#EF5350',
    'Flywheel':     '#AB47BC',
}

# 콘솔용 간단 문자 패턴
JOB_CHARS = {
    'Engine_Block': '█',
    'Crankshaft':   '▓',
    'Piston':       '░',
    'Camshaft':     '▒',
    'Flywheel':     '■',
}


def draw_gantt_console(result_df, title="Schedule"):
    """콘솔 ASCII 간트 차트 출력"""
    if result_df is None or result_df.empty:
        print("  시각화할 데이터가 없습니다.")
        return

    makespan = int(result_df['End_Time'].max())
    machines = sorted(result_df['Machine'].unique())

    print()
    print(f"  ┌─ {title} (Makespan: {makespan}) ─┐")
    print()

    # 시간 축 스케일: 1칸 = 1시간단위
    scale = 1
    if makespan > 60:
        scale = 2

    for m in machines:
        m_data = result_df[result_df['Machine'] == m].sort_values('Start_Time')
        line = [' '] * (makespan // scale + 1)

        for _, row in m_data.iterrows():
            start = int(row['Start_Time']) // scale
            end = int(row['End_Time']) // scale
            ch = JOB_CHARS.get(row['Job'], '#')
            for i in range(start, min(end, len(line))):
                line[i] = ch

        bar = ''.join(line)
        print(f"  {m:>8} │{bar}│")

    # 시간 눈금
    tick_line = ''.join([str(i % 10) for i in range(makespan // scale + 1)])
    print(f"  {'':>8} └{tick_line}┘")

    # 범례
    legend = "  범례: "
    for job, ch in JOB_CHARS.items():
        legend += f"{ch}{ch}{ch}={job}  "
    print(legend)
    print()


def draw_gantt_matplotlib(result_df, title="Schedule", save_path=None):
    """matplotlib 간트 차트"""
    if result_df is None or result_df.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 4))

    machines = sorted(result_df['Machine'].unique(), reverse=True)
    jobs = sorted(result_df['Job'].unique())
    machine_y = {m: i * 10 for i, m in enumerate(machines)}
    colors = plt.cm.tab10.colors
    job_color = {job: colors[i % len(colors)] for i, job in enumerate(jobs)}

    for _, row in result_df.iterrows():
        job, machine = row['Job'], row['Machine']
        start, duration = row['Start_Time'], row['Processing_Time']
        ax.broken_barh([(start, duration)], (machine_y[machine] + 2, 6),
                        facecolors=job_color[job], edgecolor='black', linewidth=1)
        ax.text(start + duration / 2, machine_y[machine] + 5,
                f"{job}\n({duration})", ha='center', va='center',
                color='white', fontsize=7, fontweight='bold')

    ax.set_yticks([machine_y[m] + 5 for m in machines])
    ax.set_yticklabels(machines, fontsize=11, fontweight='bold')
    ax.set_xlabel('Time', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)

    makespan = int(result_df['End_Time'].max())
    ax.set_xticks(range(0, makespan + 2, max(1, makespan // 15)))
    ax.grid(True, axis='x', linestyle='--', alpha=0.5)

    legend_patches = [mpatches.Patch(color=job_color[j], label=j) for j in jobs]
    ax.legend(handles=legend_patches, title="Jobs",
              bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  📁 간트 차트 저장: {save_path}")
        plt.close(fig)
    else:
        plt.show()


def draw_gantt(result_df, title="Schedule", save_path=None):
    """환경에 맞춰 자동 선택하여 간트 차트 출력"""
    draw_gantt_console(result_df, title)
    if HAS_MPL:
        try:
            draw_gantt_matplotlib(result_df, title, save_path)
        except Exception as e:
            print(f"  [matplotlib 오류] {e}")


# ══════════════════════════════════════════════════════════════
#  3. 결과 비교 테이블
# ══════════════════════════════════════════════════════════════

def print_comparison_table(results):
    """
    여러 알고리즘의 결과를 비교 테이블로 출력합니다.
    results: [{'name': str, 'makespan': float, 'time': float, 'df': DataFrame}, ...]
    """
    if not results:
        return

    best_makespan = min(r['makespan'] for r in results)

    print()
    print("=" * 70)
    print("  📊 스케줄링 기법 비교 요약")
    print("=" * 70)
    print()
    print(f"  {'기법':<28} │ {'Makespan':>10} │ {'Gap':>8} │ {'실행시간':>10}")
    print("  " + "─" * 65)

    for r in results:
        gap = ((r['makespan'] - best_makespan) / best_makespan * 100) if best_makespan > 0 else 0
        gap_str = f"+{gap:.1f}%" if gap > 0 else "최적"
        marker = " ★" if gap == 0 else ""
        print(f"  {r['name']:<28} │ {r['makespan']:>8.0f}분 │ {gap_str:>8} │ {r['time']:>8.4f}초{marker}")

    print("  " + "─" * 65)
    print()


def draw_comparison_charts(results, save_path="comparison_summary.png"):
    """여러 알고리즘의 Makespan과 실행 시간을 막대 그래프로 비교 시각화"""
    if not HAS_MPL or not results:
        return
        
    names = [r['name'] for r in results]
    makespans = [r['makespan'] for r in results]
    times = [r['time'] for r in results]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Makespan 비교
    ax1 = axes[0]
    bars1 = ax1.bar(names, makespans, color='#42A5F5', edgecolor='black', linewidth=1)
    ax1.set_title("Makespan 비교 (낮을수록 좋음)", fontsize=13, fontweight='bold')
    ax1.set_ylabel("Makespan (분)", fontsize=11)
    ax1.tick_params(axis='x', rotation=15)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval + (max(makespans)*0.02), f"{yval:.0f}", ha='center', va='bottom', fontsize=10, fontweight='bold')
        
    # 2. 실행 시간 비교
    ax2 = axes[1]
    bars2 = ax2.bar(names, times, color='#EF5350', edgecolor='black', linewidth=1)
    ax2.set_title("실행 시간 비교 (초)", fontsize=13, fontweight='bold')
    ax2.set_ylabel("Time (초)", fontsize=11)
    ax2.tick_params(axis='x', rotation=15)
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, yval + (max(times)*0.02), f"{yval:.4f}", ha='center', va='bottom', fontsize=10, fontweight='bold')
        
    plt.tight_layout()
    try:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  📁 비교 요약 차트 저장: {save_path}")
        plt.close(fig)
    except Exception as e:
        print(f"  [matplotlib 오류] {e}")


# ══════════════════════════════════════════════════════════════
#  4. 추가 분석용 시각화 (가동률, 학습 곡선)
# ══════════════════════════════════════════════════════════════

def draw_utilization_chart(results, save_path="utilization_analysis.png"):
    """
    각 알고리즘별 설비 가동률(Utilization) 분석 차트 생성
    """
    if not HAS_MPL or not results:
        return
        
    names = []
    # Machine -> [Utilizations for each algorithm]
    machine_utils = {m: [] for m in MACHINES}
    
    for r in results:
        df = r.get('df')
        ms = r.get('makespan')
        names.append(r['name'])
        
        if df is None or df.empty or ms >= 999:
            for m in MACHINES:
                machine_utils[m].append(0)
            continue
            
        for m in MACHINES:
            m_df = df[df['Machine'] == m]
            active_time = m_df['Processing_Time'].sum()
            util_rate = (active_time / ms) * 100 if ms > 0 else 0
            machine_utils[m].append(util_rate)
            
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(names))
    width = 0.8 / len(MACHINES)
    
    for i, m in enumerate(MACHINES):
        offset = (i - len(MACHINES)/2 + 0.5) * width
        ax.bar(x + offset, machine_utils[m], width, label=m, edgecolor='black', alpha=0.8)
        
    ax.set_ylabel('가동률 (Utilization %)', fontsize=11)
    ax.set_title('알고리즘별 설비 가동률 비교 (유휴시간 분석)', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha='right')
    ax.set_ylim(0, 110)
    ax.legend(title='Machines', bbox_to_anchor=(1.01, 1), loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    try:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  📁 설비 가동률 차트 저장: {save_path}")
        plt.close(fig)
    except Exception as e:
        print(f"  [matplotlib 오류] {e}")


def draw_learning_curve(training_results, save_path="drl_learning_curve.png", optimal_ms=None):
    """
    DRL 학습량(Timesteps)에 따른 Makespan 개선 곡선
    """
    if not HAS_MPL or not training_results:
        return
        
    valid_results = [r for r in training_results if r['makespan'] < 999]
    if not valid_results:
        return
        
    ts = [r['timesteps'] for r in valid_results]
    ms = [r['makespan'] for r in valid_results]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ts, ms, marker='o', linestyle='-', linewidth=2.5, color='#8E24AA', markersize=8)
    
    if optimal_ms is not None:
        ax.axhline(y=optimal_ms, color='r', linestyle='--', label=f'ILP 최적해 ({optimal_ms:.0f}분)')
    
    for i, txt in enumerate(ms):
        ax.annotate(f"{txt:.0f}분", (ts[i], ms[i]), textcoords="offset points", xytext=(0,10), ha='center', fontweight='bold')
        
    ax.set_title("DRL 에이전트 학습 곡선 (Learning Curve)", fontsize=13, fontweight='bold')
    ax.set_xlabel("학습량 (Timesteps)", fontsize=11)
    ax.set_ylabel("Makespan (분)", fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()
    
    plt.tight_layout()
    try:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  📁 DRL 학습 곡선 저장: {save_path}")
        plt.close(fig)
    except Exception as e:
        print(f"  [matplotlib 오류] {e}")


# ══════════════════════════════════════════════════════════════
#  단독 실행 시 문제 출력
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print_problem()
