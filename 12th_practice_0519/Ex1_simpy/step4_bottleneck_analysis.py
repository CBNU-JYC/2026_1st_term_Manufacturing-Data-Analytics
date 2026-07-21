# -*- coding: utf-8 -*-
"""
=============================================================
  Step 4: 병목 분석 & What-if 실험
=============================================================
"설비 대수를 몇 대로 운영해야 최적인가?"

설비 수를 1대~5대로 자동 반복 시뮬레이션하고,
각 시나리오별 KPI를 비교하여 병목 곡선을 분석합니다.

실행 방법:
  conda activate mfg_data
  python step4_bottleneck_analysis.py
=============================================================
"""

import simpy
import random
import time
from pathlib import Path

RESULT_DIR = Path(__file__).resolve().parent / "0_result"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# ── matplotlib (있으면 그래프, 없으면 콘솔) ──
try:
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ══════════════════════════════════════════════════════════════
#  시뮬레이션 실행 엔진
# ══════════════════════════════════════════════════════════════

def run_simulation(num_machines, process_time=3.0, arrival_interval=2.0,
                   sim_time=200, seed=42):
    """
    주어진 파라미터로 제조 시뮬레이션을 실행하고 KPI를 반환합니다.

    Args:
        num_machines: 병렬 설비 수
        process_time: 평균 가공 시간 (초)
        arrival_interval: 평균 부품 도착 간격 (초)
        sim_time: 총 시뮬레이션 시간 (초)
        seed: 랜덤 시드 (재현성)

    Returns:
        dict: KPI 결과 (처리량, 평균대기, 가동률, WIP 등)
    """
    rng = random.Random(seed)

    # 데이터 수집용
    completed = 0
    total_wait = 0
    total_busy = 0
    wip_samples = []

    buffer = []

    def part_process(env, name, machines):
        nonlocal completed, total_wait, total_busy
        arrival = env.now
        buffer.append(name)

        with machines.request() as req:
            yield req
            start = env.now
            buffer.remove(name)
            wait = start - arrival
            total_wait += wait

            proc = rng.expovariate(1.0 / process_time)
            yield env.timeout(proc)
            total_busy += proc
            completed += 1

    def arrivals(env, machines):
        i = 0
        while True:
            yield env.timeout(rng.expovariate(1.0 / arrival_interval))
            i += 1
            env.process(part_process(env, f"P{i}", machines))

    def wip_monitor(env):
        """주기적으로 WIP(재공) 수를 기록"""
        while True:
            wip_samples.append(len(buffer))
            yield env.timeout(1.0)

    # 실행
    env = simpy.Environment()
    machines = simpy.Resource(env, capacity=num_machines)
    env.process(arrivals(env, machines))
    env.process(wip_monitor(env))
    env.run(until=sim_time)

    # KPI 계산
    throughput = completed / sim_time
    avg_wait = total_wait / completed if completed > 0 else 0
    utilization = (total_busy / (num_machines * sim_time)) * 100
    avg_wip = sum(wip_samples) / len(wip_samples) if wip_samples else 0
    max_wip = max(wip_samples) if wip_samples else 0

    return {
        'num_machines': num_machines,
        'completed': completed,
        'throughput': throughput,
        'avg_wait': avg_wait,
        'utilization': utilization,
        'avg_wip': avg_wip,
        'max_wip': max_wip,
    }


# ══════════════════════════════════════════════════════════════
#  병목 분석 실행
# ══════════════════════════════════════════════════════════════

def run_bottleneck_analysis(process_time=3.0, arrival_interval=2.0,
                             sim_time=500, max_machines=5):
    """
    설비 수를 1대~max_machines대로 변경하며 시뮬레이션을 반복합니다.
    """
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║     🏭 병목 분석 — 설비 대수 vs 성능 (What-if 실험)     ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()
    print(f"  시뮬레이션 조건:")
    print(f"  • 평균 부품 도착 간격: {arrival_interval}초")
    print(f"  • 평균 가공 시간: {process_time}초")
    print(f"  • 시뮬레이션 시간: {sim_time}초")
    print(f"  • 설비 수 범위: 1대 ~ {max_machines}대")
    print()

    results = []

    for n in range(1, max_machines + 1):
        print(f"  [{n}/{max_machines}] 설비 {n}대 시뮬레이션 실행 중...", end="", flush=True)
        t0 = time.time()
        result = run_simulation(n, process_time, arrival_interval, sim_time)
        elapsed = time.time() - t0
        results.append(result)
        print(f" 완료 ({elapsed:.2f}초)")

    return results


def print_results_table(results):
    """결과 비교 테이블 출력"""
    print()
    print("=" * 78)
    print("  📊 설비 대수별 KPI 비교 요약표")
    print("=" * 78)
    print()
    print(f"  {'설비':>4} │ {'생산량':>6} │ {'처리량':>10} │ {'평균대기':>8} │ {'가동률':>8} │ {'평균WIP':>7} │ {'최대WIP':>7}")
    print("  " + "─" * 72)

    for r in results:
        # 병목 표시
        marker = ""
        if r['utilization'] > 90:
            marker = " ← 병목!"
        elif r['utilization'] < 40:
            marker = " ← 과잉"

        print(f"  {r['num_machines']:>3}대 │ {r['completed']:>5}개 │"
              f" {r['throughput']:>8.3f}/s │ {r['avg_wait']:>6.2f}초 │"
              f" {r['utilization']:>6.1f}% │ {r['avg_wip']:>5.1f}개 │"
              f" {r['max_wip']:>5}개{marker}")

    print("  " + "─" * 72)
    print()


def print_ascii_charts(results):
    """콘솔 ASCII 차트 출력"""

    # ── 처리량 차트 ──
    print("  ┌────────────────────────────────────────────────┐")
    print("  │        📈 처리량 (Throughput)                   │")
    print("  └────────────────────────────────────────────────┘")
    max_tp = max(r['throughput'] for r in results)
    for r in results:
        bar_len = int(r['throughput'] / max_tp * 35)
        bar = "█" * bar_len + "░" * (35 - bar_len)
        print(f"  {r['num_machines']}대 {bar} {r['throughput']:.3f}/s")
    print()

    # ── 평균 대기 시간 차트 ──
    print("  ┌────────────────────────────────────────────────┐")
    print("  │        ⏳ 평균 대기 시간                        │")
    print("  └────────────────────────────────────────────────┘")
    max_wait = max(r['avg_wait'] for r in results)
    if max_wait == 0:
        max_wait = 1
    for r in results:
        bar_len = int(r['avg_wait'] / max_wait * 35)
        bar = "█" * bar_len + "░" * (35 - bar_len)
        print(f"  {r['num_machines']}대 {bar} {r['avg_wait']:.2f}초")
    print()

    # ── 가동률 차트 ──
    print("  ┌────────────────────────────────────────────────┐")
    print("  │        ⚙️  설비 가동률 (Utilization)            │")
    print("  └────────────────────────────────────────────────┘")
    for r in results:
        bar_len = int(r['utilization'] / 100 * 35)
        bar = "█" * bar_len + "░" * (35 - bar_len)
        zone = "🔴" if r['utilization'] > 90 else ("🟡" if r['utilization'] > 70 else "🟢")
        print(f"  {r['num_machines']}대 {bar} {r['utilization']:.1f}% {zone}")
    print()


def print_recommendation(results):
    """최적 설비 수 추천"""
    print("=" * 60)
    print("  💡 분석 결론 & 의사결정 가이드")
    print("=" * 60)
    print()

    # 처리량 포화 지점 찾기 (증가율이 5% 미만이 되는 지점)
    prev_tp = results[0]['throughput']
    saturated_n = results[-1]['num_machines']
    for i, r in enumerate(results[1:], 1):
        improvement = (r['throughput'] - prev_tp) / prev_tp * 100 if prev_tp > 0 else 0
        if improvement < 5:
            saturated_n = results[i - 1]['num_machines']
            break
        prev_tp = r['throughput']

    # 가동률 70~85% 범위 찾기
    optimal_n = None
    for r in results:
        if 60 <= r['utilization'] <= 85:
            optimal_n = r['num_machines']
            break

    print(f"  📌 처리량 포화 지점: 설비 {saturated_n}대")
    print(f"     → {saturated_n}대 이상 추가해도 생산량이 거의 증가하지 않음")
    print()

    if optimal_n:
        print(f"  📌 적정 가동률(60~85%) 달성: 설비 {optimal_n}대")
        print(f"     → 너무 높으면(>90%) 대기 과다, 너무 낮으면(<40%) 유휴 과다")
    print()

    print("  🏭 제조 최적화의 핵심:")
    print("     '설비를 무한정 늘리는 것이 답이 아니다.'")
    print("     → 투자 비용 대비 효과(ROI)를 고려한 적정 설비 수를 찾아야 한다!")
    print()


def plot_matplotlib_charts(results):
    """matplotlib 그래프 생성"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('병목 분석 — 설비 대수 vs 성능', fontsize=14, fontweight='bold')

    ns = [r['num_machines'] for r in results]
    colors = ['#EF5350', '#FF9800', '#4CAF50', '#42A5F5', '#9C27B0'][:len(ns)]

    # 1. 처리량
    ax = axes[0][0]
    ax.bar(ns, [r['throughput'] for r in results], color=colors)
    ax.set_title('처리량 (개/초)')
    ax.set_xlabel('설비 수')
    ax.set_ylabel('처리량')
    ax.set_xticks(ns)
    ax.grid(axis='y', alpha=0.3)

    # 2. 평균 대기시간
    ax = axes[0][1]
    ax.bar(ns, [r['avg_wait'] for r in results], color=colors)
    ax.set_title('평균 대기 시간 (초)')
    ax.set_xlabel('설비 수')
    ax.set_ylabel('초')
    ax.set_xticks(ns)
    ax.grid(axis='y', alpha=0.3)

    # 3. 가동률
    ax = axes[1][0]
    bars = ax.bar(ns, [r['utilization'] for r in results], color=colors)
    ax.axhline(y=85, color='red', linestyle='--', alpha=0.7, label='위험 영역 (85%)')
    ax.axhline(y=60, color='green', linestyle='--', alpha=0.7, label='적정 하한 (60%)')
    ax.set_title('설비 가동률 (%)')
    ax.set_xlabel('설비 수')
    ax.set_ylabel('%')
    ax.set_xticks(ns)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    # 4. 평균 WIP
    ax = axes[1][1]
    ax.bar(ns, [r['avg_wip'] for r in results], color=colors, label='평균')
    ax.plot(ns, [r['max_wip'] for r in results], 'ro-', linewidth=2, label='최대')
    ax.set_title('재공(WIP) 수')
    ax.set_xlabel('설비 수')
    ax.set_ylabel('개')
    ax.set_xticks(ns)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(RESULT_DIR / 'bottleneck_analysis.png', dpi=150, bbox_inches='tight')
    print(f"  📁 그래프 저장됨: bottleneck_analysis.png")
    plt.show()


# ══════════════════════════════════════════════════════════════
#  What-if 시나리오 실험
# ══════════════════════════════════════════════════════════════

def whatif_experiment():
    """도착 간격을 변경했을 때의 영향 분석"""
    print()
    print("=" * 60)
    print("  🔬 What-if 시나리오: 수요 변동의 영향")
    print("=" * 60)
    print()
    print("  '투입 간격을 줄이면(수요 증가) 어떻게 될까?'")
    print("  설비 2대 고정, 투입 간격을 변경하며 비교:")
    print()

    intervals = [3.0, 2.5, 2.0, 1.5, 1.0]

    print(f"  {'투입간격':>8} │ {'생산량':>6} │ {'평균대기':>8} │ {'가동률':>8} │ {'평균WIP':>7}")
    print("  " + "─" * 55)

    for interval in intervals:
        r = run_simulation(2, process_time=3.0, arrival_interval=interval, sim_time=300)
        marker = ""
        if r['utilization'] > 90:
            marker = " ⚠️ 과부하!"
        print(f"  {interval:>6.1f}초 │ {r['completed']:>5}개 │"
              f" {r['avg_wait']:>6.2f}초 │ {r['utilization']:>6.1f}% │"
              f" {r['avg_wip']:>5.1f}개{marker}")

    print()
    print("  📌 투입 간격이 줄어들수록(수요 증가) 대기시간과 WIP가 급증!")
    print("     → 수요 예측에 따라 사전에 설비 증설을 계획해야 함")
    print()


# ══════════════════════════════════════════════════════════════
#  메인 실행
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    # 1. 병목 분석
    results = run_bottleneck_analysis(
        process_time=3.0,
        arrival_interval=2.0,
        sim_time=500,
        max_machines=5
    )

    # 2. 결과 출력
    print_results_table(results)
    print_ascii_charts(results)
    print_recommendation(results)

    # 3. What-if 시나리오
    whatif_experiment()

    # 4. 그래프 생성 (matplotlib 있는 경우)
    if HAS_MPL:
        print("  📈 matplotlib 그래프를 생성합니다...")
        plot_matplotlib_charts(results)
    else:
        print("  [참고] pip install matplotlib 설치 시 그래프도 출력됩니다.")
