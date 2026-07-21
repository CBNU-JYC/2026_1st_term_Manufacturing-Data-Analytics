"""
=============================================================
  비교 시각화 모듈 (Visualizer)
=============================================================
6가지 알고리즘의 결과를 시각화합니다.

matplotlib이 설치된 경우: 4가지 비교 차트 (그래프)
matplotlib이 없는 경우: 콘솔 기반 ASCII 차트 (외부 의존성 없음)
=============================================================
"""

try:
    import os as _os
    import matplotlib
    if 'MPLBACKEND' not in _os.environ:
        matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    import numpy as np
    # 한글 폰트 설정
    matplotlib.rcParams['font.family'] = 'Malgun Gothic'
    matplotlib.rcParams['axes.unicode_minus'] = False
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from result_utils import get_result_dir


# ── 알고리즘별 색상/표시 설정 ──
ALGO_KEYS = {
    'LP': ('LP', '#4FC3F7'),
    'IP': ('IP', '#4DB6AC'),
    'DP': ('DP', '#FFB74D'),
    'Greedy': ('Greedy', '#E57373'),
    'GA': ('GA', '#BA68C8'),
    'AI': ('AI/Optuna', '#7986CB'),
}


def _get_short_name(name):
    for key in ALGO_KEYS:
        if key in name:
            return ALGO_KEYS[key][0]
    return name[:8]


def _get_color(name):
    for key in ALGO_KEYS:
        if key in name:
            return ALGO_KEYS[key][1]
    return '#90A4AE'


def _calculate_dynamic_scores(results):
    """결과 데이터를 바탕으로 동적으로 알고리즘 특성 점수(1~5)를 계산합니다."""
    scores = {}
    if not results:
        return scores
        
    max_profit = max(r['profit'] for r in results) if results else 1
    
    # 실행 결과로 알 수 없는 내재적 특성은 기본값 사용
    base_chars = {
        'LP':       {'범용성': 2, '구현용이': 4},
        'IP':       {'범용성': 2, '구현용이': 4},
        'DP':       {'범용성': 3, '구현용이': 2},
        'Greedy':   {'범용성': 2, '구현용이': 5},
        'GA':       {'범용성': 5, '구현용이': 3},
        'AI/Optuna':{'범용성': 5, '구현용이': 4},
    }
    
    for r in results:
        short = _get_short_name(r['name'])
        
        # 1. 최적성: 최대 이익 대비 비율
        ratio = r['profit'] / max_profit if max_profit > 0 else 0
        if ratio >= 0.999:
            opt_score = 5
        elif ratio >= 0.95:
            opt_score = 4
        elif ratio >= 0.90:
            opt_score = 3
        elif ratio >= 0.80:
            opt_score = 2
        else:
            opt_score = 1
            
        # 2. 속도: 실행 시간 (초) 기준 절대 평가
        t = r['time']
        if t < 0.001:
            spd_score = 5
        elif t < 0.01:
            spd_score = 4
        elif t < 0.1:
            spd_score = 3
        elif t < 1.0:
            spd_score = 2
        else:
            spd_score = 1
            
        # 3. 현실성: 정수해 여부
        real_score = 5 if r.get('is_integer', False) else 2
        
        # 4 & 5. 범용성 및 구현용이성
        gen_score = base_chars.get(short, {}).get('범용성', 3)
        imp_score = base_chars.get(short, {}).get('구현용이', 3)
        
        scores[short] = {
            '최적성': opt_score,
            '속도': spd_score,
            '현실성': real_score,
            '범용성': gen_score,
            '구현용이': imp_score
        }
        
    return scores


# ══════════════════════════════════════════════════════════════
#  콘솔 기반 ASCII 시각화 (matplotlib 불필요)
# ══════════════════════════════════════════════════════════════

def _console_bar(label, value, max_value, width=35, fill='█', empty='░', fmt='.1f'):
    """콘솔용 수평 막대 하나 생성"""
    ratio = value / max_value if max_value > 0 else 0
    filled = int(ratio * width)
    bar = fill * filled + empty * (width - filled)
    return f"  {label:<14} {bar} {value:{fmt}}"


def create_console_charts(results, env):
    """콘솔에서 ASCII 차트로 결과를 시각화합니다."""
    names = [_get_short_name(r['name']) for r in results]
    profits = [r['profit'] for r in results]
    times = [r['time'] for r in results]

    max_profit = max(profits) * 1.05

    # ── 차트 1: 이익 비교 ──
    print()
    print("  ┌────────────────────────────────────────────────────┐")
    print("  │            📊 이익 비교 (만원)                      │")
    print("  └────────────────────────────────────────────────────┘")
    for name, profit in zip(names, profits):
        print(_console_bar(name, profit, max_profit))
    print()

    # ── 차트 2: 실행 시간 비교 ──
    max_time = max(times) * 1.05 if max(times) > 0 else 1
    print("  ┌────────────────────────────────────────────────────┐")
    print("  │            ⏱️  실행 시간 비교 (초)                   │")
    print("  └────────────────────────────────────────────────────┘")
    for name, t in zip(names, times):
        print(_console_bar(name, t, max_time, fmt='.4f'))
    print()

    # ── 차트 3: 제품별 생산량 비교 ──
    print("  ┌────────────────────────────────────────────────────┐")
    print("  │            📦 제품별 생산량 비교                     │")
    print("  └────────────────────────────────────────────────────┘")

    header = f"  {'알고리즘':<12}"
    for p_name in env.products:
        header += f" │ {p_name:>6}"
    header += " │ 합계"
    print(header)
    print("  " + "─" * 50)

    for r in results:
        short = _get_short_name(r['name'])
        row = f"  {short:<12}"
        total = 0
        for i, p_name in enumerate(env.products):
            q = r['quantities'][i]
            q_str = f"{q:.1f}" if isinstance(q, float) and q != int(q) else f"{int(q)}"
            row += f" │ {q_str:>6}"
            total += q
        total_str = f"{total:.1f}" if isinstance(total, float) and total != int(total) else f"{int(total)}"
        row += f" │ {total_str}"
        print(row)
    print()

    # ── 차트 4: 알고리즘 특성 비교 (스파이더 차트 텍스트 버전) ──
    print("  ┌────────────────────────────────────────────────────┐")
    print("  │         🕸️  알고리즘 특성 비교 (1~5점)              │")
    print("  └────────────────────────────────────────────────────┘")

    characteristics = _calculate_dynamic_scores(results)
    cats = ['최적성', '속도', '현실성', '범용성', '구현용이']
    header = f"  {'알고리즘':<12}"
    for cat in cats:
        header += f" │ {cat:>6}"
    print(header)
    print("  " + "─" * 55)

    for r in results:
        short = _get_short_name(r['name'])
        if short in characteristics:
            row = f"  {short:<12}"
            for cat in cats:
                score = characteristics[short][cat]
                stars = '★' * score + '☆' * (5 - score)
                row += f" │ {stars}"
            print(row)
    print()


# ══════════════════════════════════════════════════════════════
#  Matplotlib 기반 그래프 시각화
# ══════════════════════════════════════════════════════════════

def create_matplotlib_charts(results, env, save_path=None):
    """matplotlib으로 4가지 비교 차트를 생성합니다."""
    if save_path is None:
        save_path = get_result_dir() / '11_7_main_compare_figure_01.png'

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('공장 생산 계획 최적화 — 알고리즘 비교',
                 fontsize=18, fontweight='bold', y=0.98)

    names = [_get_short_name(r['name']) for r in results]
    colors = [_get_color(r['name']) for r in results]

    # ── 차트 1: 이익 비교 ──
    ax1 = axes[0][0]
    profits = [r['profit'] for r in results]
    bars = ax1.bar(names, profits, color=colors, edgecolor='white', linewidth=1.5)
    ip_r = next((r for r in results if 'IP' in r['name']), None)
    if ip_r:
        ax1.axhline(y=ip_r['profit'], color='#2E7D32', linestyle='--',
                     linewidth=1.5, alpha=0.7, label=f"IP 최적 ({ip_r['profit']:.1f})")
        ax1.legend(fontsize=10)
    for bar, p in zip(bars, profits):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f'{p:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    ax1.set_title('이익 비교 (만원)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('이익 (만원)')
    ax1.set_ylim(0, max(profits) * 1.15)
    ax1.grid(axis='y', alpha=0.3)

    # ── 차트 2: 실행 시간 비교 ──
    ax2 = axes[0][1]
    # 0인 값은 로그 스케일에서 표시 불가 → 최소 0.0001초로 보정
    times = [max(r['time'], 0.0001) for r in results]
    times_real = [r['time'] for r in results]
    bars2 = ax2.barh(names, times, color=colors, edgecolor='white', linewidth=1.5)
    ax2.set_xscale('log')
    for bar, t in zip(bars2, times_real):
        x_pos = max(bar.get_width() * 1.3, 0.0002)
        ax2.text(x_pos, bar.get_y() + bar.get_height()/2,
                 f'{t:.4f}s', ha='left', va='center', fontsize=10)
    ax2.set_title('실행 시간 비교 (로그 스케일)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('시간 (초)')
    ax2.grid(axis='x', alpha=0.3)
    ax2.invert_yaxis()

    # ── 차트 3: 레이더 차트 ──
    ax3 = axes[1][0]
    ax3.remove()
    ax3 = fig.add_subplot(2, 2, 3, projection='polar')

    categories = ['최적성', '속도', '현실성\n(정수해)', '범용성', '구현\n용이성']
    n_cats = len(categories)
    angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist()
    angles += angles[:1]

    dynamic_scores = _calculate_dynamic_scores(results)
    
    for r in results:
        short = _get_short_name(r['name'])
        if short in dynamic_scores:
            s = dynamic_scores[short]
            vals = [s['최적성'], s['속도'], s['현실성'], s['범용성'], s['구현용이']]
            vals += vals[:1]  # 레이더 차트 닫기
            col = _get_color(r['name'])
            ax3.plot(angles, vals, 'o-', linewidth=2, color=col, label=short)
            ax3.fill(angles, vals, alpha=0.1, color=col)
    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(categories, fontsize=10)
    ax3.set_ylim(0, 5.5)
    ax3.set_yticks([1, 2, 3, 4, 5])
    ax3.set_title('알고리즘 특성 비교', fontsize=14, fontweight='bold', pad=20)
    ax3.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)

    # ── 차트 4: 제품별 생산량 비교 ──
    ax4 = axes[1][1]
    x = np.arange(len(names))
    width = 0.25
    p_colors = ['#EF5350', '#42A5F5', '#66BB6A']
    for j, pname in enumerate(env.products):
        quantities = [r['quantities'][j] for r in results]
        ax4.bar(x + (j - 1) * width, quantities, width,
                label=pname, color=p_colors[j], edgecolor='white')
    ax4.set_title('제품별 생산량 비교', fontsize=14, fontweight='bold')
    ax4.set_ylabel('생산량 (개)')
    ax4.set_xticks(x)
    ax4.set_xticklabels(names, fontsize=10)
    ax4.legend(fontsize=10)
    ax4.grid(axis='y', alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  차트 저장됨: {save_path}")
    plt.show()


# ══════════════════════════════════════════════════════════════
#  통합 함수: 환경에 따라 자동 선택
# ══════════════════════════════════════════════════════════════

def create_comparison_charts(results, env, save_path=None):
    """
    결과를 시각화합니다.
    matplotlib이 있으면 그래프, 없으면 콘솔 ASCII 차트를 출력합니다.
    """
    # 항상 콘솔 차트는 출력
    create_console_charts(results, env)

    # matplotlib이 있으면 추가로 그래프 생성
    if HAS_MATPLOTLIB:
        print("  matplotlib 감지됨 → 그래프를 생성합니다...")
        try:
            create_matplotlib_charts(results, env, save_path)
        except Exception as e:
            print(f"  그래프 생성 실패 ({e}), 콘솔 차트만 표시합니다.")
    else:
        print("  [참고] pip install matplotlib 을 설치하면 그래프 차트도 출력됩니다.")


# ── 단독 실행 시 ──
if __name__ == '__main__':
    from environment import ProductionEnv
    env = ProductionEnv()

    dummy_results = [
        {'name': 'LP (선형 계획법)', 'quantities': [5.5, 10.5, 2.0], 'profit': 104.5,
         'time': 0.002, 'feasible': True, 'is_integer': False},
        {'name': 'IP (정수 계획법)', 'quantities': [5, 10, 2], 'profit': 101.0,
         'time': 0.005, 'feasible': True, 'is_integer': True},
        {'name': 'DP (동적 계획법)', 'quantities': [5, 10, 2], 'profit': 101.0,
         'time': 0.320, 'feasible': True, 'is_integer': True},
        {'name': 'Greedy (탐욕법)', 'quantities': [8, 8, 0], 'profit': 96.0,
         'time': 0.001, 'feasible': True, 'is_integer': True},
        {'name': 'GA (유전 알고리즘)', 'quantities': [5, 10, 1], 'profit': 98.0,
         'time': 0.150, 'feasible': True, 'is_integer': True},
        {'name': 'AI/Optuna (500 trials)', 'quantities': [6, 9, 2], 'profit': 99.0,
         'time': 2.100, 'feasible': True, 'is_integer': True},
    ]
    create_comparison_charts(dummy_results, env)
