"""
=============================================================
  통합 비교 실행기 (Main Compare)
=============================================================
6가지 최적화 알고리즘으로 동일한 공장 생산 계획 문제를 풀고,
결과를 비교 요약표로 출력합니다.

실행 방법:
  python main_compare.py
=============================================================
"""

import time
from environment import ProductionEnv
from result_utils import get_result_dir, start_terminal_log


def run_all_solvers(env):
    """모든 솔버를 순차적으로 실행하고 결과를 수집합니다."""
    results = []

    solvers_info = [
        ('1_lp_solver', 'LPSolver', 'LP (선형 계획법)', {}),
        ('2_ip_solver', 'IPSolver', 'IP (정수 계획법)', {}),
        ('3_dp_solver', 'DPSolver', 'DP (동적 계획법)', {}),
        ('4_greedy_solver', 'GreedySolver', 'Greedy (탐욕법)', {}),
        ('5_ga_solver', 'GASolver', 'GA (유전 알고리즘)', {}),
        ('6_ai_solver', 'AISolver', 'AI/Optuna (베이지안)', {}),
    ]

    print()
    print("=" * 62)
    print("  🏭 공장 생산 계획 최적화 — 6가지 알고리즘 비교")
    print("=" * 62)
    print()

    for idx, (module_name, class_name, display_name, kwargs) in enumerate(solvers_info, 1):
        print(f"  [{idx}/6] {display_name} 실행 중...", end="", flush=True)

        try:
            # 동적으로 모듈과 클래스를 로드
            module = __import__(module_name)
            solver_class = getattr(module, class_name)
            solver = solver_class(env, **kwargs)
            result = solver.solve()

            if result and result['feasible']:
                emoji = "✅"
            elif result:
                emoji = "⚠️"
            else:
                emoji = "❌"

            print(f" {emoji} 이익: {result['profit']:.1f}만원 | "
                  f"시간: {result['time']:.4f}초")

            # 생산량 출력
            for i, name in enumerate(env.products):
                q = result['quantities'][i]
                q_str = f"{q:.1f}" if isinstance(q, float) and q != int(q) else f"{int(q)}"
                print(f"        {name}: {q_str}개", end="")
            print()

            results.append(result)

        except Exception as e:
            print(f" ❌ 오류: {e}")
            results.append(None)

    return results


def print_comparison_table(results, env):
    """결과 비교 요약표를 출력합니다."""
    print()
    print("=" * 80)
    print("  📊 알고리즘 비교 요약표")
    print("=" * 80)
    print()

    # 헤더
    header = f"  {'알고리즘':<22} │ {'이익(만원)':>10} │ {'최적해?':>7} │ {'정수해?':>7} │ {'시간(초)':>10}"
    print(header)
    print("  " + "─" * 74)

    # IP 결과를 기준으로 최적해 판정 (IP가 정수 최적해)
    ip_result = results[1] if len(results) > 1 and results[1] else None
    ip_profit = ip_result['profit'] if ip_result else None

    for r in results:
        if r is None:
            continue

        name = r['name']
        profit = r['profit']
        is_int = '✅ 정수' if r['is_integer'] else '❌ 실수'

        # 최적성 판정
        if ip_profit is not None:
            if abs(profit - ip_profit) < 0.01:
                optimal = '✅ 최적'
            elif profit > ip_profit:
                optimal = '🔵 상한'  # LP relaxation
            else:
                optimal = '❌ 미달'
        else:
            optimal = '?'

        time_str = f"{r['time']:.4f}"

        print(f"  {name:<22} │ {profit:>10.1f} │ {optimal:>7} │ {is_int:>7} │ {time_str:>10}")

    print("  " + "─" * 74)

    # 자원 사용 비교 (IP 기준)
    if ip_result:
        print()
        print("  [IP 최적해의 자원 사용률]")
        for key, usage in ip_result['resource_usage'].items():
            pct = usage['used'] / usage['limit'] * 100
            bar_len = int(pct / 5)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            print(f"  {usage['name']:<12}: {bar} {pct:.0f}% ({usage['used']}/{usage['limit']}{usage['unit']})")

    # 알고리즘 특성 비교
    print()
    print("=" * 80)
    print("  📚 알고리즘 특성 비교 (교육 정리)")
    print("=" * 80)
    print()
    print("  ┌───────────────────────┬──────────┬──────────┬──────────┐")
    print("  │ 알고리즘              │ 최적 보장 │ 정수 처리 │ 범용성   │")
    print("  ├───────────────────────┼──────────┼──────────┼──────────┤")
    print("  │ LP (선형계획)         │  ✅ 보장  │ ❌ 실수  │ 선형만   │")
    print("  │ IP (정수계획)         │  ✅ 보장  │ ✅ 정수  │ 선형만   │")
    print("  │ DP (동적계획)         │  ✅ 보장  │ ✅ 정수  │ 구조필요 │")
    print("  │ Greedy (탐욕)         │  ❌ 미보장│ ✅ 정수  │ 제한적   │")
    print("  │ GA (유전)             │  ❌ 확률적│ ✅ 정수  │ 매우넓음 │")
    print("  │ AI/Optuna (베이지안)  │ ❌ 확률적│ ⚙️ 설정  │ 매우넓음 │")
    print("  └───────────────────────┴──────────┴──────────┴──────────┘")
    print()
    print("  ⚠️  DP/Greedy가 LP/IP보다 빨라 보이는 이유:")
    print("     ① 구현 방식: LP/IP는 PuLP→CBC 외부 프로세스를 호출(~20ms 오버헤드)")
    print("        DP/Greedy는 순수 Python 직접 구현이라 이런 오버헤드가 없음")
    print("     ② 문제 크기: 이 문제(3제품)의 DP 탐색공간은 겨우 136개")
    print("        제품 10종 이상이면 DP는 지수적으로 느려져 LP/IP에 역전!")
    print()
    print("  💡 결론: 문제의 구조를 알면 LP/IP가 최선!")
    print("     구조를 모르거나 비선형이면 GA/AI가 유용!")
    print()

    return results


def main():
    """메인 실행 함수"""
    log_path = start_terminal_log("11_7_main_compare_console.txt")

    env = ProductionEnv()
    env.print_problem()

    results = run_all_solvers(env)
    valid_results = [r for r in results if r is not None]

    print_comparison_table(valid_results, env)

    print(f"  💾 터미널 비교 결과 저장: {log_path}")

    # 시각화 생성
    try:
        from visualizer import create_comparison_charts
        print("  📈 비교 시각화 차트를 생성합니다...")
        create_comparison_charts(valid_results, env, get_result_dir() / "11_7_main_compare_figure_01.png")
        print("  ✅ 차트가 저장/표시되었습니다.")
    except ImportError:
        print("  ⚠️ visualizer.py를 찾을 수 없어 시각화를 건너뜁니다.")
    except Exception as e:
        print(f"  ⚠️ 시각화 오류: {e}")


if __name__ == '__main__':
    main()
