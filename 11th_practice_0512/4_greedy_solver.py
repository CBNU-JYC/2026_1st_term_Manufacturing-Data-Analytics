"""
=============================================================
  Solver 4: Greedy (탐욕 알고리즘)
=============================================================
특징:
  - 매 순간 가장 좋아 보이는 선택을 함 (지역 최적)
  - 매우 빠름 (O(n log n))
  - 최적해를 보장하지 않음!
  - 직접 구현

교육 포인트:
  "탐욕법은 직관적이고 빠르지만, '지금 당장 좋은 선택'이
   '전체적으로 최선의 선택'과 다를 수 있다!"
=============================================================
"""

import time
from environment import ProductionEnv
from result_utils import start_terminal_log


class GreedySolver:
    """탐욕 알고리즘 기반 솔버"""

    def __init__(self, env: ProductionEnv):
        self.env = env

    def solve(self):
        """
        탐욕법으로 풀이합니다.

        전략: '자원 효율이 가장 높은 제품'부터 최대한 생산
          - 효율 = 이익 / (모든 자원 소모량의 가중 합)
          - 가장 효율 높은 제품을 먼저 최대한 생산
          - 남은 자원으로 다음 효율 제품 생산
          - 반복
        """
        start_time = time.perf_counter()

        # ─── 1단계: 제품별 효율 계산 ───
        efficiency = []
        for i in range(self.env.n_products):
            total_resource = sum(
                res['usage'][i] / res['limit']  # 정규화된 자원 소모 비율
                for res in self.env.resources.values()
            )
            eff = self.env.profit[i] / total_resource if total_resource > 0 else 0
            efficiency.append((eff, i))

        # ─── 2단계: 효율 순으로 정렬 (높은 것부터) ───
        efficiency.sort(reverse=True)

        # ─── 3단계: 탐욕적으로 생산량 결정 ───
        quantities = [0] * self.env.n_products
        remaining = {key: res['limit'] for key, res in self.env.resources.items()}

        print_steps = []  # 과정 기록용

        for eff_val, i in efficiency:
            product_name = self.env.products[i]

            # 이 제품을 최대 몇 개 만들 수 있는지 계산
            max_possible = float('inf')
            for key, res in self.env.resources.items():
                if res['usage'][i] > 0:
                    max_possible = min(max_possible, remaining[key] // res['usage'][i])

            max_possible = int(max_possible)
            quantities[i] = max_possible

            # 자원 차감
            for key, res in self.env.resources.items():
                remaining[key] -= res['usage'][i] * max_possible

            print_steps.append({
                'product': product_name,
                'efficiency': eff_val,
                'quantity': max_possible,
                'remaining': dict(remaining)
            })

        elapsed = time.perf_counter() - start_time

        result = self.env.evaluate(quantities)
        return {
            'name': 'Greedy (탐욕법)',
            'quantities': quantities,
            'profit': result['profit'],
            'time': elapsed,
            'feasible': result['feasible'],
            'is_integer': True,
            'resource_usage': result['resource_usage'],
            'details': {
                'steps': print_steps,
                'efficiency_order': [(self.env.products[i], f"{e:.2f}") for e, i in efficiency]
            }
        }


# ── 단독 실행 시 ──
if __name__ == '__main__':
    log_path = start_terminal_log("11_4_greedy_solver_console.txt")

    env = ProductionEnv()
    env.print_problem()

    print("\n🏃 Greedy (탐욕법) 풀이 시작...\n")
    solver = GreedySolver(env)
    result = solver.solve()

    # 과정 출력
    print("  [탐욕 선택 과정]")
    print(f"  효율 순위: ", end="")
    for name, eff in result['details']['efficiency_order']:
        print(f"{name}({eff}) > ", end="")
    print("순서대로 선택\n")

    for i, step in enumerate(result['details']['steps'], 1):
        print(f"  {i}. {step['product']} → {step['quantity']}개 생산 (효율: {step['efficiency']:.2f})")

    print(f"\n  ✅ 총 이익: {result['profit']}만원")
    for i, name in enumerate(env.products):
        print(f"  {name}: {result['quantities'][i]}개")
    print(f"  풀이 시간: {result['time']:.6f}초")

    print("\n  ⚠️ 주의: 탐욕법은 최적해를 보장하지 않습니다!")
    print("  → IP/DP 결과와 비교해 보세요.")

    print(f"\n  💾 터미널 결과 저장: {log_path}")
