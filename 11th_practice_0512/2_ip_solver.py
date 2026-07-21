"""
=============================================================
  Solver 2: IP (Integer Programming, 정수 계획법)
=============================================================
특징:
  - LP와 동일한 구조이지만, 변수가 반드시 정수 → "의자 13개" 같은 현실적 답
  - 수학적으로 정수 최적해를 보장
  - LP보다 느릴 수 있음 (NP-hard)
  - 라이브러리: PuLP (CBC 솔버의 Branch & Bound)

교육 포인트:
  "LP의 최적값을 단순히 반올림하면 제약 조건을 위반할 수 있다!
   IP는 정수 제약을 직접 처리하여 현실적인 최적해를 구한다."
=============================================================
"""

import time
from environment import ProductionEnv
from result_utils import start_terminal_log


class IPSolver:
    """PuLP를 사용한 정수 계획법 솔버"""

    def __init__(self, env: ProductionEnv):
        self.env = env

    def solve(self):
        """
        IP 문제를 정의하고 풀이합니다.
        LP와 거의 동일하지만, 변수 타입이 'Integer'입니다.
        """
        import pulp

        # ─── 1단계: 문제 정의 ───
        prob = pulp.LpProblem("Production_Planning_IP", pulp.LpMaximize)

        # ─── 2단계: 결정 변수 선언 (★ 정수 변수!) ───
        x = []
        for i, name in enumerate(self.env.products):
            var = pulp.LpVariable(f'x_{name}', lowBound=0, cat='Integer')  # ← Integer!
            x.append(var)

        # ─── 3단계: 목적 함수 (LP와 동일) ───
        prob += pulp.lpSum(x[i] * self.env.profit[i]
                           for i in range(self.env.n_products)), "Total_Profit"

        # ─── 4단계: 제약 조건 (LP와 동일) ───
        for key, res in self.env.resources.items():
            prob += (pulp.lpSum(x[i] * res['usage'][i]
                                for i in range(self.env.n_products))
                     <= res['limit']), f"Constraint_{key}"

        # ─── 5단계: 풀이 실행 ───
        start_time = time.perf_counter()
        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        elapsed = time.perf_counter() - start_time

        # ─── 6단계: 결과 반환 ───
        if pulp.LpStatus[prob.status] == 'Optimal':
            quantities = [int(var.varValue) for var in x]
            result = self.env.evaluate(quantities)
            return {
                'name': 'IP (정수 계획법)',
                'quantities': quantities,
                'profit': pulp.value(prob.objective),
                'time': elapsed,
                'feasible': result['feasible'],
                'is_integer': True,
                'resource_usage': result['resource_usage']
            }
        else:
            return None


# ── 단독 실행: LP와 IP 결과 직접 비교 ──
if __name__ == '__main__':
    log_path = start_terminal_log("11_2_ip_solver_lp_vs_ip_console.txt")

    env = ProductionEnv()
    env.print_problem()

    # LP 풀이
    from importlib import import_module
    lp_mod = import_module('1_lp_solver')
    lp_result = lp_mod.LPSolver(env).solve()

    # IP 풀이
    ip_result = IPSolver(env).solve()

    print("\n" + "=" * 55)
    print("  📊 LP vs IP 결과 비교")
    print("=" * 55)

    headers = ['항목', 'LP (연속)', 'IP (정수)']
    rows = []
    for i, name in enumerate(env.products):
        lp_q = f"{lp_result['quantities'][i]:.2f}개"
        ip_q = f"{ip_result['quantities'][i]}개"
        rows.append([name, lp_q, ip_q])
    rows.append(['─' * 8, '─' * 12, '─' * 12])
    rows.append(['총 이익', f"{lp_result['profit']:.2f}만원", f"{ip_result['profit']:.0f}만원"])
    rows.append(['정수 해?', '❌' if not lp_result['is_integer'] else '✅', '✅'])
    rows.append(['풀이 시간', f"{lp_result['time']:.4f}초", f"{ip_result['time']:.4f}초"])

    # 간단한 표 출력 (tabulate 없이도 동작)
    for row in [headers] + rows:
        print(f"  {row[0]:<10} {row[1]:<14} {row[2]:<14}")

    print()
    print("  📌 LP 결과를 반올림하면 제약 조건을 위반할 수 있습니다!")
    print("     IP는 정수 제약을 직접 고려하여 현실적인 최적해를 보장합니다.")

    print(f"\n  💾 터미널 결과 저장: {log_path}")
