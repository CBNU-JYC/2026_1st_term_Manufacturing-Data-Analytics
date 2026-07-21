"""
=============================================================
  Solver 1: LP (Linear Programming, 선형 계획법)
=============================================================
특징:
  - 변수가 연속(실수)값을 가질 수 있음 → "의자 12.5개" 같은 답이 나올 수 있음
  - 수학적으로 최적해를 보장 (전역 최적)
  - 매우 빠름 (다항 시간)
  - 라이브러리: PuLP (CBC 솔버 사용)

교육 포인트:
  "LP는 수학적으로 완벽한 최적해를 주지만,
   현실에서는 의자 12.5개를 만들 수 없다 → IP가 필요한 이유!"
=============================================================
"""

import time
from environment import ProductionEnv
from result_utils import start_terminal_log


class LPSolver:
    """PuLP를 사용한 선형 계획법 솔버"""

    def __init__(self, env: ProductionEnv):
        self.env = env

    def solve(self):
        """
        LP 문제를 정의하고 풀이합니다.

        LP 정식화:
          최대화: 5*x1 + 7*x2 + 3*x3   (이익)
          제약:
            2*x1 + 3*x2 + 1*x3 ≤ 40   (기계A)
            1*x1 + 2*x2 + 2*x3 ≤ 30   (기계B)
            3*x1 + 2*x2 + 4*x3 ≤ 45   (원자재)
            x1, x2, x3 ≥ 0
        """
        import pulp

        # ─── 1단계: 문제 정의 (이익 최대화) ───
        prob = pulp.LpProblem("Production_Planning_LP", pulp.LpMaximize)

        # ─── 2단계: 결정 변수 선언 (연속 변수, 0 이상) ───
        # LpContinuous가 기본값 → 실수 값 허용
        x = []
        for i, name in enumerate(self.env.products):
            var = pulp.LpVariable(f'x_{name}', lowBound=0, cat='Continuous')
            x.append(var)

        # ─── 3단계: 목적 함수 설정 (이익 최대화) ───
        prob += pulp.lpSum(x[i] * self.env.profit[i]
                           for i in range(self.env.n_products)), "Total_Profit"

        # ─── 4단계: 제약 조건 추가 ───
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
            quantities = [var.varValue for var in x]
            result = self.env.evaluate(quantities)
            return {
                'name': 'LP (선형 계획법)',
                'quantities': quantities,
                'profit': pulp.value(prob.objective),
                'time': elapsed,
                'feasible': result['feasible'],
                'is_integer': all(q == int(q) for q in quantities),
                'resource_usage': result['resource_usage']
            }
        else:
            return None


# ── 단독 실행 시 ──
if __name__ == '__main__':
    log_path = start_terminal_log("11_1_lp_solver_console.txt")

    env = ProductionEnv()
    env.print_problem()

    print("\n🔧 LP (선형 계획법) 풀이 시작...\n")
    solver = LPSolver(env)
    result = solver.solve()

    if result:
        print("✅ 최적해를 찾았습니다!")
        print(f"   이익: {result['profit']:.2f}만원")
        for i, name in enumerate(env.products):
            print(f"   {name}: {result['quantities'][i]:.2f}개")
        print(f"   정수 해 여부: {'예 ✅' if result['is_integer'] else '아니오 ❌ (소수점 존재!)'}")
        print(f"   풀이 시간: {result['time']:.4f}초")

        print("\n📌 교훈: LP는 수학적 최적해를 보장하지만,")
        print("   실수 해가 나올 수 있어 현실 제조에 바로 적용하기 어렵습니다.")
    else:
        print("❌ 해를 찾을 수 없습니다.")

    print(f"\n💾 터미널 결과 저장: {log_path}")
