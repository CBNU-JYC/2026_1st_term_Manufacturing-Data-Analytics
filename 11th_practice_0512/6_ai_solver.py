"""
=============================================================
  Solver 6: AI (Optuna — 베이지안 최적화)
=============================================================
특징:
  - 블랙박스 최적화: 문제의 수학적 구조를 몰라도 적용 가능
  - 과거 탐색 결과를 학습하여 다음 탐색 방향을 결정 (베이지안)
  - 실행마다 결과가 다를 수 있음 (확률적)
  - 라이브러리: Optuna (TPE 샘플러)

교육 포인트:
  "AI 최적화는 가장 범용적이지만, 수학적 최적해를 보장하지 않는다.
   LP/IP처럼 문제 구조를 활용할 수 있다면 그것이 더 효율적이다.
   하지만 복잡한 비선형 문제에서는 AI가 유일한 선택지가 될 수 있다."
=============================================================
"""

import time
from environment import ProductionEnv
from result_utils import start_terminal_log


class AISolver:
    """Optuna를 사용한 AI(베이지안) 최적화 솔버"""

    def __init__(self, env: ProductionEnv, n_trials=500):
        self.env = env
        self.n_trials = n_trials

    def _objective(self, trial):
        """
        Optuna가 반복 호출하는 목적 함수.
        각 trial에서 AI가 생산량을 제안하고, 우리는 그 점수를 반환합니다.
        """
        # ─── AI에게 각 제품의 생산량을 제안받음 ───
        quantities = []
        for i, name in enumerate(self.env.products):
            max_q = self.env.get_max_quantity(i)
            q = trial.suggest_int(f'x_{name}', 0, max_q)
            quantities.append(q)

        # ─── 환경에서 평가 ───
        result = self.env.evaluate(quantities)

        # ─── 제약 위반 시 소프트 페널티 부여 ───
        # (AI는 제약조건을 직접 처리할 수 없으므로, 위반 시 점수를 감소시킴)
        penalty = 0
        for usage in result['resource_usage'].values():
            if usage['over'] > 0:
                penalty += usage['over'] * 100  # 초과량에 비례한 강한 페널티

        # Optuna는 minimize가 기본 → 이익을 음수로 반환 (최소화 = 이익 최대화)
        return -(result['profit'] - penalty)

    def solve(self):
        """Optuna 최적화를 실행합니다."""
        import optuna

        # Optuna 로그 숨김 (출력이 너무 많아지는 것을 방지)
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Study 생성 및 실행
        study = optuna.create_study(
            direction='minimize',  # -(이익)을 최소화 = 이익 최대화
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        start_time = time.perf_counter()
        study.optimize(self._objective, n_trials=self.n_trials)

        elapsed = time.perf_counter() - start_time

        # 최적 결과 추출
        best_params = study.best_params
        quantities = [best_params[f'x_{name}'] for name in self.env.products]
        result = self.env.evaluate(quantities)

        return {
            'name': f'AI/Optuna ({self.n_trials} trials)',
            'quantities': quantities,
            'profit': result['profit'],
            'time': elapsed,
            'feasible': result['feasible'],
            'is_integer': True,
            'resource_usage': result['resource_usage'],
            'details': {
                'n_trials': self.n_trials,
                'best_score': -study.best_value,
                'study': study
            }
        }


# ── 단독 실행 시 ──
if __name__ == '__main__':
    log_path = start_terminal_log("11_6_ai_solver_console.txt")

    env = ProductionEnv()
    env.print_problem()

    print("\n🤖 AI/Optuna 최적화 실행 중 (500 trials)...\n")
    solver = AISolver(env, n_trials=500)
    result = solver.solve()

    print(f"  ✅ 최적 이익: {result['profit']}만원")
    for i, name in enumerate(env.products):
        print(f"  {name}: {result['quantities'][i]}개")
    print(f"  실현 가능: {'✅' if result['feasible'] else '❌'}")
    print(f"  풀이 시간: {result['time']:.4f}초")
    print(f"  총 시도 횟수: {result['details']['n_trials']}회")

    print(f"\n  💾 터미널 결과 저장: {log_path}")
