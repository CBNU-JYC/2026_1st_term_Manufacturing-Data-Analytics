"""
=============================================================
  공장 생산 계획 문제 — 공통 환경 (Environment)
=============================================================
시나리오:
  가구 공장에서 3종류의 제품(의자, 책상, 선반)을 생산합니다.
  각 제품은 기계A 가공 → 기계B 조립 → 원자재 소모를 거칩니다.
  한정된 자원 내에서 **총 이익을 최대화**하는 생산 계획을 세우세요.

  이 파일은 모든 최적화 알고리즘(Solver)이 공유하는 문제 정의입니다.
=============================================================
"""


from result_utils import start_terminal_log


class ProductionEnv:
    """
    공장 생산 계획 문제를 정의하는 환경 클래스.
    모든 Solver는 이 클래스의 인스턴스를 받아 동일한 문제를 풀게 됩니다.
    """

    def __init__(self):
        # ── 제품 정보 ──────────────────────────────────
        self.products = ['의자', '책상', '선반']
        self.n_products = 3

        # 제품별 이익 (만원/개)
        self.profit = [5, 7, 3]

        # ── 자원 제약 조건 ─────────────────────────────
        # 각 자원에 대해: name(이름), unit(단위), usage(제품별 소모량), limit(총 가용량)
        self.resources = {
            'machine_a': {
                'name': '기계A (가공)',
                'unit': '시간',
                'usage': [2, 3, 1],   # 의자 2h, 책상 3h, 선반 1h
                'limit': 40
            },
            'machine_b': {
                'name': '기계B (조립)',
                'unit': '시간',
                'usage': [1, 2, 2],   # 의자 1h, 책상 2h, 선반 2h
                'limit': 30
            },
            'material': {
                'name': '원자재',
                'unit': 'kg',
                'usage': [3, 2, 4],   # 의자 3kg, 책상 2kg, 선반 4kg
                'limit': 45
            }
        }

    def evaluate(self, quantities):
        """
        주어진 생산량(리스트)에 대해 이익, 자원 사용량, 실현 가능 여부를 평가합니다.

        Args:
            quantities: [의자 수, 책상 수, 선반 수] 형태의 리스트

        Returns:
            dict: profit, quantities, resource_usage, feasible 정보
        """
        # 총 이익 계산
        total_profit = sum(q * p for q, p in zip(quantities, self.profit))

        # 자원별 사용량 확인
        resource_usage = {}
        feasible = True

        for key, res in self.resources.items():
            used = sum(q * u for q, u in zip(quantities, res['usage']))
            over = max(0, used - res['limit'])
            resource_usage[key] = {
                'name': res['name'],
                'used': used,
                'limit': res['limit'],
                'unit': res['unit'],
                'over': over
            }
            if used > res['limit'] + 1e-6:  # 부동소수점 오차 허용
                feasible = False

        return {
            'profit': total_profit,
            'quantities': list(quantities),
            'resource_usage': resource_usage,
            'feasible': feasible
        }

    def get_max_quantity(self, product_idx):
        """특정 제품의 이론적 최대 생산 가능량 (모든 자원을 해당 제품에만 투입)"""
        max_q = float('inf')
        for res in self.resources.values():
            if res['usage'][product_idx] > 0:
                max_q = min(max_q, res['limit'] / res['usage'][product_idx])
        return int(max_q)

    def print_problem(self):
        """문제를 보기 좋게 콘솔에 출력합니다."""
        print("=" * 60)
        print("  🏭 공장 생산 계획 최적화 문제")
        print("=" * 60)
        print()
        print("  목표: 총 이익을 최대화하는 제품별 생산량 결정")
        print()

        # 제품 정보 테이블
        print("  [제품 정보]")
        print(f"  {'제품':>6} │ {'이익(만원/개)':>12} │", end="")
        for res in self.resources.values():
            print(f" {res['name']:>10}({res['unit']}/개) │", end="")
        print()
        print("  " + "─" * 70)

        for i, name in enumerate(self.products):
            print(f"  {name:>6} │ {self.profit[i]:>12} │", end="")
            for res in self.resources.values():
                print(f" {res['usage'][i]:>18} │", end="")
            print()

        # 자원 한계
        print()
        print("  [자원 제약]")
        for res in self.resources.values():
            print(f"  • {res['name']}: 최대 {res['limit']}{res['unit']}")
        print()
        print("=" * 60)


if __name__ == '__main__':
    log_path = start_terminal_log("11_0_environment_test_console.txt")

    env = ProductionEnv()
    env.print_problem()

    # 테스트: 의자 5개, 책상 5개, 선반 5개
    result = env.evaluate([5, 5, 5])
    print(f"\n테스트 생산량: 의자=5, 책상=5, 선반=5")
    print(f"  이익: {result['profit']}만원")
    print(f"  실현 가능: {'✅ 예' if result['feasible'] else '❌ 아니오'}")
    for key, usage in result['resource_usage'].items():
        status = "✅" if usage['over'] == 0 else f"❌ {usage['over']}{usage['unit']} 초과"
        print(f"  {usage['name']}: {usage['used']}/{usage['limit']}{usage['unit']} {status}")

    print(f"\n💾 터미널 결과 저장: {log_path}")
