"""
=============================================================
  Solver 5: GA (Genetic Algorithm, 유전 알고리즘)
=============================================================
특징:
  - 자연의 진화 과정(선택, 교차, 변이)을 모방
  - 복잡한 비선형 문제에도 적용 가능
  - 최적해에 가까운 '좋은 해'를 찾지만, 보장은 없음
  - 직접 구현 (외부 라이브러리 불필요)

교육 포인트:
  "유전 알고리즘은 다양한 '후보 해'를 동시에 탐색하며,
   세대를 거듭할수록 점점 더 좋은 해를 향해 진화한다."

진화 과정:
  1. 초기 집단 생성 (무작위 생산 계획들)
  2. 적합도 평가 (이익 - 제약 위반 페널티)
  3. 선택 (토너먼트 방식)
  4. 교차 (두 부모의 유전자를 섞음)
  5. 변이 (일부 유전자를 무작위 변경)
  6. 2~5를 여러 세대 반복
=============================================================
"""

import time
import random
from environment import ProductionEnv
from result_utils import start_terminal_log


class GASolver:
    """유전 알고리즘 기반 최적화 솔버"""

    def __init__(self, env: ProductionEnv, seed=42):
        self.env = env
        self.rng = random.Random(seed)

        # GA 하이퍼파라미터
        self.pop_size = 60        # 집단 크기 (한 세대의 개체 수)
        self.n_generations = 100  # 세대 수
        self.crossover_rate = 0.8 # 교차 확률
        self.mutation_rate = 0.15 # 변이 확률
        self.tournament_size = 3  # 토너먼트 선택 크기
        self.penalty_weight = 50  # 제약 위반 페널티 가중치

        # 각 제품의 최대 가능 생산량 (탐색 범위 제한용)
        self.max_quantities = [env.get_max_quantity(i) for i in range(env.n_products)]

    # ── 1. 개체(Individual) = 염색체(Chromosome) = 생산 계획 ──

    def _create_individual(self):
        """무작위 개체 생성: [의자 수, 책상 수, 선반 수]"""
        return [self.rng.randint(0, mq) for mq in self.max_quantities]

    # ── 2. 적합도(Fitness) 평가 ──

    def _fitness(self, individual):
        """
        적합도 = 이익 - 제약 위반 페널티.
        제약을 만족하면 페널티=0 → 순수 이익이 적합도.
        """
        result = self.env.evaluate(individual)
        penalty = 0
        for usage in result['resource_usage'].values():
            if usage['over'] > 0:
                penalty += usage['over'] * self.penalty_weight
        return result['profit'] - penalty

    # ── 3. 선택(Selection): 토너먼트 방식 ──

    def _tournament_select(self, population, fitness_scores):
        """
        토너먼트 선택: k개의 개체를 무작위로 뽑고, 그 중 가장 좋은 개체를 선택.
        → 너무 강한 선택압을 피하면서도 우수한 개체가 선호됨.
        """
        candidates = self.rng.sample(range(len(population)), self.tournament_size)
        best_idx = max(candidates, key=lambda i: fitness_scores[i])
        return population[best_idx][:]  # 복사본 반환

    # ── 4. 교차(Crossover): 균일 교차 ──

    def _crossover(self, parent1, parent2):
        """
        균일 교차: 각 유전자(제품 생산량)를 50% 확률로 부모1 또는 부모2에서 선택.
        → 두 부모의 좋은 특성이 자녀에게 전달될 수 있음.
        """
        if self.rng.random() > self.crossover_rate:
            return parent1[:], parent2[:]

        child1, child2 = [], []
        for g1, g2 in zip(parent1, parent2):
            if self.rng.random() < 0.5:
                child1.append(g1)
                child2.append(g2)
            else:
                child1.append(g2)
                child2.append(g1)
        return child1, child2

    # ── 5. 변이(Mutation): 가우시안 변이 ──

    def _mutate(self, individual):
        """
        변이: 각 유전자를 mutation_rate 확률로 무작위 변경.
        → 탐색 다양성을 유지하여 지역 최적에 빠지는 것을 방지.
        """
        for i in range(len(individual)):
            if self.rng.random() < self.mutation_rate:
                # 현재 값에서 ±30% 범위 내에서 변이
                delta = int(self.max_quantities[i] * 0.3)
                individual[i] = max(0, min(
                    self.max_quantities[i],
                    individual[i] + self.rng.randint(-delta, delta)
                ))
        return individual

    # ── 전체 진화 실행 ──

    def solve(self):
        """유전 알고리즘을 실행하여 최적 생산 계획을 탐색합니다."""
        start_time = time.perf_counter()

        # 초기 집단 생성
        population = [self._create_individual() for _ in range(self.pop_size)]

        best_ever = None
        best_ever_fitness = float('-inf')
        history = []  # 세대별 최고 적합도 기록

        for gen in range(self.n_generations):
            # 적합도 평가
            fitness_scores = [self._fitness(ind) for ind in population]

            # 현 세대 최고 개체 기록
            gen_best_idx = max(range(len(population)), key=lambda i: fitness_scores[i])
            gen_best_fitness = fitness_scores[gen_best_idx]

            if gen_best_fitness > best_ever_fitness:
                best_ever_fitness = gen_best_fitness
                best_ever = population[gen_best_idx][:]

            history.append(gen_best_fitness)

            # 다음 세대 생성
            new_population = []

            # 엘리트 보존: 최고 개체 2개는 그대로 다음 세대로
            sorted_indices = sorted(range(len(population)),
                                    key=lambda i: fitness_scores[i], reverse=True)
            new_population.append(population[sorted_indices[0]][:])
            new_population.append(population[sorted_indices[1]][:])

            # 나머지는 선택 → 교차 → 변이
            while len(new_population) < self.pop_size:
                p1 = self._tournament_select(population, fitness_scores)
                p2 = self._tournament_select(population, fitness_scores)
                c1, c2 = self._crossover(p1, p2)
                c1 = self._mutate(c1)
                c2 = self._mutate(c2)
                new_population.extend([c1, c2])

            population = new_population[:self.pop_size]

        elapsed = time.perf_counter() - start_time

        # 최종 결과
        result = self.env.evaluate(best_ever)
        return {
            'name': 'GA (유전 알고리즘)',
            'quantities': best_ever,
            'profit': result['profit'] if result['feasible'] else result['profit'],
            'time': elapsed,
            'feasible': result['feasible'],
            'is_integer': True,
            'resource_usage': result['resource_usage'],
            'details': {
                'generations': self.n_generations,
                'population_size': self.pop_size,
                'history': history,
                'final_fitness': best_ever_fitness
            }
        }


# ── 단독 실행 시 ──
if __name__ == '__main__':
    log_path = start_terminal_log("11_5_ga_solver_console.txt")

    env = ProductionEnv()
    env.print_problem()

    print("\n🧬 GA (유전 알고리즘) 풀이 시작...\n")
    solver = GASolver(env)
    result = solver.solve()

    print(f"  ✅ 최적 이익: {result['profit']}만원")
    for i, name in enumerate(env.products):
        print(f"  {name}: {result['quantities'][i]}개")
    print(f"  실현 가능: {'✅' if result['feasible'] else '❌'}")
    print(f"  풀이 시간: {result['time']:.4f}초")
    print(f"  세대 수: {result['details']['generations']}")

    # 수렴 과정 간략 출력
    h = result['details']['history']
    print(f"\n  [수렴 과정]")
    print(f"  1세대: {h[0]:.1f} → 25세대: {h[24]:.1f} → "
          f"50세대: {h[49]:.1f} → 100세대: {h[-1]:.1f}")

    print(f"\n  💾 터미널 결과 저장: {log_path}")
