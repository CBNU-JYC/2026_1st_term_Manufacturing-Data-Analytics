import optuna

class AIOptimizer:
    def __init__(self, env):
        self.env = env

    def objective(self, trial):
        # 1. AI에게 가중치 비율 탐색 지시 (0 ~ 10 사이의 임의의 값)
        w_a = trial.suggest_float('w_a', 0, 10)
        w_b = trial.suggest_float('w_b', 0, 10)
        w_c = trial.suggest_float('w_c', 0, 10)

        # 2. 비율을 바탕으로 100g에 맞게 정규화 변환
        total_w = w_a + w_b + w_c
        if total_w == 0:
            return float('inf') # 모두 0인 경우 최악의 점수 반환

        x_a = (w_a / total_w) * self.env.target_weight
        x_b = (w_b / total_w) * self.env.target_weight
        x_c = (w_c / total_w) * self.env.target_weight

        # 3. 환경 모듈을 통해 결과 평가
        _, cost, total_active, total_impurity = self.env.evaluate(x_a, x_b, x_c)

        # 4. 제약 조건 확인 및 소프트 페널티(Soft Penalty) 부여
        # AI는 제약조건을 강제할 수 없으므로, 위반 시 비용(점수)을 폭증시킵니다.
        penalty = 0
        
        target_active = self.env.target_weight * self.env.min_active_ratio
        if total_active < target_active:
            penalty += (target_active - total_active) * 5000  # 부족한 만큼 강력한 페널티
            
        target_impurity = self.env.target_weight * self.env.max_impurity_ratio
        if total_impurity > target_impurity:
            penalty += (total_impurity - target_impurity) * 5000 # 초과한 만큼 강력한 페널티

        # AI가 최소화하려는 최종 점수 (실제 비용 + 페널티)
        return cost + penalty

    def optimize(self, n_trials=500):
        # 로그 숨기기 (너무 많은 출력을 방지)
        optuna.logging.set_verbosity(optuna.logging.WARNING) 
        
        # 탐색 방향 설정 (최소화)
        study = optuna.create_study(direction='minimize')
        study.optimize(self.objective, n_trials=n_trials)

        # 최적 파라미터로 실제 투입량 계산
        best = study.best_params
        total_w = best['w_a'] + best['w_b'] + best['w_c']
        
        x_a = (best['w_a'] / total_w) * self.env.target_weight
        x_b = (best['w_b'] / total_w) * self.env.target_weight
        x_c = (best['w_c'] / total_w) * self.env.target_weight
        
        # 최종 평가된 순수 비용만 반환 (페널티 제외)
        _, final_cost, _, _ = self.env.evaluate(x_a, x_b, x_c)

        return {
            'x_A': x_a,
            'x_B': x_b,
            'x_C': x_c,
            'cost': final_cost,
            'ai_score': study.best_value # 페널티가 포함된 점수 (정상적이라면 final_cost와 동일해야 함)
        }