import pulp

class PuLPOptimizer:
    def __init__(self, env):
        self.env = env

    def optimize(self):
        # 1. 문제 정의 (비용 최소화)
        prob = pulp.LpProblem("Formulation_Cost_Minimization", pulp.LpMinimize)

        # 2. 변수 선언 (투입량은 0 이상)
        x_a = pulp.LpVariable('x_A', lowBound=0)
        x_b = pulp.LpVariable('x_B', lowBound=0)
        x_c = pulp.LpVariable('x_C', lowBound=0)

        m = self.env.materials
        target_w = self.env.target_weight

        # 3. 목적 함수 설정 (비용)
        prob += (x_a * m['A']['cost'] + x_b * m['B']['cost'] + x_c * m['C']['cost']), "Total_Cost"

        # 4. 제약 조건 설정
        # 4-1. 총 중량 100g
        prob += (x_a + x_b + x_c == target_w), "Total_Weight_Constraint"
        
        # 4-2. 유효성분 30% 이상 (즉, 100g 중 30g 이상)
        prob += (x_a * m['A']['active'] + x_b * m['B']['active'] + x_c * m['C']['active'] 
                 >= target_w * self.env.min_active_ratio), "Min_Active_Constraint"
                 
        # 4-3. 불순물 3% 이하 (즉, 100g 중 3g 이하)
        prob += (x_a * m['A']['impurity'] + x_b * m['B']['impurity'] + x_c * m['C']['impurity'] 
                 <= target_w * self.env.max_impurity_ratio), "Max_Impurity_Constraint"

        # 5. 풀이 실행
        prob.solve(pulp.PULP_CBC_CMD(msg=False)) # 콘솔 출력 숨김

        if pulp.LpStatus[prob.status] == 'Optimal':
            return {
                'x_A': x_a.varValue,
                'x_B': x_b.varValue,
                'x_C': x_c.varValue,
                'cost': pulp.value(prob.objective)
            }
        else:
            return None