class FormulationEnv:
    def __init__(self):
        # 원료 데이터: [비용(원/g), 유효성분 비율, 불순물 비율]
        self.materials = {
            'A': {'cost': 10, 'active': 0.10, 'impurity': 0.05},
            'B': {'cost': 15, 'active': 0.20, 'impurity': 0.02},
            'C': {'cost': 20, 'active': 0.50, 'impurity': 0.01}
        }
        # 제약 조건
        self.target_weight = 100.0     # 목표 총 중량 (g)
        self.min_active_ratio = 0.30   # 최소 유효성분 30%
        self.max_impurity_ratio = 0.03 # 최대 불순물 3%

    def evaluate(self, x_a, x_b, x_c):
        """
        주어진 원료 투입량에 대한 비용과 성분량을 계산하여 반환합니다.
        """
        total_weight = x_a + x_b + x_c
        
        # 비용 계산
        cost = (x_a * self.materials['A']['cost'] +
                x_b * self.materials['B']['cost'] +
                x_c * self.materials['C']['cost'])
        
        # 유효성분 및 불순물 총량 계산
        total_active = (x_a * self.materials['A']['active'] +
                        x_b * self.materials['B']['active'] +
                        x_c * self.materials['C']['active'])
        
        total_impurity = (x_a * self.materials['A']['impurity'] +
                          x_b * self.materials['B']['impurity'] +
                          x_c * self.materials['C']['impurity'])
                          
        return total_weight, cost, total_active, total_impurity