from environment import FormulationEnv
from pulp_optimizer import PuLPOptimizer
from optuna_optimizer import AIOptimizer

if __name__ == "__main__":
    env = FormulationEnv()

    print("=== 1. PuLP (수리적 최적화) 실행 ===")
    pulp_opt = PuLPOptimizer(env)
    pulp_result = pulp_opt.optimize()
    print(f"최적 투입량: A={pulp_result['x_A']:.2f}g, B={pulp_result['x_B']:.2f}g, C={pulp_result['x_C']:.2f}g")
    print(f"최소 비용: {pulp_result['cost']:.2f}원\n")

    print("=== 2. AI (Optuna) 최적화 실행 ===")
    ai_opt = AIOptimizer(env)
    ai_result = ai_opt.optimize(n_trials=1000) # 1000번의 시행착오 학습
    print(f"최적 투입량: A={ai_result['x_A']:.2f}g, B={ai_result['x_B']:.2f}g, C={ai_result['x_C']:.2f}g")
    print(f"최소 비용: {ai_result['cost']:.2f}원")