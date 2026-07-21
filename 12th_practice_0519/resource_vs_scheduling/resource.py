import pulp
import matplotlib.pyplot as plt
import numpy as np

# --- 1. 최적화 모델링 및 풀이 (PuLP) ---
model_resource = pulp.LpProblem("Resource_Optimization", pulp.LpMaximize)

x_A = pulp.LpVariable('Product_A_qty', lowBound=0, cat='Integer')
x_B = pulp.LpVariable('Product_B_qty', lowBound=0, cat='Integer')

model_resource += 10 * x_A + 15 * x_B, "Total_Profit"
model_resource += 2 * x_A + 3 * x_B <= 10, "Time_Capacity"

model_resource.solve()

print("=== 실습 1. 자원 최적화 결과 ===")
print(f"제품 A 생산량: {x_A.varValue}개")
print(f"제품 B 생산량: {x_B.varValue}개")
print(f"최대 이익: {pulp.value(model_resource.objective)}만 원\n")

# --- 2. 시각화 (Matplotlib) ---
plt.figure(figsize=(8, 6))

# 제약조건 선 그리기: 2*x_A + 3*x_B = 10 -> x_B = (10 - 2*x_A) / 3
x = np.linspace(0, 6, 100)
y = (10 - 2 * x) / 3

# 실현 가능 영역(Feasible Region) 칠하기
plt.plot(x, y, label=r'$2x_A + 3x_B \leq 10$ (Time Capacity)', color='blue')
plt.fill_between(x, 0, y, where=(y >= 0), alpha=0.2, color='blue', label='Feasible Region')

# 최적해 점 찍기
plt.plot(x_A.varValue, x_B.varValue, 'ro', markersize=10, label='Optimal Solution (A, B)')
plt.annotate(f'({int(x_A.varValue)}, {int(x_B.varValue)})', 
             (x_A.varValue, x_B.varValue), textcoords="offset points", xytext=(10,10), ha='center', fontsize=12, color='red')

# 그래프 설정
plt.xlim(0, 6)
plt.ylim(0, 5)
plt.xlabel('Product A Quantity (x_A)')
plt.ylabel('Product B Quantity (x_B)')
plt.title('Resource Optimization: Feasible Region & Optimal Solution')
plt.axhline(0, color='black',linewidth=1)
plt.axvline(0, color='black',linewidth=1)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()