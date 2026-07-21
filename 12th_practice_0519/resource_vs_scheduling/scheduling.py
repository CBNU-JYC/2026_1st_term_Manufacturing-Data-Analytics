import pulp
import matplotlib.pyplot as plt

# --- 1. 최적화 모델링 및 풀이 (PuLP) ---
model_schedule = pulp.LpProblem("Scheduling_Optimization", pulp.LpMinimize)

t_A = pulp.LpVariable('Start_Time_A', lowBound=0, cat='Continuous')
t_B = pulp.LpVariable('Start_Time_B', lowBound=0, cat='Continuous')
C_max = pulp.LpVariable('Makespan', lowBound=0, cat='Continuous')
y = pulp.LpVariable('Sequence_A_then_B', cat='Binary')

M = 100

model_schedule += C_max, "Minimize_Makespan"
model_schedule += C_max >= t_A + 2, "Cmax_after_A"
model_schedule += C_max >= t_B + 3, "Cmax_after_B"

# 겹침 방지 (Either-Or) 제약조건
model_schedule += t_B >= t_A + 2 - M * (1 - y), "No_Overlap_A_first"
model_schedule += t_A >= t_B + 3 - M * y, "No_Overlap_B_first"

model_schedule.solve()

print("=== 실습 2. 스케줄링 결과 ===")
print(f"작업 A 시작시간: {t_A.varValue}시")
print(f"작업 B 시작시간: {t_B.varValue}시")
print(f"모든 작업 종료(Makespan): {pulp.value(model_schedule.objective)}시간\n")

# --- 2. 간트 차트 시각화 (Matplotlib) ---
fig, ax = plt.subplots(figsize=(10, 4))

# 작업 데이터 정리
tasks = ['Task A', 'Task B']
start_times = [t_A.varValue, t_B.varValue]
durations = [2, 3]
colors = ['skyblue', 'lightgreen']

# 가로 막대그래프(barh)를 이용해 간트 차트 그리기
for i in range(2):
    ax.barh(tasks[i], durations[i], left=start_times[i], color=colors[i], edgecolor='black', height=0.5)
    # 막대 가운데에 소요 시간 텍스트 추가
    ax.text(start_times[i] + durations[i] / 2, i, f'{durations[i]} hours', 
            ha='center', va='center', color='black', fontweight='bold')

# 차트 꾸미기
ax.set_xlabel('Time (Hours)', fontsize=12)
ax.set_title('Scheduling Optimization: Gantt Chart', fontsize=14)
ax.set_xlim(0, pulp.value(model_schedule.objective) + 1) # x축 범위를 총 소요시간 + 1 로 설정
ax.set_yticks([0, 1])
ax.set_yticklabels(tasks, fontsize=12)

# 시간에 따른 세로 그리드 선 추가
ax.xaxis.grid(True, linestyle='--', alpha=0.7)
ax.yaxis.grid(False)

plt.tight_layout()
plt.show()