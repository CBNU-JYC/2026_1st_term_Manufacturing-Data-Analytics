import time
import os

def clear_screen():
    """운영체제에 맞게 콘솔 화면을 지우는 함수"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_table_step(dp, current_i, current_w, weights, values, capacity):
    """현재 DP 테이블의 상태와 계산 과정을 화면에 출력하는 함수"""
    clear_screen()
    print("=== 🎒 0/1 배낭 문제 DP 테이블 채우기 (순차적) ===\n")
    print(f"배낭 최대 용량: {capacity}")
    print("아이템: A(무게 2, 가치 3), B(무게 3, 가치 4), C(무게 4, 가치 5), D(무게 5, 가치 8)\n")
    
    print("    용량: 0   1   2   3   4   5   6   7")
    print("-" * 43)
    
    # 표 그리기
    for i in range(len(dp)):
        if i == 0:
            row_label = "물건 0개:"
        else:
            row_label = f"물건 ~{chr(64+i)}:" # 1->A, 2->B...
        
        print(f"{row_label:9}", end="")
        for w in range(len(dp[0])):
            # 1. 방금 막 계산을 마친 현재 칸 (대괄호로 강조)
            if i == current_i and w == current_w:
                print(f"[{dp[i][w]:2d}]", end="")
            # 2. 아직 계산하지 않은 미래의 칸 (점으로 표시)
            elif i > current_i or (i == current_i and w > current_w):
                print("  . ", end="")
            # 3. 이미 계산이 끝난 과거의 칸
            else:
                print(f" {dp[i][w]:2d} ", end="")
        print()
        
    print("-" * 43)
    
    # 하단에 계산 과정(점화식) 설명 출력
    if current_i > 0 and current_w > 0:
        weight = weights[current_i-1]
        value = values[current_i-1]
        print(f"\n▶ 현재 차례: 물건 {chr(64+current_i)} (무게: {weight}, 가치: {value}) / 현재 배낭 용량: {current_w}")
        
        if weight > current_w:
            print(f"  - ❌ 너무 무거워서 배낭에 못 넣음!")
            print(f"  - 위쪽 칸의 값({dp[current_i-1][current_w]})을 그대로 가져옵니다.")
        else:
            not_included_val = dp[current_i-1][current_w]
            included_val = value + dp[current_i-1][current_w-weight]
            print(f"  - ⭕ 배낭에 넣을 수 있음! 둘 중 더 큰 가치를 선택합니다.")
            print(f"  - 선택 1 (안 넣기) : {not_included_val}")
            print(f"  - 선택 2 (넣기)    : 내 가치({value}) + 남은 용량({current_w-weight})의 최대 가치({dp[current_i-1][current_w-weight]}) = {included_val}")
            print(f"  - 결과: MAX({not_included_val}, {included_val}) = {dp[current_i][current_w]}")
    
def knapsack_dp_animated(weights, values, capacity):
    n = len(weights)
    dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]
    
    # 0행 0열 초기 상태를 한 번 보여줌
    print_table_step(dp, 0, 0, weights, values, capacity)
    time.sleep(2) # 2초 대기
    
    # 2중 for문으로 표 채우기
    for i in range(1, n + 1):
        for w in range(1, capacity + 1):
            weight = weights[i-1]
            value = values[i-1]
            
            # 동적 계획법 핵심 점화식
            if weight > w:
                dp[i][w] = dp[i-1][w]
            else:
                dp[i][w] = max(dp[i-1][w], value + dp[i-1][w - weight])
            
            # 한 칸을 계산할 때마다 화면을 갱신해서 보여줌
            print_table_step(dp, i, w, weights, values, capacity)
            time.sleep(1.2) # 1.2초 동안 멈춰서 보여줌 (시간을 조절해서 속도 변경 가능)
            
    return dp[n][capacity]

# ==========================================
# 실행 부분
# ==========================================
weights = [2, 3, 4, 5]
values = [3, 4, 5, 8]
capacity = 7

print("순차적 시각화를 시작합니다. 콘솔 창을 크게 띄워주세요...")
time.sleep(2)
knapsack_dp_animated(weights, values, capacity)
print("\n✅ DP 테이블 채우기 완료!")