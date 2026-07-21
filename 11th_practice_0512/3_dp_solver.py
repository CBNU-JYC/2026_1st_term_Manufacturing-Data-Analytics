"""
=============================================================
  Solver 3: DP (Dynamic Programming, 동적 계획법)
=============================================================
특징:
  - 큰 문제를 작은 하위 문제로 나누어 점화식으로 해결
  - 정수 최적해 보장 (모든 가능한 조합을 체계적으로 탐색)
  - 상태 공간이 크면 메모리/시간이 많이 소모될 수 있음
  - 직접 구현 (외부 라이브러리 불필요)

교육 포인트:
  "DP는 '표를 채워가며' 최적해를 찾는 방법이다.
   모든 가능성을 빠짐없이 탐색하되, 중복 계산을 제거하여 효율적이다."

구현 방식:
  1. 간소화 데모 (원자재 제약만, 애니메이션) → DP 원리 학습
  2. 풀 시나리오 데모 (3개 제약, 2D DP 테이블 애니메이션) → 실전 적용
  3. 전체 풀이 (3개 제약 모두, 다단계 결정) → 비교용 결과 반환
=============================================================
"""

import time
import os
from environment import ProductionEnv
from result_utils import start_terminal_log


class DPSolver:
    """동적 계획법 기반 최적화 솔버"""

    def __init__(self, env: ProductionEnv):
        self.env = env

    def solve(self):
        """
        전체 문제를 DP(다단계 열거)로 풀이합니다.

        접근법 (Multi-stage Decision):
          - 1단계: 의자를 몇 개 만들지 결정 (0 ~ 최대)
          - 2단계: 남은 자원으로 책상을 몇 개 만들지 결정
          - 3단계: 남은 자원으로 선반을 최대한 생산
          → 모든 (의자, 책상) 조합을 탐색하고, 선반은 자동 결정

        ⚠️ 교육 참고 (실행 시간 해석):
          DP/Greedy가 LP/IP보다 빠르게 보이는 이유는 2가지입니다:

          ① 구현 방식 차이 (라이브러리 오버헤드):
             LP/IP는 PuLP 라이브러리가 내부적으로 CBC 외부 프로세스를 호출합니다.
             (모델 직렬화 → 프로세스 생성 → 결과 파싱 = ~20ms 오버헤드)
             DP/Greedy는 순수 Python 코드로 직접 구현하여 이런 오버헤드가 없습니다.

          ② 문제 크기가 매우 작음:
             이 문제(3제품, 자원한계 40~45)의 DP 탐색 공간은 ~136개뿐입니다.
             제품이 10종 이상이거나 자원 한계가 수백 이상이면
             DP 탐색 공간이 지수적으로 증가하여 LP/IP보다 훨씬 느려집니다.
             (LP/IP는 문제가 커져도 다항 시간에 풀 수 있습니다)
        """
        start_time = time.perf_counter()

        res = self.env.resources
        usage_a = res['machine_a']['usage']
        usage_b = res['machine_b']['usage']
        usage_m = res['material']['usage']
        limit_a = res['machine_a']['limit']
        limit_b = res['machine_b']['limit']
        limit_m = res['material']['limit']
        profit = self.env.profit

        best_profit = 0
        best_quantities = [0, 0, 0]

        # 의자(제품0)의 최대 생산 가능량
        max_x0 = min(limit_a // usage_a[0],
                     limit_b // usage_b[0],
                     limit_m // usage_m[0])

        for x0 in range(max_x0 + 1):
            # x0개 의자 생산 후 남은 자원
            rem_a = limit_a - usage_a[0] * x0
            rem_b = limit_b - usage_b[0] * x0
            rem_m = limit_m - usage_m[0] * x0

            # 책상(제품1)의 최대 생산 가능량 (남은 자원 기준)
            max_x1 = min(rem_a // usage_a[1] if usage_a[1] > 0 else 9999,
                         rem_b // usage_b[1] if usage_b[1] > 0 else 9999,
                         rem_m // usage_m[1] if usage_m[1] > 0 else 9999)

            for x1 in range(max_x1 + 1):
                # x0 의자 + x1 책상 생산 후 남은 자원
                rem_a2 = rem_a - usage_a[1] * x1
                rem_b2 = rem_b - usage_b[1] * x1
                rem_m2 = rem_m - usage_m[1] * x1

                # 선반(제품2)은 남은 자원으로 최대한 생산 (DP 최적 부분 구조)
                x2 = min(rem_a2 // usage_a[2] if usage_a[2] > 0 else 9999,
                         rem_b2 // usage_b[2] if usage_b[2] > 0 else 9999,
                         rem_m2 // usage_m[2] if usage_m[2] > 0 else 9999)

                total_profit = profit[0] * x0 + profit[1] * x1 + profit[2] * x2

                if total_profit > best_profit:
                    best_profit = total_profit
                    best_quantities = [x0, x1, x2]

        elapsed = time.perf_counter() - start_time

        result = self.env.evaluate(best_quantities)
        return {
            'name': 'DP (동적 계획법)',
            'quantities': best_quantities,
            'profit': best_profit,
            'time': elapsed,
            'feasible': result['feasible'],
            'is_integer': True,
            'resource_usage': result['resource_usage']
        }

    # ══════════════════════════════════════════════════════════
    #  교육용 애니메이션 1: 간소화 (원자재 제약만, 무한 배낭)
    # ══════════════════════════════════════════════════════════

    def demo_animation_simple(self, capacity=20):
        """
        원자재 제약만 고려한 간소화 DP (무한 배낭 문제) 애니메이션.
        학생들에게 DP 테이블이 채워지는 과정을 시각적으로 보여줍니다.
        """
        import time as t

        usage = self.env.resources['material']['usage']
        profit = self.env.profit
        names = self.env.products

        dp = [0] * (capacity + 1)
        choice = [-1] * (capacity + 1)

        print()
        print("=" * 60)
        print("  📚 DP 원리 학습: 무한 배낭 문제 (원자재 제약만)")
        print("=" * 60)
        print(f"  원자재 용량: {capacity}kg")
        for i in range(len(names)):
            print(f"  {names[i]}: {usage[i]}kg → {profit[i]}만원")
        print()
        print("  점화식: dp[w] = max(제품i의 이익 + dp[w - 제품i의 무게])")
        print("  콘솔을 크게 띄워주세요...")
        t.sleep(2)

        for w in range(1, capacity + 1):
            best_val = dp[w]
            best_item = -1
            for i in range(len(names)):
                if usage[i] <= w:
                    val = profit[i] + dp[w - usage[i]]
                    if val > best_val:
                        best_val = val
                        best_item = i
            dp[w] = best_val
            choice[w] = best_item
            self._print_dp_step(dp, w, capacity, best_item, names, usage, profit)
            t.sleep(0.5)

        print("\n" + "=" * 60)
        print("  🔍 역추적 (Backtracking): 최적 생산 계획 복원")
        print("=" * 60)

        selected = []
        w = capacity
        while w > 0 and choice[w] != -1:
            item = choice[w]
            selected.append(names[item])
            w -= usage[item]

        from collections import Counter
        count = Counter(selected)
        print(f"  최대 이익: {dp[capacity]}만원")
        print(f"  생산 계획: ", end="")
        for name, cnt in count.items():
            print(f"{name} {cnt}개  ", end="")
        print()

    def _print_dp_step(self, dp, current_w, capacity, chosen_item, names, usage, profit):
        """DP 테이블의 현재 상태를 출력 (간소화 애니메이션용)"""
        os.system('cls' if os.name == 'nt' else 'clear')
        print("  📚 DP 테이블 채우기 (무한 배낭 — 원자재 제약)")
        print()

        cols_per_row = 21
        for start in range(0, capacity + 1, cols_per_row):
            end = min(start + cols_per_row, capacity + 1)
            header = "  용량: "
            for w in range(start, end):
                header += f"{w:>4}"
            print(header)
            print("  " + "─" * (7 + 4 * (end - start)))
            row = "  dp[w]: "
            for w in range(start, end):
                if w < current_w:
                    row += f"{dp[w]:>4}"
                elif w == current_w:
                    row += f"[{dp[w]:>2}]"
                else:
                    row += "   ."
            print(row)
            print()

        if chosen_item >= 0:
            i = chosen_item
            print(f"  ▶ 용량 {current_w}kg: {names[i]} 선택!")
            print(f"    이익({profit[i]}) + dp[{current_w}-{usage[i]}]({dp[current_w - usage[i]]}) = {dp[current_w]}만원")
        elif current_w > 0:
            print(f"  ▶ 용량 {current_w}kg: 이전 값 유지 ({dp[current_w]}만원)")

    # ══════════════════════════════════════════════════════════
    #  교육용 애니메이션 2: 풀 시나리오 (3개 자원 제약 + 2D DP 테이블)
    # ══════════════════════════════════════════════════════════

    def demo_animation(self, delay=0.3):
        """
        가구 공장 전체 시나리오 DP 애니메이션 (2D DP 테이블 포함).

        DP 테이블 구조:
          행(Row) = 의자 생산량,  열(Col) = 책상 생산량
          셀 값   = 해당 (의자,책상) 조합의 최대 이익 (선반은 자동 결정)
          [괄호]  = 현재 계산 중인 셀,  ★ = 현재까지 최적인 셀

        Args:
            delay: 각 단계 사이의 대기 시간(초)
        """
        import time as t

        res = self.env.resources
        usage_a = res['machine_a']['usage']
        usage_b = res['machine_b']['usage']
        usage_m = res['material']['usage']
        limit_a = res['machine_a']['limit']
        limit_b = res['machine_b']['limit']
        limit_m = res['material']['limit']
        profit = self.env.profit
        names = self.env.products

        max_x0 = min(limit_a // usage_a[0],
                     limit_b // usage_b[0],
                     limit_m // usage_m[0])

        global_max_x1 = min(
            limit_a // usage_a[1] if usage_a[1] > 0 else 999,
            limit_b // usage_b[1] if usage_b[1] > 0 else 999,
            limit_m // usage_m[1] if usage_m[1] > 0 else 999)

        # 2D DP 테이블: dp_table[의자][책상] = 이익 (None=미탐색)
        dp_table = [[None] * (global_max_x1 + 1) for _ in range(max_x0 + 1)]

        # ── 인트로 ──
        os.system('cls' if os.name == 'nt' else 'clear')
        print()
        print("╔══════════════════════════════════════════════════════════╗")
        print("║  🏭 DP 애니메이션: 가구 공장 전체 시나리오               ║")
        print("╚══════════════════════════════════════════════════════════╝")
        print()
        print(f"  ┌─────────┬───────┬────────────┬────────────┬──────────┐")
        print(f"  │  제품    │ 이익   │ 기계A(가공) │ 기계B(조립) │ 원자재   │")
        print(f"  ├─────────┼───────┼────────────┼────────────┼──────────┤")
        for i in range(3):
            print(f"  │ {names[i]:<6}  │ {profit[i]:>3}만원│ {usage_a[i]:>5}시간   │ {usage_b[i]:>5}시간   │ {usage_m[i]:>4}kg   │")
        print(f"  └─────────┴───────┴────────────┴────────────┴──────────┘")
        print(f"  자원 한계: 기계A={limit_a}h  기계B={limit_b}h  원자재={limit_m}kg")
        print()
        print("  [2D DP 테이블] 행=의자(0~{0}), 열=책상(0~{1})".format(max_x0, global_max_x1))
        print("  셀 값 = 최대 이익 (선반은 자동 결정)")
        print()
        print("  Enter를 눌러 시작...")
        input()

        # ── 탐색 시작 ──
        best_profit = 0
        best_combo = [0, 0, 0]
        step_count = 0

        for x0 in range(max_x0 + 1):
            rem_a = limit_a - usage_a[0] * x0
            rem_b = limit_b - usage_b[0] * x0
            rem_m = limit_m - usage_m[0] * x0

            max_x1_local = min(
                rem_a // usage_a[1] if usage_a[1] > 0 else 999,
                rem_b // usage_b[1] if usage_b[1] > 0 else 999,
                rem_m // usage_m[1] if usage_m[1] > 0 else 999)

            # 비실현 가능 셀을 -로 표시
            for c in range(max_x1_local + 1, global_max_x1 + 1):
                if dp_table[x0][c] is None:
                    dp_table[x0][c] = -1  # -1 = 비실현 가능

            for x1 in range(max_x1_local + 1):
                rem_a2 = rem_a - usage_a[1] * x1
                rem_b2 = rem_b - usage_b[1] * x1
                rem_m2 = rem_m - usage_m[1] * x1

                x2 = min(
                    rem_a2 // usage_a[2] if usage_a[2] > 0 else 999,
                    rem_b2 // usage_b[2] if usage_b[2] > 0 else 999,
                    rem_m2 // usage_m[2] if usage_m[2] > 0 else 999)

                total = profit[0] * x0 + profit[1] * x1 + profit[2] * x2
                is_new_best = total > best_profit
                if is_new_best:
                    best_profit = total
                    best_combo = [x0, x1, x2]

                dp_table[x0][x1] = total
                step_count += 1

                used_a = usage_a[0]*x0 + usage_a[1]*x1 + usage_a[2]*x2
                used_b = usage_b[0]*x0 + usage_b[1]*x1 + usage_b[2]*x2
                used_m = usage_m[0]*x0 + usage_m[1]*x1 + usage_m[2]*x2

                self._print_full_dp_step(
                    step_count, [x0, x1, x2], total,
                    used_a, used_b, used_m,
                    limit_a, limit_b, limit_m,
                    best_combo, best_profit, is_new_best,
                    names, profit,
                    dp_table, x0, x1, max_x0, global_max_x1)
                t.sleep(delay if not is_new_best else delay * 3)

        # ── 최종 결과 ──
        os.system('cls' if os.name == 'nt' else 'clear')
        print()
        print("╔══════════════════════════════════════════════════════════╗")
        print("║  🏆 DP 탐색 완료! 최적 생산 계획 발견                    ║")
        print("╚══════════════════════════════════════════════════════════╝")
        print()
        print(f"  총 탐색 조합 수: {step_count}개")
        print()

        # 최종 2D DP 테이블
        print("  📊 완성된 DP 테이블 (행=의자, 열=책상, 셀=이익)")
        self._print_dp_table(dp_table, max_x0, global_max_x1,
                             best_combo[0], best_combo[1], -1, -1)
        print()
        print(f"  ┌─────────┬──────────┬───────────────────────────────┐")
        print(f"  │  제품    │  생산량   │  이익 계산                     │")
        print(f"  ├─────────┼──────────┼───────────────────────────────┤")
        for i in range(3):
            q = best_combo[i]
            p = profit[i] * q
            print(f"  │ {names[i]:<6}  │  {q:>4}개   │  {q}개 × {profit[i]}만원 = {p:>3}만원            │")
        print(f"  ├─────────┼──────────┼───────────────────────────────┤")
        print(f"  │  합계    │          │  총 이익 = {best_profit:>3}만원 ★          │")
        print(f"  └─────────┴──────────┴───────────────────────────────┘")
        print()

        final_a = sum(usage_a[i]*best_combo[i] for i in range(3))
        final_b = sum(usage_b[i]*best_combo[i] for i in range(3))
        final_m = sum(usage_m[i]*best_combo[i] for i in range(3))
        self._print_resource_bar("기계A(가공)", final_a, limit_a, "시간")
        self._print_resource_bar("기계B(조립)", final_b, limit_b, "시간")
        self._print_resource_bar("원자재     ", final_m, limit_m, "kg")
        print()
        print("  📌 DP의 핵심:")
        print("     '모든 가능한 조합을 체계적으로 탐색'하여 최적해를 보장합니다.")
        print(f"     총 {step_count}개 조합만 확인하면 되므로 효율적입니다!")
        print()

    # ══════════════════════════════════════════════════════════
    #  공통 헬퍼 메서드
    # ══════════════════════════════════════════════════════════

    def _print_resource_bar(self, name, used, limit, unit):
        """자원 사용률을 게이지 바로 출력"""
        pct = used / limit * 100 if limit > 0 else 0
        bar_len = int(pct / 100 * 25)
        bar = "█" * bar_len + "░" * (25 - bar_len)
        color = "🔴" if pct > 95 else ("🟡" if pct > 80 else "🟢")
        print(f"  {name} {bar} {used:>3}/{limit}{unit} ({pct:.0f}%) {color}")

    def _print_dp_table(self, dp_table, max_x0, max_x1,
                        best_r, best_c, cur_r, cur_c):
        """
        2D DP 테이블 출력 (행=의자, 열=책상).
        테이블이 넓으면 열을 블록으로 나누어 출력합니다.
        [괄호] = 현재 계산 중,  ★ = 현재 최적
        """
        cols_per_block = min(max_x1 + 1, 11)

        for col_start in range(0, max_x1 + 1, cols_per_block):
            col_end = min(col_start + cols_per_block, max_x1 + 1)

            # 헤더 (책상 수)
            hdr = "  의자\\책상"
            for c in range(col_start, col_end):
                hdr += f"{c:>5}"
            print(hdr)
            print("  " + "─" * (10 + 5 * (col_end - col_start)))

            # 의자 행이 너무 많으면 현재 근처만 표시
            if max_x0 > 18 and cur_r >= 0:
                r_lo = max(0, cur_r - 6)
                r_hi = min(max_x0, cur_r + 6)
                rows = list(range(r_lo, r_hi + 1))
                if r_lo > 0:
                    print("       ...  (생략)")
            else:
                rows = list(range(max_x0 + 1))

            for r in rows:
                row_str = f"  {r:>5}    │"
                for c in range(col_start, col_end):
                    val = dp_table[r][c] if c < len(dp_table[r]) else None
                    if r == cur_r and c == cur_c:
                        row_str += f"[{val:>3}]" if val is not None and val >= 0 else "[ ? ]"
                    elif r == best_r and c == best_c and val is not None and val >= 0:
                        row_str += f" {val:>2}★ "
                    elif val is not None and val >= 0:
                        row_str += f" {val:>3} "
                    elif val == -1:
                        row_str += "   - "
                    else:
                        row_str += "   · "
                print(row_str)

            if max_x0 > 18 and cur_r >= 0 and r_hi < max_x0:
                print("       ...  (생략)")
            print()

    def _print_full_dp_step(self, step, combo, total,
                            used_a, used_b, used_m,
                            lim_a, lim_b, lim_m,
                            best_combo, best_profit, is_new_best,
                            names, profit,
                            dp_table, cur_x0, cur_x1, max_x0, max_x1):
        """풀 시나리오 DP 애니메이션 — 한 프레임 (2D DP 테이블 포함)"""
        os.system('cls' if os.name == 'nt' else 'clear')

        print(f"  🏭 DP 다단계 탐색  [조합 #{step}]")
        print()

        # ── 2D DP 테이블 ──
        print("  📊 DP 테이블 (셀=이익, [괄호]=현재, ★=최적)")
        self._print_dp_table(dp_table, max_x0, max_x1,
                             best_combo[0], best_combo[1],
                             cur_x0, cur_x1)

        # 현재 시도 정보
        x0, x1, x2 = combo
        print(f"  현재: {names[0]}={x0}, {names[1]}={x1}, {names[2]}={x2}(자동) → {total}만원")

        # 자원 게이지
        self._print_resource_bar("기계A", used_a, lim_a, "h")
        self._print_resource_bar("기계B", used_b, lim_b, "h")
        self._print_resource_bar("원자재", used_m, lim_m, "kg")

        # 최적 표시
        if is_new_best:
            print(f"\n  ⭐ 새로운 최적해! {total}만원")
        else:
            print(f"\n  현재 최적: {best_profit}만원 ({names[0]}={best_combo[0]}, {names[1]}={best_combo[1]}, {names[2]}={best_combo[2]})")


# ── 단독 실행 시 ──
if __name__ == '__main__':
    log_path = start_terminal_log("11_3_dp_solver_console.txt")

    env = ProductionEnv()
    env.print_problem()

    solver = DPSolver(env)

    print()
    print("  어떤 애니메이션을 실행할까요?")
    print("  [1] 풀 시나리오 (3개 제약 + 2D DP 테이블)")
    print("  [2] 간소화 버전 (원자재 제약만, 배낭 문제)")
    print("  [3] 애니메이션 없이 풀이만")
    choice = input("  선택 (1/2/3): ").strip()

    if choice == '2':
        print("\n[간소화 애니메이션]")
        solver.demo_animation_simple(capacity=20)
    elif choice != '3':
        print("\n[풀 시나리오 애니메이션]")
        solver.demo_animation(delay=0.3)

    # 전체 문제 풀이 (비교용)
    print("\n\n[전체 문제 DP 풀이 결과]")
    result = solver.solve()
    print(f"  ✅ 최적 이익: {result['profit']}만원")
    for i, name in enumerate(env.products):
        print(f"  {name}: {result['quantities'][i]}개")
    print(f"  풀이 시간: {result['time']:.4f}초")

    print(f"\n  💾 터미널 결과 저장: {log_path}")
