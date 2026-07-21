# -*- coding: utf-8 -*-
"""
=============================================================
  Step 1: SimPy 기초 개념 학습 (콘솔 전용)
=============================================================
SimPy의 3가지 핵심 개념을 단계별로 학습합니다:
  1. env.timeout()  — 시간의 흐름 모사
  2. simpy.Resource  — 자원 경쟁과 대기
  3. env.process()   — 여러 프로세스 동시 실행

실행 방법:
  conda activate mfg_data
  python step1_simpy_basics.py
=============================================================
"""

import simpy
import random
import time


def pause(seconds=1.5):
    """강의 중 설명할 시간을 주기 위한 대기"""
    time.sleep(seconds)


def wait_for_enter(message):
    """터미널에서는 Enter를 기다리고, 자동 실행 환경에서는 바로 넘어갑니다."""
    try:
        input(message)
    except EOFError:
        print(message)


# ══════════════════════════════════════════════════════════════
#  예제 1: env.timeout() — "시간의 흐름" 이해하기
# ══════════════════════════════════════════════════════════════

def example1_timeout():
    """
    SimPy에서 시간은 자동으로 흐르지 않습니다.
    yield env.timeout(시간) 을 호출해야 그만큼 시간이 전진합니다.
    """
    print("=" * 60)
    print("  [예제 1] env.timeout() — 시간의 흐름")
    print("=" * 60)
    print()
    print("  ★ 핵심: yield env.timeout(N) = 'N초 뒤에 깨워달라'")
    print("  ★ SimPy에서 시간은 자동으로 흐르지 않음!")
    print()
    pause()

    def 부품가공(env, 부품명, 가공시간):
        """하나의 부품이 가공되는 과정"""
        print(f"  t={env.now:5.1f} | {부품명} 가공 시작 ({가공시간}초 소요 예정)")

        yield env.timeout(가공시간)  # ← 핵심! 가공시간만큼 시간이 흐름

        print(f"  t={env.now:5.1f} | {부품명} 가공 완료! ✅")

    # SimPy 환경 생성 및 실행
    env = simpy.Environment()
    env.process(부품가공(env, "부품A", 3))
    env.process(부품가공(env, "부품B", 5))
    env.process(부품가공(env, "부품C", 2))

    print("  [실행 결과]")
    env.run()

    print()
    print("  📌 주목: A, B, C 모두 t=0에서 동시에 시작!")
    print("     → 서로 자원 경쟁 없이 각자 독립적으로 실행됨")
    print("     → 자원 제한이 필요하면? → 예제 2로!")
    print()


# ══════════════════════════════════════════════════════════════
#  예제 2: simpy.Resource — "자원 경쟁과 대기" 이해하기
# ══════════════════════════════════════════════════════════════

def example2_resource():
    """
    simpy.Resource(capacity=N)은 동시에 N개까지만 사용 가능한 자원입니다.
    설비가 1대인데 부품이 3개 오면? → 줄 서서 기다림!
    """
    print("=" * 60)
    print("  [예제 2] simpy.Resource — 자원 경쟁과 대기")
    print("=" * 60)
    print()
    print("  ★ 핵심: Resource(capacity=1) = 설비 1대")
    print("  ★ 설비가 사용 중이면 다음 부품은 줄을 서서 기다림!")
    print()
    pause()

    def 부품가공(env, 부품명, 설비, 가공시간):
        """설비를 요청하고, 배정받으면 가공하는 과정"""
        도착시간 = env.now
        print(f"  t={env.now:5.1f} | {부품명} 도착 → 설비 요청...")

        with 설비.request() as req:
            yield req  # ← 설비가 빌 때까지 대기!

            대기시간 = env.now - 도착시간
            if 대기시간 > 0:
                print(f"  t={env.now:5.1f} | {부품명} 대기 완료 (⏳ {대기시간:.1f}초 기다림) → 가공 시작")
            else:
                print(f"  t={env.now:5.1f} | {부품명} 바로 배정! → 가공 시작")

            yield env.timeout(가공시간)
            print(f"  t={env.now:5.1f} | {부품명} 가공 완료! ✅")

    env = simpy.Environment()
    설비 = simpy.Resource(env, capacity=1)  # ← 설비 1대!

    env.process(부품가공(env, "부품A", 설비, 3))
    env.process(부품가공(env, "부품B", 설비, 2))
    env.process(부품가공(env, "부품C", 설비, 4))

    print("  [실행 결과] 설비 1대, 부품 3개")
    env.run()

    print()
    print("  📌 주목: 설비가 1대이므로 B는 A가 끝날 때까지 기다림!")
    print("     → '대기 시간'이 병목의 시작!")
    print("     → 설비를 2대로 늘리면? → 예제 3으로!")
    print()


# ══════════════════════════════════════════════════════════════
#  예제 3: capacity 변경 — "설비 대수의 영향" 체감하기
# ══════════════════════════════════════════════════════════════

def example3_capacity():
    """
    같은 상황에서 설비를 1대 → 2대로 늘리면 어떻게 될까?
    """
    print("=" * 60)
    print("  [예제 3] 설비 대수 변경 — 병목 해소 체험")
    print("=" * 60)
    print()
    pause()

    def 부품가공(env, 부품명, 설비, 가공시간):
        도착시간 = env.now
        with 설비.request() as req:
            yield req
            대기시간 = env.now - 도착시간
            yield env.timeout(가공시간)
            return 대기시간

    def 시나리오_실행(설비_대수):
        env = simpy.Environment()
        설비 = simpy.Resource(env, capacity=설비_대수)

        부품들 = [
            ("부품A", 3), ("부품B", 2), ("부품C", 4),
            ("부품D", 3), ("부품E", 2)
        ]

        결과 = []

        def 실행_부품(env, 이름, 설비, 시간):
            대기 = yield env.process(부품가공(env, 이름, 설비, 시간))
            결과.append(대기)

        for 이름, 시간 in 부품들:
            env.process(실행_부품(env, 이름, 설비, 시간))

        env.run()
        return env.now, sum(결과) / len(결과)

    # 비교 실행
    for n in [1, 2, 3]:
        총시간, 평균대기 = 시나리오_실행(n)
        막대 = "█" * int(총시간 / 0.5)
        print(f"  설비 {n}대: 총 소요 {총시간:5.1f}초 | 평균 대기 {평균대기:4.1f}초 | {막대}")

    print()
    print("  📌 설비 1대→2대: 대기 시간이 크게 줄어듦!")
    print("     설비 2대→3대: 개선 효과가 작아짐 (수확 체감)")
    print("     → '적정 설비 수'를 찾는 것이 제조 최적화의 핵심!")
    print()


# ══════════════════════════════════════════════════════════════
#  예제 4: 연속 투입 — "실제 공장처럼 부품이 계속 오는 상황"
# ══════════════════════════════════════════════════════════════

def example4_continuous():
    """
    실제 공장에서는 부품이 한꺼번에 오는 것이 아니라,
    랜덤한 간격으로 계속 도착합니다.
    """
    print("=" * 60)
    print("  [예제 4] 연속 투입 — 실제 공장 모사")
    print("=" * 60)
    print()
    print("  ★ 핵심: while True + yield env.timeout(랜덤)")
    print("  ★ 이것이 실제 시뮬레이션의 기본 구조!")
    print()
    pause()

    완료_수 = 0
    대기_합 = 0

    def 부품_도착(env, 설비):
        """부품이 랜덤 간격으로 계속 도착하는 프로세스"""
        번호 = 0
        while True:
            yield env.timeout(random.expovariate(1.0 / 2.0))  # 평균 2초 간격
            번호 += 1
            env.process(부품_가공(env, f"부품{번호}", 설비))

    def 부품_가공(env, 이름, 설비):
        nonlocal 완료_수, 대기_합
        도착 = env.now
        with 설비.request() as req:
            yield req
            대기 = env.now - 도착
            대기_합 += 대기
            가공시간 = random.expovariate(1.0 / 3.0)  # 평균 3초
            yield env.timeout(가공시간)
            완료_수 += 1

    env = simpy.Environment()
    설비 = simpy.Resource(env, capacity=2)
    env.process(부품_도착(env, 설비))

    # 50초간 시뮬레이션 실행
    env.run(until=50)

    print(f"  [결과] 시뮬레이션 시간: 50초")
    print(f"  생산 완료: {완료_수}개")
    print(f"  평균 대기: {대기_합/완료_수:.2f}초")
    print(f"  처리량(Throughput): {완료_수/50:.2f}개/초")
    print()
    print("  📌 이것이 step2에서 GUI로 시각화할 시뮬레이션의 핵심 구조!")
    print("     → random.expovariate()는 '지수 분포' 랜덤 (공정 산포 모사)")
    print()


# ══════════════════════════════════════════════════════════════
#  메인 실행
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║     📚 SimPy 기초 학습 — 제조 시뮬레이션 입문           ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()
    print("  SimPy = '이산 사건 시뮬레이션' 라이브러리")
    print("  제조 공장의 부품 흐름, 설비 경쟁, 대기 현상을 모사합니다.")
    print()

    wait_for_enter("  Enter를 누르면 [예제 1: 시간의 흐름]을 시작합니다...")
    example1_timeout()

    wait_for_enter("  Enter를 누르면 [예제 2: 자원 경쟁]을 시작합니다...")
    example2_resource()

    wait_for_enter("  Enter를 누르면 [예제 3: 설비 대수 변경]을 시작합니다...")
    example3_capacity()

    wait_for_enter("  Enter를 누르면 [예제 4: 연속 투입]을 시작합니다...")
    example4_continuous()

    print("=" * 60)
    print("  ✅ Step 1 완료!")
    print("  다음: step2_simpy_gui.py (GUI 시각화)")
    print("=" * 60)
