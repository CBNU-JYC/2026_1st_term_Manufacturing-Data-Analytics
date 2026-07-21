# -*- coding: utf-8 -*-
"""
=============================================================
  Step 3: 기초 강화학습 스케줄링 (DRL Basic)
=============================================================
"AI가 스스로 스케줄링 규칙을 학습할 수 있을까?"

이 Step에서는 두 가지를 비교합니다:
  1. Random Agent: 무작위로 작업을 선택 (기준선)
  2. PPO Agent:    신경망이 학습한 정책으로 작업을 선택

핵심 교훈:
  → 순수 PPO는 "이미 끝난 작업을 또 선택"하는 실수를 합니다
  → 이것이 Step 4에서 Action Masking이 필요한 이유입니다!

실행: conda activate mfg_data && python step3_drl_basic.py
=============================================================
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import time
from utils import MACHINES, JOB_DATA, print_problem, draw_gantt_console


# ══════════════════════════════════════════════════════════════
#  1. JSSP 강화학습 환경 (Gymnasium 인터페이스)
# ══════════════════════════════════════════════════════════════

class JSSPGymEnv(gym.Env):
    """
    Job Shop Scheduling을 강화학습 환경으로 변환합니다.

    ┌─────────────────────────────────────────────────────┐
    │  상태(State):  [각 Job의 진행도, 설비/작업 가용시간]  │
    │  행동(Action): "다음에 스케줄링할 Job 번호 선택"      │
    │  보상(Reward): 가공시간만큼 마이너스 (빠를수록 좋음)   │
    │  종료(Done):   모든 Job의 모든 공정이 완료되면 True   │
    └─────────────────────────────────────────────────────┘
    """

    def __init__(self, job_data, machines):
        super().__init__()
        self.job_data = job_data
        self.machines = machines
        self.job_names = list(job_data.keys())
        self.num_jobs = len(self.job_names)
        self.num_machines = len(machines)

        # Action: Job 인덱스 선택 (0, 1, 2, ...)
        self.action_space = spaces.Discrete(self.num_jobs)

        # State: [공정 진행도(num_jobs) + 설비 가용시간(num_machines) + 작업 가용시간(num_jobs)]
        obs_dim = self.num_jobs + self.num_machines + self.num_jobs
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.job_op_idx = np.zeros(self.num_jobs, dtype=int)
        self.machine_avail = np.zeros(self.num_machines)
        self.job_avail = np.zeros(self.num_jobs)
        self.log = []
        return self._get_obs(), {}

    def _get_obs(self):
        return np.concatenate([
            self.job_op_idx, self.machine_avail, self.job_avail
        ]).astype(np.float32)

    def step(self, action):
        job_idx = action
        job_name = self.job_names[job_idx]
        ops = self.job_data[job_name]

        # ⚠️ 이미 끝난 Job을 선택한 경우 → 강한 페널티 + 에피소드 종료
        if self.job_op_idx[job_idx] >= len(ops):
            return self._get_obs(), -100.0, True, False, {"msg": "Invalid"}

        op_idx = self.job_op_idx[job_idx]
        m_name, p_time = ops[op_idx]
        m_idx = self.machines.index(m_name)

        # 시작 시간 = max(설비가 비는 시간, Job이 준비되는 시간)
        start = max(self.machine_avail[m_idx], self.job_avail[job_idx])
        end = start + p_time

        # 상태 갱신
        self.machine_avail[m_idx] = end
        self.job_avail[job_idx] = end
        self.job_op_idx[job_idx] += 1

        self.log.append({
            'Job': job_name, 'Machine': m_name,
            'Start_Time': start, 'End_Time': end, 'Processing_Time': p_time
        })

        # 보상: 가공시간만큼 마이너스 (빨리 끝낼수록 좋음)
        reward = -p_time

        # 종료 확인
        done = bool(np.all(
            self.job_op_idx == [len(self.job_data[j]) for j in self.job_names]))

        if done:
            reward += 50.0  # 성공적 완료 보너스

        return self._get_obs(), reward, done, False, {}

    def get_result_df(self):
        if not self.log:
            return pd.DataFrame()
        return pd.DataFrame(self.log).sort_values(
            by=['Start_Time', 'Machine']).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════
#  2. Random Agent (기준선)
# ══════════════════════════════════════════════════════════════

def run_random_agent(job_data, machines, n_episodes=50, verbose=True):
    """
    무작위로 행동을 선택하는 에이전트입니다.
    여러 번 시도해서 그 중 가장 좋은 결과를 반환합니다.
    """
    if verbose:
        print("  🎲 Random Agent: 무작위 스케줄링 시작...")

    t0 = time.time()
    best_makespan = float('inf')
    best_df = None
    success_count = 0

    for ep in range(n_episodes):
        env = JSSPGymEnv(job_data, machines)
        obs, _ = env.reset()
        done = False

        while not done:
            action = env.action_space.sample()
            obs, reward, done, _, info = env.step(action)
            if info.get("msg") == "Invalid":
                break

        df = env.get_result_df()
        if not df.empty and len(df) == sum(len(ops) for ops in job_data.values()):
            success_count += 1
            ms = df['End_Time'].max()
            if ms < best_makespan:
                best_makespan = ms
                best_df = df

    elapsed = time.time() - t0

    if verbose:
        print(f"     성공률: {success_count}/{n_episodes} ({success_count/n_episodes*100:.0f}%)")
        if best_df is not None:
            print(f"     최고 Makespan: {best_makespan:.0f}분 ({elapsed:.3f}초)")
        else:
            print(f"     ❌ 유효한 스케줄을 생성하지 못했습니다.")

    return best_df, best_makespan if best_df is not None else 999, elapsed


# ══════════════════════════════════════════════════════════════
#  3. PPO Agent (기초 강화학습)
# ══════════════════════════════════════════════════════════════

def run_ppo_agent(job_data, machines, total_timesteps=20000, verbose=True):
    """
    PPO(Proximal Policy Optimization)로 학습한 에이전트입니다.

    문제점: "이미 끝난 Job을 또 선택"하는 Invalid Action이 빈번히 발생합니다.
    → 이것이 Step 4에서 Action Masking이 필요한 이유!
    """
    from stable_baselines3 import PPO

    if verbose:
        print(f"  🧠 PPO Agent: {total_timesteps:,}스텝 학습 시작...")

    t0 = time.time()
    env = JSSPGymEnv(job_data, machines)
    model = PPO("MlpPolicy", env, verbose=0, learning_rate=0.001)
    model.learn(total_timesteps=total_timesteps)
    train_time = time.time() - t0

    if verbose:
        print(f"     학습 완료! ({train_time:.1f}초)")
        print(f"     학습된 모델로 추론 중...")

    # 추론
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)
        if info.get("msg") == "Invalid":
            if verbose:
                print(f"     ⚠️ Invalid Action 발생! (이미 끝난 Job을 선택)")
            break

    result_df = env.get_result_df()
    total_ops = sum(len(ops) for ops in job_data.values())

    if result_df.empty or len(result_df) < total_ops:
        if verbose:
            print(f"     ❌ 스케줄링 실패 (완료된 공정: {len(result_df)}/{total_ops})")
            print(f"     → Action Masking 없이는 유효한 행동을 보장할 수 없습니다!")
        return result_df, 999, train_time

    makespan = result_df['End_Time'].max()
    if verbose:
        print(f"     ✅ Makespan: {makespan:.0f}분")

    return result_df, makespan, train_time


# ══════════════════════════════════════════════════════════════
#  메인 실행
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print_problem()

    print("=" * 60)
    print("  Step 3: 기초 강화학습 (Random vs PPO)")
    print("=" * 60)
    print()

    # ── Random Agent ──
    print("  ─── 실험 1: Random Agent ───")
    rand_df, rand_ms, rand_time = run_random_agent(JOB_DATA, MACHINES, n_episodes=100)
    if rand_df is not None:
        draw_gantt_console(rand_df, f"Random Agent Best (Makespan: {rand_ms:.0f})")
    print()

    # ── PPO Agent ──
    print("  ─── 실험 2: PPO Agent (20,000 스텝) ───")
    ppo_df, ppo_ms, ppo_time = run_ppo_agent(JOB_DATA, MACHINES, total_timesteps=20000)
    if ppo_df is not None and ppo_ms < 999:
        draw_gantt_console(ppo_df, f"PPO Agent (Makespan: {ppo_ms:.0f})")
    print()

    # ── 비교 요약 ──
    print("  ┌──────────────────┬────────────┬──────────────────────────────┐")
    print("  │      기법         │ Makespan   │ 비고                          │")
    print("  ├──────────────────┼────────────┼──────────────────────────────┤")
    print(f"  │ ILP 최적해        │     27분   │ 수학적 증명 (Ground Truth)     │")
    if rand_df is not None:
        print(f"  │ Random Agent     │ {rand_ms:>6.0f}분   │ 100회 중 최선 (우연에 의존)    │")
    print(f"  │ PPO (기초)        │ {ppo_ms:>6.0f}분   │ ", end="")
    if ppo_ms >= 999:
        print("실패 (Invalid Action 문제)   │")
    else:
        print(f"학습 {ppo_time:.1f}초                    │")
    print("  └──────────────────┴────────────┴──────────────────────────────┘")
    print()
    print("  📌 PPO의 한계:")
    print("     '이미 끝난 Job을 또 선택'하는 실수 → 에피소드 강제 종료")
    print("     → 해결책: Step 4의 Action Masking (불가능한 행동을 사전 차단)")
