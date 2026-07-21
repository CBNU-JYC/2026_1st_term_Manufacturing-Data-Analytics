# -*- coding: utf-8 -*-
"""
=============================================================
  Step 4: 고도화된 강화학습 (Maskable PPO + Reward Shaping)
=============================================================
Step 3의 문제를 해결하기 위해 두 가지 기법을 추가합니다:

  1. Action Masking — "불가능한 행동"을 사전에 차단
     → 이미 끝난 Job은 선택지에서 아예 제거

  2. Reward Shaping — "설비 유휴시간 최소화" 보상 설계
     → 에이전트가 "설비가 쉬지 않도록 촘촘히 배치"하는 법을 학습

실행: conda activate mfg_data && python step4_drl_advanced.py
=============================================================
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import time
from utils import MACHINES, JOB_DATA, print_problem, draw_gantt_console


# ══════════════════════════════════════════════════════════════
#  1. 고도화된 JSSP 강화학습 환경
# ══════════════════════════════════════════════════════════════

class JSSPMaskableEnv(gym.Env):
    """
    Action Masking과 Reward Shaping이 적용된 고도화 환경.

    Step 3 대비 개선점:
    ┌──────────────────────────────────────────────────────┐
    │ [Action Masking]                                    │
    │  • action_masks() 메서드가 유효한 행동만 True 반환    │
    │  • MaskablePPO가 이 정보를 활용하여 학습             │
    │                                                     │
    │ [확장된 상태(Observation)]                            │
    │  • 기존: 공정 진행도 + 설비/작업 가용시간              │
    │  • 추가: 각 Job의 남은 총 가공시간                    │
    │                                                     │
    │ [Reward Shaping]                                    │
    │  • 매 스텝: 설비 유휴시간(Idle) 만큼 마이너스 보상    │
    │  • 종료 시: (기준 Makespan - 실제 Makespan) 보상     │
    └──────────────────────────────────────────────────────┘
    """

    def __init__(self, job_data, machines):
        super().__init__()
        self.job_data = job_data
        self.machines = machines
        self.job_names = list(job_data.keys())
        self.num_jobs = len(self.job_names)
        self.num_machines = len(machines)

        self.action_space = spaces.Discrete(self.num_jobs)

        # 상태: 공정진행도 + 설비가용 + 작업가용 + 남은가공시간
        obs_dim = self.num_jobs * 3 + self.num_machines
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.job_op_idx = np.zeros(self.num_jobs, dtype=int)
        self.machine_avail = np.zeros(self.num_machines)
        self.job_avail = np.zeros(self.num_jobs)

        # 남은 총 가공시간 (에이전트에게 중요한 정보!)
        self.job_remaining = np.array([
            sum(t for _, t in self.job_data[j]) for j in self.job_names
        ], dtype=np.float32)

        self.log = []
        return self._get_obs(), {}

    def _get_obs(self):
        return np.concatenate([
            self.job_op_idx, self.machine_avail,
            self.job_avail, self.job_remaining
        ]).astype(np.float32)

    def action_masks(self):
        """
        ★ 핵심: 유효한 행동만 True로 반환
        → MaskablePPO는 False인 행동을 절대 선택하지 않음
        """
        return np.array([
            self.job_op_idx[j] < len(self.job_data[self.job_names[j]])
            for j in range(self.num_jobs)
        ], dtype=bool)

    def step(self, action):
        job_idx = action
        job_name = self.job_names[job_idx]
        ops = self.job_data[job_name]

        op_idx = self.job_op_idx[job_idx]
        m_name, p_time = ops[op_idx]
        m_idx = self.machines.index(m_name)

        # 시간 계산
        start = max(self.machine_avail[m_idx], self.job_avail[job_idx])
        end = start + p_time

        # ★ Reward Shaping: 설비 유휴시간만큼 페널티
        idle_time = max(0, self.job_avail[job_idx] - self.machine_avail[m_idx])
        reward = -idle_time

        # 상태 갱신
        self.machine_avail[m_idx] = end
        self.job_avail[job_idx] = end
        self.job_op_idx[job_idx] += 1
        self.job_remaining[job_idx] -= p_time

        self.log.append({
            'Job': job_name, 'Machine': m_name,
            'Start_Time': start, 'End_Time': end, 'Processing_Time': p_time
        })

        # 종료 확인
        done = bool(np.all(
            self.job_op_idx == [len(self.job_data[j]) for j in self.job_names]))

        if done:
            makespan = max(self.machine_avail)
            reward += (50 - makespan) * 2  # 최적해(27)에 가까울수록 높은 보상

        return self._get_obs(), reward, done, False, {}

    def get_result_df(self):
        if not self.log:
            return pd.DataFrame()
        return pd.DataFrame(self.log).sort_values(
            by=['Start_Time', 'Machine']).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════
#  2. Action Masking 래퍼
# ══════════════════════════════════════════════════════════════

def mask_fn(env):
    """ActionMasker 래퍼가 호출하는 함수"""
    return env.unwrapped.action_masks()


# ══════════════════════════════════════════════════════════════
#  3. Maskable PPO 학습 및 추론
# ══════════════════════════════════════════════════════════════

def run_maskable_ppo(job_data, machines, total_timesteps=50000, verbose=True):
    """
    Action Masking이 적용된 PPO로 학습 및 추론합니다.
    """
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
    from sb3_contrib.common.wrappers import ActionMasker

    if verbose:
        print(f"  🧠 Maskable PPO: {total_timesteps:,}스텝 학습 시작...")

    t0 = time.time()

    # 환경 생성 + 마스킹 래퍼
    raw_env = JSSPMaskableEnv(job_data, machines)
    env = ActionMasker(raw_env, mask_fn)

    # 모델 생성 및 학습
    model = MaskablePPO(
        MaskableActorCriticPolicy, env,
        verbose=0, ent_coef=0.01, learning_rate=0.0003
    )
    model.learn(total_timesteps=total_timesteps)
    train_time = time.time() - t0

    if verbose:
        print(f"     학습 완료! ({train_time:.1f}초)")

    # 추론
    obs, _ = env.reset()
    done = False

    while not done:
        masks = mask_fn(env)
        action, _ = model.predict(obs, action_masks=masks, deterministic=True)
        obs, reward, done, _, info = env.step(action)

    result_df = env.unwrapped.get_result_df()
    makespan = result_df['End_Time'].max() if not result_df.empty else 999

    if verbose:
        print(f"     ✅ Makespan: {makespan:.0f}분")

    return result_df, makespan, train_time


# ══════════════════════════════════════════════════════════════
#  4. 학습량 비교 실험
# ══════════════════════════════════════════════════════════════

def training_curve_experiment(job_data, machines, verbose=True):
    """
    학습량(timesteps)에 따른 Makespan 변화를 보여줍니다.
    → "학습을 더 하면 더 좋아지는가?"
    """
    if verbose:
        print()
        print("  ─── 학습량 비교 실험 ───")
        print()

    timestep_list = [5000, 10000, 20000, 50000, 100000]
    results = []

    for ts in timestep_list:
        df, ms, et = run_maskable_ppo(job_data, machines,
                                       total_timesteps=ts, verbose=False)
        results.append({'timesteps': ts, 'makespan': ms, 'time': et})
        if verbose:
            bar_len = max(0, int((50 - ms) / 50 * 30)) if ms < 999 else 0
            bar = "█" * bar_len + "░" * (30 - bar_len)
            status = f"{ms:.0f}분" if ms < 999 else "실패"
            print(f"  {ts:>8,}스텝 │ {bar} │ Makespan: {status} ({et:.1f}초)")

    if verbose:
        print()
        best = min(results, key=lambda x: x['makespan'])
        print(f"  📌 최선: {best['timesteps']:,}스텝에서 Makespan {best['makespan']:.0f}분")
        print(f"     ILP 최적해(27분) 대비 Gap: +{best['makespan']-27:.0f}분")

    return results


# ══════════════════════════════════════════════════════════════
#  메인 실행
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print_problem()

    print("=" * 60)
    print("  Step 4: 고도화된 강화학습 (Maskable PPO)")
    print("=" * 60)
    print()

    print("  ★ Step 3 대비 개선점:")
    print("    1. Action Masking: 불가능한 행동을 사전 차단")
    print("    2. Reward Shaping: 설비 유휴시간 최소화 유도")
    print("    3. 확장된 상태: 남은 가공시간 정보 추가")
    print()

    # ── Maskable PPO 실행 ──
    print("  ─── Maskable PPO (50,000 스텝) ───")
    df, ms, et = run_maskable_ppo(JOB_DATA, MACHINES, total_timesteps=50000)

    if df is not None and not df.empty:
        draw_gantt_console(df, f"Maskable PPO (Makespan: {ms:.0f})")

    # ── 학습량 비교 실험 ──
    training_curve_experiment(JOB_DATA, MACHINES)

    print()
    print("  📌 핵심 교훈:")
    print("     1. Action Masking = '도메인 지식'으로 AI를 가이드")
    print("     2. Reward Shaping = '무엇이 좋은 스케줄인지' AI에게 가르침")
    print("     3. 학습량이 충분해야 좋은 결과를 얻을 수 있음")
    print("     4. 하지만 ILP(수학적 최적해)를 항상 따라잡는 것은 아님!")
