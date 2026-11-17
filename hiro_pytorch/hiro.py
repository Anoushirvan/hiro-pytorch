"""HIRO agent: two TD3 learners glued by goals (paper Sec. 3, Fig. 2).

  high level  mu_hi(obs)      -> g_t in goal space, sampled every c steps     obs = [s, task target]
  low level   mu_lo(s, g)     -> a_t, every step                               input = [s (x,y zeroed), g]
  between high-level decisions the goal evolves with h (Eq. 2); low reward = Eq. 3.
  high-level reward = hi_reward_scale * sum of c env rewards (official train.py:192 uses 0.1).

Step 3: everything except the off-policy correction (`correction=False`).
Step 4 fills in `off_policy_correction`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from .buffers import HighReplayBuffer, ReplayBuffer
from .goals import goal_transition, intrinsic_reward, low_level_input
from .td3 import TD3, TD3Config


@dataclass
class HIROConfig:
    c: int = 10
    hi_reward_scale: float = 0.1
    low_noise: float = 0.1          # exploration std, fraction of the action half-range
    high_noise: float = 1.0         # exploration std, ABSOLUTE (official META_EXPLORE_NOISE = 1.0)
    zero_idx: tuple = (0, 1)        # ZERO_OBS
    correction: bool = False        # Step 4
    n_candidates: int = 10          # Step 4: 8 Gaussian + (s_{t+c}-s_t) + original g
    candidate_std: float = 0.5      # Step 4: fraction of goal half-range (official InverseDynamics.sample sc=0.5)
    batch_size: int = 100
    low: TD3Config = field(default_factory=TD3Config)
    high: TD3Config = field(default_factory=TD3Config)


class HIRO:
    def __init__(self, obs_dim: int, state_dim: int, act_low, act_high, goal_idx, goal_low, goal_high,
                 cfg: HIROConfig = HIROConfig(), device: str = "cpu"):
        self.cfg, self.device = cfg, torch.device(device)
        self.state_dim, self.goal_idx = state_dim, np.asarray(goal_idx)
        self.act_low, self.act_high = np.asarray(act_low, np.float32), np.asarray(act_high, np.float32)
        self.goal_low, self.goal_high = np.asarray(goal_low, np.float32), np.asarray(goal_high, np.float32)
        act_dim, goal_dim = len(self.act_low), len(self.goal_low)

        self.low = TD3(state_dim + goal_dim, act_dim, act_low, act_high, cfg.low, device)
        self.high = TD3(obs_dim, goal_dim, goal_low, goal_high, cfg.high, device)
        self.low_buf = ReplayBuffer(state_dim + goal_dim, act_dim, device=device)
        self.high_buf = HighReplayBuffer(obs_dim, goal_dim, state_dim, act_dim, cfg.c, device=device)
        self.low_noise_std = cfg.low_noise * (self.act_high - self.act_low) / 2
        self.n_relabel_changed = 0      # Step 4 diagnostics

    # ------------------------------------------------------------------ acting
    def state(self, obs):
        return obs[..., : self.state_dim]

    def select_goal(self, obs, explore: bool) -> np.ndarray:
        return self.high.act(obs, self.cfg.high_noise if explore else 0.0)

    def random_goal(self) -> np.ndarray:
        return np.random.uniform(self.goal_low, self.goal_high).astype(np.float32)

    def select_action(self, obs, goal, explore: bool) -> np.ndarray:
        x = low_level_input(self.state(obs), goal, self.cfg.zero_idx)
        return self.low.act(x, self.low_noise_std if explore else 0.0)

    def next_goal(self, obs, goal, next_obs):
        return goal_transition(self.state(obs), goal, self.state(next_obs), self.goal_idx)

    # ------------------------------------------------------------------ storing
    def store_low(self, obs, goal, action, next_obs, next_goal, terminated: bool) -> float:
        s, s2 = self.state(obs), self.state(next_obs)
        r = float(intrinsic_reward(s, goal, s2, self.goal_idx))
        z = self.cfg.zero_idx
        self.low_buf.add(low_level_input(s, goal, z), action, r, low_level_input(s2, next_goal, z), terminated)
        return r

    def store_high(self, obs, goal, env_reward_sum, next_obs, terminated, states, actions):
        self.high_buf.add(obs, goal, self.cfg.hi_reward_scale * env_reward_sum, next_obs, terminated,
                          np.asarray(states)[:, : self.state_dim], actions)

    # ------------------------------------------------------------------ learning
    def update_low(self) -> dict:
        return self.low.update(self.low_buf.sample(self.cfg.batch_size))

    def update_high(self) -> dict:
        batch = self.high_buf.sample(self.cfg.batch_size)
        if self.cfg.correction:
            batch["act"] = self.off_policy_correction(batch)
        return self.high.update(batch)

    def off_policy_correction(self, batch) -> torch.Tensor:
        raise NotImplementedError("Step 4")

    # ------------------------------------------------------------------ io
    def save(self, path: str) -> None:
        torch.save({"low": self.low.state_dict(), "high": self.high.state_dict()}, path)

    def load(self, path: str) -> None:
        d = torch.load(path, map_location=self.device)
        self.low.actor.load_state_dict(d["low"]["actor"])
        self.high.actor.load_state_dict(d["high"]["actor"])


@torch.no_grad()
def rollout(env, agent: HIRO, explore: bool = False, reset_options=None, seed=None):
    """Run one episode with the hierarchical policy. Returns a trajectory dict for evaluation/visualisation:
    positions (T+1, 2), subgoal_abs (list of (t, start_xy, absolute target s_t+g_t)), final info."""
    obs, _ = env.reset(seed=seed, options=reset_options)
    c, gi = agent.cfg.c, agent.goal_idx
    pos, subgoals, t, done, info = [obs[gi].copy()], [], 0, False, {}
    while not done:
        if t % c == 0:
            g = agent.select_goal(obs, explore)
            subgoals.append((t, obs[gi].copy(), obs[gi] + g))
        a = agent.select_action(obs, g, explore)
        obs2, _, term, trunc, info = env.step(a)
        g = agent.next_goal(obs, g, obs2)
        obs, t, done = obs2, t + 1, term or trunc
        pos.append(obs[gi].copy())
    return dict(positions=np.array(pos), subgoals=subgoals, info=info)
