"""Replay buffers.

Step 1 only needs a plain transition buffer (s, a, r, s', not_done) for flat TD3.
The HIRO-specific high-level buffer (which also stores the c-step state/action
sequences needed by the off-policy correction) is added in Step 3.
"""
from __future__ import annotations

import numpy as np
import torch


class ReplayBuffer:
    """Fixed-size circular buffer of transitions stored as numpy arrays.

    Official code: `agents/circular_buffer.py`, size 200 000 (`configs/train_uvf.gin`).
    """

    def __init__(self, obs_dim: int, act_dim: int, capacity: int = 200_000, device: str = "cpu"):
        self.capacity = capacity
        self.device = torch.device(device)
        self.ptr = 0      # next write position
        self.size = 0     # number of valid entries
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rew = np.zeros((capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        # not_done = 1 - terminated. A *time-limit truncation* is NOT a terminal state,
        # so we still bootstrap from s' there (important for 500-step HIRO episodes).
        self.not_done = np.zeros((capacity, 1), dtype=np.float32)

    def add(self, obs, act, rew, next_obs, terminated: bool) -> None:
        i = self.ptr
        self.obs[i] = obs
        self.act[i] = act
        self.rew[i] = rew
        self.next_obs[i] = next_obs
        self.not_done[i] = 1.0 - float(terminated)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        idx = np.random.randint(0, self.size, size=batch_size)
        to_t = lambda x: torch.as_tensor(x[idx], device=self.device)
        return dict(obs=to_t(self.obs), act=to_t(self.act), rew=to_t(self.rew),
                    next_obs=to_t(self.next_obs), not_done=to_t(self.not_done))

    def __len__(self) -> int:
        return self.size


class HighReplayBuffer:
    """High-level buffer. Besides the usual (obs_t, g_t, R_{t:t+c-1}, obs_{t+c}) it keeps the raw low-level
    sequences that the off-policy correction (Step 4, paper Eq. 5) needs to re-label g_t:

        states  : s_t, s_{t+1}, ..., s_{t+c}        shape (c+1, state_dim)   (state part of obs only)
        actions : a_t, ..., a_{t+c-1}               shape (c,   act_dim)

    This is the paper's transition tuple (s_{t:t+c-1}, g_{t:t+c-1}, a_{t:t+c-1}, R_{t:t+c-1}, s_{t+c}) (Sec. 3.1);
    the intermediate goals g_{t+1..} are not stored because they are recomputed from g_t with h (Eq. 2).
    """

    def __init__(self, obs_dim: int, goal_dim: int, state_dim: int, act_dim: int, c: int,
                 capacity: int = 200_000, device: str = "cpu"):
        self.capacity, self.c, self.device = capacity, c, torch.device(device)
        self.ptr = self.size = 0
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.goal = np.zeros((capacity, goal_dim), dtype=np.float32)
        self.rew = np.zeros((capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.not_done = np.zeros((capacity, 1), dtype=np.float32)
        self.states = np.zeros((capacity, c + 1, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, c, act_dim), dtype=np.float32)

    def add(self, obs, goal, rew, next_obs, terminated: bool, states, actions) -> None:
        assert len(states) == self.c + 1 and len(actions) == self.c, "segments must be exactly c steps"
        i = self.ptr
        self.obs[i], self.goal[i], self.rew[i] = obs, goal, rew
        self.next_obs[i], self.not_done[i] = next_obs, 1.0 - float(terminated)
        self.states[i], self.actions[i] = states, actions
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        idx = np.random.randint(0, self.size, size=batch_size)
        to_t = lambda x: torch.as_tensor(x[idx], device=self.device)
        # key "act" = the high-level action (the goal), so the batch can be fed to TD3.update unchanged
        return dict(obs=to_t(self.obs), act=to_t(self.goal), rew=to_t(self.rew), next_obs=to_t(self.next_obs),
                    not_done=to_t(self.not_done), states=to_t(self.states), actions=to_t(self.actions))

    def __len__(self) -> int:
        return self.size
