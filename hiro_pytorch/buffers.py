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
