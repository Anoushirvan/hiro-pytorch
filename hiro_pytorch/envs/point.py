"""A tiny pure-numpy 2D point mass (fast, no MuJoCo) used for Steps 2-4.

State  s = (x, y, vx, vy).   Action a in [-1, 1]^2 (acceleration).
Dynamics (per step):  v <- 0.7 v + 0.3 a ;  p <- p + v      => top speed 1 unit/step.
Having velocity in the state makes the low-level task non-trivial: to *stop* on a goal it has to brake.

Goal space for HIRO = position (x, y)  -> goal_idx = [0, 1].
Walls (for PointMaze) are added in Step 3; with `walls=None` it is an open plane.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np


class PointEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, max_steps: int = 100, damping: float = 0.7):
        self.max_steps = max_steps
        self.damping = damping
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.goal_idx = np.array([0, 1])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.pos = np.zeros(2)
        self.vel = np.zeros(2)
        self.t = 0
        return self._obs(), {}

    def _obs(self):
        return np.concatenate([self.pos, self.vel]).astype(np.float32)

    def step(self, action):
        a = np.clip(action, -1.0, 1.0)
        self.vel = self.damping * self.vel + (1 - self.damping) * a
        self.pos = self.pos + self.vel
        self.t += 1
        # No task reward in the open plane: Step 2 trains the low level on intrinsic reward only.
        return self._obs(), 0.0, False, self.t >= self.max_steps, {}
