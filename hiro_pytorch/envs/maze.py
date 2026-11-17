"""Maze layouts (copied from official `environments/maze_env_utils.py::construct_maze`) and a
pure-numpy PointMaze environment.

Layout -> world coordinates (official `maze_env.py`): cell (i, j) has centre
    x = j * scaling - x_r,   y = i * scaling - y_r
where (x_r, y_r) is the centre of the start cell 'r', so the agent starts at (0, 0).
For 'Maze' (the paper's ⊃-shaped Ant Maze) with scaling s the free cells are
    (0,0) (s,0) (2s,0) | (2s,s) | (2s,2s) (s,2s) (0,2s)
so the "end of the maze" target is (0, 2s): (0, 16) for the ant (s=8), (0, 8) for our point (s=4).
A greedy agent heading straight for (0, 2s) gets stuck against the wall at y = s: this is the
deceptive local optimum that makes the task need exploration / temporal abstraction.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np

LAYOUTS = {
    "Maze": [
        [1, 1, 1, 1, 1],
        [1, "r", 0, 0, 1],
        [1, 1, 1, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1],
    ],
}


def block_centres(layout, scaling: float) -> np.ndarray:
    """World-coordinate centres (x, y) of all wall cells (value 1)."""
    (ir, jr), = [(i, j) for i, row in enumerate(layout) for j, v in enumerate(row) if v == "r"]
    return np.array([[(j - jr) * scaling, (i - ir) * scaling]
                     for i, row in enumerate(layout) for j, v in enumerate(row) if v == 1], dtype=np.float64)


class PointMazeEnv(gym.Env):
    """Point mass (same dynamics as envs/point.py) inside a maze of square wall blocks.

    obs = (x, y, vx, vy, target_x, target_y)   -> `state_dim = 4` (the part HIRO goals live in / the
                                                  low level sees), the target is appended for the high level.
    reward = -||target - (x, y)||               (paper App. B.2: negative L2 distance to target)
    Training targets ~ U(target_low, target_high) (official point_maze.gin: ((-2,-2),(10,10)));
    evaluation passes options={"target": (0, 8)} etc.
    success = final distance < success_radius (we use 1.5 = inside the target cell, see plan.md).
    """

    metadata = {"render_modes": []}
    state_dim = 4
    goal_idx = np.array([0, 1])

    def __init__(self, layout: str = "Maze", scaling: float = 4.0, max_steps: int = 200,
                 damping: float = 0.7, radius: float = 0.3, reset_noise: float = 0.1,
                 target_low=(-2.0, -2.0), target_high=(10.0, 10.0), success_radius: float = 1.5):
        self.layout, self.scaling = LAYOUTS[layout], scaling
        self.blocks = block_centres(self.layout, scaling)
        self.max_steps, self.damping, self.radius, self.reset_noise = max_steps, damping, radius, reset_noise
        self.target_low, self.target_high = np.array(target_low), np.array(target_high)
        self.success_radius = success_radius
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

    # ------------------------------------------------------------------ physics
    def collides(self, p) -> bool:
        half = self.scaling / 2 + self.radius
        d = np.abs(self.blocks - p)
        return bool(np.any((d[:, 0] < half) & (d[:, 1] < half)))

    def step(self, action):
        a = np.clip(action, -1.0, 1.0)
        self.vel = self.damping * self.vel + (1 - self.damping) * a
        # move one axis at a time -> the point slides along walls instead of sticking (speed<=1 < block, no tunnelling)
        for k in range(2):
            p = self.pos.copy()
            p[k] += self.vel[k]
            if self.collides(p):
                self.vel[k] = 0.0
            else:
                self.pos = p
        self.t += 1
        dist = float(np.linalg.norm(self.target - self.pos))
        info = {"dist": dist, "success": dist < self.success_radius}
        return self._obs(), -dist, False, self.t >= self.max_steps, info

    # ------------------------------------------------------------------ episode
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.pos = self.np_random.uniform(-self.reset_noise, self.reset_noise, size=2)
        self.vel = np.zeros(2)
        self.t = 0
        if options and "target" in options:
            self.target = np.asarray(options["target"], dtype=np.float64)
        else:
            self.target = self.np_random.uniform(self.target_low, self.target_high)
        return self._obs(), {}

    def _obs(self):
        return np.concatenate([self.pos, self.vel, self.target]).astype(np.float32)
