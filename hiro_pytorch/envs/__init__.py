"""Environment registry: per-env settings shared by the HIRO and flat-TD3 training scripts."""
from __future__ import annotations

from .maze import PointMazeEnv

ENV_SPECS = {
    "PointMaze": dict(
        make=lambda: PointMazeEnv(),
        goal_range=(6.0, 6.0),               # high-level action space: relative (dx, dy) in [-6, 6]^2
        eval_targets={"8_0": (8.0, 0.0), "8_8": (8.0, 8.0), "0_8": (0.0, 8.0)},
        main_target="0_8",                   # "end of the maze" (paper reports this one for Ant Maze)
        hiro_kwargs=dict(low_noise=0.1, high_noise=1.0),
    ),
}


def make_env(name: str):
    return ENV_SPECS[name]["make"]()
