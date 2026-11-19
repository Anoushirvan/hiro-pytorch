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


# ---------------------------------------------------------------- Ant tasks (official hiro_orig.gin + *_single.gin)
# High-level action = relative goal in qpos[:15]: (x, y, z, quaternion(4), 8 joint angles)
ANT_GOAL_RANGE = (10, 10, 0.5, 1, 1, 1, 1, 0.5, 0.3, 0.5, 0.3, 0.5, 0.3, 0.5, 0.3)
ANT_HIRO = dict(low_noise=1.0 / 30.0,   # official: sigma = 1.0 on actions in [-30, 30]
                high_noise=1.0)          # official META_EXPLORE_NOISE = 1.0


def _ant(maze_id):
    from .ant_maze import AntMazeEnv     # lazy import: PointMaze runs don't need mujoco
    return lambda: AntMazeEnv(maze_id)


ENV_SPECS.update({
    "AntMaze": dict(make=_ant("Maze"), goal_range=ANT_GOAL_RANGE, hiro_kwargs=ANT_HIRO, main_target="0_16",
                    eval_targets={"16_0": (16.0, 0.0), "16_16": (16.0, 16.0), "0_16": (0.0, 16.0)}),
    "AntPush": dict(make=_ant("Push"), goal_range=ANT_GOAL_RANGE, hiro_kwargs=ANT_HIRO, main_target="0_19",
                    eval_targets={"0_19": (0.0, 19.0)}),
    "AntFall": dict(make=_ant("Fall"), goal_range=ANT_GOAL_RANGE, hiro_kwargs=ANT_HIRO, main_target="0_27",
                    eval_targets={"0_27": (0.0, 27.0, 4.5)}),
})


def make_env(name: str):
    return ENV_SPECS[name]["make"]()
