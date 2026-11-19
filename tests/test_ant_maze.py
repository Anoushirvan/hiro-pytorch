"""Step 5 tests: Ant maze environments behave like the official ones (geometry, obs layout, physics)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from hiro_pytorch.envs.ant_maze import AntMazeEnv  # noqa: E402


def _launch(env, vy, steps=30):
    """Give the torso a large +y velocity and let physics run with zero torques."""
    env.data.qvel[1] = vy
    for _ in range(steps):
        obs, *_ = env.step(np.zeros(8))
    return obs


def test_obs_layout_and_spaces():
    env = AntMazeEnv("Maze")
    obs, _ = env.reset(seed=0, options={"target": (0, 16)})
    assert obs.shape == (32,) and env.action_space.shape == (8,)
    assert np.allclose(env.action_space.high, 30) and np.allclose(env.action_space.low, -30)
    for _ in range(3):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
    assert np.allclose(obs[:15], env.data.qpos[:15], atol=1e-5)
    assert np.allclose(obs[15:29], env.data.qvel[:14], atol=1e-5)
    assert np.isclose(obs[29], 0.003) and np.allclose(obs[30:], [0, 16])
    assert np.isclose(r, -np.linalg.norm(obs[:2] - [0, 16]), atol=1e-4) and not term


def test_episode_is_500_steps_and_dt():
    env = AntMazeEnv("Maze")
    env.reset(seed=0)
    assert np.isclose(env.model.opt.timestep * env.frame_skip, 0.1)
    n = 0
    while True:
        *_, trunc, _ = env.step(np.zeros(8))
        n += 1
        if trunc:
            break
    assert n == 500


def test_maze_wall_blocks_the_ant():
    env = AntMazeEnv("Maze")
    env.reset(seed=0)
    obs = _launch(env, vy=15.0)            # thrown straight at the inner wall (cell at y=8, face at y=4)
    assert obs[1] < 4.0


def test_push_block_is_movable():
    env = AntMazeEnv("Push")
    env.reset(seed=0)
    bid = env.model.body("movable_2_2").id
    y0 = env.data.xpos[bid][1]
    _launch(env, vy=15.0)                  # ant hits the red block above it
    assert env.data.xpos[bid][1] > y0 + 0.1


def test_fall_chasm_drops_the_ant():
    env = AntMazeEnv("Fall")
    obs, _ = env.reset(seed=0)
    assert obs[2] > 4.0                    # elevated platform
    env.data.qpos[:2] = [0.0, 16.0]        # teleport over the chasm cell (row 3)
    for _ in range(30):
        obs, *_ = env.step(np.zeros(8))
    assert obs[2] < 2.0
