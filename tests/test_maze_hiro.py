"""Step 3 tests: maze geometry/collisions and HIRO high-level bookkeeping."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from hiro_pytorch.envs.maze import LAYOUTS, PointMazeEnv, block_centres  # noqa: E402
from hiro_pytorch.hiro import HIRO, HIROConfig, rollout  # noqa: E402


def test_layout_coordinates_match_official():
    b = block_centres(LAYOUTS["Maze"], 8.0)
    walls = {tuple(x) for x in b}
    for free in [(0, 0), (8, 0), (16, 0), (16, 8), (16, 16), (8, 16), (0, 16)]:   # the ⊃ corridor
        assert free not in walls
    assert (0, 8) in walls and (8, 8) in walls                                       # the inner wall


def test_point_cannot_cross_inner_wall():
    env = PointMazeEnv(reset_noise=0.0)
    env.reset(seed=0, options={"target": (0, 8)})
    for _ in range(100):                     # push straight "up" towards the target through the wall
        obs, r, *_ = env.step(np.array([0.0, 1.0]))
    assert obs[1] < 2.0 - env.radius + 1e-6  # stopped at the wall face (wall spans y in [2, 6])
    assert r < -5.9                           # still ~6 away from (0, 8)


def test_point_can_walk_the_corridor():
    env = PointMazeEnv(reset_noise=0.0)
    env.reset(seed=0, options={"target": (0, 8)})
    for a, n in [((1, 0), 15), ((0, 1), 15), ((-1, 0), 15)]:   # right, up, left (sliding along walls)
        for _ in range(n):
            obs, r, term, trunc, info = env.step(np.array(a, dtype=np.float32))
    x, y = obs[0], obs[1]
    assert y > 6 and x < 2                   # reached the end cell of the ⊃ (around (0, 8))


def _agent(env, c=10):
    return HIRO(obs_dim=6, state_dim=4, act_low=env.action_space.low, act_high=env.action_space.high,
                goal_idx=env.goal_idx, goal_low=[-6, -6], goal_high=[6, 6], cfg=HIROConfig(c=c))


def test_rollout_makes_one_decision_every_c_steps_and_subgoal_is_absolute():
    env = PointMazeEnv()
    tr = rollout(env, _agent(env), explore=True, seed=0)
    assert len(tr["positions"]) == 201 and len(tr["subgoals"]) == 20
    t, start, target_abs = tr["subgoals"][3]
    assert t == 30 and np.allclose(start, tr["positions"][30], atol=1e-5)
    assert np.all(np.abs(target_abs - start) <= 6 + 1e-5)


def test_high_buffer_stores_sequences_and_scaled_reward():
    env = PointMazeEnv()
    ag = _agent(env, c=3)
    obs = np.zeros(6, np.float32)
    states = [np.full(4, k, np.float32) for k in range(4)]
    ag.store_high(obs, np.ones(2), -20.0, obs, False, states, [np.zeros(2)] * 3)
    b = ag.high_buf.sample(2)
    assert b["states"].shape == (2, 4, 4) and b["actions"].shape == (2, 3, 2)
    assert np.isclose(b["rew"][0, 0].item(), -2.0)          # 0.1 * sum of env rewards
