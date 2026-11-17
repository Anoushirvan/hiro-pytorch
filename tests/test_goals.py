"""Unit tests for Step 2: goal transition h (Eq. 2), intrinsic reward (Eq. 3), low-level input."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from hiro_pytorch.envs.point import PointEnv  # noqa: E402
from hiro_pytorch.goals import goal_transition, intrinsic_reward, low_level_input  # noqa: E402

GI = np.array([0, 1])


def test_h_keeps_absolute_target_fixed():
    rng = np.random.default_rng(0)
    s = rng.normal(size=4)
    g = np.array([3.0, -2.0])
    target = s[GI] + g
    for _ in range(10):                      # agent wanders randomly; target must not move
        s2 = s + rng.normal(size=4)
        g = goal_transition(s, g, s2, GI)
        s = s2
        assert np.allclose(s[GI] + g, target)


def test_reward_is_negative_distance_to_target():
    s = np.array([1.0, 1.0, 9.0, 9.0])       # velocity dims must be ignored
    s2 = np.array([2.0, 1.0, -5.0, 0.0])
    g = np.array([4.0, 5.0])                 # absolute target (5, 6); landed at (2, 1)
    assert np.isclose(intrinsic_reward(s, g, s2, GI), -np.hypot(3, 5))
    assert intrinsic_reward(s, s2[GI] - s[GI], s2, GI) > -1e-4   # goal reached => r ~ 0


def test_reward_numpy_torch_agree_and_batch():
    rng = np.random.default_rng(1)
    s, s2, g = rng.normal(size=(8, 4)), rng.normal(size=(8, 4)), rng.normal(size=(8, 2))
    r_np = intrinsic_reward(s, g, s2, GI)
    r_t = intrinsic_reward(torch.tensor(s), torch.tensor(g), torch.tensor(s2), GI)
    assert r_np.shape == (8,) and np.allclose(r_np, r_t.numpy())


def test_low_level_input_zeroes_xy_without_mutating():
    s, g = np.array([5.0, 6.0, 0.1, 0.2]), np.array([1.0, 2.0])
    x = low_level_input(s, g)
    assert np.allclose(x, [0, 0, 0.1, 0.2, 1, 2]) and s[0] == 5.0
    xt = low_level_input(torch.tensor(s), torch.tensor(g))
    assert torch.allclose(xt, torch.tensor(x, dtype=xt.dtype))
    assert np.allclose(low_level_input(s, g, zero_idx=()), [5, 6, 0.1, 0.2, 1, 2])


def test_point_env_top_speed_is_one():
    env = PointEnv()
    s, _ = env.reset(seed=0)
    for _ in range(50):
        s, *_ = env.step(np.array([1.0, 0.0]))
    assert np.isclose(s[2], 1.0, atol=1e-6) and s[3] == 0.0
