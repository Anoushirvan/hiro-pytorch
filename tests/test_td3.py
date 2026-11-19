"""Unit tests for Step 1: replay buffer + DDPG/TD3 update mechanics."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from hiro_pytorch.buffers import ReplayBuffer  # noqa: E402
from hiro_pytorch.td3 import TD3, TD3Config  # noqa: E402


def test_buffer_wraps_and_samples():
    buf = ReplayBuffer(obs_dim=3, act_dim=2, capacity=5)
    for i in range(7):
        buf.add(np.full(3, i), np.zeros(2), float(i), np.full(3, i + 1), terminated=(i == 6))
    assert len(buf) == 5 and buf.ptr == 2           # circular: entries 5,6 overwrote 0,1
    assert buf.rew[0, 0] == 5 and buf.not_done[1, 0] == 0
    b = buf.sample(8)
    assert b["obs"].shape == (8, 3) and b["act"].shape == (8, 2) and b["rew"].shape == (8, 1)


def test_actor_respects_action_bounds():
    agent = TD3(4, 2, act_low=[-30, -1], act_high=[30, 1])
    a = agent.act(np.random.randn(4) * 100, noise_std=1000)
    assert np.all(a >= [-30, -1]) and np.all(a <= [30, 1])


def test_target_uses_min_of_twin_critics():
    torch.manual_seed(0)
    agent = TD3(3, 1, [-1], [1], TD3Config(target_noise=0.0))
    s2, r, nd = torch.randn(16, 3), torch.randn(16, 1), torch.ones(16, 1)
    y = agent._target_q(r, s2, nd)
    a2 = agent.actor_target(s2)
    q = torch.min(agent.critic1_target(s2, a2), agent.critic2_target(s2, a2))
    assert torch.allclose(y, r + 0.99 * q)


def test_terminal_transition_does_not_bootstrap():
    agent = TD3(3, 1, [-1], [1], TD3Config.ddpg())
    r = torch.ones(4, 1)
    y = agent._target_q(r, torch.randn(4, 3), torch.zeros(4, 1))
    assert torch.allclose(y, r)


def test_policy_delay_and_soft_update():
    agent = TD3(3, 1, [-1], [1], TD3Config(policy_delay=2, tau=0.5))
    batch = dict(obs=torch.randn(8, 3), act=torch.rand(8, 1) * 2 - 1, rew=torch.randn(8, 1),
                 next_obs=torch.randn(8, 3), not_done=torch.ones(8, 1))
    actor_before = [p.clone() for p in agent.actor.parameters()]
    target_before = [p.clone() for p in agent.actor_target.parameters()]
    info = agent.update(batch)                       # update 1: critic only
    assert "actor_loss" not in info
    assert all(torch.equal(a, b) for a, b in zip(actor_before, agent.actor.parameters()))
    info = agent.update(batch)                       # update 2: actor + soft target update
    assert "actor_loss" in info
    new = list(agent.actor.parameters())
    for tb, tp, p in zip(target_before, agent.actor_target.parameters(), new):
        assert torch.allclose(tp, 0.5 * tb + 0.5 * p)


def test_ddpg_config_has_single_critic():
    agent = TD3(3, 1, [-1], [1], TD3Config.ddpg())
    assert not hasattr(agent, "critic2") and agent.cfg.policy_delay == 1


def test_official_init_starts_at_action_centre_and_q_near_zero():
    agent = TD3(10, 3, act_low=[-30, -30, -30], act_high=[30, 30, 30])
    o = torch.randn(64, 10) * 5
    a = agent.actor(o)
    assert a.abs().max() < 3.0                               # tanh(small) * 30 -> close to 0, not saturated
    assert agent.critic1(o, a).abs().max() < 1.0


def test_legacy_architecture_flag():
    cfg = TD3Config(final_init=0.0, action_in_first=False)
    agent = TD3(10, 3, [-1] * 3, [1] * 3, cfg)
    assert agent.critic1.l1.in_features == 10                # action only enters at layer 2
    assert TD3(10, 3, [-1] * 3, [1] * 3).critic1.l1.in_features == 13
