"""Step 4 tests: off-policy correction (paper Eq. 5).

Key trick: the score only uses the stored states and actions, not the dynamics. So we can take *any*
state sequence, generate the actions the current low policy would take for a known true goal g*
(goals evolving with h), and check that the correction prefers g*.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from hiro_pytorch.goals import low_level_input  # noqa: E402
from hiro_pytorch.hiro import HIRO, HIROConfig  # noqa: E402

C, SD, GD, AD, B = 10, 4, 2, 2, 16


def make_agent():
    torch.manual_seed(0)
    np.random.seed(0)
    return HIRO(obs_dim=6, state_dim=SD, act_low=[-1, -1], act_high=[1, 1], goal_idx=[0, 1],
                goal_low=[-6, -6], goal_high=[6, 6], cfg=HIROConfig(c=C))


def synthetic_batch(agent, g_true):
    """Random states; actions = mu_lo(s_i, g_i) with g_i = s_0 + g_true - s_i (Eq. 2 unrolled)."""
    states = torch.randn(B, C + 1, SD) * 3
    acts = []
    for i in range(C):
        g_i = states[:, 0, :2] + g_true - states[:, i, :2]
        with torch.no_grad():
            acts.append(agent.low.actor(low_level_input(states[:, i], g_i)))
    return dict(states=states, actions=torch.stack(acts, 1))


def test_true_goal_has_max_score_zero():
    ag = make_agent()
    g_true = torch.rand(B, GD) * 8 - 4
    b = synthetic_batch(ag, g_true)
    cands = torch.stack([g_true, g_true + 1.0, g_true - 2.0, torch.zeros_like(g_true)])
    sc = ag.correction_scores(b["states"], b["actions"], cands)
    assert sc.shape == (4, B)
    assert torch.allclose(sc[0], torch.zeros(B), atol=1e-6)          # exact match => log-prob 0 (max)
    assert torch.all(sc[1:] < sc[0])


def test_correction_keeps_correct_original_goal():
    ag = make_agent()
    g_true = torch.rand(B, GD) * 8 - 4
    b = synthetic_batch(ag, g_true)
    b["act"] = g_true.clone()
    new = ag.off_policy_correction(b)
    assert torch.allclose(new, g_true) and ag.relabel_stats["relabel_frac_changed"] == 0.0


def test_correction_replaces_wrong_goal_with_true_one(monkeypatch=None):
    ag = make_agent()
    g_true = torch.rand(B, GD) * 8 - 4
    b = synthetic_batch(ag, g_true)
    b["act"] = -g_true                                               # stale goal from an "old" low policy
    orig_fn = ag.candidate_goals

    def with_true(batch):                                            # make sure g* is among the candidates
        c = orig_fn(batch)
        c[5] = g_true
        return c

    ag.candidate_goals = with_true
    new = ag.off_policy_correction(b)
    assert torch.allclose(new, g_true)


def test_candidates_shape_order_and_clipping():
    ag = make_agent()
    states = torch.zeros(B, C + 1, SD)
    states[:, -1, :2] = 20.0                                         # moved far beyond the goal range
    batch = dict(states=states, act=torch.full((B, GD), 0.5))
    c = ag.candidate_goals(batch)
    assert c.shape == (10, B, GD)
    assert torch.allclose(c[0], batch["act"])                        # index 0: original goal
    assert torch.allclose(c[1], torch.full((B, GD), 6.0))            # index 1: s_{t+c}-s_t, clipped to 6
    assert c.abs().max() <= 6.0
