"""Goal machinery of HIRO (paper Sec. 3.1-3.2). Works on numpy arrays and torch tensors, batched or not.

Notation: s = full observation, g = *relative* goal living in a subspace of s picked by `goal_idx`
(for the ant: qpos[:15] = obs[:15]; for the point: x, y).

  Eq. 2  goal transition   h(s_t, g_t, s_{t+1}) = s_t + g_t - s_{t+1}
         -> keeps the *absolute* target  s_t + g_t  fixed while the agent moves.
  Eq. 3  intrinsic reward  r(s_t, g_t, a_t, s_{t+1}) = -|| s_t + g_t - s_{t+1} ||_2
         -> negative distance from where the agent landed to the absolute target.
  Eq. 4  the low-level policy/critic simply take (s_t, g_t) as input: mu_lo(s, g), Q_lo(s, g, a).

Official code: context/context_transition_functions.py (relative_context_transition_fn),
               context/rewards_functions.py (negative_distance, relative_context=True, L2, eps=1e-10).
"""
from __future__ import annotations

import numpy as np
import torch

EPS = 1e-10  # official code: sqrt(d^2 + 1e-10) so the gradient of sqrt at 0 is finite


def _sub(x, idx):
    return x[..., idx]


def goal_transition(s, g, s_next, goal_idx):
    """Eq. 2: h(s, g, s') = s + g - s'  (only on the goal dimensions)."""
    return _sub(s, goal_idx) + g - _sub(s_next, goal_idx)


def intrinsic_reward(s, g, s_next, goal_idx, scale: float = 1.0):
    """Eq. 3: r = -||s + g - s'||_2 (times `scale`). Returns shape s.shape[:-1]."""
    diff = goal_transition(s, g, s_next, goal_idx)
    if isinstance(diff, torch.Tensor):
        return -scale * torch.sqrt((diff ** 2).sum(-1) + EPS)
    return -scale * np.sqrt((diff ** 2).sum(-1) + EPS)


def low_level_input(s, g, zero_idx=(0, 1)):
    """Input to mu_lo / Q_lo: concat(s with `zero_idx` dims set to 0, g).

    ZERO_OBS (official `ddpg_networks.py:133`): the low level does not see the absolute x, y.
    Because goals are *relative*, absolute position is irrelevant to the low-level task
    ("move by g"), and hiding it makes the skill position-independent -> generalises across the maze.
    """
    if isinstance(s, torch.Tensor):
        s = s.clone()
        if zero_idx:
            s[..., list(zero_idx)] = 0.0
        return torch.cat([s, g.to(s.dtype)], dim=-1)
    s = np.array(s, dtype=np.float32, copy=True)
    if zero_idx:
        s[..., list(zero_idx)] = 0.0
    return np.concatenate([s, np.asarray(g, dtype=np.float32)], axis=-1)
