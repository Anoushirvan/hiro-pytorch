"""DDPG (ref [27]) and TD3 (ref [12]) in one class.

TD3 = DDPG + three independent fixes, each switchable by a flag so we can study them:

  1. twin_q          Clipped double-Q learning: two critics, target uses min(Q1', Q2').
                     Fights the *overestimation bias* of max-style targets (TD3 paper Sec. 4.1).
  2. target_noise    Target policy smoothing: a' = clip(mu'(s') + clip(eps, -c, c)), eps~N(0, sigma).
                     Regularises Q so it can't exploit narrow peaks (TD3 Sec. 5.3).
  3. policy_delay    Update actor + target nets only every d critic updates (TD3 Sec. 5.2).

With twin_q=False, target_noise=0, policy_delay=1 this is exactly DDPG's update:
    critic:  minimise ( Q(s,a) - [r + gamma * not_done * Q'(s', mu'(s'))] )^2          (HIRO paper Eq. 1)
    actor:   maximise Q(s, mu(s))  (deterministic policy gradient)
    targets: theta' <- tau*theta + (1-tau)*theta'                                       (soft update)

Architecture follows the official HIRO code (`configs/base_uvf.gin`, `agents/ddpg_networks.py`):
    actor : obs -> 300 -> 300 -> act (tanh, scaled to action range); final layer init U(±3e-3)
    critic: [obs, action] -> 300, concat(action) -> 300 -> 1; final layer init U(±3e-3)
    loss  : Huber (delta=1) by default, like the official code (paper writes squared error).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- networks
def official_init(layers, final: nn.Linear, final_scale: float) -> None:
    """Initialisation of the official HIRO code (agents/ddpg_networks.py), which follows the DDPG paper (Sec. 7):
    hidden weights U(±1/sqrt(fan_in)) (variance_scaling factor 1/3, uniform) and zero biases;
    final layer weights U(±final_scale) with final_scale = 3e-3, zero bias.
    => at initialisation the actor outputs ~the centre of the action range and Q ~ 0.
    final_scale <= 0 keeps PyTorch's default initialisation (what we used before the fix)."""
    if final_scale <= 0:
        return
    for l in layers:
        nn.init.uniform_(l.weight, -1 / l.in_features ** 0.5, 1 / l.in_features ** 0.5)
        nn.init.zeros_(l.bias)
    nn.init.uniform_(final.weight, -final_scale, final_scale)
    nn.init.zeros_(final.bias)


class Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, act_low, act_high, hidden=(300, 300), final_init=3e-3):
        super().__init__()
        self.l1 = nn.Linear(obs_dim, hidden[0])
        self.l2 = nn.Linear(hidden[0], hidden[1])
        self.l3 = nn.Linear(hidden[1], act_dim)
        official_init([self.l1, self.l2], self.l3, final_init)
        low = torch.as_tensor(act_low, dtype=torch.float32)
        high = torch.as_tensor(act_high, dtype=torch.float32)
        # buffers move with .to(device) and are saved in state_dict, but are not trained
        self.register_buffer("act_center", (high + low) / 2)
        self.register_buffer("act_scale", (high - low) / 2)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.l1(obs))
        x = F.relu(self.l2(x))
        return self.act_center + self.act_scale * torch.tanh(self.l3(x))


class Critic(nn.Module):
    """Q(s, a). The action is injected after the first layer (DDPG paper, Sec. 7) and, if `action_in_first`,
    also into the first layer — as in the official HIRO code (`ddpg_networks.py`, marked "#TD3")."""

    def __init__(self, obs_dim: int, act_dim: int, hidden=(300, 300), final_init=3e-3, action_in_first=True):
        super().__init__()
        self.action_in_first = action_in_first
        self.l1 = nn.Linear(obs_dim + (act_dim if action_in_first else 0), hidden[0])
        self.l2 = nn.Linear(hidden[0] + act_dim, hidden[1])
        self.l3 = nn.Linear(hidden[1], 1)
        official_init([self.l1, self.l2], self.l3, final_init)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.l1(torch.cat([obs, act], dim=-1) if self.action_in_first else obs))
        x = F.relu(self.l2(torch.cat([x, act], dim=-1)))
        return self.l3(x)


# --------------------------------------------------------------------------- config
@dataclass
class TD3Config:
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 1e-4
    critic_lr: float = 1e-3
    hidden: tuple = (300, 300)
    # --- the three TD3 fixes (defaults = full TD3) ---
    twin_q: bool = True
    target_noise: float = 0.2      # sigma, as a FRACTION of the action half-range
    noise_clip: float = 0.5        # c,     as a FRACTION of the action half-range
    policy_delay: int = 2
    # --- loss ---
    huber: bool = True             # official HIRO code uses Huber(delta=1)
    # --- architecture details of the official code (added after the Step 6 failure, see plan.md) ---
    final_init: float = 3e-3       # final-layer init U(±3e-3); <= 0 = PyTorch default
    action_in_first: bool = True   # critic also sees the action in its first layer

    @classmethod
    def ddpg(cls, **kw) -> "TD3Config":
        """Vanilla DDPG update (all TD3 fixes off)."""
        return cls(twin_q=False, target_noise=0.0, policy_delay=1, **kw)


# --------------------------------------------------------------------------- agent
class TD3:
    def __init__(self, obs_dim: int, act_dim: int, act_low, act_high,
                 cfg: TD3Config = TD3Config(), device: str = "cpu"):
        self.cfg = cfg
        self.device = torch.device(device)
        self.act_low = np.asarray(act_low, dtype=np.float32)
        self.act_high = np.asarray(act_high, dtype=np.float32)

        self.actor = Actor(obs_dim, act_dim, act_low, act_high, cfg.hidden, cfg.final_init).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        mk_critic = lambda: Critic(obs_dim, act_dim, cfg.hidden, cfg.final_init, cfg.action_in_first)
        self.critic1 = mk_critic().to(self.device)
        self.critic1_target = copy.deepcopy(self.critic1)
        critic_params = list(self.critic1.parameters())
        if cfg.twin_q:
            self.critic2 = mk_critic().to(self.device)
            self.critic2_target = copy.deepcopy(self.critic2)
            critic_params += list(self.critic2.parameters())
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_opt = torch.optim.Adam(critic_params, lr=cfg.critic_lr)
        self.n_updates = 0

    # ---------------------------------------------------------------- acting
    @torch.no_grad()
    def act(self, obs: np.ndarray, noise_std=0.0) -> np.ndarray:
        """Deterministic mu(s), plus optional Gaussian exploration noise (absolute std, scalar or per-dim), clipped."""
        o = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        a = self.actor(o)[0].cpu().numpy()
        if np.any(np.asarray(noise_std) > 0):
            a = a + np.random.normal(0.0, noise_std, size=a.shape)
        return np.clip(a, self.act_low, self.act_high).astype(np.float32)

    @torch.no_grad()
    def q_value(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        """Q estimate used for the overestimation study (critic 1, like the TD3 paper Fig. 1)."""
        return self.critic1(obs, act)

    # ---------------------------------------------------------------- learning
    @torch.no_grad()
    def _target_q(self, rew, next_obs, not_done) -> torch.Tensor:
        a_next = self.actor_target(next_obs)
        if self.cfg.target_noise > 0:                                   # fix 2: target smoothing
            scale = self.actor_target.act_scale
            eps = torch.randn_like(a_next) * self.cfg.target_noise * scale
            eps = torch.max(torch.min(eps, self.cfg.noise_clip * scale), -self.cfg.noise_clip * scale)
            a_next = torch.max(torch.min(a_next + eps, self.actor_target.act_center + scale),
                               self.actor_target.act_center - scale)
        q_next = self.critic1_target(next_obs, a_next)
        if self.cfg.twin_q:                                             # fix 1: clipped double Q
            q_next = torch.min(q_next, self.critic2_target(next_obs, a_next))
        return rew + self.cfg.gamma * not_done * q_next                # Bellman target (Eq. 1)

    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """One gradient step on the critic(s); actor/targets every `policy_delay` steps."""
        obs, act = batch["obs"], batch["act"]
        y = self._target_q(batch["rew"], batch["next_obs"], batch["not_done"])

        loss_fn = F.smooth_l1_loss if self.cfg.huber else F.mse_loss   # smooth_l1 == Huber(delta=1)
        q1 = self.critic1(obs, act)
        critic_loss = loss_fn(q1, y)
        if self.cfg.twin_q:
            critic_loss = critic_loss + loss_fn(self.critic2(obs, act), y)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        info = {"critic_loss": critic_loss.item(), "q_mean": q1.mean().item()}
        self.n_updates += 1
        if self.n_updates % self.cfg.policy_delay == 0:                 # fix 3: delayed actor
            actor_loss = -self.critic1(obs, self.actor(obs)).mean()     # deterministic policy gradient
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()
            info["actor_loss"] = actor_loss.item()
            self._soft_update()
        return info

    @torch.no_grad()
    def _soft_update(self) -> None:
        pairs = [(self.actor, self.actor_target), (self.critic1, self.critic1_target)]
        if self.cfg.twin_q:
            pairs.append((self.critic2, self.critic2_target))
        for net, tgt in pairs:
            for p, tp in zip(net.parameters(), tgt.parameters()):
                tp.mul_(1 - self.cfg.tau).add_(self.cfg.tau * p)

    # ---------------------------------------------------------------- io
    def state_dict(self) -> dict:
        d = {"actor": self.actor.state_dict(), "critic1": self.critic1.state_dict()}
        if self.cfg.twin_q:
            d["critic2"] = self.critic2.state_dict()
        return d
