"""Train a flat (non-hierarchical) DDPG / TD3 agent on a Gymnasium env.

Step 1 sanity check:  python scripts/train_flat.py --env Pendulum-v1 --variant td3 --steps 30000
Later reused as the "No HRL" baseline on Ant Maze.

Every --eval-every steps we run deterministic episodes and log
  * eval_return       : undiscounted episode return
  * q_est / q_true    : overestimation probe (TD3 paper Fig. 1). For the first `probe_t` steps of each
                        eval episode we compare the critic's Q(s_t, a_t) to the discounted Monte-Carlo
                        return actually obtained from s_t. q_est > q_true  => overestimation.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import gymnasium as gym
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from hiro_pytorch.buffers import ReplayBuffer  # noqa: E402
from hiro_pytorch.envs import ENV_SPECS, make_env  # noqa: E402
from hiro_pytorch.td3 import TD3, TD3Config  # noqa: E402
from hiro_pytorch.utils import CSVLogger, set_seed  # noqa: E402


def make_config(args) -> TD3Config:
    cfg = TD3Config.ddpg() if args.variant == "ddpg" else TD3Config()
    # individual overrides for ablations, e.g. --variant td3 --no-twin
    if args.no_twin:
        cfg.twin_q = False
    if args.no_smooth:
        cfg.target_noise = 0.0
    if args.delay is not None:
        cfg.policy_delay = args.delay
    if args.mse:
        cfg.huber = False
    return cfg


def evaluate(env, agent: TD3, episodes: int, gamma: float, probe_t: int, seed: int, reset_options=None):
    returns, q_est, q_true, success = [], [], [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep, options=reset_options)
        obs_list, act_list, rews = [], [], []
        done = False
        while not done:
            a = agent.act(obs)
            obs_list.append(obs)
            act_list.append(a)
            obs, r, term, trunc, info = env.step(a)
            rews.append(r)
            done = term or trunc
        returns.append(sum(rews))
        success.append(float(info.get("success", np.nan)))
        # discounted return-to-go G_t = r_t + gamma * G_{t+1}
        g, gs = 0.0, np.zeros(len(rews))
        for t in reversed(range(len(rews))):
            g = rews[t] + gamma * g
            gs[t] = g
        n = min(probe_t, len(rews))
        o = torch.as_tensor(np.array(obs_list[:n]), dtype=torch.float32, device=agent.device)
        a = torch.as_tensor(np.array(act_list[:n]), dtype=torch.float32, device=agent.device)
        q_est.append(agent.q_value(o, a).mean().item())
        q_true.append(gs[:n].mean())
    return float(np.mean(returns)), float(np.mean(q_est)), float(np.mean(q_true)), float(np.mean(success))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="Pendulum-v1")
    p.add_argument("--variant", choices=["ddpg", "td3"], default="td3")
    p.add_argument("--no-twin", action="store_true")
    p.add_argument("--no-smooth", action="store_true")
    p.add_argument("--delay", type=int, default=None)
    p.add_argument("--mse", action="store_true", help="use MSE instead of Huber critic loss")
    p.add_argument("--steps", type=int, default=30_000)
    p.add_argument("--start-steps", type=int, default=1_000, help="uniform-random warm-up steps")
    p.add_argument("--expl-noise", type=float, default=0.1, help="exploration std, fraction of half-range")
    p.add_argument("--batch", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=2_000)
    p.add_argument("--eval-episodes", type=int, default=5)
    p.add_argument("--probe-t", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu", help="cpu | cuda | cuda:0 | cuda:1 | mps")
    p.add_argument("--threads", type=int, default=1,
                   help="torch threads per run; 1 is ~1.5x faster in total when running several seeds in parallel")
    p.add_argument("--out", default="results/runs")
    p.add_argument("--tag", default=None)
    args = p.parse_args()
    torch.set_num_threads(args.threads)

    set_seed(args.seed)
    if args.env in ENV_SPECS:   # our maze envs; evaluate on the main ("end of maze") target
        env, eval_env = make_env(args.env), make_env(args.env)
        spec = ENV_SPECS[args.env]
        eval_opts = {"target": spec["eval_targets"][spec["main_target"]]}
    else:
        env, eval_env, eval_opts = gym.make(args.env), gym.make(args.env), None
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    low, high = env.action_space.low, env.action_space.high
    cfg = make_config(args)
    agent = TD3(obs_dim, act_dim, low, high, cfg, device=args.device)
    buf = ReplayBuffer(obs_dim, act_dim, device=args.device)
    noise_std = args.expl_noise * (high - low) / 2

    tag = args.tag or args.variant
    logger = CSVLogger(os.path.join(args.out, args.env, tag, f"seed{args.seed}.csv"))
    print(f"[{tag}] {cfg}")

    obs, _ = env.reset(seed=args.seed)
    env.action_space.seed(args.seed)
    t0 = time.time()
    info = {"critic_loss": float("nan")}
    for step in range(1, args.steps + 1):
        if step <= args.start_steps:
            a = env.action_space.sample()
        else:
            a = agent.act(obs, noise_std)
        next_obs, r, term, trunc, _ = env.step(a)
        buf.add(obs, a, r, next_obs, term)        # only true termination stops bootstrapping
        obs = next_obs
        if term or trunc:
            obs, _ = env.reset()

        if step > args.start_steps:
            info = agent.update(buf.sample(args.batch))

        if step % args.eval_every == 0:
            ret, q_est, q_true, succ = evaluate(eval_env, agent, args.eval_episodes, cfg.gamma,
                                                args.probe_t, seed=10_000, reset_options=eval_opts)
            logger.log(dict(step=step, eval_return=ret, q_est=q_est, q_true=q_true, success=succ,
                            critic_loss=info["critic_loss"], sps=step / (time.time() - t0)))
            print(f"step {step:7d} | return {ret:9.2f} | success {succ:.0%} | Q est {q_est:8.2f} vs true {q_true:8.2f}"
                  f" | {step / (time.time() - t0):.0f} sps", flush=True)


if __name__ == "__main__":
    main()
