"""Train HIRO (paper Fig. 2 loop).

  python scripts/train_hiro.py --env PointMaze --steps 200000                  # Step 3 (no correction)
  python scripts/train_hiro.py --env PointMaze --steps 200000 --correction    # Step 4

Loop per environment step t:
  if t % c == 0:   g_t ~ mu_hi(obs_t) + noise            (new high-level decision)
  a_t ~ mu_lo(s_t, g_t) + noise ;  step env -> R_t, s_{t+1}
  low transition  (s_t, g_t, a_t, r_t = -||s_t+g_t-s_{t+1}||, s_{t+1}, g_{t+1} = h(...))  -> low buffer
  after c steps:  high transition (obs_t, g_t, 0.1*sum R, obs_{t+c}, s_{t:t+c}, a_{t:t+c-1}) -> high buffer
  update low every step, high every c steps.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from hiro_pytorch.envs import ENV_SPECS, make_env  # noqa: E402
from hiro_pytorch.hiro import HIRO, HIROConfig, rollout  # noqa: E402
from hiro_pytorch.utils import CSVLogger, set_seed  # noqa: E402


def evaluate(env, agent, targets, episodes):
    out = {}
    for name, tgt in targets.items():
        trajs = [rollout(env, agent, explore=False, reset_options={"target": tgt}, seed=1000 + k)
                 for k in range(episodes)]
        out[f"success_{name}"] = float(np.mean([tr["info"]["success"] for tr in trajs]))
        out[f"dist_{name}"] = float(np.mean([tr["info"]["dist"] for tr in trajs]))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="PointMaze", choices=list(ENV_SPECS))
    p.add_argument("--steps", type=int, default=200_000)
    p.add_argument("--correction", action="store_true")
    p.add_argument("--start-steps", type=int, default=None, help="warm-up steps without updates (default: 5 episodes)")
    p.add_argument("--legacy", action="store_true",
                   help="reproduce our pre-fix setup: PyTorch default init, action only in critic layer 2, "
                        "uniform-random warm-up (goals over the full range, random torques)")
    p.add_argument("--eval-every", type=int, default=5_000)
    p.add_argument("--eval-episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu", help="cpu | cuda | cuda:0 | cuda:1 | mps")
    p.add_argument("--threads", type=int, default=1,
                   help="torch threads per run; 1 is ~1.5x faster in total when running several seeds in parallel")
    p.add_argument("--out", default="results/runs")
    p.add_argument("--tag", default=None)
    p.add_argument("--age-analysis", action="store_true",
                   help="at the end, measure the correction's relabel rate vs. age of the buffer data")
    args = p.parse_args()
    torch.set_num_threads(args.threads)

    set_seed(args.seed)
    spec = ENV_SPECS[args.env]
    env, eval_env = make_env(args.env), make_env(args.env)
    cfg = HIROConfig(correction=args.correction, **spec.get("hiro_kwargs", {}))
    if args.legacy:
        for c_ in (cfg.low, cfg.high):
            c_.final_init, c_.action_in_first = 0.0, False
    agent = HIRO(obs_dim=env.observation_space.shape[0], state_dim=env.state_dim,
                 act_low=env.action_space.low, act_high=env.action_space.high, goal_idx=env.goal_idx,
                 goal_low=-np.asarray(spec["goal_range"]), goal_high=np.asarray(spec["goal_range"]), cfg=cfg, device=args.device)
    c = cfg.c
    start_steps = args.start_steps if args.start_steps is not None else 5 * env.max_steps
    assert env.max_steps % c == 0, "episode length must be a multiple of c"

    tag = args.tag or (("hiro" if args.correction else "hiro_no_correction") + ("_legacy" if args.legacy else ""))
    run_dir = os.path.join(args.out, args.env, tag)
    logger = CSVLogger(os.path.join(run_dir, f"seed{args.seed}.csv"))
    ckpt_dir = os.path.join("results", "checkpoints", args.env, tag)
    os.makedirs(ckpt_dir, exist_ok=True)

    obs, _ = env.reset(seed=args.seed)
    t_ep, ep_return, recent_returns, t0 = 0, 0.0, [], time.time()
    lo_info, hi_info = {}, {}
    for step in range(1, args.steps + 1):
        warmup = step <= start_steps
        if t_ep % c == 0:                                         # ---- high-level decision
            # warm-up: the official code collects its first episodes with the noisy *policies* (whose freshly
            # initialised actors output ~0, i.e. small, reachable goals); legacy = uniform over the goal range
            g = agent.random_goal() if (warmup and args.legacy) else agent.select_goal(obs, explore=True)
            seg_obs, seg_goal, seg_R = obs, g, 0.0
            seg_states, seg_actions = [agent.state(obs)], []
        a = env.action_space.sample() if (warmup and args.legacy) else agent.select_action(obs, g, explore=True)
        obs2, R, term, trunc, info = env.step(a)
        g2 = agent.next_goal(obs, g, obs2)                        # Eq. 2
        agent.store_low(obs, g, a, obs2, g2, term)                # Eq. 3/4
        seg_R += R
        seg_states.append(agent.state(obs2))
        seg_actions.append(a)
        ep_return += R
        obs, g, t_ep = obs2, g2, t_ep + 1
        if t_ep % c == 0 or term:                                 # ---- close high-level segment
            agent.store_high(seg_obs, seg_goal, seg_R, obs, term, seg_states, seg_actions)
        if term or trunc:
            recent_returns = (recent_returns + [ep_return])[-10:]
            obs, _ = env.reset()
            t_ep, ep_return = 0, 0.0

        if step > start_steps:                                    # ---- learning
            lo_info = agent.update_low()
            if step % c == 0:
                hi_info = agent.update_high()

        if step % args.eval_every == 0:
            row = dict(step=step, train_return=float(np.mean(recent_returns)) if recent_returns else np.nan,
                       **evaluate(eval_env, agent, spec["eval_targets"], args.eval_episodes),
                       low_q=lo_info.get("q_mean", np.nan), high_q=hi_info.get("q_mean", np.nan),
                       relabel_frac_changed=agent.relabel_stats.get("relabel_frac_changed", np.nan),
                       relabel_frac_delta=agent.relabel_stats.get("relabel_frac_delta", np.nan), sps=step / (time.time() - t0))
            logger.log(row)
            succ = " ".join(f"{k[8:]}={v:.0%}" for k, v in row.items() if k.startswith("success_"))
            print(f"step {step:7d} | train ret {row['train_return']:8.1f} | success {succ} | "
                  f"dist {row[f'dist_{spec['main_target']}']:.2f} | relabeled {row['relabel_frac_changed']:.0%} | {row['sps']:.0f} sps", flush=True)
            agent.save(os.path.join(ckpt_dir, f"seed{args.seed}.pt"))

    if args.age_analysis:
        relabel_vs_age(agent, os.path.join(run_dir, f"relabel_vs_age_seed{args.seed}.csv"))


@torch.no_grad()
def relabel_vs_age(agent, path, n_bins=10):
    """Apply the correction (with the FINAL low policy) to every segment in the high buffer, grouped by
    insertion order: bin 0 = oldest data (collected by the earliest low policies), last bin = newest."""
    hb = agent.high_buf
    assert hb.size < hb.capacity, "buffer wrapped; insertion order != age"
    torch.manual_seed(0)
    rows = []
    for b, idx in enumerate(np.array_split(np.arange(hb.size), n_bins)):
        batch = {k: torch.as_tensor(v[idx], device=agent.device) for k, v in
                 dict(act=hb.goal, states=hb.states, actions=hb.actions).items()}
        cands = agent.candidate_goals(batch)
        best = agent.correction_scores(batch["states"], batch["actions"], cands).argmax(0)
        moved = (cands[best, torch.arange(len(idx))] - batch["act"]).norm(dim=-1)
        rows.append(dict(bin=b, first_step=int(idx[0]) * agent.cfg.c, last_step=int(idx[-1]) * agent.cfg.c,
                         frac_changed=(best != 0).float().mean().item(), mean_moved=moved.mean().item()))
        print(f"data from env steps {rows[-1]['first_step']:7d}-{rows[-1]['last_step']:7d}: "
              f"relabelled {rows[-1]['frac_changed']:.0%}, mean ||g_new-g_orig|| {rows[-1]['mean_moved']:.2f}")
    logger = CSVLogger(path)
    for r in rows:
        logger.log(r)


if __name__ == "__main__":
    main()
