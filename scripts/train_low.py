"""Step 2: train ONLY the goal-conditioned low-level policy on random relative goals.

No high level yet: every c steps a goal g ~ U(-R, R)^2 is sampled (this is the official `DirectionSampler`,
and the paper's "with pre-training" variant). In between, the goal follows Eq. 2 (h), the reward is Eq. 3,
and the low-level TD3 learns on (s, g, a, r, s', h(s, g, s')) as in Eq. 4.

Metric: at the end of each c-step segment, distance between where the point is and the absolute target
s_t + g_t that was set at the start of the segment. Doing nothing scores ||g|| (~3.8 for R=5).

  python scripts/train_low.py --steps 50000
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from hiro_pytorch.buffers import ReplayBuffer  # noqa: E402
from hiro_pytorch.envs.point import PointEnv  # noqa: E402
from hiro_pytorch.goals import goal_transition, intrinsic_reward, low_level_input  # noqa: E402
from hiro_pytorch.td3 import TD3, TD3Config  # noqa: E402
from hiro_pytorch.utils import CSVLogger, set_seed  # noqa: E402


def make_goal_sampler(env_name, R, rest):
    """Point: g ~ U(-R, R)^2.  Ant (15-D goal = qpos[:15]): x,y ~ U(-R, R); the other 13 dims (z, orientation,
    joints) are 0 ("keep posture", rest='zero') or uniform over the official goal range (rest='full')."""
    if env_name == "point":
        return lambda rng: rng.uniform(-R, R, size=2).astype(np.float32)
    from hiro_pytorch.envs import ANT_GOAL_RANGE
    hi = np.asarray(ANT_GOAL_RANGE, dtype=np.float32)

    def sample(rng):
        g = np.zeros(15, np.float32) if rest == "zero" else rng.uniform(-hi, hi).astype(np.float32)
        g[:2] = rng.uniform(-R, R, size=2)
        return g
    return sample


def run_segments(env, agent, c, sample_goal, zero_idx, n_episodes, rng, noise_std=0.0, buf=None):
    """Roll out episodes made of c-step goal segments.
    Returns list of (||g_xy||, final x,y distance to the absolute target, final distance in all goal dims)."""
    gi = env.goal_idx
    sd = getattr(env, "state_dim", None)
    stats = []
    for _ in range(n_episodes):
        o, _ = env.reset()
        s = o[:sd] if sd else o
        done, t = False, 0
        while not done:
            if t % c == 0:                                   # new goal every c steps
                g = sample_goal(rng)
                target, g_norm = s[gi] + g, np.linalg.norm(g[:2])
            a = agent.act(low_level_input(s, g, zero_idx), noise_std)
            o2, _, term, trunc, _ = env.step(a)
            s2 = o2[:sd] if sd else o2
            r = intrinsic_reward(s, g, s2, gi)               # Eq. 3
            g2 = goal_transition(s, g, s2, gi)               # Eq. 2
            if buf is not None:                              # Eq. 4 transition
                buf.add(low_level_input(s, g, zero_idx), a, r, low_level_input(s2, g2, zero_idx), term)
            s, g, t, done = s2, g2, t + 1, term or trunc
            if t % c == 0:
                stats.append((g_norm, np.linalg.norm(s[gi][:2] - target[:2]), np.linalg.norm(s[gi] - target)))
    return stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", choices=["point", "AntRun"], default="point")
    p.add_argument("--goal-rest", choices=["zero", "full"], default="zero",
                   help="Ant only: non-xy goal dims = 0 (keep posture) or uniform over the official range")
    p.add_argument("--steps", type=int, default=50_000)
    p.add_argument("--c", type=int, default=10)
    p.add_argument("--goal-range", type=float, default=5.0)
    p.add_argument("--expl-noise", type=float, default=0.1)
    p.add_argument("--no-zero-obs", action="store_true")
    p.add_argument("--start-steps", type=int, default=1_000)
    p.add_argument("--eval-every", type=int, default=2_500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu", help="cpu | cuda | cuda:0 | cuda:1 | mps")
    p.add_argument("--threads", type=int, default=1,
                   help="torch threads per run; 1 is ~1.5x faster in total when running several seeds in parallel")
    p.add_argument("--out", default="results/runs/PointGoalReach")
    p.add_argument("--tag", default=None)
    args = p.parse_args()
    torch.set_num_threads(args.threads)

    set_seed(args.seed)
    rng, eval_rng = np.random.default_rng(args.seed), np.random.default_rng(12345)
    if args.env == "point":
        env, eval_env = PointEnv(), PointEnv()
    else:
        from hiro_pytorch.envs import make_env
        env, eval_env = make_env(args.env), make_env(args.env)
    sd = getattr(env, "state_dim", None)
    zero_idx = () if args.no_zero_obs else (0, 1)
    obs_dim = sd or env.observation_space.shape[0]
    goal_dim, act_dim = len(env.goal_idx), env.action_space.shape[0]
    half = (env.action_space.high - env.action_space.low) / 2
    sample_goal = make_goal_sampler(args.env, args.goal_range, args.goal_rest)
    agent = TD3(obs_dim + goal_dim, act_dim, env.action_space.low, env.action_space.high, TD3Config(), device=args.device)
    buf = ReplayBuffer(obs_dim + goal_dim, act_dim, device=args.device)
    tag = args.tag or ("low_no_zero_obs" if args.no_zero_obs else "low")
    logger = CSVLogger(os.path.join(args.out, tag, f"seed{args.seed}.csv"))

    # Collect with one segment-episode at a time so that updates interleave with env steps.
    class _Random:  # uniform-random warm-up policy (point); the Ant warms up with the noisy policy (official)
        def act(self, _x, _n=0.0):
            return rng.uniform(-half, half).astype(np.float32)

    t0, step = time.time(), 0
    o, _ = env.reset(seed=args.seed)
    s = o[:sd] if sd else o
    gi, t_ep = env.goal_idx, 0
    noise = args.expl_noise * half
    while step < args.steps:
        if t_ep % args.c == 0:
            g = sample_goal(rng)
        pol = _Random() if (step < args.start_steps and args.env == "point") else agent
        a = pol.act(low_level_input(s, g, zero_idx), noise)
        o2, _, term, trunc, _ = env.step(a)
        s2 = o2[:sd] if sd else o2
        r = intrinsic_reward(s, g, s2, gi)
        g2 = goal_transition(s, g, s2, gi)
        buf.add(low_level_input(s, g, zero_idx), a, r, low_level_input(s2, g2, zero_idx), term)
        s, g, t_ep, step = s2, g2, t_ep + 1, step + 1
        if term or trunc:
            o, _ = env.reset()
            s = o[:sd] if sd else o
            t_ep = 0
        if step >= args.start_steps:
            agent.update(buf.sample(100))
        if step % args.eval_every == 0:
            n_ep = 5 if args.env == "point" else 2
            st = np.array(run_segments(eval_env, agent, args.c, sample_goal, zero_idx, n_ep, eval_rng))
            row = dict(step=step, goal_norm=st[:, 0].mean(), final_dist=st[:, 1].mean(),
                       final_dist_all=st[:, 2].mean(), success=(st[:, 1] < 0.5).mean(),
                       sps=step / (time.time() - t0))
            logger.log(row)
            print(f"step {step:6d} | |g_xy| {row['goal_norm']:.2f} -> final xy dist {row['final_dist']:.2f}"
                  f" (all dims {row['final_dist_all']:.2f}) | within 0.5: {row['success']:.0%} | {row['sps']:.0f} sps",
                  flush=True)

    os.makedirs("results/checkpoints", exist_ok=True)
    torch.save(agent.state_dict(), f"results/checkpoints/{tag}_seed{args.seed}.pt")


if __name__ == "__main__":
    main()
