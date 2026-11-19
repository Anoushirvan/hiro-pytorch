"""Visualise what the hierarchy does: maze walls, the agent's path, and every high-level subgoal
(arrow from s_t to the absolute target s_t + g_t chosen at that decision).

  python scripts/viz_goals.py --env PointMaze --ckpt results/checkpoints/PointMaze/hiro/seed0.pt \
         --target 0 8 --out results/step3_goals.png
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from hiro_pytorch.envs import ENV_SPECS, make_env  # noqa: E402
from hiro_pytorch.hiro import HIRO, HIROConfig, rollout  # noqa: E402


def draw_maze(ax, env):
    s = env.scaling
    for cx, cy in env.blocks:
        ax.add_patch(Rectangle((cx - s / 2, cy - s / 2), s, s, color="#555", zorder=1))
    if hasattr(env, "maze_id"):          # Ant envs: also draw chasms and the movable block's start cell
        from hiro_pytorch.envs.ant_maze import LAYOUTS
        lay = LAYOUTS[env.maze_id]
        (ri, rj), = [(i, j) for i, row in enumerate(lay) for j, v in enumerate(row) if v == "r"]
        for i, row in enumerate(lay):
            for j, v in enumerate(row):
                x, y = (j - rj) * s, (i - ri) * s
                if v == -1:
                    ax.add_patch(Rectangle((x - s / 2, y - s / 2), s, s, color="#bcd4ee", zorder=1))
                elif v in ("XY", "YZ"):
                    ax.add_patch(Rectangle((x - s / 2, y - s / 2), s, s, fill=False, hatch="//",
                                           ec="#d03030", lw=1.2, zorder=2))
    ax.set_aspect("equal")
    ax.set_xlim(env.blocks[:, 0].min() - s / 2, env.blocks[:, 0].max() + s / 2)
    ax.set_ylim(env.blocks[:, 1].min() - s / 2, env.blocks[:, 1].max() + s / 2)


def draw_episode(ax, env, tr, target, title):
    draw_maze(ax, env)
    p = tr["positions"][:, :2]          # goals may be 15-D (ant): plot the x, y part
    ax.plot(p[:, 0], p[:, 1], "-", color="#2a6fdb", lw=1.5, zorder=3, label="agent path")
    for k, (_, start, goal_abs) in enumerate(tr["subgoals"]):
        start, goal_abs = start[:2], goal_abs[:2]
        ax.annotate("", xy=goal_abs, xytext=start, zorder=4,
                    arrowprops=dict(arrowstyle="->", color="#e0703a", lw=1, alpha=0.8))
        ax.plot(*goal_abs, "o", ms=3, color="#e0703a", zorder=4, label="subgoal $s_t+g_t$" if k == 0 else None)
    ax.plot(*p[0], "s", color="k", ms=6, zorder=5, label="start")
    ax.plot(*target[:2], "*", color="#2e9e5b", ms=14, zorder=5, label="target")
    ok = "success" if tr["info"]["success"] else "fail"
    ax.set_title(f"{title}\nfinal dist {tr['info']['dist']:.2f} ({ok})", fontsize=9)


def load_agent(env_name, ckpt):
    spec, env = ENV_SPECS[env_name], make_env(env_name)
    agent = HIRO(env.observation_space.shape[0], env.state_dim, env.action_space.low, env.action_space.high,
                 env.goal_idx, -np.asarray(spec["goal_range"]), np.asarray(spec["goal_range"]),
                 HIROConfig(**spec.get("hiro_kwargs", {})))
    agent.load(ckpt)
    return env, agent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="PointMaze")
    p.add_argument("--ckpt", nargs="+", required=True, help="one or more checkpoints -> one panel each")
    p.add_argument("--labels", nargs="*", default=None)
    p.add_argument("--target", nargs="+", type=float, default=None)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    spec = ENV_SPECS[args.env]
    target = args.target or spec["eval_targets"][spec["main_target"]]
    fig, axes = plt.subplots(1, len(args.ckpt), figsize=(4.2 * len(args.ckpt), 4.4), squeeze=False)
    for i, ck in enumerate(args.ckpt):
        env, agent = load_agent(args.env, ck)
        tr = rollout(env, agent, explore=False, reset_options={"target": target}, seed=args.seed)
        draw_episode(axes[0, i], env, tr, target, args.labels[i] if args.labels else os.path.dirname(ck))
    axes[0, 0].legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print("saved", args.out)


if __name__ == "__main__":
    main()
