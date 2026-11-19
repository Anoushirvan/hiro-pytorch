"""Animated GIF of a HIRO episode on the PointMaze (for the slides): maze, path so far, the current subgoal
(absolute target s_t + g_t, orange) and the task target (green star).

  python scripts/make_point_gif.py --ckpt results/checkpoints/PointMaze/hiro/seed0.pt --out results/pointmaze_hiro.gif
"""
from __future__ import annotations

import argparse
import io
import os
import sys

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.viz_goals import draw_maze, load_agent  # noqa: E402


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--target", nargs=2, type=float, default=(0.0, 8.0))
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--every", type=int, default=2)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    env, agent = load_agent("PointMaze", args.ckpt)
    obs, _ = env.reset(seed=args.seed, options={"target": tuple(args.target)})
    c, gi = agent.cfg.c, agent.goal_idx
    path, frames, t, done = [obs[gi].copy()], [], 0, False
    while not done:
        if t % c == 0:
            g = agent.select_goal(obs, explore=False)
            sub = obs[gi] + g
        if t % args.every == 0:
            fig, ax = plt.subplots(figsize=(4.2, 4.2), dpi=100)
            draw_maze(ax, env)
            pa = np.array(path)
            ax.plot(pa[:, 0], pa[:, 1], "-", color="#2a6fdb", lw=2, zorder=3)
            ax.plot(*pa[-1], "o", color="#2a6fdb", ms=10, zorder=5)
            ax.plot(*sub, "o", color="#e0703a", ms=11, zorder=4, alpha=0.9)
            ax.annotate("", xy=sub, xytext=pa[-1], zorder=4,
                        arrowprops=dict(arrowstyle="->", color="#e0703a", lw=1.5, ls="--"))
            ax.plot(*args.target, "*", color="#2e9e5b", ms=20, zorder=5)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"step {t:3d}   ●  subgoal $s_t+g_t$", fontsize=10, color="#333")
            fig.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            plt.close(fig)
            frames.append(np.array(Image.open(buf).convert("RGB")))
        a = agent.select_action(obs, g, explore=False)
        obs2, _, term, trunc, info = env.step(a)
        g = agent.next_goal(obs, g, obs2)
        obs, t, done = obs2, t + 1, term or trunc
        path.append(obs[gi].copy())
    frames += [frames[-1]] * args.fps            # hold the last frame for 1 s
    imageio.mimsave(args.out, frames, duration=1000 / args.fps, loop=0)
    print(f"saved {args.out}: {len(frames)} frames, final dist {info['dist']:.2f}, success {info['success']}")


if __name__ == "__main__":
    main()
