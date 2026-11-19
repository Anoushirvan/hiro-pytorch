"""Render a HIRO episode on an Ant env as a GIF: top-down view, current subgoal (blue sphere, the absolute
target s_t + g_t of the active high-level decision) and task target (green sphere).

  python scripts/make_gif.py --env AntMaze --ckpt results/checkpoints/AntMaze/hiro/seed0.pt \
         --out results/antmaze_hiro.gif
"""
from __future__ import annotations

import argparse
import os
import sys

import imageio.v2 as imageio
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.viz_goals import load_agent  # noqa: E402
from hiro_pytorch.envs import ENV_SPECS  # noqa: E402

SUBGOAL_RGBA, TARGET_RGBA = (0.15, 0.35, 0.95, 0.85), (0.1, 0.7, 0.2, 1.0)


@torch.no_grad()
def render_episode(env, agent, target, seed, every=2, size=400, max_steps=None):
    obs, _ = env.reset(seed=seed, options={"target": target})
    c, gi = agent.cfg.c, agent.goal_idx
    z_marker = env.wall_top + 0.5               # above the walls, so subgoals proposed inside a wall stay visible
    tgt3 = (target[0], target[1], z_marker)
    frames, t, done, info = [], 0, False, {}
    while not done and (max_steps is None or t < max_steps):
        if t % c == 0:
            g = agent.select_goal(obs, explore=False)
            sub_abs = obs[gi][:2] + g[:2]       # absolute x, y of the subgoal
        if t % every == 0:
            frames.append(env.render(size, size, markers=[((*sub_abs, z_marker), SUBGOAL_RGBA, 0.8),
                                                          (tgt3, TARGET_RGBA, 1.0)]))
        a = agent.select_action(obs, g, explore=False)
        obs2, _, term, trunc, info = env.step(a)
        g = agent.next_goal(obs, g, obs2)      # sub_abs stays fixed within a segment: Eq. 2 keeps s + g constant
        obs, t, done = obs2, t + 1, term or trunc
    return frames, info


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="AntMaze")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--target", nargs="+", type=float, default=None)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--every", type=int, default=2, help="render every k-th step (500 steps / 2 = 250 frames)")
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--size", type=int, default=400)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    spec = ENV_SPECS[args.env]
    target = tuple(args.target or spec["eval_targets"][spec["main_target"]])
    env, agent = load_agent(args.env, args.ckpt)
    frames, info = render_episode(env, agent, target, args.seed, args.every, args.size, args.max_steps)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    imageio.mimsave(args.out, frames, duration=1000 / args.fps, loop=0)
    print(f"saved {args.out}: {len(frames)} frames, final dist {info.get('dist', float('nan')):.2f}, "
          f"success {info.get('success')}")


if __name__ == "__main__":
    main()
