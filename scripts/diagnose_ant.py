"""Diagnose a HIRO checkpoint on an Ant task (written after the failed first Step 6 batch).

Checks:  (1) a deterministic evaluation episode: how far does the ant get, does it flip?
         (2) are the high-level goals saturated at the edge of the goal range?
         (3) can the low level walk on command? (goal "move +3 in x / y" re-issued every c steps)

  python scripts/diagnose_ant.py --env AntMaze --ckpt results/checkpoints/AntMaze/hiro/seed0.pt
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.viz_goals import load_agent  # noqa: E402
from hiro_pytorch.hiro import rollout  # noqa: E402


def walk_test(env, agent, gxy, steps=200, seed=2):
    obs, _ = env.reset(seed=seed)
    start, c = obs[:2].copy(), agent.cfg.c
    for t in range(steps):
        if t % c == 0:
            g = np.zeros(len(agent.goal_low), np.float32)
            g[:2] = gxy
        a = agent.select_action(obs, g, explore=False)
        obs2, *_ = env.step(a)
        g = agent.next_goal(obs, g, obs2)
        obs = obs2
    return obs[:2] - start, float(np.abs(a).mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="AntMaze")
    p.add_argument("--ckpt", nargs="+", required=True)
    p.add_argument("--target", nargs="+", type=float, default=(16.0, 0.0))
    args = p.parse_args()
    for ck in args.ckpt:
        env, ag = load_agent(args.env, ck)
        tr = rollout(env, ag, reset_options={"target": tuple(args.target)}, seed=1)
        pos = tr["positions"]
        goals = np.array([sg[2] - sg[1] for sg in tr["subgoals"]])[:, :2]
        sat = (np.abs(goals) > 0.99 * ag.goal_high[:2]).mean()
        print(f"== {ck}")
        print(f"  eval to {tuple(args.target)}: final dist {tr['info']['dist']:.2f}, "
              f"max xy displacement {np.abs(pos[:, :2] - pos[0, :2]).max():.2f}, final quat-w {pos[-1, 3]:.2f}")
        print(f"  high-level goals (x,y): mean |g| {np.linalg.norm(goals, axis=1).mean():.2f}, saturated {sat:.0%}")
        for name, gxy in [("+x", (3, 0)), ("+y", (0, 3)), ("-x", (-3, 0))]:
            d, amag = walk_test(env, ag, gxy)
            print(f"  low level 'walk {name}' 200 steps: moved ({d[0]:+.2f}, {d[1]:+.2f}), mean |a| {amag:.1f}")


if __name__ == "__main__":
    main()
