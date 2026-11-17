"""Plot learning curves (mean ± standard error over seeds) from CSV logs.

Usage:
  python scripts/plot.py --root results/runs/Pendulum-v1 --metric eval_return --out results/pendulum_return.png
  python scripts/plot.py --root results/runs/Pendulum-v1 --overestimation --out results/pendulum_q.png

Each sub-directory of --root is one variant (e.g. ddpg/, td3/), containing seed*.csv files.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os

import matplotlib.pyplot as plt
import numpy as np

COLORS = ["#2a6fdb", "#e0703a", "#2e9e5b", "#9b59b6", "#c0392b"]


def load(variant_dir: str, metric: str):
    runs = []
    for f in sorted(glob.glob(os.path.join(variant_dir, "seed*.csv"))):
        with open(f) as fh:
            rows = list(csv.DictReader(fh))
        # `metric` may be "a,b": use the first column that exists (HIRO logs success_0_8, flat TD3 logs success)
        col = next(m for m in metric.split(",") if m in rows[0])
        runs.append((np.array([float(r["step"]) for r in rows]), np.array([float(r[col]) for r in rows])))
    if not runs:
        return None
    n = min(len(r[0]) for r in runs)                      # align to the shortest run
    steps = runs[0][0][:n]
    ys = np.stack([r[1][:n] for r in runs])
    return steps, ys.mean(0), ys.std(0) / np.sqrt(len(runs)), len(runs)


def band(ax, steps, mean, se, label, color, ls="-"):
    ax.plot(steps, mean, ls, color=color, label=label, lw=2)
    ax.fill_between(steps, mean - se, mean + se, color=color, alpha=0.2, lw=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--metric", default="eval_return", help="column, or comma-separated fallbacks")
    p.add_argument("--ylabel", default=None)
    p.add_argument("--overestimation", action="store_true", help="plot q_est (solid) vs q_true (dashed)")
    p.add_argument("--variants", nargs="*", default=None)
    p.add_argument("--title", default=None)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    variants = args.variants or sorted(d for d in os.listdir(args.root) if os.path.isdir(os.path.join(args.root, d)))
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, v in enumerate(variants):
        c = COLORS[i % len(COLORS)]
        if args.overestimation:
            est, true = load(os.path.join(args.root, v), "q_est"), load(os.path.join(args.root, v), "q_true")
            band(ax, est[0], est[1], est[2], f"{v}: critic Q estimate", c)
            band(ax, true[0], true[1], true[2], f"{v}: true discounted return", c, "--")
        else:
            d = load(os.path.join(args.root, v), args.metric)
            band(ax, d[0], d[1], d[2], f"{v} (n={d[3]})", c)
    ax.set_xlabel("environment steps")
    ax.set_ylabel(args.ylabel or ("value" if args.overestimation else args.metric))
    ax.set_title(args.title or os.path.basename(args.root.rstrip("/")))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print("saved", args.out)


if __name__ == "__main__":
    main()
