"""All Step 6 figures + a summary table from the CSV logs, in one go.

  python scripts/plot_step6.py --root results/runs --out results/step6

Produces
  antmaze_success.png    success at (0,16) vs steps: HIRO / HIRO-no-correction / flat TD3 (mean ± s.e.)
  antmaze_dist.png       final distance to (0,16) (more informative while success is still 0)
  antmaze_all_targets.png HIRO success on the three Ant Maze evaluation targets
  pushfall.png           Ant Push / Ant Fall: success and final distance
  relabel.png            fraction of goals changed by the correction over training
  summary.md             table: final (mean of last 3 evals) and best success / distance per variant
"""
from __future__ import annotations

import argparse
import csv
import glob
import os

import matplotlib.pyplot as plt
import numpy as np

COL = {"hiro": "#2a6fdb", "hiro_no_correction": "#e0703a", "td3": "#2e9e5b"}
LABEL = {"hiro": "HIRO", "hiro_no_correction": "HIRO, no correction", "td3": "flat TD3 (no HRL)"}


def load(env_dir, variant, cols):
    """-> steps (T,), values (n_seeds, T) for the first existing column in `cols`. None if no data."""
    runs = []
    for f in sorted(glob.glob(os.path.join(env_dir, variant, "seed*.csv"))):
        rows = list(csv.DictReader(open(f)))
        if not rows:
            continue
        col = next((c for c in cols if c in rows[0]), None)
        if col is None:
            continue
        runs.append((np.array([float(r["step"]) for r in rows]), np.array([float(r[col]) for r in rows])))
    if not runs:
        return None
    n = min(len(r[0]) for r in runs)
    return runs[0][0][:n], np.stack([r[1][:n] for r in runs])


def band(ax, steps, ys, label, color):
    m, se = ys.mean(0), ys.std(0) / np.sqrt(len(ys))
    ax.plot(steps / 1e6, m, color=color, lw=2, label=f"{label} (n={len(ys)})")
    ax.fill_between(steps / 1e6, m - se, m + se, color=color, alpha=0.2, lw=0)


def curve_plot(root, env, cols, ylabel, title, out, variants=("hiro", "hiro_no_correction", "td3")):
    fig, ax = plt.subplots(figsize=(6, 3.8))
    any_data = False
    for v in variants:
        d = load(os.path.join(root, env), v, cols)
        if d is not None:
            band(ax, *d, LABEL[v], COL[v])
            any_data = True
    if not any_data:
        plt.close(fig)
        return
    ax.set_xlabel("environment steps (millions)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("saved", out)


def summary_rows(root):
    rows = []
    for env, tgt in [("AntMaze", "0_16"), ("AntPush", "0_19"), ("AntFall", "0_27")]:
        for v in ("hiro", "hiro_no_correction", "td3"):
            s = load(os.path.join(root, env), v, [f"success_{tgt}", "success"])
            d = load(os.path.join(root, env), v, [f"dist_{tgt}", "dist"])
            if s is None:
                continue
            fin = s[1][:, -3:].mean(1)                     # per seed: mean of the last 3 evaluations
            row = dict(env=env, variant=LABEL[v], n=len(fin), steps=int(s[0][-1]),
                       final=f"{fin.mean():.2f} ± {fin.std() / np.sqrt(len(fin)):.2f}",
                       best=f"{s[1].max(1).mean():.2f}")
            row["final_dist"] = f"{d[1][:, -3:].mean():.1f}" if d is not None else "–"
            rows.append(row)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="results/runs")
    p.add_argument("--out", default="results/step6")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    R, O = args.root, args.out

    curve_plot(R, "AntMaze", ["success_0_16", "success"], "success rate, target (0,16)",
               "Ant Maze: HIRO vs ablation vs flat TD3", f"{O}/antmaze_success.png")
    curve_plot(R, "AntMaze", ["dist_0_16"], "final distance to (0,16)",
               "Ant Maze: final distance to target (lower is better)", f"{O}/antmaze_dist.png",
               variants=("hiro", "hiro_no_correction"))

    # HIRO on the 3 Ant Maze evaluation targets (easy -> hard)
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for tgt, c in [("16_0", "#9bb7e8"), ("16_16", "#5a87d6"), ("0_16", "#1d3f8f")]:
        d = load(os.path.join(R, "AntMaze"), "hiro", [f"success_{tgt}"])
        if d is not None:
            band(ax, *d, f"target ({tgt.replace('_', ',')})", c)
    ax.set(xlabel="environment steps (millions)", ylabel="success rate")
    ax.set_title("Ant Maze, HIRO: easy (16,0) → hard (0,16) targets", fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(f"{O}/antmaze_all_targets.png", dpi=150); plt.close(fig); print("saved", f"{O}/antmaze_all_targets.png")

    # Push / Fall
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for env, tgt, c in [("AntPush", "0_19", "#9b59b6"), ("AntFall", "0_27", "#c0392b")]:
        s = load(os.path.join(R, env), "hiro", [f"success_{tgt}"])
        d = load(os.path.join(R, env), "hiro", [f"dist_{tgt}"])
        if s is not None:
            band(axes[0], *s, f"{env} HIRO", c)
            band(axes[1], *d, f"{env} HIRO", c)
    axes[0].set(xlabel="environment steps (millions)", ylabel="success rate", title="Ant Push / Ant Fall: success")
    axes[1].set(xlabel="environment steps (millions)", ylabel="final distance to target",
                title="Ant Push / Ant Fall: final distance")
    for a in axes:
        a.grid(alpha=0.3); a.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{O}/pushfall.png", dpi=150); plt.close(fig); print("saved", f"{O}/pushfall.png")

    # relabel fraction over training for all HIRO-with-correction runs
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for env, c in [("AntMaze", "#2a6fdb"), ("AntPush", "#9b59b6"), ("AntFall", "#c0392b")]:
        d = load(os.path.join(R, env), "hiro", ["relabel_frac_changed"])
        if d is not None:
            band(ax, d[0], d[1] * 100, env, c)
    ax.set(xlabel="environment steps (millions)", ylabel="% of sampled goals relabelled")
    ax.set_title("Off-policy correction: how often it changes the stored goal", fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(f"{O}/relabel.png", dpi=150); plt.close(fig); print("saved", f"{O}/relabel.png")

    rows = summary_rows(R)
    with open(f"{O}/summary.md", "w") as f:
        f.write("| Task | Variant | seeds | steps | final success (last 3 evals, mean ± s.e.) | best success "
                "| final distance |\n|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['env']} | {r['variant']} | {r['n']} | {r['steps']:,} | {r['final']} | {r['best']} "
                    f"| {r['final_dist']} |\n")
    print(open(f"{O}/summary.md").read())


if __name__ == "__main__":
    main()
