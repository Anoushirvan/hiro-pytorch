"""Explanatory diagrams for the report/slides (no experiment data):
  results/fig_hiro_diagram.png   two-level structure of HIRO (paper Fig. 2, redrawn)
  results/fig_correction.png     illustration of the off-policy correction (synthetic example)
"""
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

BLUE, ORANGE, GREEN, GREY, DARK = "#2a6fdb", "#e0703a", "#2e9e5b", "#9aa0a6", "#222"


def box(ax, xy, w, h, text, fc, ec=DARK, fs=11, tc="white"):
    ax.add_patch(FancyBboxPatch((xy[0] - w / 2, xy[1] - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=1.2))
    ax.text(*xy, text, ha="center", va="center", fontsize=fs, color=tc)


def arrow(ax, a, b, color=DARK, style="-|>", lw=1.4, ls="-"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=12, color=color, lw=lw, linestyle=ls))


def hiro_diagram(out):
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.set_xlim(-0.6, 10.6)
    ax.set_ylim(-0.4, 4.6)
    ax.axis("off")
    xs = [0.6, 2.4, 4.6, 6.4, 8.4, 10.0]
    labels_t = ["t", "t+1", "…", "t+c−1", "t+c", ""]
    # high level: decisions only at t and t+c
    box(ax, (xs[0], 4.0), 1.2, 0.6, r"$\mu^{hi}$", BLUE, fs=13)
    box(ax, (xs[4], 4.0), 1.2, 0.6, r"$\mu^{hi}$", BLUE, fs=13)
    ax.text(5.0, 4.38, "high level: picks a goal every c = 10 steps, reward = 0.1 · Σ env reward",
            ha="center", fontsize=9.5, color=BLUE)
    # goals row
    gl = [r"$g_t$", r"$g_{t+1}$", "…", r"$g_{t+c-1}$", r"$g_{t+c}$"]
    for i, x in enumerate(xs[:5]):
        if gl[i] == "…":
            ax.text(x, 3.0, "…", ha="center", va="center", fontsize=14)
            continue
        ax.add_patch(plt.Circle((x, 3.0), 0.36, fc="#fdecdf", ec=ORANGE, lw=1.2))
        ax.text(x, 3.0, gl[i], ha="center", va="center", fontsize=11)
    arrow(ax, (xs[0], 3.68), (xs[0], 3.28), BLUE)
    arrow(ax, (xs[4], 3.68), (xs[4], 3.28), BLUE)
    for a_, b_ in [(0, 1), (1, 2), (2, 3)]:
        arrow(ax, (xs[a_] + 0.38, 3.0), (xs[b_] - 0.38, 3.0), ORANGE)
        ax.text((xs[a_] + xs[b_]) / 2, 3.17, "h", ha="center", fontsize=9, color=ORANGE)
    ax.text(4.5, 3.62, r"h: $g_{t+1} = s_t + g_t - s_{t+1}$  (absolute target $s_t+g_t$ stays fixed)",
            ha="center", fontsize=9.5, color=ORANGE)
    # low level
    for i, x in enumerate(xs[:5]):
        if i == 2:
            ax.text(x, 1.75, "…", ha="center", va="center", fontsize=14)
            continue
        box(ax, (x, 1.75), 1.0, 0.55, r"$\mu^{lo}$", GREEN, fs=12)
        arrow(ax, (x, 2.62), (x, 2.05), ORANGE)
        arrow(ax, (x, 1.47), (x, 0.95), GREEN)
        ax.text(x + 0.12, 1.2, "a", fontsize=9, color=GREEN)
    ax.text(9.7, 1.75, "low level: acts every step,\nreward  $-\\|s_t+g_t-s_{t+1}\\|$",
            ha="left", va="center", fontsize=9.5, color=GREEN)
    # environment
    ax.add_patch(FancyBboxPatch((0.0, 0.25), 9.0, 0.6, boxstyle="round,pad=0.02", fc="#eef0f2", ec=GREY))
    ax.text(4.5, 0.55, "environment  →  states $s_t$, task reward $R_t$", ha="center", va="center", fontsize=11)
    for i, x in enumerate(xs[:5]):
        if i != 2:
            ax.text(x, 0.05, labels_t[i], ha="center", fontsize=9, color=GREY)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def correction_figure(out):
    rng = np.random.default_rng(3)
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    s0 = np.array([0.0, 0.0])
    path = np.array([[0, 0], [0.5, 0.25], [1.1, 0.35], [1.8, 0.6], [2.4, 1.0], [2.9, 1.5], [3.2, 2.1],
                     [3.35, 2.6], [3.4, 3.0], [3.45, 3.3], [3.5, 3.5]])
    sc = path[-1]
    g_orig = np.array([5.0, 0.5])
    delta = sc - s0
    cands = delta + rng.normal(0, 0.9, size=(8, 2))
    best = cands[np.argmin(np.linalg.norm(cands - (delta + [0.25, 0.1]), axis=1))]
    ax.plot(path[:, 0], path[:, 1], "-", color=GREEN, lw=2.5, label="stored low-level behaviour $s_{t:t+c}$")
    ax.plot(*s0, "ks", ms=8)
    ax.text(-0.2, -0.45, "$s_t$", fontsize=12)
    ax.plot(*sc, "o", color=GREEN, ms=8)
    ax.text(sc[0] + 0.15, sc[1] + 0.1, "$s_{t+c}$", fontsize=12, color=GREEN)
    ax.plot(*g_orig, "X", color=ORANGE, ms=13, label="original goal $g_t$ (old low policy)")
    ax.annotate("", xy=g_orig, xytext=s0, arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.5, ls="--"))
    ax.scatter(cands[:, 0], cands[:, 1], s=40, color=GREY, zorder=3, label="8 Gaussian candidates")
    ax.plot(*delta, "D", color=BLUE, ms=8, label="candidate $s_{t+c}-s_t$")
    ax.plot(*best, "*", color="#c0392b", ms=20, zorder=4, label=r"chosen $\tilde g_t$ (max log-likelihood)")
    ax.annotate("", xy=best, xytext=s0, arrowprops=dict(arrowstyle="->", color="#c0392b", lw=2))
    ax.text(0.2, 4.65, r"$\tilde g_t=\arg\max_{g}\;-\frac{1}{2}\sum_{i}\|a_{t+i}-\mu^{lo}(s_{t+i},\,\tilde g_{t+i})\|^2$",
            fontsize=11)
    ax.set_xlim(-1, 9.6)
    ax.set_ylim(-1, 5.2)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("Off-policy correction: relabel the goal so the current low level explains the stored actions",
                 fontsize=9.5)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    hiro_diagram("results/fig_hiro_diagram.png")
    correction_figure("results/fig_correction.png")
    print("saved results/fig_hiro_diagram.png, results/fig_correction.png")
