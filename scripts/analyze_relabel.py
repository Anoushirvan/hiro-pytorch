"""Why does the correction re-label 50-70% of goals? Measure HOW FAR re-labelled goals move and WHY.

Collect c-step segments with a trained HIRO agent, then apply the off-policy correction and report
  * fraction re-labelled, which candidate won (original / s_{t+c}-s_t / random sample)
  * ||g_new - g_orig|| compared to ||g_orig|| and to the high-level exploration noise (sigma = 1)
  * score (log-prob, Eq. 5) gain of the chosen goal over the original
under three data-collection conditions:
  A  "train-like"     high noise 1.0, low noise 10%           (what the buffer actually contains)
  B  "no low noise"   high noise 1.0, low noise 0             (actions = mu_lo exactly)
  C  "stale low"      like A, but actions come from ANOTHER seed's low policy (a very changed mu_lo)

  python scripts/analyze_relabel.py --ckpt results/checkpoints/PointMaze/hiro/seed0.pt \
         --stale results/checkpoints/PointMaze/hiro_no_correction/seed0.pt
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from hiro_pytorch.goals import low_level_input  # noqa: E402
from scripts.viz_goals import load_agent  # noqa: E402


def collect(env, agent, actor_agent, n_segments, low_noise, seed):
    """Run the hierarchy (goals from `agent`'s high level, actions from `actor_agent`'s low level),
    return segments in the high-buffer format."""
    rng = np.random.default_rng(seed)
    c, gi = agent.cfg.c, agent.goal_idx
    noise = low_noise * (agent.act_high - agent.act_low) / 2
    goals, states, actions = [], [], []
    while len(goals) < n_segments:
        obs, _ = env.reset(seed=int(rng.integers(1 << 30)))
        done, t = False, 0
        while not done:
            if t % c == 0:
                g = agent.select_goal(obs, explore=True)
                seg_g, seg_s, seg_a = g.copy(), [agent.state(obs)], []
            a = actor_agent.low.act(low_level_input(agent.state(obs), g, agent.cfg.zero_idx), noise)
            obs2, _, term, trunc, _ = env.step(a)
            g = agent.next_goal(obs, g, obs2)
            seg_s.append(agent.state(obs2))
            seg_a.append(a)
            obs, t, done = obs2, t + 1, term or trunc
            if t % c == 0:
                goals.append(seg_g)
                states.append(np.array(seg_s))
                actions.append(np.array(seg_a))
    as_t = lambda x: torch.as_tensor(np.array(x[:n_segments]), dtype=torch.float32)
    return dict(act=as_t(goals), states=as_t(states), actions=as_t(actions))


def analyse(agent, batch, name):
    torch.manual_seed(0)
    cands = agent.candidate_goals(batch)
    scores = agent.correction_scores(batch["states"], batch["actions"], cands)
    best = scores.argmax(0)
    idx = torch.arange(len(best))
    g_new, g_orig = cands[best, idx], batch["act"]
    moved = (g_new - g_orig).norm(dim=-1)
    changed = best != 0
    gain = scores[best, idx] - scores[0]
    s = dict(name=name, n=len(best), frac_changed=changed.float().mean().item(),
             frac_delta=(best == 1).float().mean().item(), frac_random=(best >= 2).float().mean().item(),
             g_norm=g_orig.norm(dim=-1).mean().item(),
             moved_mean_changed=moved[changed].mean().item() if changed.any() else 0.0,
             moved_median_changed=moved[changed].median().item() if changed.any() else 0.0,
             orig_score=scores[0].mean().item(), gain_mean=gain[changed].mean().item() if changed.any() else 0.0,
             moved=moved.numpy())
    print(f"[{name}] relabelled {s['frac_changed']:.0%} (delta {s['frac_delta']:.0%}, random {s['frac_random']:.0%})"
          f" | ||g_orig|| {s['g_norm']:.2f} | ||g_new-g_orig|| (changed only) mean {s['moved_mean_changed']:.2f}"
          f" median {s['moved_median_changed']:.2f} | log-prob of original {s['orig_score']:.3f},"
          f" gain {s['gain_mean']:.3f}")
    return s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="PointMaze")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--stale", required=True, help="checkpoint whose LOW policy plays the role of an old mu_lo")
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--out", default="results/step4_relabel_analysis.png")
    args = p.parse_args()

    env, agent = load_agent(args.env, args.ckpt)
    _, stale = load_agent(args.env, args.stale)
    res = [analyse(agent, collect(env, agent, agent, args.n, 0.1, 1), "A train-like (low noise 10%)"),
           analyse(agent, collect(env, agent, agent, args.n, 0.0, 2), "B no low-level noise"),
           analyse(agent, collect(env, agent, stale, args.n, 0.1, 3), "C stale low policy")]

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    bins = np.linspace(0, 8, 41)
    for r, col in zip(res, ["#2a6fdb", "#2e9e5b", "#e0703a"]):
        ax.hist(r["moved"], bins=bins, alpha=0.55, color=col,
                label=f"{r['name']}: relabelled {r['frac_changed']:.0%}")
    ax.axvline(res[0]["g_norm"], color="k", ls="--", lw=1, label=f"mean ||g_orig|| = {res[0]['g_norm']:.1f}")
    ax.set_yscale("log")
    ax.set_xlabel("||g_relabelled - g_original||  (0 = kept)")
    ax.set_ylabel("segments (log)")
    ax.set_title("How far does the off-policy correction move goals? (PointMaze)", fontsize=10)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print("saved", args.out)


if __name__ == "__main__":
    main()
