# Work report — outline and draft text

> Final format: Word (.docx), English. This file is the source text; it will be converted to .docx in Step 8
> with figures embedded. `[TODO-ANT]` marks content that waits for the Step 6 (Ant) results.
> Target length: ~12–15 pages including figures.

**Title:** Re-implementing HIRO: Data-Efficient Hierarchical Reinforcement Learning in PyTorch
**Subtitle:** A step-by-step study of Nachum, Gu, Lee & Levine (NeurIPS 2018)
**Course:** Reinforcement Learning — author, date

---

## Abstract (≈150 words)

We re-implement HIRO (HIerarchical Reinforcement learning with Off-policy correction), a two-level
hierarchical agent in which a high-level policy proposes goal states every *c* = 10 steps and a goal-conditioned
low-level policy is rewarded for reaching them; both levels are trained off-policy with TD3. The paper's key
idea is an *off-policy correction* that relabels stored high-level actions so that old experience stays valid
while the low-level policy changes. We build the method incrementally in PyTorch — DDPG/TD3, goal machinery, the
hierarchy, the correction, and MuJoCo Ant environments ported from the authors' TensorFlow code — with unit tests
at each stage. On a small point-mass maze (5 seeds) the correction raises final success from 0.29 to 0.91, while
flat TD3 reaches 0.20. An analysis of the replay buffer shows that the correction mostly changes *old*
experience (92 % of the oldest vs 17 % of the newest segments), as the paper's motivation predicts.
[TODO-ANT: one sentence on Ant Maze/Push/Fall results at 2 M steps.]

---

## 1. Introduction (≈1 page)

- Why hierarchy: long-horizon tasks (e.g. a quadruped navigating a maze) need both low-level motor control and
  high-level planning; flat RL explores at the wrong time scale.
- Three open questions the paper names: how to train the lower level to induce distinct behaviour, how to define
  high-level actions, and how to train without huge amounts of on-policy data.
- HIRO's answers: (i) **goals are raw states** (no learned representation, no hand-designed skills),
  (ii) **both levels trained off-policy** (TD3) for sample efficiency, (iii) an **off-policy correction** to deal
  with the non-stationarity this creates for the high level.
- Our goal: understand the method by re-implementing it from scratch, verify each component in isolation, and
  reproduce the main qualitative claims at laptop/PC scale (not the paper's 10 M steps × 10 seeds).
- Contributions of this report: a tested PyTorch implementation; a list of implementation details that matter
  but are only in the official code; a small-scale reproduction; an analysis of *when* the correction relabels.

## 2. Background (≈2 pages)

### 2.1 Off-policy actor–critic: DDPG [27]
- Deterministic policy μ_φ, critic Q_θ. Critic minimises the Bellman error (paper Eq. 1):
  (Q_θ(s_t,a_t) − R_t − γ Q_θ'(s_{t+1}, μ_φ'(s_{t+1})))². Actor maximises Q_θ(s, μ_φ(s)).
- Target networks with soft updates θ' ← τθ + (1−τ)θ'; exploration with Gaussian noise
  a_t ~ N(μ(s_t), σ²) — HIRO later relies on this behaviour policy being stochastic (Sec. 3.3).

### 2.2 TD3 [12]
- Function-approximation error → overestimation bias in actor-critic, which accumulates through bootstrapping.
- Three fixes: clipped double-Q (min of two critics in the target), target policy smoothing (clipped noise on
  the target action), delayed policy and target updates.
- **Our check (Fig. 1):** Pendulum-v1, 30 k steps, 3 seeds: both reach ≈ −112. Overestimation probe (critic's
  Q(s,a) vs the discounted Monte-Carlo return actually obtained): DDPG −29 vs −32 (overestimates by ≈ 3), TD3 −35
  vs −33 (slight underestimate) — qualitatively as in the TD3 paper; the effect is small on this easy task.

## 3. The HIRO method (≈2.5 pages)

### 3.1 Two-level structure (paper Fig. 2)
- High level μ_hi(s_t) outputs a goal g_t ∈ R^{d_g} every c steps; between decisions the goal follows a fixed
  transition h. Low level μ_lo(s_t, g_t) outputs torques every step. Only the low level acts on the environment.
- Low-level transition stored: (s_t, g_t, a_t, r_t, s_{t+1}, g_{t+1}).
  High-level transition: (s_{t:t+c−1}, g_t, a_{t:t+c−1}, Σ R_{t:t+c−1}, s_{t+c}).

### 3.2 Goals as relative states (Eqs. 2–3)
- g_t = desired *change* in the (positional part of the) state.
  Goal transition h(s_t, g_t, s_{t+1}) = s_t + g_t − s_{t+1} keeps the absolute target s_t + g_t fixed.
- Intrinsic reward r = −‖s_t + g_t − s_{t+1}‖₂: the low level gets dense reward from the very first step,
  before any task reward is seen.
- Low-level learner = standard TD3 with g as an extra input (Eq. 4).

### 3.3 The off-policy correction (Eq. 5)
- Problem: a stored high-level transition (s_t, g_t, R, s_{t+c}) was produced by an *older* low-level policy;
  under the current μ_lo the same g_t would lead elsewhere → the high-level "dynamics" are non-stationary.
- Fix: relabel g_t with the goal g̃_t that makes the stored low-level actions most likely under the current μ_lo:
  log μ_lo(a_{t:t+c−1} | s_{t:t+c−1}, g̃_{t:t+c−1}) ∝ −½ Σ_i ‖a_i − μ_lo(s_i, g̃_i)‖² + const.
- Argmax approximated over 10 candidates: the original g_t, s_{t+c} − s_t, and 8 samples from a Gaussian centred
  at s_{t+c} − s_t.
- Diagram: [figure — timeline of one high-level segment with original vs relabelled goal].

### 3.4 Algorithm summary
- Pseudo-code box: collect → store low/high transitions → update low every step → sample high batch, correct,
  update high every c steps.

## 4. Implementation (≈2.5 pages)

### 4.1 Overview
- PyTorch, Gymnasium API, official MuJoCo Python bindings. Repository layout table
  (`td3.py`, `goals.py`, `buffers.py`, `hiro.py`, `envs/`, `scripts/`, `tests/`).
- Built in 8 steps, each with unit tests (25 tests total) and a small experiment; development log in `plan.md`.

### 4.2 Paper ↔ code map
| Paper | Code |
|---|---|
| Eq. 1, TD3 fixes | `td3.py: TD3._target_q`, `update`, `_soft_update` (each fix switchable) |
| Eq. 2 h | `goals.py: goal_transition` |
| Eq. 3 r | `goals.py: intrinsic_reward` |
| Eq. 4 | `goals.py: low_level_input` + unchanged TD3 |
| Eq. 5 | `hiro.py: candidate_goals`, `correction_scores`, `off_policy_correction` |
| App. B environments | `envs/ant_maze.py` (Maze, Push, Fall) |

### 4.3 Details found only in the official code (worth knowing)
1. Critic loss is **Huber** (δ = 1), not the squared error of Eq. 1.
2. High-level reward is **0.1 ×** the sum of the c environment rewards.
3. The low-level networks see the state with **x, y zeroed** (`ZERO_OBS`) — consistent with relative goals.
4. In the correction the action error is **normalised by the action range**; candidate std = 0.5 × goal half-range.
5. The ant starts from the model's default pose (z = 0.75, joints 0); the `init_qpos` in `ant.xml` is unused.
6. Goal space = qpos[:15] with ranges ±(10, 10, 0.5, 1×4, 0.5/0.3 for joints); low noise σ = 1 on ±30 actions;
   high noise σ = 1; Adam 1e-4 / 1e-3; τ = 0.005; batch 100; buffers 200 k.

### 4.4 Our deviations from the paper (honest list)
- Budget: 2 M steps and 2–3 seeds for Ant tasks (paper: 10 M steps, 10 seeds); no Ant Gather.
- Extra toy environment (our own, numpy): PointMaze with 4-unit blocks, 200-step episodes, goal range ±6,
  success radius 1.5 (the official point maze differs).
- Success = final distance < 5 at the end of a 500-step episode (official `eval_utils.py`).
- Compute: Apple M2 Pro (Steps 1–5), a PC with i7-7700 + 2× GTX 1080 (Step 6). Throughput findings:
  small networks → CPU faster on the Mac (MPS 147 vs CPU 391 steps/s) but GPU faster on the PC (250 vs 99);
  1 torch thread per run gives ≈1.5× more total throughput when running seeds in parallel.

### 4.5 Testing strategy
- Each equation has a unit test (e.g. h keeps s + g fixed while the agent moves; r = −distance; the correction
  recovers a known goal from actions generated by the current policy; walls block the ant; Push block moves;
  Fall chasm drops the ant).

## 5. Experiments and results (≈4 pages)

### 5.1 TD3 sanity check — Pendulum (Fig. 1a, 1b)
- `results/step1_pendulum_return.png`, `results/step1_pendulum_q.png`. Numbers in §2.2.

### 5.2 Goal-conditioned low level alone (Fig. 2)
- Random relative goals U(−5,5)², c = 10, point mass with velocity.
- 30 k steps × 3 seeds: final distance to target 3.81 (do nothing) → 0.02 with ZERO_OBS, 0.08 without; both ≈ 100 %
  within 0.5. `results/step2_low_final_dist.png`.

### 5.3 PointMaze: hierarchy and correction (Figs. 3–4)
- Setup: ⊃-shaped maze, deceptive local optimum (greedy path to (0,8) hits the inner wall). 200 k steps, 5 seeds.
- Final success at (0,8) (mean of last 4 evaluations):
  | Variant | final success | seeds reaching ≥ 0.8 at some point |
  |---|---|---|
  | HIRO | **0.91** | 5 / 5 |
  | HIRO without correction | 0.29 | 2 / 5 |
  | flat TD3 | 0.20 | 1 / 5 |
- Learning curves `results/step4_pointmaze_success.png`; subgoal plots `results/step3_goals_nocorr.png`
  (failure mode: subgoals through the wall) and `results/step4_goals_corr.png` (subgoals around the ⊃).
- Note: with 3 seeds (Step 3) HIRO-no-correction and TD3 looked equal; 5 seeds were needed for a clear picture.

### 5.4 When does the correction relabel? (Figs. 5–6) — our own analysis
- During training 50–70 % of sampled goals are relabelled. Hypothesis "caused by exploration noise" **rejected**:
  fresh data with training noise → 0–2 % relabelled; data from a different (stale) low policy → 59–67 %.
- Applying the final low policy to the whole buffer, grouped by age: oldest 20 k steps 92 % relabelled (mean goal
  shift 4.45), newest 17 % (0.22), monotonic. `results/step4_relabel_vs_age.png`,
  `results/step4_relabel_analysis_seed{0,2}.png`.
- Interpretation: the correction is doing what the paper claims — repairing experience made stale by the changing
  low level. As far as we can tell, the paper does not show this analysis.

### 5.5 Ant Maze, Ant Push, Ant Fall [TODO-ANT]
- Setup: 2 M steps; Maze: HIRO / HIRO-no-correction / flat TD3 × 3 seeds; Push & Fall: HIRO × 2 seeds.
- Figures: `results/step6/antmaze_success.png`, `antmaze_dist.png`, `antmaze_all_targets.png`, `pushfall.png`,
  `relabel.png`; subgoal plots and GIF stills from `results/step7/`.
- Table: `results/step6/summary.md`, side by side with the paper's Table 1 (10 M steps, best policy):
  HIRO Ant Maze 0.99 ± 0.01, Ant Push 0.92 ± 0.04, Ant Fall 0.66 ± 0.07.
- Discussion points to fill in: does HIRO beat flat TD3 (paper: flat TD3 makes almost no progress)? Does the
  correction help on the Ant (paper: small effect on Maze, larger on Push/Fall)? Where are we on the paper's
  learning curves at 2 M steps?

## 6. Discussion (≈1 page)
- What made the method work: dense intrinsic reward from raw-state goals; relabelling old experience.
- What was surprising: hierarchy gave no clear benefit on the directly-controllable point without the correction;
  how much the official code differs in small details from the paper text; GPU vs CPU depends on the machine.
- Limitations: few seeds, short budgets, one toy env of our own design, no hyper-parameter search, no Ant Gather.
- Possible extensions: better approximation of the argmax in Eq. 5 (more candidates / gradient-based),
  learning the goal space (the authors' follow-up work), prioritising recent data instead of correcting old data.

## 7. Conclusion (≈0.3 page)

## References
- [HIRO] O. Nachum, S. Gu, H. Lee, S. Levine. Data-Efficient Hierarchical Reinforcement Learning. NeurIPS 2018.
- [12] S. Fujimoto, H. van Hoof, D. Meger. Addressing Function Approximation Error in Actor-Critic Methods. ICML 2018.
- [27] T. Lillicrap et al. Continuous Control with Deep Reinforcement Learning. ICLR 2016.
- Official code: github.com/tensorflow/models/tree/master/research/efficient-hrl
- MuJoCo (Todorov et al. 2012); Gymnasium.

## Appendix
- A. Full hyper-parameter table (from `docs/notes.md`).
- B. How to reproduce: commands for every figure (`scripts/train_*.py`, `run_step6.sh`, `make_step7_figures.sh`).
- C. Compute and wall-clock times.
