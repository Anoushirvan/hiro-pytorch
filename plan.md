# plan.md — HIRO re-implementation roadmap & progress log

> Living document. Update the checkboxes and the log at the end of every step.

## Context
Course project (RL). Goal: understand HIRO deeply by re-implementing it step by step, then produce
(1) a private GitHub repo with our implementation, (2) a Word work report (English), (3) a ~15-min
PowerPoint for an RL class (~12–15 slides). Accuracy parity with the paper is **not** required — this is
educational; reduced step budgets/seeds are fine and will be stated honestly.

Inputs in `presentation/`: main paper PDF, ref [12] TD3 (Fujimoto et al.), ref [27] DDPG (Lillicrap et al.).
Official TF code (reference only): tensorflow/models/research/efficient-hrl.

Decisions made with the user:
- Envs: ladder — 2D PointMaze (fast debugging) → paper's Ant Maze → Ant Push / Ant Fall.
- Experiments: HIRO vs flat TD3; with vs without off-policy correction; Push/Fall; goal-trajectory visualizations.
- Style: I (Claude) write each module, explain paper-equation ↔ code, add tests, then **stop for the user's review before next step**.
- Hardware: Apple M2 Pro 16 GB, conda env `new_torch` (Python 3.12, torch 2.12, numpy, matplotlib). No GPU cluster.
- Repo: local git from the start; private GitHub repo via `gh` at the end (name confirmed with user then).

## Project layout (new folder `presentation/hiro-pytorch/`)
```
hiro_pytorch/
  envs/        point_maze.py, maze_env.py (port of official maze_env/ant_maze_env), ant.py, assets/ant.xml
  td3.py       Actor/Critic, TD3 update (clipped double-Q, target smoothing, delayed actor)   [ref 12]
  buffers.py   LowReplayBuffer, HighReplayBuffer (stores full c-step state/action sequences)
  hiro.py      HIRO agent: goal transition h, intrinsic reward r, off-policy correction, hi/lo loop
  utils.py     seeding, logging (CSV), normalisation helpers
scripts/       train.py (--algo {td3,hiro} --correction {on,off} --env ...), eval.py, plot.py, viz_goals.py
configs/       yaml per env/variant (hyperparams lifted from official configs/*.gin)
tests/         pytest: buffer shapes, h/r math, correction picks true goal on synthetic policy, TD3 on Pendulum
notebooks/     01_ddpg_td3.ipynb ... walkthroughs mirroring each step (for learning + report figures)
results/       csv logs, plots, gifs
docs/          report.docx, slides.pptx
```

## Step-by-step implementation (each step ends with a checkpoint for user review + git commit)
0. **Setup & reading map.** git init; install into `new_torch`: `gymnasium[mujoco]`, `mujoco`, `pytest`, `pyyaml`,
   `imageio`, `python-docx`/`python-pptx` later. Clone official repo into a separate read-only folder; extract exact
   hyperparams from `configs/hiro_orig.gin`, `agent.py`, `context/`, `environments/` into a cheat-sheet (`docs/notes.md`).
1. **DDPG → TD3 (refs [27], [12]).** Implement DDPG update (Eq. 1) then TD3's 3 fixes. Verify on Gymnasium
   `Pendulum-v1` (should reach ≈ −200 within ~20k steps). Explain overestimation bias.
2. **Goal-conditioned low level.** Goal transition h(s,g,s') = s+g−s' (Eq. 2), intrinsic reward −‖s+g−s'‖ (Eq. 3)
   on the goal subspace, goal-conditioned critic (Eq. 4). Unit tests for h and r.
3. **PointMaze env + two-level HIRO without correction.** High level acts every c=10 steps, outputs relative goal;
   high replay stores (s_t, g_t, ΣR, s_{t+c}) plus raw state/action sequences. Train on PointMaze; visualize subgoals.
4. **Off-policy correction (Eq. 5, the key contribution).** Candidates: original g, s_{t+c}−s_t, 8 Gaussian samples
   around it (clipped to goal range) = 10; roll each through h over c steps, score
   −½ Σ‖a_i − μ_lo(s_i, g̃_i)‖² ; relabel with argmax. Test: with a synthetic known low policy the true goal wins.
5. **Ant + Ant Maze port.** Port ant.xml (gear ±30), maze builder (8×8×8 blocks, ⊃-maze), obs = qpos, qvel, t;
   dt=0.02, frame_skip=5, 500-step episodes, training target uniform in maze, eval target (0,16), success = dist < 5;
   goal space = positional obs. Sanity: random/scripted ant moves, maze collision works, render a frame.
6. **Experiments (reduced budget).** Ant Maze: HIRO, HIRO-no-correction, flat TD3 — e.g. ~1–2M steps, 2–3 seeds
   (exact budget set after measuring steps/sec in step 5). Then Ant Push (eval target (0,19)) and Ant Fall
   (3-D target (0,27,4.5)) with HIRO (+ ablation if time). Long runs in background with checkpointing/resume.
7. **Visualization & analysis.** Learning curves (mean ± stderr), goal-trajectory plots over maze map, GIFs of ant + subgoal markers.
8. **Deliverables.** README with how-to-run + results; Word report (intro, background DDPG/TD3, HIRO method, our
   implementation & differences from official code, experiments, discussion, limitations); 12–15 slide .pptx
   (using the docx/pptx skills); create private GitHub repo via `gh` (confirm name first) and push.

## Hyperparameters to start from (verify against official gin configs in step 0)
c=10; γ=0.99; τ=0.005; actor lr 1e-4, critic lr 1e-3; MLP 300-300 (verify); batch 100; replay 200k;
TD3 target noise 0.2 clip 0.5, policy delay 2; low expl. noise σ=1.0 (action range ±30), high noise per-dim
scaled to goal range; high-level reward scale 0.1; low trains every step, high every c steps; correction with 10 candidates.

## Verification
- `pytest tests/` passes after each step.
- Step 1: TD3 solves Pendulum. Step 3/4: HIRO reaches goals in PointMaze; correction ablation shows a difference.
- Step 5: env renders, episode lengths/rewards/success logic match paper Appendix B.
- Step 6: success-rate curves for Ant Maze above flat TD3 (flat TD3 expected ≈0 success, as in paper).
- Final: fresh clone of the private repo runs `python scripts/train.py --env point_maze --algo hiro` end-to-end.

## Progress
- [x] Step 0 — setup, official code cloned to `../reference_official/`, cheat-sheet `docs/notes.md`
- [x] Step 1 — DDPG → TD3 (flags for each TD3 fix), verified on Pendulum-v1
- [ ] Step 2 — goal-conditioned low level (h, r, Eq. 2–4)
- [ ] Step 3 — PointMaze + HIRO without correction
- [ ] Step 4 — off-policy correction (Eq. 5)
- [ ] Step 5 — Ant + Ant Maze port
- [ ] Step 6 — experiments (Maze: HIRO / no-corr / flat TD3; Push; Fall)
- [ ] Step 7 — plots, goal-trajectory visualisations, GIFs
- [ ] Step 8 — README, Word report, PowerPoint, private GitHub repo

## Log / decisions
- 2026-11-15: plan approved. Claude writes each module, user reviews at each checkpoint.
- 2026-11-15: Step 0 done. Paper-vs-code differences found: Huber critic loss; high reward ×0.1; low-level
  sees x,y zeroed (ZERO_OBS); correction error normalised by action range; candidate σ = 0.5·goal half-range.
- 2026-11-17: Step 1 started — DDPG and TD3 in one class with per-fix flags (twin Q, target smoothing, delayed actor).
- 2026-11-17: Step 1 done. `hiro_pytorch/{td3,buffers,utils}.py`, `scripts/{train_flat,plot}.py`, 6 tests pass.
  Pendulum 30k steps × 3 seeds: DDPG −113, TD3 −112 (both solve; DDPG reaches it faster here).
  Overestimation probe at 30k: DDPG Q −29 vs true −32 (over by ≈3); TD3 Q −35 vs true −33 (slightly under) —
  matches TD3 paper's claim qualitatively; effect small on Pendulum. ~500 sps single run, ~200 sps with 6 in parallel.
  Plots: results/step1_pendulum_{return,q}.png.
