# Slides — outline (~15 min RL class, 14 slides)

> Final format: PowerPoint (.pptx). One idea per slide; figures from `results/`. `[TODO-ANT]` = waits for Step 6.
> Timing budget in brackets; speaker notes under each slide.

---

### 1. Title [0:30]
Re-implementing HIRO — Data-Efficient Hierarchical RL (Nachum et al., NeurIPS 2018). Name, course, date.
Visual: Ant Maze render (`results/step5_envs.png`, left panel) or GIF still.

### 2. The problem: long-horizon control [1:00]
- An ant must *walk* (torques, 0.1 s) **and** *navigate* (a maze, hundreds of steps).
- Flat RL explores at the wrong time scale → paper: flat agents make almost no progress.
- Visual: Ant Maze with the deceptive direct path crossed out.
Notes: the greedy direction to the target runs into the wall — exploration has to go *away* first.

### 3. HIRO in one picture [1:30]
- High level: every c = 10 steps propose a goal g = desired change of the state.
- Low level: reach it; reward = −‖s + g − s′‖.
- Both trained off-policy with TD3.
- Visual: paper Fig. 2-style diagram (redrawn).

### 4. Background: DDPG → TD3 [1:30]
- DDPG: deterministic actor + critic, Bellman target (Eq. 1).
- TD3: twin critics (min), target smoothing, delayed actor → less overestimation.
- Visual: `results/step1_pendulum_q.png` — "our check: DDPG over-, TD3 slightly under-estimates".

### 5. Goals as states (Eqs. 2–3) [1:00]
- h(s, g, s′) = s + g − s′ keeps the absolute target fixed.
- Dense intrinsic reward from step 1, no learned representation.
- Visual: `results/step2_low_final_dist.png` (low level alone learns to reach goals in ~2 k steps).

### 6. The non-stationarity problem [1:00]
- Old high-level transition: "I chose g, the agent went *there*". The low level has since changed → with the same
  g it would now go elsewhere. Replaying it teaches the high level wrong "dynamics".
- Visual: simple sketch: same g, old vs new low-level trajectory.

### 7. Off-policy correction (Eq. 5) [1:30]
- Relabel g with the goal that best explains the stored actions under the *current* low policy.
- Score = −½ Σ‖a_i − μ_lo(s_i, g̃_i)‖²; 10 candidates (original, s_{t+c} − s_t, 8 Gaussian samples); argmax.
- Visual: candidate goals as dots, the winner highlighted.

### 8. Our implementation [1:00]
- PyTorch, built in 8 steps, 25 unit tests, paper ↔ code map.
- "Hidden" details from the official code: Huber loss, high reward × 0.1, low level ignores x,y, normalised
  correction score.
- Visual: repo tree + equation→function table (small).

### 9. PointMaze: does the correction matter? [1:30]
- 5 seeds, 200 k steps: HIRO 0.91 · no correction 0.29 · flat TD3 0.20.
- Visual: `results/step4_pointmaze_success.png`.

### 10. What the agent does [1:00]
- Failure without correction: subgoals through the wall. Success: subgoals lead around the ⊃.
- Visual: `results/step3_goals_nocorr.png` (one failing panel) next to `results/step4_goals_corr.png` (one success).

### 11. When does the correction act? (our analysis) [1:00]
- Old data: 92 % relabelled; newest: 17 %. Exploration noise alone: 0–2 %.
- → it repairs experience made stale by the changing low level, exactly the paper's motivation.
- Visual: `results/step4_relabel_vs_age.png`.

### 12. Ant Maze results [1:30] [TODO-ANT]
- HIRO vs no correction vs flat TD3 at 2 M steps (paper: 10 M steps, 10 seeds).
- Visual: `results/step6/antmaze_success.png` + GIF still / GIF of best seed.

### 13. Ant Push & Ant Fall [0:45] [TODO-ANT]
- Visual: `results/step6/pushfall.png`, subgoal plot.
- Table vs paper's Table 1 (Maze 0.99, Push 0.92, Fall 0.66 at 10 M).

### 14. Takeaways & limitations [1:00]
- Raw-state goals + off-policy training = simple and sample-efficient; the correction is what makes off-policy
  high-level training work.
- Limitations: compute (2 M vs 10 M steps, 2–3 seeds), toy env of our own, no Ant Gather.
- Lessons: verify each piece in isolation; read the official code; seeds matter (3 vs 5 changed our conclusion).

(Backup slides: hyper-parameter table; throughput CPU vs GPU; relabel histogram.)
