# HIRO cheat-sheet (paper ↔ official code)

Sources: paper arXiv:1805.08296 (Nachum et al., NeurIPS 2018); official TF code
`tensorflow/models/research/efficient-hrl` (files referenced below are relative to it).

## 1. Base learner: TD3 (ref [12]) on top of DDPG (ref [27])
| Item | Value | Where |
|---|---|---|
| Actor MLP | 300-300, ReLU, tanh output scaled to action range | `configs/base_uvf.gin` |
| Critic MLP | state→300, then concat action →300 →1 (two critics, TD3) | `configs/base_uvf.gin` |
| Critic loss | **Huber** (δ=1), not MSE | `configs/base_uvf.gin` |
| Optimizer | Adam; actor lr 1e-4, critic lr 1e-3 (both levels) | `configs/train_uvf.gin` |
| γ | 0.99 (both levels) | `configs/train_uvf.gin` |
| τ (soft target update) | 0.005 | `td3_update_targets.tau` |
| Target/actor update period | 2 (TD3 delayed policy update) | `train_uvf.target_update_period` |
| Batch size | 100 | `train_uvf.batch_size` |
| Replay size | 200 000 per level | `configs/train_uvf.gin` |
| Warm-up | `initial_episodes=5` random episodes (2 500 steps) | `configs/train_uvf.gin` |

## 2. Hierarchy
| Item | Value | Where |
|---|---|---|
| c (high-level period) | 10 | `meta_action_every_n` |
| Low trains | every env step | `num_collect_per_update=1` |
| High trains | every 10 env steps | `num_collect_per_meta_update=10` |
| Goal dim | 15 = ant qpos[:15] (x,y,z, quaternion(4), 8 joint angles) | `hiro_orig.gin` SUBGOAL_DIM |
| Goal range (high action space) | ±(10,10,0.5,1,1,1,1,0.5,0.3,0.5,0.3,0.5,0.3,0.5,0.3) | `hiro_orig.gin` |
| Goal transition h | g' = s + g − s'  (Eq. 2) | `relative_context_transition_fn` |
| Low reward r | −‖s + g − s'‖₂ on goal dims (Eq. 3) | `uvf/negative_distance, relative_context=True` |
| High reward | **0.1 ×** Σ env rewards over c steps | `train.py:192` |
| Low expl. noise | Gaussian σ = 1.0 (actions in ±30) | `uvf_add_noise_fn.stddev` |
| High expl. noise | Gaussian σ = 1.0 | `META_EXPLORE_NOISE` |
| ZERO_OBS | low-level networks see x,y zeroed out | `ddpg_networks.py:133` |

## 3. Off-policy correction (Eq. 5)
* 10 candidates: 8 samples ~ N(s_{t+c} − s_t, (0.5·half_range)²), plus s_{t+c} − s_t, plus original g; clipped to goal range
  (`agent.py: InverseDynamics.sample`, `sc=0.5`).
* Each candidate rolled through h over the c steps, low policy queried (deterministic, no noise).
* Score = −Σ_i ‖(a_i − μ_lo(s_i, g̃_i)) / half_action_range‖² (`agent.py: log_probs`), argmax (`train.py:253-261`).

## 4. Environments (paper App. B + `environments/`)
* MuJoCo, timestep 0.02, frame_skip 5; ant gear reduced to ±30. Episodes 500 steps (no termination).
* Obs = qpos[:15], qvel[:14], t·0.001  (+ target appended by the high level context).
* Maze scaling 8 (each block 8×8 units, height 0.5 for Maze).
* Ant Maze layout (`maze_env_utils.construct_maze`): `⊃` shape, start 'r' at (0,0);
  train targets uniform in [−4,20]²; eval targets (16,0), (16,16), **(0,16)** (paper reports (0,16)).
* Ant Push: target (0,19); Ant Fall: 3-D target (0,27,4.5).
* Env reward = −‖target − xy(z)‖; success = final distance < 5.
