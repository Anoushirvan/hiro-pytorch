"""Ant Maze / Ant Push / Ant Fall (paper App. B.2) on the official `mujoco` Python bindings.

Port of the official `environments/{ant.py, maze_env.py, maze_env_utils.py, create_maze_env.py}`:
  * Ant model = official ant.xml (motors ctrlrange ±30, i.e. the paper's reduced "gear ±30"),
    timestep 0.02 (RK4), frame_skip 5  -> one env step = 0.1 s.
  * Maze built from 8×8 blocks (scaling 8), height 0.5*8 = 4. Agent starts at (0, 0).
      Maze : ⊃ corridor                         eval target (0, 16)
      Push : a movable block (x,y) in the way   target (0, 19)
      Fall : a chasm; a block (y,z) must be pushed into it to form a bridge; maze elevated; target (0, 27, 4.5)
  * obs = [qpos[:15], qvel[:14], t*0.001, target]   (n_bins=0 and observe_blocks=False in the official
    config for these tasks, so block positions are NOT observed).  state_dim = 30, goal dims = qpos[:15].
  * reward = -||target - (x,y[,z])||; episode = 500 steps, no termination; success = final distance < 5.
  * reset: ant qpos += U(-0.1, 0.1), qvel = N(0, 0.1²)  (official ant.py reset_model).
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET

import gymnasium as gym
import mujoco
import numpy as np

ASSET = os.path.join(os.path.dirname(__file__), "assets", "ant.xml")

# Movable-block types (official maze_env_utils.Move)
MOVE_XY, MOVE_YZ = "XY", "YZ"

LAYOUTS = {   # copied from official maze_env_utils.construct_maze (1 = wall, -1 = chasm, 'r' = start)
    "Maze": [[1, 1, 1, 1, 1],
             [1, "r", 0, 0, 1],
             [1, 1, 1, 0, 1],
             [1, 0, 0, 0, 1],
             [1, 1, 1, 1, 1]],
    "Push": [[1, 1, 1, 1, 1],
             [1, 0, "r", 1, 1],
             [1, 0, MOVE_XY, 0, 1],
             [1, 1, 0, 1, 1],
             [1, 1, 1, 1, 1]],
    "Fall": [[1, 1, 1, 1],
             [1, "r", 0, 1],
             [1, 0, MOVE_YZ, 1],
             [1, -1, -1, 1],
             [1, 0, 0, 1],
             [1, 1, 1, 1]],
}

LAYOUTS["Open"] = [["r"]]   # diagnostics only: no walls, just the floor

TASKS = {   # target sampling for training, the main evaluation target, which coords the reward uses
    "Maze": dict(train_low=(-4, -4), train_high=(20, 20), eval_target=(0, 16), dims=2),
    "Push": dict(train_low=None, train_high=None, eval_target=(0, 19), dims=2),       # fixed target
    "Fall": dict(train_low=None, train_high=None, eval_target=(0, 27, 4.5), dims=3),  # fixed target
    "Open": dict(train_low=None, train_high=None, eval_target=(0, 0), dims=2),        # diagnostics
}


def build_xml(maze_id: str, scaling: float = 8.0, height: float = 0.5) -> str:
    """Insert maze geoms into ant.xml, mirroring official MazeEnv.__init__."""
    tree = ET.parse(ASSET)
    world = tree.find(".//worldbody")
    world.find(".//geom[@name='floor']").set("size", "100 100 40")   # cosmetic: no sky visible in renders
    layout = LAYOUTS[maze_id]
    (ri, rj), = [(i, j) for i, row in enumerate(layout) for j, v in enumerate(row) if v == "r"]
    tx, ty = rj * scaling, ri * scaling
    elevated = any(-1 in row for row in layout)
    has_blocks = any(v in (MOVE_XY, MOVE_YZ) for row in layout for v in row)
    h_off = height * scaling if elevated else 0.0
    if elevated:
        tree.find(".//body[@name='torso']").set("pos", f"0 0 {0.75 + h_off:.2f}")
    if has_blocks:  # better contact detection with movable blocks (official)
        tree.find(".//default").find(".//geom").set("solimp", ".995 .995 .01")
    half, hh = 0.5 * scaling, height / 2 * scaling
    for i, row in enumerate(layout):
        for j, v in enumerate(row):
            x, y = j * scaling - tx, i * scaling - ty
            if elevated and v != -1:   # platform under every non-chasm cell
                ET.SubElement(world, "geom", name=f"elevated_{i}_{j}", type="box", pos=f"{x} {y} {hh}",
                              size=f"{half} {half} {hh}", contype="1", conaffinity="1", rgba="0.9 0.9 0.9 1")
            if v == 1:
                ET.SubElement(world, "geom", name=f"block_{i}_{j}", type="box", pos=f"{x} {y} {h_off + hh}",
                              size=f"{half} {half} {hh}", contype="1", conaffinity="1", rgba="0.4 0.4 0.4 1")
            elif v in (MOVE_XY, MOVE_YZ):
                falling = v == MOVE_YZ
                shrink = 0.99 if falling else 1.0
                body = ET.SubElement(world, "body", name=f"movable_{i}_{j}", pos=f"{x} {y} {h_off + hh}")
                ET.SubElement(body, "geom", name=f"mblock_{i}_{j}", type="box", pos="0 0 0",
                              size=f"{half * shrink} {half * shrink} {hh}", mass="0.001" if falling else "0.0002",
                              contype="1", conaffinity="1", rgba="0.9 0.1 0.1 1")
                axes = {"x": "1 0 0", "y": "0 1 0", "z": "0 0 1"}
                for ax in ("xy" if v == MOVE_XY else "yz"):
                    rng = f"{-h_off} 0" if ax == "z" else f"{-scaling} {scaling}"
                    ET.SubElement(body, "joint", name=f"movable_{ax}_{i}_{j}", type="slide", axis=axes[ax],
                                  armature="0", damping="0.0", margin="0.01", pos="0 0 0", range=rng,
                                  limited="true" if (falling or ax == "z") else "false")
    return ET.tostring(tree.getroot(), encoding="unicode")


class AntMazeEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    state_dim = 30
    goal_idx = np.arange(15)          # goals live in qpos[:15]  (hiro_orig.gin SUBGOAL_DIM = 15)
    frame_skip = 5

    def __init__(self, maze_id: str = "Maze", max_steps: int = 500, success_radius: float = 5.0, scaling=8.0):
        self.maze_id, self.task = maze_id, TASKS[maze_id]
        self.scaling = scaling
        self.model = mujoco.MjModel.from_xml_string(build_xml(maze_id, scaling))
        self.data = mujoco.MjData(self.model)
        # Official gym MujocoEnv uses the model's default qpos0 (torso at its XML pos, z=0.75 (+4 if elevated),
        # hinge joints 0); the <custom> init_qpos in ant.xml is NOT used by it, so we don't either.
        self.init_qpos = self.model.qpos0.copy()
        self.max_steps, self.success_radius = max_steps, success_radius
        self.dims = self.task["dims"]
        self.target = np.array(self.task["eval_target"], dtype=np.float64)
        obs_dim = self.state_dim + self.dims
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        lo, hi = self.model.actuator_ctrlrange[:, 0], self.model.actuator_ctrlrange[:, 1]
        self.action_space = gym.spaces.Box(lo.astype(np.float32), hi.astype(np.float32), dtype=np.float32)
        self._renderer = None
        self._style()
        elevated = any(-1 in row for row in LAYOUTS[maze_id])
        self.wall_top = 0.5 * scaling * (2 if elevated else 1)   # z of wall tops (for drawing markers above them)
        # wall-block centres for plotting (same convention as envs/maze.py)
        layout = LAYOUTS[maze_id]
        (ri, rj), = [(i, j) for i, row in enumerate(layout) for j, v in enumerate(row) if v == "r"]
        self.blocks = np.array([[(j - rj) * scaling, (i - ri) * scaling] for i, row in enumerate(layout)
                                for j, v in enumerate(row) if v == 1], dtype=np.float64).reshape(-1, 2)

    # ------------------------------------------------------------------ gym API
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        qpos, qvel = self.init_qpos.copy(), np.zeros(self.model.nv)
        qpos[:15] += self.np_random.uniform(-0.1, 0.1, size=15)
        qvel[:14] = self.np_random.normal(0.0, 0.1, size=14)
        self.data.qpos[:], self.data.qvel[:] = qpos, qvel
        mujoco.mj_forward(self.model, self.data)
        self.t = 0
        if options and "target" in options:
            self.target = np.asarray(options["target"], dtype=np.float64)
        elif self.task["train_low"] is not None:
            self.target = self.np_random.uniform(self.task["train_low"], self.task["train_high"])
        else:
            self.target = np.array(self.task["eval_target"], dtype=np.float64)
        return self._obs(), {}

    def step(self, action):
        self.data.ctrl[:] = np.clip(action, self.action_space.low, self.action_space.high)
        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)
        self.t += 1
        dist = float(np.linalg.norm(self.target - self.data.qpos[: self.dims]))
        info = {"dist": dist, "success": dist < self.success_radius}
        return self._obs(), -dist, False, self.t >= self.max_steps, info

    def _obs(self):
        return np.concatenate([self.data.qpos[:15], self.data.qvel[:14], [self.t * 0.001],
                               self.target]).astype(np.float32)

    # ------------------------------------------------------------------ rendering (cosmetic only)
    def _style(self):
        """Softer lighting and clearer colours for top-down renders. Does not affect physics."""
        m = self.model
        m.light_diffuse[:], m.light_ambient[:], m.light_specular[:] = 0.55, 0.25, 0.05
        m.vis.headlight.ambient[:], m.vis.headlight.diffuse[:], m.vis.headlight.specular[:] = 0.15, 0.25, 0.0
        for i in range(m.ngeom):
            n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
            if n.startswith("block_"):
                m.geom_rgba[i] = [0.25, 0.25, 0.3, 1]        # walls
            elif n.startswith("elevated_"):
                m.geom_rgba[i] = [0.75, 0.75, 0.72, 1]       # Fall platform
            elif n == "floor":
                m.geom_rgba[i] = [0.92, 0.93, 0.9, 1]
            elif n.startswith("mblock_"):
                m.geom_rgba[i] = [0.85, 0.2, 0.2, 1]         # movable block
            else:
                m.geom_rgba[i] = [0.85, 0.55, 0.15, 1]       # ant

    def render(self, width=480, height=480, markers=()):
        """Top-down RGB frame of the whole maze. `markers` = [(xyz, rgba, radius), ...] drawn as spheres
        (e.g. the current subgoal and the target)."""
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height, width)
            self._cam = mujoco.MjvCamera()
            self._cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            lo, hi = self.blocks.min(0) - self.scaling / 2, self.blocks.max(0) + self.scaling / 2
            self._cam.lookat[:] = [*(lo + hi) / 2, 0]
            half_fov = np.deg2rad(self.model.vis.global_.fovy / 2)
            self._cam.distance = 1.05 * max(hi - lo) / 2 / np.tan(half_fov)
            self._cam.elevation, self._cam.azimuth = -90, 90
        self._renderer.update_scene(self.data, camera=self._cam)
        scn = self._renderer.scene
        for pos, rgba, radius in markers:
            if scn.ngeom >= scn.maxgeom:
                break
            mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, [radius, 0, 0],
                                np.asarray(pos, dtype=np.float64), np.eye(3).flatten(),
                                np.asarray(rgba, dtype=np.float32))
            scn.ngeom += 1
        return self._renderer.render()


class AntRunEnv(AntMazeEnv):
    """Diagnostic task (not in the paper): open floor, reward = forward velocity dx/dt (the locomotion part of
    gym's Ant reward, without control cost). Tests whether our Ant port can learn to walk at all."""

    def __init__(self, max_steps: int = 500):
        super().__init__("Open", max_steps=max_steps)

    def step(self, action):
        x0 = self.data.qpos[0]
        obs, _, term, trunc, info = super().step(action)
        dt = self.model.opt.timestep * self.frame_skip
        info["x"] = float(self.data.qpos[0])
        return obs, float((self.data.qpos[0] - x0) / dt), term, trunc, info
