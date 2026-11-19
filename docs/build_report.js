// Builds docs/report.docx (work report) from the text below and the figures in results/.
//   NODE_PATH=<dir with node_modules/docx> node docs/build_report.js
// Inline markup in paragraphs: **bold**, *italic*, _{subscript}, ^{superscript}, `code`.
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, ImageRun, Table, TableRow, TableCell,
  WidthType, BorderStyle, ShadingType, LevelFormat, PageBreak, TableOfContents, Footer, PageNumber,
} = require("docx");

const ROOT = path.join(__dirname, "..");
const RES = (f) => path.join(ROOT, "results", f);
const FONT = "Calibri";
const CONTENT_W = 9026; // A4 width 11906 - 2 * 1440 margins (DXA)

// ------------------------------------------------------------------ inline markup -> runs
function runs(text, base = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*|_\{[^}]+\}|\^\{[^}]+\}|`[^`]+`)/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), font: FONT, ...base }));
    const t = m[0];
    if (t.startsWith("**")) out.push(new TextRun({ text: t.slice(2, -2), bold: true, font: FONT, ...base }));
    else if (t.startsWith("_{")) out.push(new TextRun({ text: t.slice(2, -1), subScript: true, font: FONT, ...base }));
    else if (t.startsWith("^{")) out.push(new TextRun({ text: t.slice(2, -1), superScript: true, font: FONT, ...base }));
    else if (t.startsWith("`")) out.push(new TextRun({ text: t.slice(1, -1), font: "Consolas", size: 20, ...base }));
    else out.push(new TextRun({ text: t.slice(1, -1), italics: true, font: FONT, ...base }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), font: FONT, ...base }));
  return out;
}

const P = (text, opts = {}) => new Paragraph({ children: runs(text), spacing: { after: 120, line: 276 },
  alignment: AlignmentType.JUSTIFIED, ...opts });
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text: t })] });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text: t })] });
const BUL = (t, level = 0) => new Paragraph({ numbering: { reference: "bullets", level }, children: runs(t),
  spacing: { after: 60, line: 276 } });
const NUM = (t, ref = "numbers") => new Paragraph({ numbering: { reference: ref, level: 0 }, children: runs(t),
  spacing: { after: 60, line: 276 } });
const EQ = (t) => new Paragraph({ children: runs(t, { italics: false }), alignment: AlignmentType.CENTER,
  spacing: { before: 80, after: 120 } });

let figNo = 0, tabNo = 0;
function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}
function figure(file, widthPx, caption) {
  const { w, h } = pngSize(RES(file));
  figNo += 1;
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 }, keepNext: true,
      children: [new ImageRun({ type: "png", data: fs.readFileSync(RES(file)),
        transformation: { width: widthPx, height: Math.round((widthPx * h) / w) },
        altText: { title: `Figure ${figNo}`, description: caption, name: file } })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 },
      children: [new TextRun({ text: `Figure ${figNo}. `, bold: true, font: FONT, size: 20 }),
        ...runs(caption, { size: 20, italics: true })] }),
  ];
}
function figurePair(f1, f2, caption) {
  figNo += 1;
  const cellW = CONTENT_W / 2;
  const none = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
  const borders = { top: none, bottom: none, left: none, right: none };
  const cell = (f) => {
    const { w, h } = pngSize(RES(f));
    return new TableCell({ width: { size: cellW, type: WidthType.DXA }, borders,
      children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new ImageRun({ type: "png",
        data: fs.readFileSync(RES(f)), transformation: { width: 295, height: Math.round((295 * h) / w) },
        altText: { title: `Figure ${figNo}`, description: caption, name: f } })] })] });
  };
  return [
    new Table({ width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: [cellW, cellW],
      rows: [new TableRow({ children: [cell(f1), cell(f2)] })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 60, after: 200 },
      children: [new TextRun({ text: `Figure ${figNo}. `, bold: true, font: FONT, size: 20 }),
        ...runs(caption, { size: 20, italics: true })] }),
  ];
}
function table(headers, rows, widths, caption) {
  tabNo += 1;
  const total = widths.reduce((a, b) => a + b, 0);
  const border = { style: BorderStyle.SINGLE, size: 4, color: "B0B7C3" };
  const borders = { top: border, bottom: border, left: border, right: border };
  const mk = (t, i, head) => new TableCell({ width: { size: widths[i], type: WidthType.DXA }, borders,
    shading: head ? { fill: "DCE6F5", type: ShadingType.CLEAR, color: "auto" } : undefined,
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    children: [new Paragraph({ children: runs(String(t), head ? { size: 19, bold: true } : { size: 19 }) })] });
  return [
    new Paragraph({ spacing: { before: 120, after: 60 }, keepNext: true,
      children: [new TextRun({ text: `Table ${tabNo}. `, bold: true, font: FONT, size: 20 }),
        ...runs(caption, { size: 20, italics: true })] }),
    new Table({ width: { size: total, type: WidthType.DXA }, columnWidths: widths,
      rows: [new TableRow({ tableHeader: true, children: headers.map((h, i) => mk(h, i, true)) }),
        ...rows.map((r) => new TableRow({ children: r.map((c, i) => mk(c, i, false)) }))] }),
    new Paragraph({ spacing: { after: 120 }, children: [] }),
  ];
}

// ------------------------------------------------------------------ title page
const titlePage = [
  new Paragraph({ spacing: { before: 2400 }, children: [] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 240 },
    children: [new TextRun({ text: "Re-implementing HIRO", font: FONT, size: 52, bold: true, color: "1F3B6E" })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 600 },
    children: [new TextRun({ text: "Data-Efficient Hierarchical Reinforcement Learning, step by step in PyTorch",
      font: FONT, size: 30, color: "2A6FDB" })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 }, children: [new TextRun({
    text: "Work report on Nachum, Gu, Lee & Levine, NeurIPS 2018 (arXiv:1805.08296)", font: FONT, size: 24, italics: true })] }),
  new Paragraph({ spacing: { before: 1800 }, children: [] }),
  ...["[Name]  ·  [Student ID]", "[Course]  ·  Instructor: [Instructor]", "[University]", "[Date]"].map((t) =>
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 100 },
      children: [new TextRun({ text: t, font: FONT, size: 24 })] })),
  new Paragraph({ children: [new PageBreak()] }),
];

// ------------------------------------------------------------------ body
const body = [
  H1("Abstract"),
  P("We study HIRO (HIerarchical Reinforcement learning with Off-policy correction), a two-level agent in which a " +
    "high-level policy proposes goal states every *c* = 10 steps and a goal-conditioned low-level policy is rewarded " +
    "for reaching them. Both levels are trained off-policy with TD3, which makes the method sample-efficient, and the " +
    "paper's key idea is an *off-policy correction* that relabels stored high-level actions so that old experience " +
    "stays valid while the low-level policy keeps changing. We re-implemented the method from scratch in PyTorch, one " +
    "component at a time, each verified by unit tests and a small experiment: DDPG/TD3, the goal machinery, the " +
    "two-level agent, the correction, and the MuJoCo Ant Maze, Ant Push and Ant Fall environments ported from the " +
    "authors' TensorFlow code. On a point-mass maze with a deceptive local optimum (5 seeds) the correction raises the " +
    "final success rate from 0.29 to 0.91, while a flat TD3 agent reaches 0.20. An analysis of the replay buffer shows " +
    "that the correction mainly rewrites *old* experience (92 % of the oldest versus 17 % of the newest segments), " +
    "exactly the non-stationarity the paper sets out to fix. For the Ant tasks we rely on the paper's reported results; " +
    "our own 2 M-step Ant runs did not learn within that budget, and we discuss why."),

  H1("1. Introduction"),
  P("Many interesting control problems are *long-horizon*: a simulated quadruped that has to walk through a maze must " +
    "choose joint torques every 0.1 s, yet the decision that matters — which corridor to take — plays out over " +
    "hundreds of steps. A flat reinforcement-learning agent explores at the time scale of torques and almost never " +
    "stumbles upon the behaviour that solves such tasks. Hierarchical reinforcement learning (HRL) addresses this by " +
    "letting a higher-level policy make decisions at a coarser time scale and delegating their execution to a " +
    "lower-level policy."),
  P("Nachum et al. [1] name three questions any HRL method has to answer: how to train the lower level so that it " +
    "produces semantically distinct behaviour, how to define the high-level actions, and how to train both levels " +
    "without an inordinate amount of experience. HIRO's answers are deliberately simple:"),
  BUL("**Goals are raw states.** The high-level action is a desired change of the state observation; no learned " +
    "goal representation, no hand-designed skills."),
  BUL("**Both levels are trained off-policy** with TD3 [2], a sample-efficient actor–critic algorithm."),
  BUL("**An off-policy correction** relabels old high-level experience, so that off-policy training stays valid " +
    "although the low-level policy keeps changing underneath the high level."),
  P("The goal of this project was to *understand* the method by re-implementing it, verifying each component in " +
    "isolation before combining them. This report describes the background (Section 2), the method (Section 3), our " +
    "implementation and the details we found only in the official code (Section 4), our experiments (Section 5) and " +
    "what we learned (Sections 6–7)."),

  H1("2. Background"),
  H2("2.1 Off-policy actor–critic: DDPG"),
  P("DDPG [3] learns a deterministic policy μ_{φ}(s) and a critic Q_{θ}(s, a). The critic minimises the Bellman error " +
    "over transitions (s_{t}, a_{t}, R_{t}, s_{t+1}) sampled from a replay buffer (Eq. 1 in [1]):"),
  EQ("( Q_{θ}(s_{t}, a_{t}) − R_{t} − γ Q_{θ′}(s_{t+1}, μ_{φ′}(s_{t+1})) )^{2}"),
  P("and the actor is trained to maximise Q_{θ}(s, μ_{φ}(s)) by following the deterministic policy gradient. Slowly " +
    "updated target networks (θ′ ← τθ + (1 − τ)θ′) stabilise the bootstrapped target. Experience is collected with a " +
    "*stochastic* behaviour policy a_{t} ~ N(μ_{φ}(s_{t}), σ^{2}); HIRO later relies on exactly this fact."),
  H2("2.2 TD3"),
  P("Fujimoto et al. [2] show that function-approximation error makes actor–critic methods *overestimate* values and " +
    "that the error accumulates through bootstrapping. TD3 adds three fixes to DDPG: (i) *clipped double-Q learning* — " +
    "two critics, and the target uses the minimum of both; (ii) *target policy smoothing* — clipped Gaussian noise is " +
    "added to the target action; (iii) *delayed policy updates* — the actor and the target networks are updated only " +
    "every second critic update."),
  P("We implemented DDPG and TD3 in a single class in which each fix can be switched on and off, and checked them on " +
    "Pendulum-v1 (30 k steps, 3 seeds; Figure 4). Both solve the task (final return ≈ −112). To measure " +
    "overestimation we compared the critic's Q(s, a) with the discounted return actually obtained from the same " +
    "states: DDPG's critic ends *above* the truth (−29 vs. −32), TD3's slightly *below* (−35 vs. −33). This is the " +
    "qualitative effect reported in [2]; on such an easy task it is small."),

  H1("3. The HIRO method"),
  H2("3.1 Two levels of policies"),
  P("HIRO has a high-level policy μ^{hi} and a low-level policy μ^{lo} (Figure 1). Every *c* steps the high level " +
    "observes the state s_{t} and outputs a goal g_{t}. Between two high-level decisions the goal is updated by a fixed " +
    "goal-transition function h. The low level observes (s_{t}, g_{t}) and outputs the action a_{t} that is applied to " +
    "the environment. Only the low level touches the environment; the high level acts through it."),
  ...figure("fig_hiro_diagram.png", 600, "Structure of HIRO (after Fig. 2 of [1]). The high level proposes a goal " +
    "every c steps; in between, the goal follows h so that the absolute target s_{t} + g_{t} stays fixed. The low " +
    "level acts every step and is rewarded for approaching that target."),
  H2("3.2 Goals as relative states"),
  P("A goal is a desired *change* of the positional part of the state: at time t the high level asks the low level to " +
    "reach an observation close to s_{t} + g_{t} after c steps. To keep that absolute target fixed while the agent " +
    "moves, the goal is updated by (Eq. 2 in [1])"),
  EQ("h(s_{t}, g_{t}, s_{t+1}) = s_{t} + g_{t} − s_{t+1}"),
  P("and the low level receives the intrinsic reward (Eq. 3 in [1])"),
  EQ("r(s_{t}, g_{t}, a_{t}, s_{t+1}) = − ‖ s_{t} + g_{t} − s_{t+1} ‖_{2}"),
  P("Because goals are raw states, the low level receives a dense, meaningful reward from the very first step — " +
    "long before the task reward provides any signal. The low level is then just a standard TD3 agent whose actor and " +
    "critic take g_{t} as an extra input (Eq. 4 in [1]). For the Ant, the goal space is the first 15 entries of the " +
    "observation: the torso position, its orientation and the eight joint angles."),
  H2("3.3 The off-policy correction"),
  P("The high level is also trained off-policy, on transitions (s_{t}, g_{t}, ΣR_{t:t+c−1}, s_{t+c}). This creates a " +
    "problem specific to HRL: a transition stored long ago was produced by an *older* low-level policy. Under the " +
    "current low level, the same goal g_{t} would lead somewhere else, so the stored transition no longer describes " +
    "the high level's actual \"dynamics\" — the high level faces a non-stationary problem."),
  P("HIRO's fix is to relabel the stored goal with the goal g̃_{t} that makes the stored low-level actions most likely " +
    "under the *current* low-level policy (Eq. 5 in [1]). With a Gaussian behaviour policy, the log-probability is, up " +
    "to a constant,"),
  EQ("log μ^{lo}(a_{t:t+c−1} | s_{t:t+c−1}, g̃_{t:t+c−1}) ∝ − ½ Σ_{i} ‖ a_{t+i} − μ^{lo}(s_{t+i}, g̃_{t+i}) ‖^{2}"),
  P("where the candidate's goal sequence g̃_{t+i} is obtained by rolling g̃_{t} forward with h. The maximisation is " +
    "approximated over ten candidates: the original goal, the observed displacement s_{t+c} − s_{t}, and eight samples " +
    "from a Gaussian centred at s_{t+c} − s_{t} (Figure 2)."),
  ...figure("fig_correction.png", 430, "The off-policy correction (illustrative example). The stored goal (orange) " +
    "was proposed to an older low-level policy; the relabelled goal (red star) is the candidate under which the " +
    "current low level best explains the stored behaviour."),
  H2("3.4 Training loop"),
  NUM("Every c steps the high level proposes g_{t} (with Gaussian exploration noise); otherwise g_{t+1} = h(s_{t}, g_{t}, s_{t+1})."),
  NUM("The low level acts, a_{t} = μ^{lo}(s_{t}, g_{t}) + noise; the transition (s_{t}, g_{t}, a_{t}, r_{t}, s_{t+1}, g_{t+1}) " +
    "goes to the low-level buffer."),
  NUM("After c steps the high-level transition — including the c states and actions needed for the correction — " +
    "goes to the high-level buffer, with reward 0.1 · ΣR."),
  NUM("The low level is updated every step; every c steps a high-level batch is sampled, relabelled, and used to " +
    "update the high level."),

  H1("4. Implementation"),
  H2("4.1 Approach"),
  P("We wrote the code in PyTorch with the Gymnasium interface and the official MuJoCo Python bindings, in eight " +
    "steps. Each step added one component, a unit test for every equation, and a small experiment; a development log " +
    "(`plan.md`) recorded every decision and result. The final code base has 27 unit tests. Table 1 maps the paper to " +
    "the code."),
  ...table(["Paper", "Code (hiro_pytorch/…)"], [
    ["Eq. 1 and the three TD3 fixes", "`td3.py`: `TD3._target_q`, `TD3.update`, `_soft_update` (each fix switchable)"],
    ["Eq. 2, goal transition h", "`goals.py`: `goal_transition`"],
    ["Eq. 3, intrinsic reward", "`goals.py`: `intrinsic_reward`"],
    ["Eq. 4, goal-conditioned low level", "`goals.py`: `low_level_input` + unchanged TD3"],
    ["High-level buffer with c-step sequences", "`buffers.py`: `HighReplayBuffer`"],
    ["Eq. 5, off-policy correction", "`hiro.py`: `candidate_goals`, `correction_scores`, `off_policy_correction`"],
    ["Appendix B environments", "`envs/ant_maze.py` (Ant Maze, Push, Fall); `envs/maze.py` (our PointMaze)"],
  ], [3300, 5726], "Where each part of the paper lives in the code."),
  P("Testing every equation separately paid off. For instance, one test lets an agent wander randomly and checks that " +
    "s_{t} + g_{t} never moves under h; another generates actions with the current low-level policy for a known goal " +
    "and checks that the correction recovers exactly that goal. Physics tests check that walls stop the ant, that the " +
    "block in Ant Push can be pushed, and that the ant falls into the chasm in Ant Fall."),
  H2("4.2 Details found only in the official code"),
  P("Reading the authors' code [4] next to the paper revealed details that the paper does not mention but that we " +
    "adopted:"),
  NUM("The critic loss is the **Huber loss** (δ = 1), not the squared error of Eq. 1.", "details"),
  NUM("The high-level reward is **0.1 ×** the sum of the c environment rewards.", "details"),
  NUM("The low-level networks see the state with the **x, y position set to zero** — consistent with relative goals, " +
    "which make the absolute position irrelevant for the low-level task.", "details"),
  NUM("In the correction the action error is **divided by the action range**, and the candidate Gaussian has a " +
    "standard deviation of **half the goal range**.", "details"),
  NUM("The **final layers** of actor and critic are initialised uniformly in ±0.003 (as recommended in [3]), so a " +
    "fresh actor outputs near-zero torques and near-zero goals; the critic also receives the action in its first " +
    "layer. We missed this at first (see Section 6).", "details"),
  NUM("The ant starts from the model's default pose; the initial pose stored in the model file is not used.", "details"),
  ...table(["Hyper-parameter", "Value"], [
    ["Networks (both levels)", "2 hidden layers × 300 units, ReLU; tanh output scaled to the action range"],
    ["Optimiser", "Adam; actor 1e-4, critic 1e-3"],
    ["Discount γ, target update τ", "0.99 (both levels), 0.005"],
    ["TD3", "target noise 0.2, clip 0.5 (fractions of the action range), policy delay 2"],
    ["Batch / replay size", "100 / 200 000 per level"],
    ["High-level period c", "10 steps"],
    ["Goal range (Ant)", "±(10, 10, 0.5, 1, 1, 1, 1, 0.5, 0.3, 0.5, 0.3, 0.5, 0.3, 0.5, 0.3)"],
    ["Exploration noise", "low: σ = 1 on torques in ±30; high: σ = 1"],
    ["Warm-up", "5 episodes collected with the noisy policies, no updates"],
  ], [3000, 6026], "Main hyper-parameters (taken from the official configuration files)."),
  H2("4.3 Environments"),
  P("We ported Ant Maze, Ant Push and Ant Fall (Figure 3) from the authors' TensorFlow code to the current MuJoCo " +
    "bindings, keeping the ant model, the 8 × 8 maze blocks, the observation (joint positions, joint velocities, time) " +
    "and the 500-step episodes. Success means ending an episode within 5 units of the target. Because Ant runs are " +
    "slow, we also built a small, fast *PointMaze* of our own: a point mass with velocity in the same ⊃-shaped layout, " +
    "with 4-unit blocks, 200-step episodes and the hard target (0, 8) at the end of the corridor. Heading straight " +
    "for the target runs into the inner wall — a deceptive local optimum."),
  ...figure("step5_envs.png", 560, "Our ports of Ant Maze, Ant Push and Ant Fall (top view). Green: target; blue: " +
    "an example subgoal; red: the movable block."),
  H2("4.4 Deviations from the paper"),
  BUL("Our PointMaze is our own design (the official point maze differs); it was used for all our quantitative results."),
  BUL("The paper trains for 10 M steps with 10 seeds per curve. We used 200 k steps and 5 seeds on the PointMaze and " +
    "2 M steps with 2–3 seeds on the Ant tasks; the Ant Gather task was not implemented."),
  BUL("Compute: an Apple M2 Pro laptop and a PC with an i7-7700 and two GTX 1080 GPUs. With networks this small the " +
    "best device depends on the machine: on the laptop the CPU was 2.7× faster than its GPU, on the PC the GPU was " +
    "2.5× faster than its CPU."),

  H1("5. Experiments and results"),
  H2("5.1 TD3 sanity check"),
  ...figurePair("step1_pendulum_return.png", "step1_pendulum_q.png", "Pendulum-v1, 3 seeds (mean ± standard error). " +
    "Left: return. Right: critic estimate (solid) vs. the discounted return actually obtained (dashed). DDPG " +
    "overestimates slightly, TD3 slightly underestimates."),
  H2("5.2 The goal-conditioned low level alone"),
  P("Before adding a high level we trained the low level alone, giving it random relative goals g ~ U(−5, 5)^{2} every " +
    "10 steps on an open plane. After 30 k steps (3 seeds) the final distance to the target falls from 3.81 (doing " +
    "nothing) to **0.02**, with nearly 100 % of goals reached within 0.5 units (Figure 5). Hiding x, y from the low " +
    "level (as the official code does) gave a 4× lower final error than showing it (0.08)."),
  ...figure("step2_low_final_dist.png", 400, "Low level alone on random relative goals: distance to the absolute " +
    "target after c = 10 steps, with (\"low\") and without x, y hidden from the policy. 3 seeds."),
  H2("5.3 PointMaze: hierarchy and correction"),
  P("We compared full HIRO, HIRO without the off-policy correction, and a flat TD3 agent on the PointMaze, " +
    "200 k steps and 5 seeds each (Table 3, Figure 6)."),
  ...table(["Variant", "Final success at (0, 8)", "Seeds that reached ≥ 0.8 at some point"], [
    ["**HIRO (with correction)**", "**0.91**", "**5 / 5**"],
    ["HIRO without correction", "0.29", "2 / 5"],
    ["Flat TD3 (no hierarchy)", "0.20", "1 / 5"],
  ], [3600, 2500, 2926], "PointMaze, 200 k steps, 5 seeds. Final success = mean of the last 4 evaluations."),
  ...figure("step4_pointmaze_success.png", 470, "PointMaze success rate at the hard target (0, 8), mean ± standard " +
    "error over 5 seeds."),
  P("The correction makes the difference: without it the hierarchy is no better than a flat agent on this task, with " +
    "it all five seeds solve the maze. The subgoal plots (Figures 7 and 8) show why. Failing runs keep proposing subgoals on " +
    "the other side of the wall and push the point against it; successful runs propose subgoals that lead around the " +
    "⊃ and then hold the point at the target."),
  ...figure("step3_goals_nocorr.png", 600, "Without correction: the agent's path (blue) and every subgoal s_{t} + g_{t} " +
    "(orange arrows) for three seeds. Seeds 0 and 1 are stuck in the local optimum; seed 2 solves the maze."),
  ...figure("step4_goals_corr.png", 600, "With correction: subgoals lead around the inner wall."),
  P("Two lessons came out of this experiment. First, with only 3 seeds (our first run) the variants looked equal; 5 " +
    "seeds were needed for a clear picture. Second, on a point that can be steered directly, a flat agent already " +
    "has a fairly easy problem, so the benefit of the hierarchy depends on training the high level correctly."),
  H2("5.4 When does the correction act?"),
  P("During training the correction changed 50–70 % of the sampled goals, more than we expected. We first suspected " +
    "the exploration noise in the stored actions, but that hypothesis was wrong: on fresh data collected with the " +
    "training noise, only 0–2 % of goals were relabelled, whereas data generated by a *different* (stale) low-level " +
    "policy was relabelled 59–67 % of the time."),
  P("To confirm this we applied the final low-level policy to the whole replay buffer of a 200 k-step run and grouped " +
    "the segments by age (Figure 9). The oldest data is relabelled 92 % of the time with a mean goal shift of 4.45 " +
    "units, the newest only 17 % with a shift of 0.22; the decrease is monotonic. The correction therefore does what " +
    "the paper claims: it repairs experience made stale by the changing low level. As far as we can tell, [1] does " +
    "not show this analysis."),
  ...figure("step4_relabel_vs_age.png", 470, "The final low-level policy applied to every segment in the replay " +
    "buffer, grouped by when the segment was collected: share of goals relabelled (bars) and mean change of the goal " +
    "(line). PointMaze, seed 0."),
  H2("5.5 Ant tasks: the paper's results"),
  P("For the Ant tasks we rely on the results reported in [1]. After 10 M steps (best policy, 10 seeds) HIRO solves " +
    "Ant Maze almost always and Ant Push most of the time, and reaches two thirds success on Ant Fall, while all " +
    "baselines — FuN, SNN4HRL and VIME variants — stay close to zero on the three navigation tasks (Table 4). In the " +
    "ablations of [1], a flat TD3 agent makes almost no progress, and removing the off-policy correction hurts most " +
    "on the harder tasks Ant Push and Ant Fall — consistent with our PointMaze result."),
  ...table(["Method", "Ant Gather", "Ant Maze", "Ant Push", "Ant Fall"], [
    ["**HIRO**", "**3.02 ± 1.49**", "**0.99 ± 0.01**", "**0.92 ± 0.04**", "**0.66 ± 0.07**"],
    ["FuN cos similarity", "0.85 ± 1.17", "0.16 ± 0.33", "0.06 ± 0.17", "0.07 ± 0.22"],
    ["FuN transition PG", "0.41 ± 0.06", "0.0 ± 0.0", "0.56 ± 0.39", "0.01 ± 0.02"],
    ["SNN4HRL", "1.92 ± 0.52", "0.0 ± 0.0", "0.02 ± 0.01", "0.0 ± 0.0"],
    ["VIME", "1.42 ± 0.90", "0.0 ± 0.0", "0.02 ± 0.02", "0.0 ± 0.0"],
  ], [2426, 1650, 1650, 1650, 1650], "Results reported in [1] (Table 1): performance of the best policy within 10 M " +
    "steps, mean ± standard error over 10 seeds (Ant Gather: reward; other tasks: success rate)."),

  H1("6. Discussion"),
  H2("6.1 Our Ant runs"),
  P("We also trained HIRO, HIRO without correction and flat TD3 on Ant Maze, and HIRO on Ant Push and Ant Fall, for " +
    "2 M steps each (13 runs, about 8 hours on the PC). None of them reached a target. A diagnosis showed that the " +
    "low level had not learned to walk and that the high level kept proposing goals at the edge of its range, about " +
    "14 units away, which the ant cannot reach in one second. We found and fixed three deviations from the official " +
    "code — the final-layer initialisation, the action input of the critic and the warm-up data (Section 4.2) — which " +
    "made learning on the PointMaze about twice as fast; on the Ant the first 100 k steps still looked the same. A " +
    "public PyTorch re-implementation of HIRO [5] only shows success on Ant Maze after several million steps (about " +
    "4.5 M in its published learning curve, read approximately), so our " +
    "budget was most likely too short. We therefore do not claim to have reproduced the Ant results; for those we rely on [1]."),
  H2("6.2 What we learned"),
  BUL("**Raw-state goals give the low level a learning signal immediately.** On its own, the low level learned to " +
    "reach goals within about 2 k steps."),
  BUL("**The correction is what makes off-policy training of the high level work.** In our experiments it was the " +
    "difference between 0.29 and 0.91 success, and it acts mostly on old data, as the paper argues."),
  BUL("**Small implementation details matter.** Several choices that decide whether learning starts at all — the loss, " +
    "reward scaling, initialisation and warm-up — are only visible in the official code."),
  BUL("**Seeds matter.** Three seeds gave us a misleading picture; five gave a clear one."),
  H2("6.3 Limitations"),
  P("Our quantitative evidence comes from a toy environment of our own design, with few seeds, short budgets and no " +
    "hyper-parameter search; the Ant results are the paper's, not ours. Ant Gather was not implemented."),

  H1("7. Conclusion"),
  P("HIRO combines three simple ideas — raw states as goals, off-policy training of both levels, and a correction that " +
    "relabels stale high-level experience — into a general and sample-efficient hierarchical agent. Re-implementing it " +
    "step by step let us verify each idea in isolation: TD3 reduces overestimation, the goal-conditioned low level " +
    "learns quickly from the dense intrinsic reward, and the off-policy correction turns an unreliable hierarchy " +
    "into one that solves our maze in every seed, by rewriting mostly old experience. Scaling the reproduction to the " +
    "Ant tasks needs a longer training budget than we could afford."),

  H1("References"),
  ...[
    "[1] O. Nachum, S. Gu, H. Lee, S. Levine. Data-Efficient Hierarchical Reinforcement Learning. NeurIPS 2018. arXiv:1805.08296.",
    "[2] S. Fujimoto, H. van Hoof, D. Meger. Addressing Function Approximation Error in Actor-Critic Methods. ICML 2018.",
    "[3] T. P. Lillicrap, J. J. Hunt, A. Pritzel, N. Heess, T. Erez, Y. Tassa, D. Silver, D. Wierstra. Continuous Control with Deep Reinforcement Learning. ICLR 2016.",
    "[4] Official code: github.com/tensorflow/models/tree/master/research/efficient-hrl.",
    "[5] watakandai. hiro_pytorch: a PyTorch implementation of HIRO. GitHub, github.com/watakandai/hiro_pytorch.",
    "[6] E. Todorov, T. Erez, Y. Tassa. MuJoCo: A Physics Engine for Model-Based Control. IROS 2012.",
  ].map((t) => new Paragraph({ children: runs(t, { size: 20 }), spacing: { after: 80 }, indent: { left: 360, hanging: 360 } })),

  H1("Appendix: reproducing the results"),
  P("All code is in the project repository. The main commands (conda environment with PyTorch, Gymnasium and MuJoCo):"),
  BUL("Tests: `python -m pytest tests`"),
  BUL("Section 5.1: `python scripts/train_flat.py --env Pendulum-v1 --variant {ddpg,td3} --seed S`"),
  BUL("Section 5.2: `python scripts/train_low.py --steps 30000 --seed S [--no-zero-obs]`"),
  BUL("Section 5.3: `python scripts/train_hiro.py --env PointMaze --steps 200000 [--correction] --seed S` and " +
    "`python scripts/train_flat.py --env PointMaze --steps 200000 --seed S`"),
  BUL("Section 5.4: `python scripts/train_hiro.py --env PointMaze --correction --age-analysis` and `scripts/analyze_relabel.py`"),
  BUL("Figures: `scripts/plot.py`, `scripts/viz_goals.py`, `scripts/make_diagrams.py`"),
];

// Figure order: 1 diagram, 2 correction, 3 envs, 4 pendulum pair, 5 low level, 6 pointmaze, 7/8 subgoals, 9 relabel.
// Table order: 1 paper->code, 2 hyper-parameters, 3 PointMaze, 4 paper's Table 1.

const doc = new Document({
  creator: "[Name]",
  title: "Re-implementing HIRO — work report",
  features: { updateFields: true },
  styles: {
    default: { document: { run: { font: FONT, size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 32, bold: true, color: "1F3B6E" },
        paragraph: { spacing: { before: 360, after: 160 }, keepNext: true, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 26, bold: true, color: "2A6FDB" },
        paragraph: { spacing: { before: 240, after: 120 }, keepNext: true, outlineLevel: 1 } },
    ],
  },
  numbering: { config: [
    { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] },
    { reference: "numbers", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 540, hanging: 300 } } } }] },
    { reference: "details", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 540, hanging: 300 } } } }] },
  ] },
  sections: [{
    properties: { page: { margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 18, color: "888888" })] })] }) },
    children: [
      ...titlePage,
      new Paragraph({ children: [new TextRun({ text: "Contents", font: FONT, size: 32, bold: true, color: "1F3B6E" })],
        spacing: { after: 200 } }),
      new TableOfContents("Contents", { hyperlink: true, headingStyleRange: "1-2" }),
      new Paragraph({ children: [new PageBreak()] }),
      ...body,
    ],
  }],
});

Packer.toBuffer(doc).then((buf) => {
  const out = path.join(ROOT, "docs", "report.docx");
  fs.writeFileSync(out, buf);
  console.log("wrote", out, `(${figNo} figures, ${tabNo} tables)`);
});
