// Builds docs/slides.pptx (~15 min RL-class talk) from the figures in results/.
//   NODE_PATH=<dir with node_modules/pptxgenjs> node docs/build_slides.js <path to pptx skill>/scripts/apply_theme.js
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const { applyTheme } = require(process.argv[2]);

const ROOT = path.join(__dirname, "..");
const RES = (f) => path.join(ROOT, "results", f);
const OUT = path.join(ROOT, "docs", "slides.pptx");

const THEME = {
  name: "HIRO",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1B2335", lt1: "FFFFFF", dk2: "1F2A44", lt2: "EEF2F7",
    accent1: "E0703A", accent2: "2A6FDB", accent3: "2E9E5B", accent4: "8E5BB5", accent5: "C0392B", accent6: "8A94A6",
    hlink: "2A6FDB", folHlink: "6D4C9F",
  },
};
const HEX = THEME.colors;

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5 in
pres.title = "Re-implementing HIRO";
pres.author = "[Name]";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;
const W = 13.333, H = 7.5;

// ------------------------------------------------------------------ layouts
pres.defineSlideMaster({
  title: "TITLE_DARK",
  background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.7, y: 2.0, w: 7.2, h: 1.6, fontFace: THEME.headFontFace,
      fontSize: 44, bold: true, color: C.background1, valign: "bottom", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.7, y: 3.75, w: 7.2, h: 1.3, fontSize: 20,
      color: C.background2, valign: "top", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: C.background1 },
  margin: [0.5, 0.6, 0.5, 0.6],
  slideNumber: { x: 12.35, y: 7.0, w: 0.6, h: 0.3, fontSize: 10, color: C.accent6, align: "right" },
  objects: [
    { text: { text: "Re-implementing HIRO", options: { x: 0.6, y: 7.0, w: 4, h: 0.3, fontSize: 10, color: C.accent6, margin: 0 } } },
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 0.85, fontFace: THEME.headFontFace,
      fontSize: 32, bold: true, color: C.text2, valign: "middle", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CLOSING_DARK",
  background: { color: C.text2 },
  slideNumber: { x: 12.35, y: 7.0, w: 0.6, h: 0.3, fontSize: 10, color: C.accent6, align: "right" },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 0.85, fontFace: THEME.headFontFace,
      fontSize: 32, bold: true, color: C.background1, valign: "middle", margin: 0 }, text: "" } },
  ],
});

// ------------------------------------------------------------------ helpers
function imgSize(file) {
  const b = fs.readFileSync(file);
  if (file.endsWith(".gif")) return { w: b.readUInt16LE(6), h: b.readUInt16LE(8) };
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}
// place an image inside a box, keeping its aspect ratio (centred)
function fitImage(slide, file, x, y, w, h, name, alt) {
  const s = imgSize(RES(file));
  const r = Math.min(w / s.w, h / s.h);
  const iw = s.w * r, ih = s.h * r;
  slide.addImage({ path: RES(file), x: x + (w - iw) / 2, y: y + (h - ih) / 2, w: iw, h: ih, objectName: name,
    altText: alt });
  return { x: x + (w - iw) / 2, y: y + (h - ih) / 2, w: iw, h: ih };
}
const T = (slide, text, o) => slide.addText(text, { isTextBox: true, fontSize: 16, color: C.text1, margin: 0,
  valign: "top", ...o });
function card(slide, x, y, w, h, name, fill = C.background2) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.12, fill: { color: fill },
    line: { color: fill, width: 0 }, objectName: name });
}
function numberDot(slide, x, y, n, color, name) {
  slide.addShape(pres.shapes.OVAL, { x, y, w: 0.5, h: 0.5, fill: { color }, line: { color, width: 0 }, objectName: name });
  slide.addText(String(n), { isTextBox: true, x, y, w: 0.5, h: 0.5, align: "center", valign: "middle", fontSize: 16,
    bold: true, color: C.background1, margin: 0, objectName: name + " number" });
}
function stat(slide, x, y, w, big, label, color, name) {
  slide.addText(big, { isTextBox: true, x, y, w, h: 0.95, fontFace: THEME.headFontFace, fontSize: 48, bold: true,
    color, margin: 0, valign: "bottom", objectName: name + " value" });
  slide.addText(label, { isTextBox: true, x, y: y + 1.0, w, h: 0.7, fontSize: 14, color: C.text1, margin: 0,
    objectName: name + " label" });
}
const caption = (slide, text, x, y, w, name) => slide.addText(text, { isTextBox: true, x, y, w, h: 0.4, fontSize: 12,
  italic: true, color: C.accent6, margin: 0, objectName: name });

// ------------------------------------------------------------------ 1. title
pres.addSection({ title: "Introduction" });
let s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Introduction" });
s.addText("Re-implementing HIRO", { placeholder: "title" });
s.addText("Data-Efficient Hierarchical Reinforcement Learning (Nachum et al., NeurIPS 2018), step by step in PyTorch",
  { placeholder: "body" });
s.addText("[Name]  ·  [Course]  ·  [Date]", { isTextBox: true, x: 0.7, y: 6.3, w: 7.2, h: 0.4, fontSize: 14,
  color: C.accent6, margin: 0, objectName: "Presenter line" });
s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 8.55, y: 1.15, w: 4.1, h: 4.1, rectRadius: 0.2, fill: { color: C.background1 },
  line: { color: C.background1, width: 0 }, objectName: "Render frame" });
fitImage(s, "fig_antmaze_render.png", 8.7, 1.3, 3.8, 3.8, "Ant Maze render", "Top view of the Ant Maze with the target");
s.addText("Ant Maze: walk to the green target", { isTextBox: true, x: 8.55, y: 5.4, w: 4.1, h: 0.4, fontSize: 13,
  italic: true, color: C.background2, align: "center", margin: 0, objectName: "Render caption" });
s.addNotes("Today: HIRO, a hierarchical RL method by Nachum, Gu, Lee and Levine (NeurIPS 2018). We learned it by " +
  "re-implementing it from scratch in PyTorch, one component at a time. I'll explain the method, show what we built, " +
  "and present our experiments.");

// ------------------------------------------------------------------ 2. problem
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Introduction" });
s.addText("Long-horizon control is hard for flat RL", { placeholder: "title" });
const probs = [
  ["Walk", "An ant robot chooses 8 joint torques every 0.1 s."],
  ["Navigate", "Reaching the target takes hundreds of steps through a maze."],
  ["Explore", "A flat agent explores at the torque time scale; in the paper it makes almost no progress."],
];
probs.forEach(([h, d], i) => {
  const y = 1.65 + i * 1.5;
  numberDot(s, 0.7, y + 0.05, i + 1, [C.accent1, C.accent2, C.accent3][i], `Problem ${i + 1} dot`);
  T(s, h, { x: 1.45, y, w: 5.6, h: 0.45, fontSize: 20, bold: true, color: C.text2, objectName: `Problem ${i + 1} title` });
  T(s, d, { x: 1.45, y: y + 0.5, w: 5.6, h: 0.8, objectName: `Problem ${i + 1} text` });
});
fitImage(s, "fig_antmaze_render.png", 7.7, 1.45, 4.9, 4.9, "Ant Maze image", "Ant Maze top view");
caption(s, "The straight line to the target runs into the wall: the agent must first move away", 7.7, 6.45, 4.9, "Maze caption");
s.addNotes("The ant has to learn two things at once: how to walk, and where to go. The direct path to the target is " +
  "blocked by the inner wall, so a greedy agent gets stuck. Flat RL explores one torque at a time and rarely finds " +
  "the long detour. Hierarchy lets one policy think in terms of where to go, and another in terms of how to move.");

// ------------------------------------------------------------------ 3. HIRO overview
pres.addSection({ title: "Method" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Method" });
s.addText("HIRO in one picture", { placeholder: "title" });
fitImage(s, "fig_hiro_diagram.png", 0.6, 1.35, 12.1, 4.5, "HIRO diagram", "Two-level structure of HIRO");
const chips = [["Goals are raw states", C.accent1], ["Both levels trained off-policy (TD3)", C.accent2],
  ["Off-policy correction", C.accent3]];
chips.forEach(([t, col], i) => {
  const x = 0.9 + i * 4.0;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 6.0, w: 3.6, h: 0.65, rectRadius: 0.3, fill: { color: col },
    line: { color: col, width: 0 }, objectName: `Idea chip ${i + 1}` });
  s.addText(t, { isTextBox: true, x, y: 6.0, w: 3.6, h: 0.65, align: "center", valign: "middle", fontSize: 16, bold: true,
    color: C.background1, margin: 0, objectName: `Idea chip ${i + 1} text` });
});
s.addNotes("The high level looks at the state every c = 10 steps and outputs a goal: a desired change of the state. " +
  "In between, the goal is updated by h so that the absolute target stays the same. The low level acts every step " +
  "and is rewarded for getting close to that target. Three ideas make it work: raw states as goals, off-policy " +
  "training of both levels, and the off-policy correction.");

// ------------------------------------------------------------------ 4. DDPG -> TD3
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Method" });
s.addText("Both levels learn with TD3", { placeholder: "title" });
T(s, "DDPG: deterministic actor μ(s) + critic Q(s, a), trained from a replay buffer. TD3 adds three fixes against " +
  "overestimation:", { x: 0.6, y: 1.4, w: 6.0, h: 0.9, objectName: "TD3 intro" });
const fixes = [["Twin critics", "the target uses the smaller of two Q estimates"],
  ["Target smoothing", "clipped noise on the target action"],
  ["Delayed actor", "actor and targets updated every 2nd step"]];
fixes.forEach(([h, d], i) => {
  const y = 2.45 + i * 1.3;
  card(s, 0.6, y, 6.0, 1.1, `Fix card ${i + 1}`);
  T(s, h, { x: 0.85, y: y + 0.15, w: 5.5, h: 0.4, fontSize: 18, bold: true, color: C.accent2, objectName: `Fix ${i + 1} title` });
  T(s, d, { x: 0.85, y: y + 0.58, w: 5.5, h: 0.4, objectName: `Fix ${i + 1} text` });
});
fitImage(s, "step1_pendulum_q.png", 7.0, 1.4, 5.7, 3.9, "Overestimation plot", "Critic estimate vs true return");
T(s, "Our check on Pendulum: DDPG's critic ends above the true return (−29 vs −32), TD3's slightly below (−35 vs −33).",
  { x: 7.0, y: 5.45, w: 5.7, h: 0.9, fontSize: 14, objectName: "Pendulum takeaway" });
s.addNotes("Both levels of HIRO are trained with TD3, an improved version of DDPG. Actor-critic methods tend to " +
  "overestimate values; TD3's three fixes reduce that. We implemented DDPG and TD3 in one class with a switch per " +
  "fix and checked them on Pendulum: DDPG overestimates a little, TD3 slightly underestimates, as in the TD3 paper.");

// ------------------------------------------------------------------ 5. goals as states
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Method" });
s.addText("Goals as states: a learning signal from step one", { placeholder: "title" });
card(s, 0.6, 1.5, 6.0, 1.45, "Equation card h");
T(s, "Goal transition (Eq. 2)", { x: 0.85, y: 1.65, w: 5.5, h: 0.4, fontSize: 16, bold: true, color: C.accent1,
  objectName: "h label" });
T(s, "h(s, g, s′) = s + g − s′", { x: 0.85, y: 2.1, w: 5.5, h: 0.55, fontFace: THEME.headFontFace, fontSize: 24,
  color: C.text2, objectName: "h formula" });
card(s, 0.6, 3.15, 6.0, 1.45, "Equation card r");
T(s, "Low-level reward (Eq. 3)", { x: 0.85, y: 3.3, w: 5.5, h: 0.4, fontSize: 16, bold: true, color: C.accent3,
  objectName: "r label" });
T(s, "r = − ‖ s + g − s′ ‖", { x: 0.85, y: 3.75, w: 5.5, h: 0.55, fontFace: THEME.headFontFace, fontSize: 24,
  color: C.text2, objectName: "r formula" });
T(s, "The absolute target s + g stays fixed while the agent moves, and the low level gets dense reward before any " +
  "task reward appears.", { x: 0.6, y: 4.85, w: 6.0, h: 1.0, objectName: "Goals takeaway" });
fitImage(s, "step2_low_final_dist.png", 7.0, 1.4, 5.7, 3.8, "Low level plot", "Distance to target after 10 steps");
stat(s, 7.2, 5.25, 5.4, "3.81 → 0.02", "distance to the target after 10 steps, low level alone, ~2 k steps", C.accent3, "Low level stat");
s.addNotes("A goal is a relative change of the state. h keeps the absolute target fixed while the agent moves. The " +
  "reward is minus the distance to that target, so the low level learns from the very first step. Trained alone on " +
  "random goals, our low level reached them in about two thousand steps.");

// ------------------------------------------------------------------ 6. the problem: stale experience
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Method" });
s.addText("The catch: old experience goes stale", { placeholder: "title" });
card(s, 0.6, 1.6, 5.4, 2.5, "Then card");
T(s, "When it was stored", { x: 0.9, y: 1.8, w: 4.8, h: 0.45, fontSize: 20, bold: true, color: C.accent1, objectName: "Then title" });
T(s, "The high level chose goal g. The low level at that time walked to place A. Stored: (s, g, reward, A).",
  { x: 0.9, y: 2.4, w: 4.8, h: 1.5, fontSize: 19, objectName: "Then text" });
s.addText("≠", { isTextBox: true, x: 6.0, y: 2.25, w: 1.3, h: 1.2, align: "center", valign: "middle", fontSize: 72,
  bold: true, color: C.accent5, margin: 0, objectName: "Not equal sign" });
card(s, 7.3, 1.6, 5.4, 2.5, "Now card");
T(s, "Now", { x: 7.6, y: 1.8, w: 4.8, h: 0.45, fontSize: 20, bold: true, color: C.accent2, objectName: "Now title" });
T(s, "The low level has kept learning. Given the same goal g it would now walk to place B.",
  { x: 7.6, y: 2.4, w: 4.8, h: 1.5, fontSize: 19, objectName: "Now text" });
T(s, "Replaying the old tuple teaches the high level wrong \"dynamics\": the problem it learns on is non-stationary. " +
  "Off-policy training of the high level needs a fix.", { x: 0.6, y: 4.6, w: 12.1, h: 1.2, fontSize: 22,
  color: C.text2, objectName: "Stale takeaway" });
s.addNotes("Off-policy training reuses old data. For the high level that is a problem: its 'action' is a goal, and " +
  "what happens after a goal depends on the low level, which keeps changing. An old transition says 'goal g led to " +
  "A', but today the same goal would lead to B. That is the non-stationarity HIRO's correction addresses.");

// ------------------------------------------------------------------ 7. the correction
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Method" });
s.addText("Off-policy correction: relabel the goal", { placeholder: "title" });
fitImage(s, "fig_correction.png", 0.6, 1.35, 6.6, 5.1, "Correction figure", "Candidate goals and chosen relabelled goal");
const steps7 = [["10 candidates", "original g, the observed change s_{t+c} − s_t, 8 Gaussian samples around it"],
  ["Score each", "how likely are the stored actions under the current low level? −½ Σ‖a − μlo(s, g̃)‖²"],
  ["Keep the best", "train the high level on the relabelled goal g̃"]];
steps7.forEach(([h, d], i) => {
  const y = 1.55 + i * 1.6;
  numberDot(s, 7.6, y + 0.02, i + 1, C.accent1, `Correction step ${i + 1} dot`);
  T(s, h, { x: 8.3, y, w: 4.4, h: 0.45, fontSize: 20, bold: true, color: C.text2, objectName: `Correction step ${i + 1} title` });
  const parts = i === 0
    ? [{ text: "original g, the observed change s" }, { text: "t+c", options: { subscript: true } }, { text: " − s" },
       { text: "t", options: { subscript: true } }, { text: ", and 8 Gaussian samples around it" }]
    : [{ text: d }];
  T(s, parts, { x: 8.3, y: y + 0.5, w: 4.4, h: 0.95, fontSize: 15, objectName: `Correction step ${i + 1} text` });
});
s.addNotes("The fix: replace the stored goal by the goal under which the current low level would most likely have " +
  "produced the stored actions. With Gaussian exploration noise, that likelihood is minus half the squared error " +
  "between stored and predicted actions. The argmax is approximated over ten candidates.");

// ------------------------------------------------------------------ 8. implementation
pres.addSection({ title: "Our work" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Our work" });
s.addText("We built it in 8 verified steps", { placeholder: "title" });
const steps8 = ["DDPG → TD3", "Goals: h and r", "PointMaze + two levels", "Off-policy correction",
  "Ant Maze / Push / Fall", "Ant experiments", "Figures", "Report + slides"];
s.addShape(pres.shapes.LINE, { x: 1.0, y: 2.05, w: 11.3, h: 0, line: { color: HEX.accent6, width: 2 }, objectName: "Timeline line" });
steps8.forEach((t, i) => {
  const x = 0.75 + i * 1.6;
  numberDot(s, x, 1.8, i + 1, C.accent2, `Timeline dot ${i + 1}`);
  T(s, t, { x: x - 0.45, y: 2.45, w: 1.4, h: 0.8, fontSize: 13, align: "center", objectName: `Timeline label ${i + 1}` });
});
stat(s, 0.8, 3.45, 3.6, "27", "unit tests: every equation checked on its own", C.accent2, "Tests stat");
stat(s, 4.85, 3.45, 3.6, "6", "details found only in the official code (loss, reward scale, init …)", C.accent1, "Details stat");
stat(s, 8.9, 3.45, 3.6, "3", "Ant tasks ported from TensorFlow to the current MuJoCo", C.accent3, "Envs stat");
T(s, "Examples: Huber critic loss · high-level reward × 0.1 · low level does not see x, y · final layers initialised in ±0.003",
  { x: 0.8, y: 5.55, w: 11.8, h: 0.5, fontSize: 15, color: C.text2, objectName: "Details examples" });
s.addNotes("We built HIRO in eight steps, each with unit tests and a small experiment. For example, one test checks " +
  "that the correction recovers a known goal from actions generated by the current policy. Reading the official " +
  "code, we found six details the paper does not mention, such as the Huber loss and the 0.1 reward scale.");

// ------------------------------------------------------------------ 9. PointMaze results
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Our work" });
s.addText("PointMaze: the correction makes the difference", { placeholder: "title" });
stat(s, 0.7, 1.5, 3.6, "0.91", "HIRO with correction (5 / 5 seeds solve it)", C.accent2, "HIRO stat");
stat(s, 0.7, 3.2, 3.6, "0.29", "HIRO without correction (2 / 5)", C.accent1, "No-correction stat");
stat(s, 0.7, 4.9, 3.6, "0.20", "flat TD3 (1 / 5)", C.accent3, "TD3 stat");
fitImage(s, "step4_pointmaze_success.png", 4.7, 1.35, 8.0, 5.3, "PointMaze success plot", "Success rate vs steps, 3 variants");
caption(s, "Success at the end of the maze, 200 k steps, 5 seeds, mean ± standard error", 4.7, 6.55, 8.0, "PointMaze caption");
s.addNotes("Our main experiment: a point-mass version of the maze, with the same trap. Five seeds each. With the " +
  "correction, HIRO succeeds 91 percent of the time and all five seeds solve the maze. Without it, 29 percent; a flat " +
  "TD3 agent, 20 percent. With only three seeds the picture was unclear, so seeds really matter.");

// ------------------------------------------------------------------ 10. what the agent does
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Our work" });
s.addText("What the high level learns to ask for", { placeholder: "title" });
fitImage(s, "fig_nocorr_seed0.png", 0.6, 1.35, 5.8, 4.9, "Failure without correction", "Subgoals proposed through the wall");
caption(s, "Without correction: subgoals through the wall, the point gets stuck", 0.6, 6.35, 5.8, "Failure caption");
fitImage(s, "pointmaze_hiro.gif", 6.9, 1.35, 5.8, 4.9, "HIRO episode animation", "Animated episode: subgoals lead around the wall");
caption(s, "With correction: subgoals (orange) lead around the wall to the target", 6.9, 6.35, 5.8, "Success caption");
s.addNotes("Left: a run without the correction. The high level keeps asking for points on the other side of the " +
  "wall, and the low level pushes against it. Right, animated: with the correction, the subgoals lead around the " +
  "corridor, and the point stops at the target.");

// ------------------------------------------------------------------ 11. when does the correction act
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Our work" });
s.addText("The correction mostly rewrites old experience", { placeholder: "title" });
fitImage(s, "step4_relabel_vs_age.png", 0.6, 1.35, 7.6, 5.0, "Relabel vs age plot", "Share of relabelled goals by data age");
stat(s, 8.7, 1.4, 4.0, "92 %", "of the oldest segments get a new goal", C.accent5, "Oldest stat");
stat(s, 8.7, 3.05, 4.0, "17 %", "of the newest segments", C.accent2, "Newest stat");
stat(s, 8.7, 4.7, 4.0, "0–2 %", "on fresh data: exploration noise alone is not the cause", C.accent3, "Fresh stat");
s.addNotes("Our own analysis. During training the correction changed half or more of the goals, which surprised us. " +
  "We tested whether exploration noise was the reason: no, fresh data is almost never relabelled. When we apply the " +
  "final low level to the whole buffer, old data is relabelled 92 percent of the time and new data 17 percent: the " +
  "correction repairs stale experience, exactly as the paper motivates.");

// ------------------------------------------------------------------ 12. paper's Ant results
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Our work" });
s.addText("On the Ant tasks, HIRO beats all baselines (paper)", { placeholder: "title" });
s.addChart(pres.charts.BAR, [
  { name: "HIRO", labels: ["Ant Maze", "Ant Push", "Ant Fall"], values: [0.99, 0.92, 0.66] },
  { name: "FuN cos similarity", labels: ["Ant Maze", "Ant Push", "Ant Fall"], values: [0.16, 0.06, 0.07] },
  { name: "SNN4HRL", labels: ["Ant Maze", "Ant Push", "Ant Fall"], values: [0.0, 0.02, 0.0] },
  { name: "VIME", labels: ["Ant Maze", "Ant Push", "Ant Fall"], values: [0.0, 0.02, 0.0] },
], { x: 0.6, y: 1.35, w: 7.6, h: 5.0, barDir: "col", barGrouping: "clustered",
  chartColors: [HEX.accent2, HEX.accent6, "B9C0CC", "D5DAE2"],
  showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 11, dataLabelFontFace: "+mn-lt", dataLabelFormatCode: "0.00",
  valAxisMinVal: 0, valAxisMaxVal: 1.1, valAxisMajorUnit: 0.25, valAxisLabelFormatCode: "0.00",
  valAxisLabelFontSize: 11, catAxisLabelFontSize: 14, valAxisLabelFontFace: "+mn-lt", catAxisLabelFontFace: "+mn-lt",
  valAxisLabelColor: HEX.accent6, catAxisLabelColor: HEX.dk1,
  valGridLine: { color: "E3E7ED", size: 0.75 }, catGridLine: { style: "none" },
  showLegend: true, legendPos: "b", legendFontSize: 12, legendFontFace: "+mn-lt",
  showTitle: true, title: "Success rate, best policy within 10 M steps", titleFontSize: 14, titleFontFace: "+mn-lt",
  titleColor: HEX.dk2, objectName: "Paper results chart", altText: "HIRO vs baselines on Ant Maze, Push, Fall" });
caption(s, "Source: Nachum et al. 2018, Table 1 (10 seeds)", 0.6, 6.45, 7.6, "Chart source");
fitImage(s, "step5_envs.png", 8.6, 1.6, 4.1, 1.4, "Ant environments", "Ant Maze, Push and Fall renders");
T(s, "Push: move the red block out of the way.\nFall: push the block into the chasm to cross it.",
  { x: 8.6, y: 3.25, w: 4.1, h: 1.2, fontSize: 15, objectName: "Task descriptions" });
T(s, "Paper's ablations: flat TD3 makes almost no progress; dropping the correction hurts most on Push and Fall.",
  { x: 8.6, y: 4.65, w: 4.1, h: 1.4, fontSize: 15, color: C.text2, bold: true, objectName: "Ablation note" });
s.addNotes("For the Ant tasks we rely on the paper's results. After 10 million steps HIRO solves Ant Maze 99 percent " +
  "of the time, Ant Push 92 percent and Ant Fall 66 percent, while the other hierarchical methods stay near zero. " +
  "The paper's ablations agree with our PointMaze finding: the correction matters most on the harder tasks.");

// ------------------------------------------------------------------ 13. takeaways
pres.addSection({ title: "Conclusion" });
s = pres.addSlide({ masterName: "CLOSING_DARK", sectionTitle: "Conclusion" });
s.addText("Takeaways", { placeholder: "title" });
const take = [["Raw-state goals", "give the low level a learning signal immediately"],
  ["The off-policy correction", "turned 0.29 into 0.91 success, by rewriting stale experience"],
  ["Small details matter", "loss, reward scale, initialisation and warm-up are only in the official code"]];
take.forEach(([h, d], i) => {
  const y = 1.5 + i * 1.25;
  numberDot(s, 0.7, y + 0.05, i + 1, [C.accent1, C.accent2, C.accent3][i], `Takeaway ${i + 1} dot`);
  s.addText([{ text: h + ": ", options: { bold: true, color: C.background1 } }, { text: d, options: { color: C.background2 } }],
    { isTextBox: true, x: 1.45, y, w: 11.2, h: 0.75, fontSize: 20, margin: 0, valign: "middle", objectName: `Takeaway ${i + 1}` });
});
s.addText([{ text: "Limitations: ", options: { bold: true } },
  { text: "toy environment of our own, few seeds, short budgets. Our 2 M-step Ant runs did not learn (a public " +
    "re-implementation needs several million steps), so the Ant results shown are the paper's." }],
  { isTextBox: true, x: 0.7, y: 5.4, w: 11.9, h: 0.9, fontSize: 15, color: C.background2, margin: 0, objectName: "Limitations" });
s.addText("Thank you. Questions?", { isTextBox: true, x: 0.7, y: 6.4, w: 11.9, h: 0.5, fontFace: THEME.headFontFace,
  fontSize: 22, bold: true, color: C.accent1, margin: 0, objectName: "Thanks" });
s.addNotes("Three takeaways. Raw states as goals give the low level an immediate learning signal. The off-policy " +
  "correction is what makes off-policy training of the high level work, and it acts on old data. And many details " +
  "that decide whether learning starts at all are only in the official code. Honest limitation: our own Ant runs " +
  "of two million steps did not learn; a public re-implementation needs several million, so for the Ant we rely on the paper.");

// ------------------------------------------------------------------ 14. backup
pres.addSection({ title: "Backup" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Backup" });
s.addText("Backup: hyper-parameters (official configuration)", { placeholder: "title" });
const hp = [["Networks", "2 × 300 ReLU, tanh output; final layers U(±0.003)"], ["Optimiser", "Adam: actor 1e-4, critic 1e-3"],
  ["γ / τ", "0.99 / 0.005"], ["TD3", "target noise 0.2, clip 0.5, policy delay 2, Huber loss"],
  ["Batch / buffer", "100 / 200 000 per level"], ["High-level period c", "10 steps; reward 0.1 · Σ env reward"],
  ["Goal space (Ant)", "qpos[:15], range ±(10, 10, 0.5, 1 ×4, 0.5/0.3 joints)"],
  ["Exploration", "σ = 1 on both levels (torques in ±30)"], ["Correction", "10 candidates, σ = 0.5 × goal range"]];
s.addTable([[{ text: "Setting", options: { bold: true, color: C.background1, fill: { color: C.text2 } } },
  { text: "Value", options: { bold: true, color: C.background1, fill: { color: C.text2 } } }],
  ...hp.map(([a, b], i) => [{ text: a, options: { bold: true, fill: { color: i % 2 ? C.background1 : C.background2 } } },
    { text: b, options: { fill: { color: i % 2 ? C.background1 : C.background2 } } }])],
{ x: 0.6, y: 1.4, w: 12.1, colW: [3.4, 8.7], fontSize: 14, color: C.text1, border: { type: "solid", pt: 0.5, color: "D5DAE2" },
  rowH: 0.48, objectName: "Hyper-parameter table" });
s.addNotes("Backup slide in case of questions about the settings.");

pres.writeFile({ fileName: OUT }).then(async () => {
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
});
