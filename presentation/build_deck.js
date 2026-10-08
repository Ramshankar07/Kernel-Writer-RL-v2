// Builds presentation/deck.pptx from results/numbers.json + presentation/figures/*.png.
// Every number comes from numbers.json.   node presentation/build_deck.js
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const ROOT = path.resolve(__dirname, "..");
const N = JSON.parse(fs.readFileSync(path.join(ROOT, "results", "numbers.json"), "utf8"));
const FIG = (f) => path.join(__dirname, "figures", f);
const has = (f) => fs.existsSync(FIG(f));

// Palette: deep navy + amber accent ("hot silicon"), light slides match figure surface.
const NAVY = "14213D", AMBER = "FCA311", INK = "0B0B0B", INK2 = "52514E", MUTED = "898781";
const LIGHT = "FCFCFB", CARD = "F1F0EC", WHITE = "FFFFFF";
const HEAD = "Cambria", BODY = "Calibri";

const ae = (tag) => Object.entries(N.agent_eval || {}).find(([k]) => k.endsWith(tag))?.[1];
const pc = (x, d = 1) => `${(100 * x).toFixed(d)}%`;
const V1 = ae("_fbv1"), V2 = ae("_fbv2"), S1 = N.stage1, KB = N.kb_sanity;
const G = N.grpo ? Object.values(N.grpo)[0] : null;
const KB_BASE = ae("_kb_base"), KB_LORA = ae("_kb_lora"), AK_LORA = ae("_ak_lora");

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5
pres.title = "AutoKernel-RLVR: teaching an LLM to write faster GPU kernels";

function base(title, { dark = false } = {}) {
  const s = pres.addSlide();
  s.background = { color: dark ? NAVY : LIGHT };
  if (title) {
    s.addText(title, { x: 0.6, y: 0.4, w: 12.1, h: 0.9, fontFace: HEAD, fontSize: 32, bold: true,
      color: dark ? WHITE : INK, margin: 0, isTextBox: true });
  }
  s.addText(N.label, { x: 0.6, y: 7.0, w: 6, h: 0.3, fontFace: BODY, fontSize: 10,
    color: dark ? "C3C2B7" : MUTED, margin: 0, isTextBox: true });
  return s;
}

function stat(s, x, y, w, value, label, { dark = false, color } = {}) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h: 1.6, rectRadius: 0.12,
    fill: { color: dark ? "22335A" : CARD }, line: { color: dark ? "22335A" : CARD } });
  s.addText(value, { x: x + 0.25, y: y + 0.15, w: w - 0.5, h: 0.85, fontFace: HEAD, fontSize: 36,
    bold: true, color: color || (dark ? AMBER : NAVY), margin: 0, isTextBox: true });
  s.addText(label, { x: x + 0.25, y: y + 0.98, w: w - 0.5, h: 0.5, fontFace: BODY, fontSize: 13,
    color: dark ? "E1E0D9" : INK2, margin: 0, isTextBox: true, valign: "top" });
}

function bullets(s, items, x, y, w, h, size = 16, color = INK) {
  s.addText(items.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < items.length - 1 } })),
    { x, y, w, h, fontFace: BODY, fontSize: size, color, paraSpaceAfter: 8, valign: "top", margin: 0,
      isTextBox: true });
}

function figure(s, f, x, y, w, h) {
  if (has(f)) s.addImage({ path: FIG(f), x, y, w, h, sizing: { type: "contain", w, h } });
  else s.addText("figure pending: stage not run yet", { x, y, w, h, align: "center", fontFace: BODY,
    fontSize: 14, color: MUTED, fill: { color: CARD }, isTextBox: true });
}

function pending(s, x, y, w) {
  s.addText("Pending: this stage had not finished when the deck was built.", { x, y, w, h: 0.5,
    fontFace: BODY, fontSize: 14, italic: true, color: MUTED, margin: 0, isTextBox: true });
}

// 1. Title -----------------------------------------------------------------
{
  const s = base(null, { dark: true });
  s.addText("Teaching an LLM to write\nfaster GPU kernels with RL", { x: 0.8, y: 1.6, w: 11.5, h: 2.2,
    fontFace: HEAD, fontSize: 48, bold: true, color: WHITE, margin: 0, isTextBox: true });
  s.addText("AutoKernel-RLVR: multi-turn GRPO with a GPU benchmark as the verifier", { x: 0.8, y: 4.0,
    w: 11.5, h: 0.6, fontFace: BODY, fontSize: 22, color: AMBER, margin: 0, isTextBox: true });
  s.addText("KernelBench v1 → AutoKernel kernels → GRPO on Qwen2.5-Coder-7B  ·  re-run on Modal, Sept 2026",
    { x: 0.8, y: 4.8, w: 11.5, h: 0.5, fontFace: BODY, fontSize: 16, color: "C3C2B7", margin: 0, isTextBox: true });
  s.addNotes("All numbers in this deck come from a fresh re-run on Modal in September 2026. The original " +
    "April 2026 results were lost with an old disk. Source of truth: results/numbers.json.");
}

// 2. The loop -----------------------------------------------------------------
{
  const s = base("The idea: a benchmark is a perfect reward function");
  const steps = [["1", "Propose", "LLM writes a full Triton kernel.py"],
                 ["2", "Verify", "bench.py on a real GPU: 5 correctness stages + timing"],
                 ["3", "Read", "correctness, speedup, failure lines come back as the next message"],
                 ["4", "Revise", "up to N turns per episode"]];
  steps.forEach(([n, t, d], i) => {
    const x = 0.6 + i * 3.1;
    s.addShape(pres.shapes.OVAL, { x, y: 1.7, w: 0.8, h: 0.8, fill: { color: NAVY }, line: { color: NAVY } });
    s.addText(n, { x, y: 1.7, w: 0.8, h: 0.8, align: "center", valign: "middle", fontFace: HEAD,
      fontSize: 24, bold: true, color: AMBER, margin: 0, isTextBox: true });
    s.addText(t, { x, y: 2.65, w: 2.8, h: 0.5, fontFace: HEAD, fontSize: 20, bold: true, color: INK, margin: 0, isTextBox: true });
    s.addText(d, { x, y: 3.15, w: 2.8, h: 1.0, fontFace: BODY, fontSize: 14, color: INK2, margin: 0, isTextBox: true, valign: "top" });
  });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y: 4.6, w: 12.1, h: 1.7, rectRadius: 0.12,
    fill: { color: CARD }, line: { color: CARD } });
  s.addText([
    { text: "Reward per episode  ", options: { bold: true, color: NAVY } },
    { text: "r = log2(best speedup vs PyTorch) if any turn PASSes, else 0;  clipped to [−1, 3]", options: { color: INK } },
  ], { x: 0.9, y: 4.8, w: 11.5, h: 0.6, fontFace: BODY, fontSize: 18, margin: 0, isTextBox: true });
  s.addText("No partial credit: a fast but wrong kernel scores 0. GRPO compares 8 episodes of the same task and pushes toward the better ones.",
    { x: 0.9, y: 5.45, w: 11.5, h: 0.7, fontFace: BODY, fontSize: 15, color: INK2, margin: 0, isTextBox: true });
  s.addNotes("reward.py is unchanged from the original design. RLVR = reinforcement learning from verifiable rewards.");
}

// 3. Project arc --------------------------------------------------------------
{
  const s = base("Project arc");
  const arc = [["KernelBench v1", "Started here: 100 Level 1 PyTorch ops, fast_p metric. Became the held-out test set."],
               ["AutoKernel kernels", `9 kernel types with a fixed 5-stage bench.py. Always valid and cheaper per reward, so used for RL.`],
               ["GRPO on Qwen-7B", "Multi-turn agent, LoRA, rewards from real GPU benchmarks."],
               ["Modal re-run", "Original results lost; rebuilt on ≤10 Modal GPUs and re-measured everything."]];
  arc.forEach(([t, d], i) => {
    const x = 0.6 + i * 3.1;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.8, w: 2.85, h: 3.4, rectRadius: 0.12,
      fill: { color: i === 3 ? NAVY : CARD }, line: { color: i === 3 ? NAVY : CARD } });
    s.addText(`0${i + 1}`, { x: x + 0.25, y: 2.0, w: 2.3, h: 0.6, fontFace: HEAD, fontSize: 28, bold: true,
      color: AMBER, margin: 0, isTextBox: true });
    s.addText(t, { x: x + 0.25, y: 2.7, w: 2.4, h: 0.6, fontFace: HEAD, fontSize: 19, bold: true,
      color: i === 3 ? WHITE : INK, margin: 0, isTextBox: true });
    s.addText(d, { x: x + 0.25, y: 3.35, w: 2.4, h: 1.7, fontFace: BODY, fontSize: 14,
      color: i === 3 ? "E1E0D9" : INK2, margin: 0, isTextBox: true, valign: "top" });
  });
  s.addNotes("Tell it chronologically: KernelBench first, then the pivot to AutoKernel for cost, then RL, then the re-run.");
}

// 4. KernelBench harness --------------------------------------------------------
{
  const s = base("KernelBench v1 on Modal: is the timing trustworthy?");
  figure(s, "03_kb_identity_noise.png", 0.5, 1.4, 7.6, 3.8);
  stat(s, 8.5, 1.4, 4.2, `${KB.valid}/${KB.n}`, "Level 1 problems that are harness-valid (identity PASSes reliably)");
  stat(s, 8.5, 3.2, 4.2, `${KB.identity_speedup_median.toFixed(3)}×`, "median measured speedup of an identical model");
  bullets(s, [
    `${KB.conv_unseeded} conv problems fail even as an identical model: weight init isn't seeded`,
    `${KB.flaky_timeouts} huge-input problems time out intermittently (CPU input generation inside a 30 s limit)`,
  ], 0.6, 5.5, 12, 1.3, 15, INK2);
  s.addNotes("Identity test = the 'answer' is the reference model itself. It must PASS at 1.0x. " +
    "AutoKernel's own KernelBench starter crashes on all 100 problems (super(Model, self) after the class rename).");
}

// 5. Why switch ---------------------------------------------------------------
{
  const s = base("Why RL moved to AutoKernel's kernels: validity and cost");
  stat(s, 0.6, 1.6, 3.8, `${KB.valid}/100`, "KernelBench L1 problems usable as a reward");
  stat(s, 4.75, 1.6, 3.8, `${S1.bench_wall_s.L4.p50.toFixed(1)} s`, "median AutoKernel bench on L4");
  stat(s, 8.9, 1.6, 3.8, `${S1.max_cv_pct}%`, "max run-to-run noise (CV) of AutoKernel speedup");
  bullets(s, [
    (N.grpo && N.grpo.grpo_v1)
      ? `RL needs thousands of rewards: one GRPO step here is ${9 * N.grpo.grpo_v1.config.group} episodes × ${N.grpo.grpo_v1.config.max_turns} turns = ${9 * N.grpo.grpo_v1.config.group * N.grpo.grpo_v1.config.max_turns} benchmarks`
      : "RL needs thousands of rewards per training run",
    "AutoKernel's bench.py is fixed and 5-stage (smoke, shape sweep incl. fp32, stability, determinism, edge sizes)",
    "KernelBench stays as the held-out evaluation",
  ], 0.6, 3.7, 12, 2.5, 17);
  s.addNotes("Cheaper verifier → more gradient per dollar. L4 bench workers match the original A10/L4 design.");
}

// 6. Architecture ----------------------------------------------------------------
{
  const s = base("Architecture: original design vs Modal re-run");
  figure(s, "01_architecture.png", 0.5, 1.4, 12.3, 5.0);
  s.addNotes("Account limit: 10 concurrent GPUs. 1 trainer H100 + 1 vLLM policy H100 + 8 L4 bench containers. " +
    "Redis queue replaced by modal.Dict content-hash cache; the whole training loop runs inside the trainer container.");
}

// 7. Verifier findings -------------------------------------------------------------
{
  const s = base("The verifier is strict: most starter kernels fail it");
  figure(s, "02_starter_baselines.png", 0.5, 1.35, 7.9, 4.9);
  stat(s, 8.8, 1.4, 3.9, `${S1.starters_pass_full.H100}/9`, "AutoKernel starter kernels that pass the full bench");
  bullets(s, [
    `quick mode skips numerical stability: ${S1.quick_pass_full_fail.H100.join(", ")} pass quick, fail full`,
    "→ reward uses the full bench, or the agent learns unsafe kernels",
    "matmul fails because tl.dot uses TF32 on fp32 inputs",
  ], 8.8, 3.3, 3.9, 3.2, 14, INK2);
  s.addNotes(`Speedups are GPU-specific: softmax starter is ${S1.per_gpu.H100.softmax.speedup_quick}x on H100 but ${S1.per_gpu.L4.softmax.speedup_quick}x on L4.`);
}

// 8. Base model ---------------------------------------------------------------------
if (V2) {
  const s = base("Before RL: the model re-submits what already works");
  figure(s, "05_per_kernel_base.png", 0.5, 1.35, 7.9, 4.9);
  stat(s, 8.8, 1.4, 3.9, pc(V2.pass_rate), "episodes with ≥1 PASS (9 kernels × 8 samples × 8 turns)");
  stat(s, 8.8, 3.2, 3.9, pc(V2.fast_p["fast_1.0"]), "fast_1.0: correct AND faster than PyTorch");
  s.addText(`${pc(V2.dup_code_frac, 0)} of turns resubmit code already tried; best speedup ${V2.best_speedup_max}×`,
    { x: 8.8, y: 5.1, w: 3.9, h: 1.0, fontFace: BODY, fontSize: 14, color: INK2, margin: 0, isTextBox: true });
  s.addNotes("Qwen2.5-Coder-7B-Instruct, temperature 0.8, rewards on L4. Only kernels whose starter already passes get solved, plus rmsnorm once.");
}

// 9. Feedback ablation -------------------------------------------------------------
if (V1 && V2) {
  const s = base("A free win: tell the model *why* its kernel failed");
  figure(s, "04_pass_at_turn_feedback.png", 0.5, 1.35, 7.4, 3.9);
  const rows = [
    ["", "v1: log tail", "v2: failure lines"],
    ["episodes with PASS", pc(V1.pass_rate), pc(V2.pass_rate)],
    ["mean reward", V1.reward_mean.toFixed(3), V2.reward_mean.toFixed(3)],
    ["PASS turns", String(V1.turn_outcomes.PASS), String(V2.turn_outcomes.PASS)],
    ["CRASH turns", String(V1.turn_outcomes.CRASH), String(V2.turn_outcomes.CRASH)],
    ["kernels solved", String(Object.values(V1.per_kernel).filter(v => v.pass_rate > 0).length),
                       String(Object.values(V2.per_kernel).filter(v => v.pass_rate > 0).length)],
  ];
  s.addTable(rows.map((r, i) => r.map((c, j) => ({ text: c, options: {
    bold: i === 0 || j === 0, color: i === 0 ? WHITE : INK, fill: { color: i === 0 ? NAVY : (i % 2 ? CARD : LIGHT) },
    fontFace: BODY, fontSize: 14, align: j ? "center" : "left" } }))),
    { x: 8.2, y: 1.45, w: 4.6, colW: [1.8, 1.4, 1.4], rowH: 0.48, border: { type: "none" } });
  s.addText("The original worker returned the tail of bench.py's log: only the summary block, never which check failed.",
    { x: 0.6, y: 5.5, w: 12, h: 0.8, fontFace: BODY, fontSize: 15, color: INK2, margin: 0, isTextBox: true });
  s.addNotes("Same model, same sampling; only the observation text changed. This is the cheapest improvement in the project.");
}

// 10. Failure taxonomy ----------------------------------------------------------
if (V2) {
  const s = base("Correctness, not speed, is the bottleneck");
  figure(s, "06_failure_taxonomy.png", 0.5, 1.35, 8.2, 4.4);
  bullets(s, ["numerical mismatch: wrong accumulation dtype, TF32, bad masking on edge sizes",
              "Triton API / compile errors: e.g. tl.math.tanh and tl.EVICT_FALSE don't exist in Triton 3.2",
              "shared-memory overflow: tile sizes chosen for H100, run on L4 (101 KB limit)"],
          9.0, 1.6, 3.8, 4.5, 14, INK2);
  s.addNotes("Categories come from the failure lines bench.py prints; see modal_app/analyze.py::failure_taxonomy.");
}

// 11. GRPO setup ---------------------------------------------------------------------
if (V2) {
  const z = V2.ablations.zero_adv_group_frac_by_group_size;
  const s = base("GRPO's hidden cost: most groups give no gradient");
  figure(s, "07_zero_advantage_groups.png", 0.5, 1.35, 7.4, 4.1);
  stat(s, 8.3, 1.4, 4.4, pc(z["4"], 0), "of 4-sample groups have identical rewards → advantage 0");
  bullets(s, [`group 8 → ${pc(z["8"], 0)}: chosen for training`, "LoRA r=64 · lr 1e-5 · 4 turns · T=1.0",
              "token-mean loss, no KL, one on-policy update (verl GRPO semantics)"], 8.3, 3.3, 4.4, 2.6, 15, INK2);
  s.addNotes("Advantage = (r − mean_group)/(std_group + 1e-6). If every sample in a group fails, the group teaches nothing.");
}

// 12. GRPO results -------------------------------------------------------------------
{
  const G1 = (N.grpo || {}).grpo_v1, G2 = (N.grpo || {}).grpo_v2;
  const s = base("GRPO: dense credit adds signal, not new kernels");
  figure(s, "09_grpo_curves.png", 0.4, 1.3, 12.5, 3.7);
  if (G1) {
    stat(s, 0.6, 5.2, 4.0, `${G1.reward_first3.toFixed(3)} → ${G1.reward_last3.toFixed(3)}`,
      `v1 original reward, ${G1.steps} steps (first vs last 3)`);
    stat(s, 4.75, 5.2, 3.8, pc(G1.zero_adv_mean, 0), "v1 groups with zero gradient");
  } else pending(s, 0.6, 5.3, 6);
  if (G2) stat(s, 8.75, 5.2, 4.0, pc(G2.zero_adv_mean, 0), `v2 groups with zero gradient (${G2.steps} steps)`);
  else pending(s, 8.75, 5.3, 4.0);
  s.addNotes("v1: original log2 reward. 5 of 9 kernels never passed in any episode, so their groups never gave gradient. " +
    "v2: partial credit for correctness progress (max 0.3), any PASS >= 0.5, and groups with std < 0.02 (timing noise) treated as ties. " +
    (G2 ? `v2 reward_v2 ${G2.reward_v2_first3} -> ${G2.reward_v2_last3}, pass ${pc(G2.pass_first3)} -> ${pc(G2.pass_last3)} vs v1 pass ${pc(G1.pass_first3)} -> ${pc(G1.pass_last3)}. ` +
          "Single seed; reward changes are within step-to-step noise. The robust effect is the zero-gradient fraction." : ""));
}

// 13. KernelBench transfer -------------------------------------------------------------
{
  const s = base("Does it transfer to KernelBench v1? Not yet");
  figure(s, "10_kernelbench_transfer.png", 0.5, 1.35, 8.2, 4.8);
  if (KB_BASE) {
    stat(s, 9.0, 1.4, 3.7, `${KB_BASE.turn_outcomes.PASS || 0}/${KB_BASE.meta.tasks * KB_BASE.meta.max_turns}`, "base: PASS turns on valid L1 problems");
    if (KB_LORA) stat(s, 9.0, 3.2, 3.7, `${KB_LORA.turn_outcomes.PASS || 0}/${KB_LORA.meta.tasks * KB_LORA.meta.max_turns}`, "after GRPO v2: PASS turns");
    s.addText("Most failures are invented or missing Triton APIs; a 7B model needs Triton API grounding before speed.",
      { x: 9.0, y: 5.1, w: 3.7, h: 1.2, fontFace: BODY, fontSize: 14, color: INK2, margin: 0, isTextBox: true });
  } else pending(s, 9.0, 1.5, 3.7);
  s.addNotes("Subset = every 2nd harness-valid Level 1 problem (55 valid of 100); 1 sample x 3 turns; H100; bench_kb full protocol.");
}

// 14. What broke --------------------------------------------------------------------
{
  const s = base("Re-running exposed five real bugs");
  const bugs = [["--kernel-type", "worker passed a flag bench.py doesn't have (it's --kernel)"],
                ["results.tsv", "worker parsed a file bench.py never writes; metrics are on stdout"],
                ["super(Model)", "AutoKernel's KernelBench starter crashes on all 100 problems"],
                ["no seed", `KernelBench mode never seeds weight init: ${KB.conv_unseeded} conv problems unpassable`],
                ["log tail", "agent was shown the summary block, not the failing check"]];
  bugs.forEach(([k, d], i) => {
    const y = 1.5 + i * 1.0;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y, w: 3.0, h: 0.75, rectRadius: 0.1,
      fill: { color: NAVY }, line: { color: NAVY } });
    s.addText(k, { x: 0.6, y, w: 3.0, h: 0.75, align: "center", valign: "middle", fontFace: "Courier New",
      fontSize: 16, bold: true, color: AMBER, margin: 0, isTextBox: true });
    s.addText(d, { x: 3.9, y, w: 8.8, h: 0.75, valign: "middle", fontFace: BODY, fontSize: 17, color: INK,
      margin: 0, isTextBox: true });
  });
  s.addNotes("Also: vLLM 0.8.5 breaks with the newest transformers (pin 4.51.3). The AMD MI300X path is design-only: no AMD GPUs on Modal.");
}

// 15. Lessons ------------------------------------------------------------------------
{
  const s = base("Takeaways and next steps", { dark: true });
  bullets(s, [
    "Verifier quality is the product: quick-mode rewards and log-tail feedback both silently hurt",
    "The reward has a floor bug: correct-but-slow (<1×) scores below a crash; use max(0, log2 s) + PASS bonus",
    (N.grpo && N.grpo.grpo_v1 && N.grpo.grpo_v2)
      ? `GRPO signal is sparse: dense correctness credit cut zero-gradient groups ${pc(N.grpo.grpo_v1.zero_adv_mean, 0)} → ${pc(N.grpo.grpo_v2.zero_adv_mean, 0)}, but solving new kernels needs more steps or a curriculum`
      : "GRPO signal is sparse: most groups are all-fail; next is a curriculum on the unsolved kernels",
    "Async rollouts: the trainer H100 idles during benchmarking",
    "Next benchmark: KernelBench Level 2 fusion problems",
  ], 0.8, 1.6, 11.8, 4.6, 19, WHITE);
  s.addNotes("30 follow-up questions with numbers: presentation/qa_30.md.");
}

const out = path.join(__dirname, "deck.pptx");
pres.writeFile({ fileName: out }).then(() => console.log("wrote", out));
