// Builds presentation/v3/deck_v3.pptx: kernel-verification job talk (storyline: presentation/v3/storyline_v3.md).
// Reframe of v2: the harness and the RMSNorm kernel are the subject; the GRPO study is the use case.
// Every measured number is read at build time from
//   results/numbers.json, results/extra_numbers.json   (v2 analyses)
//   results/v3/v3_numbers.json                        (python3 presentation/v3/make_v3_numbers.py)
//   results/v3/rmsnorm_remeasure.json                 (remeasure job; slides show "PENDING" boxes if it is absent)
// Analytic quantities (bytes moved, bandwidth floors, overflow thresholds) are computed here from shapes/specs.
//   node presentation/v3/build_deck_v3.js
const fs = require("fs");
const path = require("path");
let pptxgen;
try { pptxgen = require("pptxgenjs"); } catch (e) { pptxgen = require(path.join(__dirname, "..", "node_modules", "pptxgenjs")); }

const ROOT = path.resolve(__dirname, "..", "..");
// tolerant JSON: the remeasure job writes Python's Infinity / NaN
const readJSON = (file) => JSON.parse(fs.readFileSync(file, "utf8")
  .replace(/:\s*-Infinity\b/g, ': "-Inf"').replace(/:\s*Infinity\b/g, ': "Inf"').replace(/:\s*NaN\b/g, ': "NaN"'));
const J = (...p) => readJSON(path.join(ROOT, ...p));
const N = J("results", "numbers.json");
const X = J("results", "extra_numbers.json");
const V3 = J("results", "v3", "v3_numbers.json");
const RM_FILE = path.join(ROOT, "results", "v3", "rmsnorm_remeasure.json");
const RM = fs.existsSync(RM_FILE) ? readJSON(RM_FILE) : null;

// ---------- data handles ----------
const ae = (tag) => Object.entries(N.agent_eval).find(([k]) => k.endsWith(tag))[1];
const V1 = ae("_fbv1"), V2 = ae("_fbv2"), KBB = ae("_kb_base"), KBL = ae("_kb_lora");
const S1 = N.stage1, KB = N.kb_sanity, G1 = N.grpo.grpo_v1, G2 = N.grpo.grpo_v2, C = G2.config;
const TB = X.time_breakdown, DR = X.dr_grpo_whatif, GE = X.group_explorer;
const BK = X.best_kernel, PS = X.pass_speedups;
const BC = V3.bench_config, LV = V3.lowvar_audit_grpo_v1, LF = V3.library_facts, WT = V3.winner_turn, VER = V3.versions;
const SR = V3.stage1_rmsnorm;

const pc = (x, d = 1) => `${(100 * x).toFixed(d)}%`;
const pp = (x, d = 1) => `${Number(x).toFixed(d)}%`;
const f = (x, d = 3) => Number(x).toFixed(d);
const log2 = Math.log2;
const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;
const nTasks = V2.meta.tasks;
const passCount = (r) => Math.round(r.pass_rate * r.n_traj);
const turnsTotal = (r) => Object.values(r.turn_outcomes).reduce((a, b) => a + b, 0);
const um = (x) => String(x).replace(/^-/, "−");
const fm = (x, d = 3) => um(f(x, d));
const sgnPct = (x, d = 1) => `${x >= 0 ? "+" : "−"}${Math.abs(100 * x).toFixed(d)}%`;
const rV1 = (pass, s) => (pass && s > 0 ? Math.max(-1, Math.min(3, log2(s))) : 0);
const rV2 = (pass, s) => (pass ? 0.5 + Math.max(0, Math.min(3, log2(s))) : 0);
const zeroAdvMean = (G) => mean(G.curve.map((c) => c.zero_adv_group_frac));
const cacheMean = (G) => mean(G.curve.map((c) => c.cache_hit_rate));

// ---------- analytic kernel quantities ----------
const ESZ = { float16: 2, bfloat16: 2, float32: 4 };
const bytesRms = (M, Nn, dt = "float16") => (2 * M * Nn + Nn) * ESZ[dt];      // bench.py bytes_fn
const L4BW = BC.L4_spec.peak_bandwidth_gb_s, L4L2 = BC.L4_spec.l2_cache_mb;
const floorUs = (bytes, bw = L4BW) => bytes / (bw * 1e9) * 1e6;
const gbs = (bytes, us) => bytes / (us * 1e-6) / 1e9;
const MB = (b) => b / 1e6;
const nextPow2 = (n) => 2 ** Math.ceil(log2(n));
const timed = BC.rmsnorm_test_sizes.find((s) => s.label === BC.timed_size_label);
const TBYTES = bytesRms(timed.M, timed.N, BC.timed_dtype);
const xMiB = timed.M * timed.N * ESZ[BC.timed_dtype] / 2 ** 20;
const L4q = SR.L4_quick, H1q = SR.H100_quick;
const stLat = mean(L4q.map((r) => r.latency_us)), ptLat = mean(L4q.map((r) => r.pytorch_latency_us));
const stSp = L4q.map((r) => r.speedup_vs_pytorch);
const stGBs = gbs(TBYTES, stLat), ptGBs = gbs(TBYTES, ptLat);
const FP16MAX = LF.fp16_max;
const sqOverflow = Math.sqrt(FP16MAX);                        // |x| where x*x overflows fp16
const rmsOverflow = (Nn) => Math.sqrt(FP16MAX / Nn);          // row RMS where an fp16 sum of N squares overflows
const h100Pct = Math.max(...H1q.map((r) => r.pct_peak_bandwidth));
const h100Lat = mean(H1q.map((r) => r.latency_us));
const h100Real = 100 * gbs(TBYTES, h100Lat) / BC.H100_SXM_spec_bw_gb_s;
const basePrev = BK.top10.find((t) => t.kernel === BK.kernel_type && t.source.startsWith("base"));
const stagesOk = (st) => Object.values(st).filter((v) => String(v).startsWith("PASS")).length;

// ---------- remeasure (results/v3/rmsnorm_remeasure.json) ----------
const RG = RM && RM.gpus && RM.gpus.L4 ? RM.gpus.L4 : null;           // L4 block
const RH = RM && RM.gpus && RM.gpus.H100 ? RM.gpus.H100 : null;
const P = RG ? RG.primary_rows : null;                                 // starter / best / fp32acc / torch_eager / torch_compile
const HL = RG ? RG.headline : null;
const castGain = HL ? HL.best_vs_starter_cold_median - 1 : null;       // speed the two casts contributed (L4, cold L2)
const overlap = P ? (P.best.cold_p10_us <= P.starter.cold_p90_us && P.starter.cold_p10_us <= P.best.cold_p90_us) : null;
const rtab = (k, M_, N_, dt = "float16") => RG && RG.table.find((r) => r.kernel === k && r.M === M_ && r.N === N_ && r.dtype === dt);
const firstFail = (k, dt, Nn) => RM && RM.numerics_first_failing_scale_vs_truth ? RM.numerics_first_failing_scale_vs_truth[`${k}|${dt}|N=${Nn}`] : undefined;
const stab = (c, k) => RG && RG.stability_replica.find((r) => r.case === c && r.kernel === k);
const hrun = (k, quick) => RG && RG.harness_runs.find((r) => r.kernel === k && r.quick === quick);
const numRow = (k, dt, Nn, sc) => RG && RG.numerics.find((r) => r.kernel === k && r.dtype === dt && r.N === Nn && r.scale === sc);
const zeroScale = RM ? firstFail("best", "float16", timed.N) : null;           // first failing scale at the timed N
const zB = zeroScale ? numRow("best", "float16", timed.N, zeroScale) : null, zS = zeroScale ? numRow("starter", "float16", timed.N, zeroScale) : null;
const eagerByteRatio = RG ? RG.gbps_primary.torch_eager.gbps_cold_on_its_own_analytic_bytes / RG.gbps_primary.torch_eager.gbps_cold : null;
const ci = (k, n) => RG && RG.compile_info[`${k}_N${n}`];
const eagerKernels = RG ? RG.eager_cuda_kernels.map((s) => {
  const m = s.match(/(pow|MeanOps|add|sqrt|Div|Mul)/i); return m ? m[1].replace("MeanOps", "mean").toLowerCase() : "?"; }) : null;
if (P) {                                                                // bytes cross-check vs the remeasure job
  if (P.best.fused_bytes !== TBYTES) console.warn(`bytes mismatch: analytic ${TBYTES} vs remeasure ${P.best.fused_bytes}`);
  console.log(`remeasure: L4 cold starter ${f(P.starter.cold_median_us, 1)} us, best ${f(P.best.cold_median_us, 1)} us, eager ${f(P.torch_eager.cold_median_us, 1)} us; casts ${sgnPct(castGain)}`);
} else console.log("remeasure: results/v3/rmsnorm_remeasure.json absent -> PENDING placeholders");

// ---------- visual identity (v2) ----------
const INKBG = "1B1A3A", INDIGO = "2E2A6B", ACC = "E8A33D", ACC_D = "9A5712", RED = "B3261E", GREEN = "1E7B34";
const LIGHT = "FCFCFB", CARD = "F2F1EC", TINT = "EDECF5", WHITE = "FFFFFF", CODEBG = "23213F";
const TXT = "1F1F24", TXT2 = "4A4A55", MUTED = "85848F", DTXT = "E6E4F0", DMUTED = "A9A6C4";
const HEAD = "Cambria", BODY = "Calibri", MONO = "Courier New";
const W = 13.333, M = 0.5, CW = W - 2 * M;

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.title = "GPU kernel verification: why correctness and timing are harder than they look";

let slideNo = 0;
function base(title, { dark = false, appendix = false, sub = null } = {}) {
  const s = pres.addSlide();
  slideNo += 1;
  s.background = { color: dark ? INKBG : LIGHT };
  if (title) {
    if (title.length > 55) console.warn(`title > 55 chars (${title.length}): ${title}`);
    s.addText(title, { x: M, y: 0.32, w: CW, h: 0.7, fontFace: HEAD, fontSize: 30, bold: true,
      color: dark ? WHITE : INKBG, margin: 0, isTextBox: true, valign: "middle", fit: "none" });
  }
  if (sub) s.addText(sub, { x: M, y: 0.98, w: CW, h: 0.36, fontFace: HEAD, fontSize: 16, italic: true,
    color: dark ? ACC : ACC_D, margin: 0, isTextBox: true, valign: "middle" });
  s.addText(`${appendix ? "Appendix · " : ""}${N.label} · ${slideNo}`, { x: M, y: 7.02, w: 4.5, h: 0.28,
    fontFace: BODY, fontSize: 10, color: dark ? DMUTED : MUTED, margin: 0, isTextBox: true });
  return s;
}
function cite(s, text, dark = false) {
  s.addText(text, { x: 5.0, y: 7.02, w: W - M - 5.0, h: 0.28, fontFace: BODY, fontSize: 10, italic: true,
    color: dark ? DMUTED : MUTED, align: "right", margin: 0, isTextBox: true });
}
function txt(s, text, x, y, w, h, o = {}) {
  s.addText(text, Object.assign({ x, y, w, h, fontFace: BODY, fontSize: 16, color: TXT, margin: 0,
    isTextBox: true, valign: "top", paraSpaceAfter: 4 }, o));
}
function bullets(s, items, x, y, w, h, o = {}) {
  // pptxgenjs/LibreOffice split mixed-format runs into separate paragraphs, so each bullet is one plain run
  s.addText(items.map((t, i) => ({ text: Array.isArray(t) ? t.map((r) => r.text).join("") : t,
    options: Object.assign({ bullet: { indent: 14 } }, i < items.length - 1 ? { breakLine: true } : {}) })),
    Object.assign({ x, y, w, h, fontFace: BODY, fontSize: o.fontSize || 15, color: TXT, margin: 0, isTextBox: true,
    valign: "top", paraSpaceAfter: 6 }, o));
}
function card(s, x, y, w, h, fill = CARD, line = null) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: fill },
    line: line ? { color: line, width: 1.5, dashType: "dash" } : { color: fill } });
}
function stat(s, x, y, w, h, value, label, o = {}) {
  const dark = !!o.dark;
  card(s, x, y, w, h, o.fill || (dark ? "2A2858" : CARD));
  const vh = o.vh || 0.6;
  s.addText(value, { x: x + 0.18, y: y + 0.12, w: w - 0.36, h: vh, fontFace: HEAD, fontSize: o.vs || 26, bold: true,
    color: o.color || (dark ? ACC : INDIGO), margin: 0, isTextBox: true, valign: "middle" });
  s.addText(label, { x: x + 0.18, y: y + 0.16 + vh, w: w - 0.36, h: h - vh - 0.24, fontFace: BODY, fontSize: o.ls || 13,
    color: dark ? DTXT : TXT2, margin: 0, isTextBox: true, valign: "top" });
}
function pngSize(file) { const b = fs.readFileSync(file); return [b.readUInt32BE(16), b.readUInt32BE(20)]; }
function fig(s, rel, x, y, w, h, align = "c", valign = "t") {
  const file = path.join(ROOT, "presentation", rel);
  const [pw, ph] = pngSize(file);
  const r = pw / ph;
  let iw = w, ih = w / r;
  if (ih > h) { ih = h; iw = h * r; }
  const ix = align === "l" ? x : align === "r" ? x + w - iw : x + (w - iw) / 2;
  const iy = valign === "m" ? y + (h - ih) / 2 : y;
  s.addImage({ path: file, x: ix, y: iy, w: iw, h: ih });
  return { x: ix, y: iy, w: iw, h: ih };
}
function table(s, rows, x, y, w, colW, o = {}) {
  s.addTable(rows.map((r, i) => r.map((c, j) => {
    const cell = typeof c === "object" ? c : { text: String(c) };
    return { text: cell.text, options: Object.assign({ bold: i === 0 || (o.boldFirstCol && j === 0),
      color: i === 0 ? WHITE : TXT, fill: { color: i === 0 ? INDIGO : (i % 2 ? CARD : LIGHT) }, fontFace: BODY,
      fontSize: o.fontSize || 14, align: j && o.centerCols ? "center" : "left", valign: "middle", margin: [2, 5, 2, 5] }, cell.options || {}) };
  })), { x, y, w, colW, rowH: o.rowH || 0.4, border: { type: "none" } });
}
function pending(s, x, y, w, h, what) {
  card(s, x, y, w, h, "FFF7E8", ACC);
  s.addText([
    { text: "PENDING remeasure", options: { bold: true, color: ACC_D, fontSize: 15, breakLine: true } },
    { text: what, options: { color: TXT2, fontSize: 13, breakLine: true } },
    { text: "Fills from results/v3/rmsnorm_remeasure.json on rebuild (node presentation/v3/build_deck_v3.js).", options: { color: MUTED, italic: true, fontSize: 11 } },
  ], { x: x + 0.2, y: y + 0.1, w: w - 0.4, h: h - 0.2, fontFace: BODY, margin: 0, isTextBox: true, valign: "middle", paraSpaceAfter: 4 });
}
function codeBlock(s, lines, x, y, w, h, o = {}) {
  card(s, x, y, w, h, CODEBG);
  if (o.caption) s.addText(o.caption, { x: x + 0.22, y: y + 0.1, w: w - 0.44, h: 0.34, fontFace: BODY, fontSize: 13, color: DMUTED, margin: 0, isTextBox: true });
  const top = o.caption ? 0.5 : 0.15;
  s.addText(lines.map((l, i) => ({ text: l.t || " ", options: { color: l.c || "D8D6EA", bold: !!l.b, breakLine: i < lines.length - 1 } })),
    { x: x + 0.22, y: y + top, w: w - 0.4, h: h - top - 0.1, fontFace: MONO, fontSize: o.fs || 12, margin: 0, isTextBox: true, valign: "top", paraSpaceAfter: 1 });
}
const us1 = (v) => (v === undefined || v === null ? "PENDING" : `${f(v, 1)} µs`);
const verdictCell = (v) => ({ text: v, options: { bold: true, color: /FAIL/.test(v) ? RED : GREEN } });

// kernel source excerpts (verbatim code from presentation/v2/best_kernel.py; strings, not numbers)
const SRC = fs.readFileSync(path.join(ROOT, "presentation", "v2", "best_kernel.py"), "utf8").split("\n");
const srcLine = (re) => SRC.map((l) => l.trim()).find((l) => re.test(l));
const diffLines = BK.diff_vs_starter.filter((l) => l.slice(1).startsWith("    ") && !/#/.test(l))
  .map((l) => ({ t: `${l[0]} ${l.slice(1).trim().replace(/tl\.load\([^)]*\)/, "tl.load(…)")}`, c: l[0] === "-" ? "F08A80" : "8FD6A0" }));

// =============================================================================
// 1. Title
{
  const s = base(null, { dark: true });
  s.addText("GPU kernel verification", { x: 0.8, y: 1.55, w: 11.8, h: 1.0, fontFace: HEAD, fontSize: 48, bold: true, color: WHITE, margin: 0, isTextBox: true });
  s.addText("why correctness and timing are harder than they look", { x: 0.8, y: 2.55, w: 11.8, h: 0.7, fontFace: HEAD, fontSize: 28, italic: true, color: ACC, margin: 0, isTextBox: true });
  s.addText("Building and validating a Triton kernel evaluation harness: AutoKernel bench.py and a KernelBench bridge, on Modal L4 / H100",
    { x: 0.8, y: 3.7, w: 11.8, h: 0.5, fontFace: BODY, fontSize: 18, color: DTXT, margin: 0, isTextBox: true });
  s.addText(`Use case: a ${C.model.split("/")[1]} GRPO kernel agent whose reward is the harness`,
    { x: 0.8, y: 4.25, w: 11.8, h: 0.45, fontFace: BODY, fontSize: 18, color: DTXT, margin: 0, isTextBox: true });
  s.addText(`Artifacts: results/ (every rollout, numbers.json, v3/), code in modal_app/ · torch ${VER.torch}, Triton ${VER.triton}, AutoKernel ${VER.autokernel_commit}`,
    { x: 0.8, y: 5.2, w: 11.8, h: 0.4, fontFace: BODY, fontSize: 14, color: DMUTED, margin: 0, isTextBox: true });
  s.addNotes(`This talk is about kernel verification. I built and debugged an evaluation harness for Triton kernels: the thing that decides whether a generated kernel is correct and how fast it is. ` +
    `The use case that forced me to take it seriously was an RL study, ${C.model} trained with GRPO, where the harness is the reward. ` +
    `Everything is re-run on Modal with torch ${VER.torch} and Triton ${VER.triton}; every number on the slides is read from results/ at build time.`);
}

// 2. Hook
{
  const s = base(P ? `The verifier paid ${f(BK.speedup, 1)}× for ${Math.abs(100 * castGain).toFixed(0)}% speed and broken numerics` : `Two casts: FAIL → a ${f(BK.speedup, 1)}× reward`, { sub: P
    ? `two dtype casts flipped FAIL → PASS; the speed was the starter's fusion, and rows with RMS > ${f(rmsOverflow(timed.N), 1)} now come back as zeros`
    : "…and no measurable speed of their own (remeasure pending)" });
  codeBlock(s, diffLines, M, 1.5, 6.3, 1.75, { caption: `${BK.kernel_type}: every code line that changed vs the starter`, fs: 13 });
  const st = S1.per_gpu.L4.rmsnorm;
  const rows = [
    ["", "starter", "model's kernel"],
    ["full bench (L4)", { text: `FAIL: ${stagesOk(st.stages_full)}/5 stages`, options: { color: RED, bold: true } }, { text: `PASS: ${stagesOk(WT.stages)}/5 stages`, options: { color: GREEN, bold: true } }],
    ["failing stage", "numerical stability", "–"],
    ["logged speedup", `${f(st.speedup_quick)}× (quick mode)`, `${f(BK.speedup)}× (full) = reward`],
    ["remeasured, cold L2", us1(P && P.starter.cold_median_us), { text: us1(P && P.best.cold_median_us), options: { bold: true, color: P ? TXT : ACC_D } }],
    ["eager PyTorch", { text: us1(P && P.torch_eager.cold_median_us), options: { colspan: 2 } }],
    [zB ? `zero rows, RMS ≈ ${zeroScale}` : "numerics vs fp64", zS ? `${zS.zero_rows}/${zS.rows}` : "PENDING", zB ? { text: `${zB.zero_rows}/${zB.rows}`, options: { bold: true, color: RED } } : "PENDING"],
  ];
  table(s, rows, M, 3.4, 6.3, [2.0, 2.15, 2.15], { rowH: 0.4, fontSize: 14, boldFirstCol: true, centerCols: true });
  const rx = 7.2, rw = W - M - rx;
  s.addText("What actually happened", { x: rx, y: 1.5, w: rw, h: 0.4, fontFace: HEAD, fontSize: 18, bold: true, color: TXT, margin: 0, isTextBox: true });
  bullets(s, [
    [{ text: "Speed was already there: ", options: { bold: true } }, { text: P
      ? `the ${f(P.torch_eager.cold_median_us / P.starter.cold_median_us, 1)}× over eager PyTorch is the starter's; the casts moved latency ${sgnPct(-castGain / (1 + castGain))}.`
      : `the starter times at ${f(st.speedup_quick)}×; on the timed ${BC.timed_dtype} path the cast is a no-op.` }],
    [{ text: "Numerics broke: ", options: { bold: true } }, { text: zB ? `Σx² now runs in fp16 and overflows; at N=${timed.N}, RMS ≈ ${zeroScale} zeroes ${zB.zero_rows}/${zB.rows} rows (fp32 starter: ${zS.zero_rows}).` : "Σx² now runs in fp16 and can overflow (remeasure pending)." }],
    [{ text: "Why the verifier paid: ", options: { bold: true } }, { text: `its reference squares in ${BC.timed_dtype} and overflows identically, so the correct fp32 starter FAILs; and speed is scored vs PyTorch, not vs the starter.` }],
  ], rx, 2.0, rw, 2.7, { fontSize: 14 });
  card(s, rx, 4.85, rw, 1.4, INKBG);
  txt(s, "This talk: the harness that surfaced this, what else it got wrong, and how I would verify and time kernels so a PASS and a speedup mean something.",
    rx + 0.25, 4.93, rw - 0.5, 1.25, { fontSize: 14, color: DTXT, valign: "middle" });
  cite(s, `${BK.source} step ${BK.step}, code ${BK.code_sha} · stage1_baselines.jsonl · ${RM ? "results/v3/rmsnorm_remeasure.json" : "remeasure pending"}`);
  s.addNotes(`Here is the hook. The fastest correct kernel in the project is an RMSNorm the verifier scored at ${f(BK.speedup)}x over PyTorch on L4. It differs from the starter in exactly two casts, float32 to float16 on the loads. ` +
    `The starter already ran at ${f(st.speedup_quick)}x in quick mode, but it fails the full bench on numerical stability; the model's version passes all five stages and so it gets the reward. ` +
    (P ? `Remeasured on L4 with a cold L2 and ${RG.reps} repetitions, the starter takes ${f(P.starter.cold_median_us, 1)} microseconds and the model's kernel ${f(P.best.cold_median_us, 1)}, against ${f(P.torch_eager.cold_median_us, 1)} for eager PyTorch. The casts are worth ${sgnPct(castGain)}, which is noise. And at N=${timed.N}, inputs with row RMS around ${zeroScale} make the model's kernel return zeros in ${zB.zero_rows} of ${zB.rows} rows, where the starter returns none. ` : `The remeasurement of starter against model kernel is pending. `) +
    `So what changed is correctness, not speed, and in the wrong direction: the reference computes x squared in fp16 and overflows on the adversarial inputs, and the fp16 kernel overflows identically, so it matches. ` +
    `Two verifier lessons. A reference with the kernel's own precision bug can't tell a better kernel from a matching bug. And a reward measured against PyTorch, rather than against the starter or the previous turn, credits an edit with speed it didn't create.`);
}

// 3. The harness path
{
  const s = base("The harness: model output → reward", { sub: "every box below gave a wrong answer at least once" });
  const steps = [
    ["Parse", "code block → kernel.py", "worker passed --kernel-type and parsed results.tsv; bench.py takes --kernel and prints to stdout"],
    ["Correctness", "5 stages: smoke, shape sweep, stability, determinism, edge sizes", `--quick skips stability; both-NaN/Inf outputs accepted (bench.py:${BC.both_nan_inf_accepted_line})`],
    ["Timing", `do_bench, ${BC.timed_dtype} ${timed.M}×${timed.N}, vs eager PyTorch`, `one shape, one dtype, PyTorch-only baseline; H100 spec lookup reports ${pp(h100Pct, 0)} of peak BW`],
    ["Cache", "sha256(gpu, mode, kernel, code)", `${LV.best_pass_records_cached}/${LV.best_pass_records} best-PASS records in low-variance groups were cache replays`],
    ["Reward", "log2 speedup if PASS, else 0", `a slow PASS scored below a crash (${V1.ablations.clip.slower_than_pytorch_pass}/${V1.ablations.clip.n_pass_traj} passing episodes)`],
  ];
  const bw = (CW - 4 * 0.18) / 5;
  steps.forEach(([h, d, bug], i) => {
    const x = M + i * (bw + 0.18);
    card(s, x, 1.55, bw, 1.95, INDIGO);
    s.addText(h, { x: x + 0.15, y: 1.65, w: bw - 0.3, h: 0.45, fontFace: HEAD, fontSize: 18, bold: true, color: ACC, margin: 0, isTextBox: true });
    s.addText(d, { x: x + 0.15, y: 2.12, w: bw - 0.3, h: 1.3, fontFace: BODY, fontSize: 13, color: WHITE, margin: 0, isTextBox: true, valign: "top" });
    if (i < 4) s.addText("→", { x: x + bw - 0.02, y: 2.25, w: 0.22, h: 0.4, fontFace: BODY, fontSize: 18, bold: true, color: ACC_D, margin: 0, isTextBox: true, align: "center" });
    card(s, x, 3.65, bw, 1.75, "FBEAE8");
    s.addText([{ text: "failure found: ", options: { bold: true, color: RED } }, { text: bug, options: { color: TXT } }],
      { x: x + 0.15, y: 3.75, w: bw - 0.3, h: 1.55, fontFace: BODY, fontSize: 13, margin: 0, isTextBox: true, valign: "top" });
  });
  card(s, M, 5.6, CW, 1.15, TINT);
  txt(s, [{ text: "KernelBench bridge (second harness, held-out eval): ", options: { bold: true } },
    { text: `starter keeps super(Model, self) → every starter crashes; weights never seeded → all ${KB.conv_unseeded} conv problems fail even for an identical model; CPU get_inputs() timed inside the 30 s trial → ${KB.flaky_timeouts} flaky timeouts. Only ${KB.valid}/${KB.n} Level 1 problems are harness-valid.` }],
    M + 0.25, 5.68, CW - 0.5, 1.0, { fontSize: 14, color: TXT2, valign: "middle" });
  cite(s, "modal_app/bench_modal.py · results/autokernel_src/bench.py · CLAUDE.md findings");
  s.addNotes(`This is the request-to-result path. The model's code block becomes kernel.py; bench.py runs five correctness stages; if they pass it times the kernel against eager PyTorch with do_bench; the result is cached by a hash of GPU, mode, kernel type and code; and the reward is computed from it. ` +
    `Each box had a real defect. Parsing: the original worker used a flag bench.py doesn't have and read a file it never writes, so no reward could be parsed at all. Correctness: quick mode skips stability, and the stability check accepts any output with NaN or Inf if the reference has them too. ` +
    `Timing: one shape, one dtype, and PyTorch is the only baseline; on H100 the spec lookup gave ${pp(h100Pct, 0)} of peak bandwidth, which is impossible. Cache: in the low-variance groups, ${LV.best_pass_records_cached} of ${LV.best_pass_records} best-PASS timings were replays. Reward: a correct but slightly slower kernel scored below a crash. ` +
    `The KernelBench bridge was worse: every starter crashed, ${KB.conv_unseeded} conv problems could never pass, and only ${KB.valid} of ${KB.n} were usable.`);
}

// 4. RMSNorm anatomy
{
  const s = base("RMSNorm, line by line", { sub: "y = x / sqrt(mean(x²) + ε) · w, one program per row" });
  const code = [
    { t: srcLine(/^row = tl\.program_id/) },
    { t: srcLine(/^offs = tl\.arange/) },
    { t: srcLine(/^mask = offs < N/) },
    { t: "x = tl.load(X_ptr + row*stride_xm + offs*stride_xn," },
    { t: "            mask=mask, other=0.0).to(tl.float16)", c: ACC, b: true },
    { t: srcLine(/^sq_mean = /), c: ACC, b: true },
    { t: srcLine(/^rms = tl\.sqrt/) },
    { t: srcLine(/^x_norm = /) },
    { t: "w = tl.load(W_ptr + offs, mask=mask, other=0.0).to(tl.float16)", c: ACC, b: true },
    { t: srcLine(/^out = x_norm \* w/) },
    { t: "tl.store(OUT_ptr + row*stride_om + offs*stride_on, out, mask=mask)" },
    { t: "" },
    { t: "# launch", c: "8C89A8" },
    { t: srcLine(/^BLOCK_SIZE = triton\.next_power_of_2/) },
    { t: "rmsnorm_kernel[(M,)](x, weight, out, M, N, …, BLOCK_SIZE=BLOCK_SIZE)" },
  ];
  codeBlock(s, code, M, 1.5, 7.3, 5.2, { caption: "model's kernel (verbatim; amber = where precision is decided)", fs: 12 });
  const rx = 8.05, rw = W - M - rx;
  const small = BC.rmsnorm_test_sizes[0];
  const waste = 1 - small.N / nextPow2(small.N);
  const cS = ci("starter", timed.N), cB = ci("best", timed.N);
  s.addText("Mapping", { x: rx, y: 1.5, w: rw, h: 0.38, fontFace: HEAD, fontSize: 17, bold: true, color: TXT, margin: 0, isTextBox: true });
  bullets(s, [
    "grid = (M,): one program owns a whole row; the reduction is one in-register tl.sum",
    `BLOCK = next_pow2(N): N=${timed.N} → ${nextPow2(timed.N)} lanes; N=${small.N} → ${nextPow2(small.N)}, ${pc(waste, 0)} masked off`,
    "tail: masked lanes load 0.0, so the sum is exact; it divides by N, not BLOCK",
    cB ? `compiled (L4, N=${timed.N}): ${cB.num_warps} warps, ${cS.n_regs}→${cB.n_regs} regs, ${cB.n_spills} spills, ${cB.ptx_ld_global} vector loads either way`
      : `no num_warps / autotune: Triton default (${LF.triton_default_num_warps} warps)`,
    cB ? `PTX: fp32 mul/fma ${cS.ptx_fp32_fma_or_mul}→${cB.ptx_fp32_fma_or_mul}, f16 ops ${cS.ptx_f16_ops}→${cB.ptx_f16_ops}: the math moved to fp16, the memory ops did not`
      : "strides are passed, so strided x works; w is re-read by every row (L2-resident)",
  ], rx, 1.9, rw, 3.1, { fontSize: 13 });
  s.addText("Contract bench.py checks", { x: rx, y: 5.05, w: rw, h: 0.38, fontFace: HEAD, fontSize: 17, bold: true, color: TXT, margin: 0, isTextBox: true });
  const tol = BC.rmsnorm_tolerances;
  bullets(s, [
    `dtypes ${BC.rmsnorm_test_dtypes.join(", ")} only; sizes ${BC.rmsnorm_test_sizes.map((z) => `${z.M}×${z.N}`).join(", ")} + 2 edge sizes`,
    `tolerance: fp16 atol = rtol = ${tol.float16.atol}; bf16 atol ${tol.bfloat16.atol}, rtol ${tol.bfloat16.rtol}`,
  ], rx, 5.45, rw, 1.35, { fontSize: 13, color: TXT2 });
  cite(s, `presentation/v2/best_kernel.py · bench.py rmsnorm config${cB ? " · remeasure compile_info" : ""}`);
  s.addNotes(`The kernel is the textbook one-row-per-program RMSNorm. Each program loads a full row into registers, BLOCK_SIZE is the next power of two of N, masked lanes load zero so the sum of squares is exact, and it divides by N. ` +
    `At the timed size N is ${timed.N}, so each program holds ${nextPow2(timed.N)} elements; at the small size ${small.N} rounds up to ${nextPow2(small.N)} and ${pc(waste, 0)} of lanes are wasted. ` +
    (cB ? `From the compiled code: ${cB.num_warps} warps, ${cB.n_regs} registers and no spills, and the same ${cB.ptx_ld_global} vectorised global loads as the starter. The fp32 multiply count halves and fp16 ops go up: the casts moved arithmetic into fp16 and left memory traffic alone. ` : "") +
    `The amber lines are where precision is decided: the two load casts and the sum of x*x. In the starter both casts are to float32, so x*x, the reduction, the divide and the weight multiply run in fp32 and only the store narrows back. ` +
    `bench.py tests fp16 and bf16 only, with fp16 tolerance ${tol.float16.atol} and a much looser bf16 tolerance of ${tol.bfloat16.atol} absolute.`);
}

// 5. Numerics: mechanism
{
  const s = base("What the two casts change numerically", { sub: "fp32 accumulation vs fp16: the model's kernel forms x*x and sums it in fp16" });
  const rows = [
    ["step (fp16 input)", "PyTorch reference", "starter", "model's kernel"],
    ["load x, w", "fp16", "fp32", "fp16"],
    ["x · x", { text: "fp16 (x ** 2)", options: { color: RED, bold: true } }, "fp32", { text: "fp16", options: { color: RED, bold: true } }],
    ["Σ over N", "mean: fp32 accum.", "fp32", { text: "fp16 (tl.sum: no widening)", options: { color: RED, bold: true } }],
    ["÷ rms, · w", "fp16", "fp32", "fp16"],
    ["store", "fp16", "fp32 → fp16", "fp16"],
    ["bf16 input", "bf16", "fp32", { text: "bf16 → fp16: range narrows", options: { color: RED } }],
  ];
  table(s, rows, M, 1.5, 7.4, [1.75, 1.9, 1.35, 2.4], { rowH: 0.43, fontSize: 13, boldFirstCol: true, centerCols: true });
  const tw = (7.4 - 0.4) / 3, y2 = 4.65;
  stat(s, M, y2, tw, 1.3, `|x| > ${f(sqOverflow, 1)}`, "x·x overflows fp16 (max 65504)", { vs: 20 });
  stat(s, M + tw + 0.2, y2, tw, 1.3, `RMS > ${f(rmsOverflow(timed.N), 1)}`, `fp16 Σx² overflows at N=${timed.N}`, { vs: 20 });
  stat(s, M + 2 * (tw + 0.2), y2, tw, 1.3, `RMS > ${f(rmsOverflow(BC.rmsnorm_test_sizes[0].N), 1)}`, `…at N=${BC.rmsnorm_test_sizes[0].N} (stability size)`, { vs: 20 });
  txt(s, `randn rows have RMS ≈ 1, so the sweep passes; activations with outlier channels need not. Worst sweep error is ${WT.stages.shape_sweep.match(/worst_err=([\d.e+-]+)/)[1]} for both kernels (bf16).`,
    M, 6.08, 7.4, 0.7, { fontSize: 13, color: TXT2 });
  const rx = 8.2, rw = W - M - rx;
  s.addText("Why the stability stage flips", { x: rx, y: 1.5, w: rw, h: 0.38, fontFace: HEAD, fontSize: 17, bold: true, color: TXT, margin: 0, isTextBox: true });
  const [hiS, loS] = BC.stability_mixed_scale;
  bullets(s, [
    `mixed_scale: every float input × ${hiS} or × ${loS}, ${BC.stability_size_label} size, first dtype (${BC.rmsnorm_test_dtypes[0]})`,
    `reference: x**2 → inf → rms = inf → output 0 (finite, so "clean")`,
    [{ text: "starter (fp32): ", options: { bold: true } }, { text: `correct values ≫ 0 → exceeds ${BC.stability_tolerance_relax_factor}× tol → FAIL` }],
    [{ text: "model (fp16): ", options: { bold: true } }, { text: "overflows the same way → 0 = 0 → PASS" }],
    `near_max (× ${BC.stability_near_max_fp16_scale}): both inf → accepted by the both-NaN/Inf rule`,
  ], rx, 1.92, rw, 3.0, { fontSize: 14 });
  card(s, rx, 5.0, rw, 1.7, TINT);
  txt(s, [{ text: "Prediction to test: ", options: { bold: true } },
    { text: P ? "the remeasure replays stage 3 and checks every kernel against an fp64 truth: next slide." : "replay stage 3 against an fp64 truth (remeasure pending)." }],
    rx + 0.2, 5.1, rw - 0.4, 1.5, { fontSize: 14, color: TXT2, valign: "middle" });
  cite(s, `bench.py stage 3 · reference.rmsnorm_ref · Triton ${VER.triton} tl.sum`);
  s.addNotes(`Here is the dtype flow for an fp16 input. The PyTorch reference materialises x ** 2 as an fp16 tensor; its mean accumulates in fp32, but by then the squares have already overflowed. The starter upcasts on load, so everything is fp32 until the store. The model's kernel stays in fp16 throughout, including x*x and the tl.sum, which in Triton ${VER.triton} does not widen floating inputs. ` +
    `That matters at modest magnitudes: x*x overflows fp16 once |x| exceeds ${f(sqOverflow, 1)}, and an fp16 sum of ${timed.N} squares overflows once the row RMS exceeds about ${f(rmsOverflow(timed.N), 1)}. Random-normal test rows have RMS near 1, so the shape sweep never sees it. On bf16 inputs the cast to fp16 narrows the range. ` +
    `The stability stage multiplies all inputs by ${hiS} or ${loS}. The fp16 reference overflows to an infinite RMS and outputs zeros; the correct fp32 starter disagrees with those zeros and fails; the fp16 kernel overflows the same way and passes. That is a prediction from reading the code, and the next slide tests it.`);
}

// 6. Numerics: measured
{
  const s = base("Measured vs fp64 truth: the oracle is the bug", { sub: P ? "the stability stage rejects every correct kernel and accepts the overflowing one" : "remeasure pending" });
  if (RG) {
    const ks = [["starter", "starter (fp32)"], ["best", "model's kernel (fp16)"], ["fp32acc", "hand-written fp32-acc"], ["torch_compile", "torch.compile"]];
    const ms = "mixed_scale", ref = stab(ms, "best");
    const rows = [["kernel", "harness verdict", "zero rows", "max |err| vs fp64"]];
    ks.forEach(([k, lab]) => { const r = stab(ms, k); rows.push([lab, verdictCell(r.harness_verdict), `${r.kernel_zero_rows}/${BC.rmsnorm_test_sizes[0].M}`, f(r.max_abs_err_vs_truth_finite, r.max_abs_err_vs_truth_finite > 100 ? 0 : 2)]); });
    rows.push([{ text: "harness reference", options: { bold: true, color: RED } }, "–", `${ref.harness_ref_zero_rows}/${BC.rmsnorm_test_sizes[0].M}`, { text: f(ref.max_abs_err_harness_ref_vs_truth_finite, 0), options: { bold: true, color: RED } }]);
    s.addText(`Stage 3 replayed: ${ms}, ${BC.rmsnorm_test_dtypes[0]}, ${BC.rmsnorm_test_sizes[0].M}×${BC.rmsnorm_test_sizes[0].N} (L4)`, { x: M, y: 1.5, w: 7.0, h: 0.38, fontFace: HEAD, fontSize: 16, bold: true, color: TXT, margin: 0, isTextBox: true });
    table(s, rows, M, 1.95, 7.0, [2.4, 1.6, 1.3, 1.7], { rowH: 0.44, fontSize: 13, boldFirstCol: true, centerCols: true });
    const nm = stab("near_max", "best");
    bullets(s, [
      `the reference zeroes all ${ref.harness_ref_zero_rows} rows; the model's kernel matches it exactly, so it PASSes with the reference's error`,
      `near_max: every kernel PASSes via "both NaN/Inf" (${nm.harness_ref_nonfinite.toLocaleString("en-US")} non-finite reference outputs, never compared)`,
      "an fp32-accumulating kernel and torch.compile both FAIL: correctness here means agreeing with an overflowing reference",
      zB ? `outside the harness: at N=${timed.N}, scale ${zeroScale}, the model's kernel zeroes ${zB.zero_rows}/${zB.rows} rows; fp32 kernels and eager zero none` : "",
    ].filter(Boolean), M, 4.72, 7.0, 2.2, { fontSize: 12, color: TXT2 });
    const rx = 7.95, rw = W - M - rx;
    s.addText("First input scale that fails vs fp64 truth", { x: rx, y: 1.5, w: rw, h: 0.38, fontFace: HEAD, fontSize: 16, bold: true, color: TXT, margin: 0, isTextBox: true });
    const Ns = [...new Set(Object.keys(RM.numerics_first_failing_scale_vs_truth).map((k) => Number(k.split("N=")[1])))].sort((a, b) => a - b);
    const t2 = [["N", "predicted RMS*", "model fp16", "model bf16", "eager fp16"]];
    Ns.forEach((n) => t2.push([String(n), f(rmsOverflow(n), 1), { text: String(firstFail("best", "float16", n) ?? "–"), options: { bold: true, color: RED } },
      String(firstFail("best", "bfloat16", n) ?? "–"), String(firstFail("torch_eager", "float16", n) ?? "–")]));
    table(s, t2, rx, 1.95, rw, [0.8, 1.2, 0.95, 0.95, rw - 3.9], { rowH: 0.42, fontSize: 13, centerCols: true });
    txt(s, `inputs = scale × randn, so row RMS ≈ scale. * fp16 Σx² overflow threshold sqrt(65504/N), computed here. The model's kernel breaks exactly where fp16 accumulation predicts; eager PyTorch holds until x² itself overflows.`,
      rx, 4.2, rw, 1.35, { fontSize: 13, color: TXT2 });
    card(s, rx, 5.6, rw, 1.1, INKBG);
    txt(s, "Fix the oracle, not the kernel: an fp64 / upcast truth reference and exact NaN/Inf masks would flip both verdicts.", rx + 0.2, 5.65, rw - 0.4, 1.0, { fontSize: 14, color: DTXT, valign: "middle" });
  } else pending(s, M, 1.6, CW, 2.4, "Stage-3 replay per kernel vs an fp64 truth (verdict, zero rows, max error) and the first input scale at which each kernel fails.");
  cite(s, "results/v3/rmsnorm_remeasure.json: stability_replica, numerics_first_failing_scale_vs_truth");
  s.addNotes(P ? (() => {
    const ref = stab("mixed_scale", "best"), st0 = stab("mixed_scale", "starter");
    return `The remeasure replays stage 3 exactly and also compares every output with an fp64 truth. On mixed_scale the harness reference zeroes all ${ref.harness_ref_zero_rows} rows and is off from the truth by ${f(ref.max_abs_err_harness_ref_vs_truth_finite, 0)}. ` +
      `The starter, a hand-written fp32-accumulating kernel and torch.compile all agree with the truth to within ${f(st0.max_abs_err_vs_truth_finite, 2)}, and all three FAIL. The model's fp16 kernel reproduces the reference's zeros and PASSes. ` +
      `On the right, the input scale at which each kernel first fails against the truth with the harness tolerance. The model's kernel fails at scale ${firstFail("best", "float16", timed.N)} for N=${timed.N}, right where the fp16 accumulation threshold of ${f(rmsOverflow(timed.N), 1)} predicts; eager PyTorch holds out to ${firstFail("torch_eager", "float16", timed.N)}, where x squared itself overflows. ` +
      `So the verifier is rewarding a kernel that is less accurate than PyTorch by more than an order of magnitude in input range. The fix belongs in the oracle.`;
  })() : "Placeholder until the remeasure JSON is present.");
}

// 7. Memory traffic
{
  const s = base("Memory traffic: same bytes, same speed", { sub: "RMSNorm is bandwidth-bound: read x, read w, write y" });
  const shapes = [BC.rmsnorm_test_sizes[0], BC.rmsnorm_test_sizes[1], BC.rmsnorm_test_sizes[3], timed, { label: "big", M: 16384, N: 8192 }]
    .filter((z) => !RG || rtab("best", z.M, z.N));
  const rows = [["shape (fp16)", "bytes (MB)", "floor @ L4 peak", "starter", "model's kernel", "eager PyTorch", "model GB/s", "% of peak"]];
  shapes.forEach((z) => {
    const b = bytesRms(z.M, z.N), rs = rtab("starter", z.M, z.N), rb = rtab("best", z.M, z.N), re = rtab("torch_eager", z.M, z.N);
    const lab = `${z.M}×${z.N}${z === timed ? " (timed)" : ""}`;
    rows.push([lab, f(MB(b), 1), `${f(floorUs(b), 1)} µs`, rs ? us1(rs.cold_median_us) : "PENDING", rb ? { text: us1(rb.cold_median_us), options: { bold: true } } : "PENDING",
      re ? us1(re.cold_median_us) : "PENDING", rb ? f(gbs(b, rb.cold_median_us), 0) : "–", rb ? pp(100 * gbs(b, rb.cold_median_us) / L4BW, 0) : "–"]);
  });
  table(s, rows, M, 1.5, CW, [2.2, 1.2, 1.55, 1.45, 1.65, 1.6, 1.3, CW - 10.95], { rowH: 0.4, fontSize: 13, centerCols: true, boldFirstCol: true });
  const yT = 1.5 + 0.4 * rows.length + 0.1;
  txt(s, `L4, cold L2, median of ${RG ? RG.reps : "n"} reps. bytes = (2·M·N + N)·2 B (bench.py bytes_fn), floor at ${L4BW} GB/s: both computed here from the shape. For fp16 input the casts change no load or store width.`,
    M, yT, CW, 0.5, { fontSize: 12, color: MUTED, italic: true });
  const tw = (CW - 0.6) / 4, y2 = 4.55;
  if (P) {
    stat(s, M, y2, tw, 1.45, `${f(P.best.gbps_cold, 0)} GB/s`, `model's kernel at ${timed.M}², cold L2: ${pp(P.best.pct_peak_cold, 0)} of ${L4BW} (starter ${f(P.starter.gbps_cold, 0)})`, { vs: 22 });
    stat(s, M + tw + 0.2, y2, tw, 1.45, `${f(P.best.gbps_warm, 0)} GB/s`, `warm L2: x is ${f(xMiB, 0)} MiB, L2 is ${RG.l2_mb} MB, so warm beats "peak" on small shapes`, { vs: 22 });
    stat(s, M + 2 * (tw + 0.2), y2, tw, 1.45, `${eagerKernels.length} kernels`, `eager PyTorch (${eagerKernels.join(", ")}) moves ${f(eagerByteRatio, 1)}× the bytes: ≈${f(2 * eagerByteRatio, 0)}·M·N vs 2·M·N`, { vs: 22 });
    stat(s, M + 3 * (tw + 0.2), y2, tw, 1.45, `≤ ${f(P.best.cold_median_us / floorUs(TBYTES), 2)}×`, `headroom to the ${f(floorUs(TBYTES), 0)} µs bandwidth floor; torch.compile is ${f(P.torch_compile.cold_median_us, 1)} µs`, { vs: 22 });
  } else {
    stat(s, M, y2, tw, 1.45, `${f(stGBs, 0)} GB/s`, `starter (stage1 quick), ${pp(100 * stGBs / L4BW, 0)} of peak`, { vs: 22 });
    stat(s, M + tw + 0.2, y2, tw, 1.45, `${f(ptGBs, 0)} GB/s`, `eager PyTorch effective, ${f(ptLat, 0)} µs`, { vs: 22 });
    pending(s, M + 2 * (tw + 0.2), y2, 2 * tw + 0.2, 1.45, "per-shape warm/cold latency for starter, model kernel, eager");
  }
  card(s, M, 6.15, CW, 0.62, TINT);
  txt(s, [{ text: "Where 2.9× comes from: ", options: { bold: true } }, { text: "fusion. Eager runs each op as a separate kernel and re-reads the row; one fused pass moves the minimum bytes. The starter already had it; the casts added nothing." }],
    M + 0.2, 6.18, CW - 0.4, 0.56, { fontSize: 13, color: TXT2, valign: "middle" });
  cite(s, "results/v3/rmsnorm_remeasure.json (gpus.L4.table, primary_rows, eager_cuda_kernels) · bench.py");
  s.addNotes(`RMSNorm does about ${f(6 * timed.M * timed.N / TBYTES, 1)} flops per byte, so it is bandwidth-bound. Minimum traffic is read x, read w, write y: ${f(MB(TBYTES), 1)} MB at the timed shape, which at L4's ${L4BW} GB/s gives a floor of ${f(floorUs(TBYTES), 0)} microseconds. ` +
    (P ? `Cold-L2 medians: starter ${f(P.starter.cold_median_us, 1)}, model ${f(P.best.cold_median_us, 1)}, eager ${f(P.torch_eager.cold_median_us, 1)} microseconds. The model's kernel runs at ${f(P.best.gbps_cold, 0)} GB/s, ${pp(P.best.pct_peak_cold, 0)} of peak, so there is at most ${f(P.best.cold_median_us / floorUs(TBYTES), 2)}x left, and torch.compile lands at the same ${f(P.torch_compile.cold_median_us, 1)}. ` +
      `Warm L2 matters: the timed x is ${f(xMiB, 0)} MiB against a ${RG.l2_mb} MB L2, so warm runs reach ${f(P.best.gbps_warm, 0)} GB/s, and on the small shapes warm numbers exceed DRAM peak. Eager PyTorch launches ${eagerKernels.length} kernels, ${eagerKernels.join(", ")}, which is the whole 2.9x. ` : `The starter reached ${f(stGBs, 0)} GB/s in stage 1; per-shape numbers are pending. `) +
    `For fp16 inputs the casts change nothing that is loaded or stored, so the bytes and the speed are the same.`);
}

// 8. How the 2.9x was measured
{
  const s = base("How the 2.9× was measured", { sub: "and what it looks like measured properly" });
  const hB = hrun("best", false), hS = hrun("starter", false);
  const rows = [
    ["", `logged: starter ${f(S1.per_gpu.L4.rmsnorm.speedup_quick)}×`, `logged: model ${f(BK.speedup)}×`, "remeasure (both)"],
    ["bench mode", "quick (stability skipped)", "full (5 stages)", "no gate; fp64 checked apart"],
    ["observations", `${L4q.length} runs: ${f(Math.min(...stSp))}–${f(Math.max(...stSp))}×`, `${BK.same_code_sha_pass_count} (this code hash)`, RG ? `${RG.reps} reps per kernel/shape` : "PENDING"],
    ["cache", "fresh", WT.cached ? "cache hit" : "fresh", RG ? "none" : "PENDING"],
    ["L2", "flushed (do_bench)", "flushed (do_bench)", RG ? "warm and cold, reported apart" : "PENDING"],
    ["baseline", "eager PyTorch", "eager PyTorch", RG ? "eager, torch.compile, starter" : "PENDING"],
    ["model ÷ starter", "–", "–", HL ? { text: `${f(HL.best_vs_starter_cold_median)}× cold · ${f(HL.best_vs_starter_warm_median)}× warm`, options: { bold: true } } : "PENDING"],
  ];
  table(s, rows, M, 1.5, 8.2, [1.55, 2.2, 1.85, 2.6], { rowH: 0.48, fontSize: 12, boldFirstCol: true });
  txt(s, HL
    ? `Geomean over all ${RG.table.filter((r) => r.kernel === "best").length} shape×dtype configs, model ÷ starter: ${f(HL.best_vs_starter_cold_geomean_all_shapes)}× cold, ${f(HL.best_vs_starter_warm_geomean_all_shapes)}× warm. On H100: ${f(RH.headline.best_vs_starter_cold_median)}× cold. Cold-L2 CV at ${timed.M}²: ${pc(P.best.cold_cv)}.`
    : `The winner sits ${pp(100 * (BK.speedup / mean(stSp) - 1), 1)} above the starter's mean, inside run-to-run spread (quick-mode CV ≤ ${S1.max_cv_pct}%).`,
    M, 5.0, 8.2, 0.75, { fontSize: 13, color: TXT2 });
  txt(s, `All ${PS.per_kernel.rmsnorm.n} RMSNorm PASS turns in the project: ${f(PS.per_kernel.rmsnorm.min)}–${f(PS.per_kernel.rmsnorm.max)}×; the base model already had ${f(basePrev.speedup)}× before any training.${hB ? ` Re-running bench.py full: model PASS at ${hB.speedup_vs_pytorch}, starter FAIL but prints ${hS.speedup_vs_pytorch}.` : ""}`,
    M, 5.8, 8.2, 0.9, { fontSize: 13, color: TXT2 });
  const rx = 9.0, rw = W - M - rx;
  s.addText("What do_bench actually does", { x: rx, y: 1.5, w: rw, h: 0.38, fontFace: HEAD, fontSize: 16, bold: true, color: TXT, margin: 0, isTextBox: true });
  bullets(s, [
    `warmup=${BC.do_bench_warmup_arg}, rep=${BC.do_bench_rep_arg} are ms budgets, not counts (Triton ${VER.triton})`,
    `returns the ${LF.do_bench_default_return_mode} by default; bench.py's docstring says median`,
    `zeroes a ${LF.do_bench_l2_flush_buffer_mb} MB buffer before each call: cold L2 only`,
    HL ? `its ratio agrees with the cold median: model ${f(HL.do_bench_best_speedup)}× vs starter ${f(HL.do_bench_starter_speedup)}×` : "one shape, one dtype decide the speedup",
    "quick and full time the same do_bench call on the same input; full only adds stages 3–5",
    "the timer was fine; the attribution was not",
  ], rx, 1.9, rw, 4.9, { fontSize: 13 });
  cite(s, "bench.py _do_bench · Triton testing.do_bench (source-read) · rmsnorm_remeasure.json headline, harness_runs");
  s.addNotes(`The two logged numbers were measured differently. The starter's ${f(S1.per_gpu.L4.rmsnorm.speedup_quick)}x is quick mode over three runs, because in full mode it fails and gets no reward. The model's ${f(BK.speedup)}x is one full-mode observation of that exact code hash. ` +
    (HL ? `The remeasure times both ${RG.reps} times per shape, warm and cold, with eager and torch.compile baselines and no correctness gate. Model over starter is ${f(HL.best_vs_starter_cold_median)} cold and ${f(HL.best_vs_starter_warm_median)} warm at the timed shape, and the geomean over all shapes is ${f(HL.best_vs_starter_cold_geomean_all_shapes)}. On H100 it is ${f(RH.headline.best_vs_starter_cold_median)}. The casts bought nothing. ` : "") +
    `Every RMSNorm pass in the project sits between ${f(PS.per_kernel.rmsnorm.min)} and ${f(PS.per_kernel.rmsnorm.max)}x, and the base model had ${f(basePrev.speedup)}x before training. ` +
    `On the timer: in Triton ${VER.triton} do_bench's arguments are millisecond budgets, it returns the mean, and it flushes L2 by zeroing a ${LF.do_bench_l2_flush_buffer_mb} MB buffer each call. Its ratio agrees with the careful cold-L2 median, so the timer was fine. The problem was attribution: comparing only against PyTorch.`);
}

// 9. Harness findings: AutoKernel bench.py
{
  const s = base("Harness findings I: AutoKernel bench.py", { sub: "correctness and timing defects, each found by re-running end to end" });
  const items = [
    ["--kernel-type / results.tsv", "FIXED", "worker passed a flag bench.py lacks (it is --kernel) and parsed a file it never writes; metrics are on stdout (bench_modal.parse_stdout)"],
    ["--quick", "AVOIDED", `skips stability, determinism, edges: ${S1.starters_pass_quick.H100}/9 starters pass quick vs ${S1.starters_pass_full.H100}/9 full on H100; ${S1.quick_pass_full_fail.H100.join(" and ")} pass quick, fail full`],
    ["both NaN/Inf → PASS", "REMAINS", `stability accepts NaN/Inf output whenever the reference has any; masks never compared (bench.py:${BC.both_nan_inf_accepted_line})`],
    ["low-precision reference", "REMAINS", `reference computes in the input dtype (x ** 2 in fp16); correct fp32 kernels FAIL, the overflowing one PASSes`],
    ["narrow stability coverage", "REMAINS", `one size (${BC.stability_size_label}), one dtype (${BC.rmsnorm_test_dtypes[0]}), tolerance × ${BC.stability_tolerance_relax_factor}`],
    ["GPU spec lookup", "IDENTIFIED", `"${H1q[0].gpu}" misses "H100 SXM", falls back to ${BC.H100_fallback_spec_bw_gb_s} GB/s: rmsnorm reports ${pp(h100Pct, 0)} of peak; vs ${BC.H100_SXM_spec_bw_gb_s} GB/s it is ${pp(h100Real, 0)}`],
  ];
  const col = { FIXED: GREEN, AVOIDED: "2E5E9A", REMAINS: RED, IDENTIFIED: ACC_D };
  items.forEach(([k, tag, d], i) => {
    const y = 1.5 + i * 0.87;
    card(s, M, y, 3.5, 0.74, INKBG);
    s.addText(k, { x: M, y, w: 3.5, h: 0.74, align: "center", valign: "middle", fontFace: MONO, fontSize: 13, bold: true, color: ACC, margin: 0, isTextBox: true });
    s.addText(tag, { x: M + 3.65, y, w: 1.35, h: 0.74, valign: "middle", fontFace: BODY, fontSize: 12, bold: true, color: col[tag], margin: 0, isTextBox: true });
    s.addText(d, { x: M + 5.05, y, w: CW - 5.05, h: 0.74, valign: "middle", fontFace: BODY, fontSize: 14, color: TXT, margin: 0, isTextBox: true });
  });
  cite(s, "results/autokernel_src/bench.py · stage1_baselines.jsonl · CLAUDE.md");
  s.addNotes(`These are the AutoKernel-side findings. The first two would have made the whole RL study meaningless: with the original worker no reward parsed at all, and a quick-mode reward would have paid kernels that fail numerical stability, since ${S1.quick_pass_full_fail.H100.join(" and ")} pass quick but fail full. ` +
    `Three remain in the harness as shipped: the both-NaN rule, the low-precision reference, and stability coverage of one size and one dtype with tolerances relaxed ${BC.stability_tolerance_relax_factor} times. Together they are exactly what let the fp16 RMSNorm pass. ` +
    `The last one is a reporting bug: Modal's H100 reports as "${H1q[0].gpu}", the substring lookup misses "H100 SXM" and falls back to ${BC.H100_fallback_spec_bw_gb_s} GB/s, so rmsnorm shows ${pp(h100Pct, 0)} of peak. Against ${BC.H100_SXM_spec_bw_gb_s} GB/s it is about ${pp(h100Real, 0)}. It doesn't affect speedup, but any roofline claim from those logs is wrong.`);
}

// 10. Harness findings: KernelBench bridge
{
  const s = base("Harness findings II: the KernelBench bridge", { sub: "an identity model must PASS at 1.0×; most problems could not" });
  fig(s, "figures/03_kb_identity_noise.png", M, 1.5, 7.2, 4.2, "l");
  const rx = 8.0, rw = W - M - rx, tw = (rw - 0.2) / 2;
  stat(s, rx, 1.5, tw, 1.35, `${KB.valid}/${KB.n}`, "Level 1 problems harness-valid", { vs: 26 });
  stat(s, rx + tw + 0.2, 1.5, tw, 1.35, `${f(KB.identity_speedup_median)}×`, `identity median (p5–p95 ${f(KB.identity_speedup_p5_p95[0])}–${f(KB.identity_speedup_p5_p95[1])})`, { vs: 26 });
  bullets(s, [
    [{ text: "super(Model, self): ", options: { bold: true } }, { text: "starter renamed to ModelNew keeps it → every starter crashes; identity test uses class ModelNew(Model): pass" }],
    [{ text: "unseeded init: ", options: { bold: true } }, { text: `reference and candidate get different weights → all ${KB.conv_unseeded} conv problems fail for an identical model` }],
    [{ text: "CPU get_inputs() in the 30 s trial: ", options: { bold: true } }, { text: `${KB.flaky_timeouts} huge-input problems time out intermittently, even at cpu=8, 32 GB` }],
    [{ text: "listing N.py and N.json: ", options: { bold: true } }, { text: `read 1–50 twice; kept as a repeat test: ${KB.repeat.same_verdict}/${KB.repeat.problems} same verdict` }],
  ], rx, 3.0, rw, 3.8, { fontSize: 13 });
  txt(s, `Timing is trustworthy where the harness is valid (identity within 1%: ${KB.identity_within_1pct}/${KB.n}); coverage is not. A KB bench costs ${f(KB.wall_s_p50, 1)} s p50 vs ${f(S1.bench_wall_s.L4.p50, 1)} s for AutoKernel on L4.`,
    M, 5.85, 7.2, 0.9, { fontSize: 13, color: TXT2 });
  cite(s, "KernelBench, arXiv 2502.10517 · results/kb_valid_problems.json");
  s.addNotes(`The KernelBench bridge was my first harness, and the identity test is the cheapest verifier test there is: submit the reference model itself, which must pass at 1.0x. ` +
    `The median identity speedup is ${f(KB.identity_speedup_median)}x with p5 to p95 of ${f(KB.identity_speedup_p5_p95[0])} to ${f(KB.identity_speedup_p5_p95[1])}, so the timer is fine. Coverage is not: the bridge's starter crashes on super(Model, self), weights are never seeded so all ${KB.conv_unseeded} conv problems fail even for an identical model, and CPU input generation is timed inside the 30 second trial so ${KB.flaky_timeouts} problems time out intermittently. ` +
    `Only ${KB.valid} of ${KB.n} Level 1 problems are usable, and that list is what the held-out eval uses. This is a limitation of the local bridge, not a claim that KernelBench itself is broken.`);
}

// 11. Timing: noise vs cache
{
  const s = base("Timing: what is noise, what is cache", { sub: "a reward difference inside a group is not automatically timing noise" });
  const tw = (CW - 0.6) / 4;
  stat(s, M, 1.5, tw, 1.4, `≤ ${S1.max_cv_pct}%`, `speedup CV, quick mode, ${S1.n_runs} stage1 runs (3 repeats)`, { vs: 24 });
  stat(s, M + (tw + 0.2), 1.5, tw, 1.4, `${LV.groups}/${DR.grpo_v1.groups_total}`, "GRPO v1 groups with 0 < σ(reward) < 0.02", { vs: 24 });
  stat(s, M + 2 * (tw + 0.2), 1.5, tw, 1.4, `${LV.mixed_pass_fail}/${LV.groups}`, "of those mix PASS and FAIL (not noise)", { vs: 24, color: RED });
  stat(s, M + 3 * (tw + 0.2), 1.5, tw, 1.4, `${LV.best_pass_records_cached}/${LV.best_pass_records}`, "best-PASS timings replayed from cache", { vs: 24 });
  s.addText("What the low-variance groups really contain", { x: M, y: 3.15, w: 7.2, h: 0.4, fontFace: HEAD, fontSize: 17, bold: true, color: TXT, margin: 0, isTextBox: true });
  bullets(s, [
    `${LV.mixed_pass_fail} mix PASS and FAIL: the floor bug put slow PASSes (log2 s < 0) next to failures at 0`,
    `${LV.all_pass} are all-PASS; ${LV.multiple_best_pass_code_sha}/${LV.groups} contain several distinct best-PASS code hashes, so "same kernel, re-timed" is rarely true`,
    "a cached result replays one earlier timing: identical code gets identical reward, new code gets a fresh sample",
    `cache hit rate averaged ${pc(cacheMean(G1), 1)} (v1) and ${pc(cacheMean(G2), 1)} (v2) of benchmarks`,
  ], M, 3.6, 7.2, 3.1, { fontSize: 14 });
  const rx = 8.0, rw = W - M - rx;
  card(s, rx, 3.15, rw, 3.55, TINT);
  const cvl = log2(1 + S1.max_cv_pct / 100);
  txt(s, [{ text: "Noise floor, derived: ", options: { bold: true } },
    { text: `max CV ${S1.max_cv_pct}% ≈ log2(1 + ${S1.max_cv_pct / 100}) = ${f(cvl, 3)} reward units, so the v2 cutoff σ ≤ ${C.adv_min_std} sits just below it.${P ? ` The remeasure's cold-L2 CV at ${timed.M}² is ${pc(P.best.cold_cv)} over ${RG.reps} reps.` : " Full-mode noise was never calibrated."}`, options: { breakLine: true } },
    { text: " ", options: { breakLine: true, fontSize: 6 } },
    { text: "Consequence: ", options: { bold: true } },
    { text: "std-normalised advantages amplify whatever differs inside a group: ordering bugs and cache/fresh mixes as well as timing. Remeasure fixed code uncached before calling anything noise." }],
    rx + 0.25, 3.3, rw - 0.5, 3.3, { fontSize: 14, color: TXT2, valign: "top" });
  cite(s, "results/v3/v3_numbers.json (lowvar audit of grpo_v1 rollouts) · bench_modal.cache_key");
  s.addNotes(`Timing noise is the obvious suspect when rewards in a group differ by a hair. The stage1 repeats put quick-mode CV at ${S1.max_cv_pct}% at worst. ` +
    `In GRPO v1, ${LV.groups} of ${DR.grpo_v1.groups_total} groups had a reward standard deviation between 0 and 0.02. I audited them from the rollouts: ${LV.mixed_pass_fail} mix PASS and FAIL, which is the floor bug, not noise. Only ${LV.all_pass} are all-PASS, and ${LV.multiple_best_pass_code_sha} of ${LV.groups} contain several different best-PASS kernels. ` +
    `And ${LV.best_pass_records_cached} of ${LV.best_pass_records} of those best-PASS timings came from the cache, which replays one earlier measurement per code hash. Inside a group you are often comparing a replayed timing of one kernel with a fresh timing of another. ` +
    `The 0.02 cutoff is just under the quick-mode CV in log units${P ? `; the careful remeasure shows cold-L2 CV of ${pc(P.best.cold_cv)}` : ""}. The fix is to remeasure fixed code uncached, repeatedly, before attributing anything to noise.`);
}

// 12. The use case
{
  const s = base("The use case: an RL kernel agent", { sub: "the harness is the reward, so its defects become the gradient" });
  fig(s, "figures/01_architecture.png", M, 1.45, 7.6, 4.3, "l");
  const rx = 8.4, rw = W - M - rx;
  const rows = [
    ["policy", `${C.model.split("/")[1]} + LoRA r=${C.lora_r}`],
    ["tasks", `${nTasks} AutoKernel kernel types`],
    ["episode", `≤ ${C.max_turns} turns: write kernel, get bench feedback`],
    ["group", `${C.group} episodes/task, ${nTasks * C.group}/step`],
    ["benchmarks", `≤ ${nTasks * C.group * C.max_turns} per step, ≤ 8 L4 containers`],
    ["runs", `v1 ${G1.steps} steps, v2 ${G2.steps} steps; one each`],
  ];
  table(s, rows, rx, 1.5, rw, [1.35, rw - 1.35], { rowH: 0.46, fontSize: 13, boldFirstCol: true });
  stat(s, rx, 4.45, rw, 1.3, pp(TB.grpo_v1.share_pct.benchmarking), `of a v1 step's wall time is benchmarking (generation ${pp(TB.grpo_v1.share_pct.generation)})`, { vs: 26 });
  txt(s, "The verifier is both the bottleneck and the signal: every defect on the last slides was paid for in wall time and in reward.",
    M, 5.95, CW, 0.7, { fontSize: 15, color: TXT2, italic: true });
  cite(s, "modal_app/train_grpo.py, policy.py, bench_modal.py");
  s.addNotes(`Now the use case. A ${C.model} policy with a rank-${C.lora_r} LoRA writes a Triton kernel for one of ${nTasks} AutoKernel tasks, gets bench.py's feedback, and tries again for up to ${C.max_turns} turns. GRPO uses groups of ${C.group}, so a step is ${nTasks * C.group} episodes and up to ${nTasks * C.group * C.max_turns} benchmarks. ` +
    `Benchmarking was ${pp(TB.grpo_v1.share_pct.benchmarking)} of a step's wall time and generation ${pp(TB.grpo_v1.share_pct.generation)}. So the harness is both the throughput bottleneck and the only source of learning signal.`);
}

// 13. RL core, compressed
{
  const z1 = zeroAdvMean(G1), z2 = zeroAdvMean(G2);
  const s = base("Verification quality set training-signal quality", { sub: `${pc(z1, 0)} of GRPO groups had zero advantage under the naive reward` });
  fig(s, "figures/09_grpo_curves.png", M, 1.5, 7.3, 3.4, "l");
  const rx = 8.1, rw = W - M - rx, tw = (rw - 0.2) / 2;
  stat(s, rx, 1.5, tw, 1.6, `${pc(z1, 0)} → ${pc(z2, 0)}`, "zero-advantage groups, v1 → v2 reward", { vs: 24 });
  stat(s, rx + tw + 0.2, 1.5, tw, 1.6, `${fm(rV1(true, V1.per_kernel.reduce.best_speedup))}`, `v1 reward for a correct reduce at ${f(V1.per_kernel.reduce.best_speedup)}× (crash = 0)`, { vs: 24, color: RED });
  stat(s, rx, 3.3, tw, 1.6, pp(DR.grpo_v1.abs_adv_mass_share_from_noise_groups_pct.grpo_std_as_trained), `of v1 |A| from σ < 0.02 groups; ${LV.mixed_pass_fail}/${LV.groups} mix PASS/FAIL`, { vs: 24 });
  stat(s, rx + tw + 0.2, 3.3, tw, 1.6, `${passCount(KBB)}/${KBB.n_traj} → ${passCount(KBL)}/${KBL.n_traj}`, "KernelBench L1 transfer, base → GRPO v2", { vs: 24 });
  bullets(s, [
    `v1 (log2 speedup, 0 on fail): all-fail groups tie; slow PASS < crash (${V1.ablations.clip.slower_than_pytorch_pass}/${V1.ablations.clip.n_pass_traj} passing episodes scored < 0)`,
    `v2 (PASS floor 0.5 + partial credit from bench stages + σ ≤ ${C.adv_min_std} cutoff) adds signal, bundled: effects not isolated`,
    `outcome: both runs pass the same ${G1.kernels_ever_passed.length} kernel types; pass rate ${pc(G1.pass_first3)} → ${pc(G1.pass_last3)} (v1), ${pc(G2.pass_first3)} → ${pc(G2.pass_last3)} (v2); one run each`,
  ], M, 4.0, 7.3, 2.8, { fontSize: 14, color: TXT2 });
  card(s, rx, 5.1, rw, 1.6, INKBG);
  txt(s, "The reward could only be as good as the PASS it was gated on, and the speedup it was measured against.", rx + 0.2, 5.15, rw - 0.4, 1.5, { fontSize: 14, color: DTXT, valign: "middle" });
  cite(s, "numbers.json grpo · extra_numbers.json dr_grpo_whatif · details in appendix");
  s.addNotes(`This is the whole RL result on one slide. Under the naive reward, log2 speedup if PASS and zero otherwise, ${pc(z1, 1)} of groups had all-equal rewards and so zero advantage: eight failures tie, and a correct reduce at ${f(V1.per_kernel.reduce.best_speedup)}x scores ${fm(rV1(true, V1.per_kernel.reduce.best_speedup))}, below a crash. ` +
    `Reward v2 floors any PASS at 0.5, gives partial credit from bench.py's own stages and ignores groups with sigma at most ${C.adv_min_std}; zero-advantage groups fell to ${pc(z2, 1)}. Those changes were bundled, so I can't attribute the drop to one of them. ` +
    `${pp(DR.grpo_v1.abs_adv_mass_share_from_noise_groups_pct.grpo_std_as_trained)} of v1's absolute advantage came from low-variance groups, and ${LV.mixed_pass_fail} of ${LV.groups} of those mix PASS and FAIL, so it is not simply timing noise. ` +
    `Neither run solved a new kernel type, pass rates were flat within noise, and KernelBench transfer was ${passCount(KBB)} of ${KBB.n_traj} before and after. The verifier's quality bounded the training signal's quality, which is why my roadmap is about the verifier.`);
}

// 14. Roadmap
{
  const s = base("Roadmap: verification before scale");
  const cols = [
    ["1  Hidden-test verification", [
      "fp32 / fp64 golden reference; accept if error ≤ k × PyTorch-same-dtype error",
      "compare NaN/Inf masks exactly; drop the both-NaN pass",
      "adversarial magnitude sweep (scale × randn up to the overflow thresholds), every dtype, hidden shapes and strides",
      "mutation tests (no eps, wrong axis, tail mask, fp16 sum) to measure false-accept rate",
    ]],
    ["2  Deterministic timing", [
      "explicit L2 policy: report warm and cold",
      "fixed clocks, interleaved candidate / baseline, randomized order",
      "n repeats → median + CI, absolute µs and GB/s vs roofline",
      "per-shape table, not one ratio; correct GPU spec lookup",
    ]],
    ["3  Uncached, attributed timing", [
      [{ text: "baselines: starter / previous turn and torch.compile, ", options: { bold: true, color: TXT } }, { text: "not only eager; reward the Δ an edit made" }],
      "cache verdicts, never final timings; re-time winners fresh; calibrate noise before any σ cutoff",
      P ? "RMSNorm remeasure done (slides 6–8): the casts were worth " + sgnPct(castGain) : "RMSNorm remeasure in progress",
    ]],
  ];
  const cw = (CW - 0.5) / 3;
  cols.forEach(([h, items], i) => {
    const x = M + i * (cw + 0.25);
    card(s, x, 1.3, cw, 4.15);
    s.addText(h, { x: x + 0.25, y: 1.42, w: cw - 0.5, h: 0.5, fontFace: HEAD, fontSize: 18, bold: true, color: INDIGO, margin: 0, isTextBox: true });
    bullets(s, items, x + 0.25, 2.0, cw - 0.5, 3.35, { fontSize: 14, color: TXT2 });
  });
  card(s, M, 5.6, CW, 1.15, INKBG);
  s.addText([
    { text: "Decision rule: ", options: { bold: true, color: ACC } },
    { text: "scale training only after independent evaluation shows a reproducible gain. Then, on a validated verifier: frozen checkpoints vs a best-of-N search control, and PASS floor, partial credit and normalisation ablated one at a time.", options: { color: DTXT } },
  ], { x: M + 0.3, y: 5.65, w: CW - 0.6, h: 1.05, fontFace: BODY, fontSize: 15, margin: 0, isTextBox: true, valign: "middle" });
  cite(s, "Planned, not measured · Measuring the Checker 2609.22220 · KernelBench-Verified 2607.16241");
  s.addNotes(`Three things before any more RL. First, hidden-test verification: a truth reference in fp64 or upcast fp32, where a candidate passes if its error is within a small multiple of what PyTorch itself achieves in the same dtype; exact NaN and Inf mask comparison; every dtype, outlier-heavy inputs and hidden shapes; and mutation tests to measure how often the checker accepts a wrong kernel. ` +
    `Second, deterministic timing: an explicit L2 policy with warm and cold reported, fixed clocks, candidate and baseline interleaved in random order, repeated with confidence intervals, in absolute microseconds and GB/s against the roofline, per shape. ` +
    `Third, uncached and attributed timing: time every candidate against the starter or the previous turn and against torch.compile, not only eager PyTorch, so the reward pays for the delta an edit made; the RMSNorm casts got a ${f(BK.speedup, 1)}x reward for ${P ? sgnPct(castGain) : "no measurable"} speed. The cache may store correctness verdicts but never the timing that becomes a reward or a headline; re-time selected winners fresh, and calibrate full-mode noise before choosing any sigma cutoff. ` +
    `The decision rule: scale training only after independent evaluation shows a reproducible gain. Then the RL questions become answerable: frozen checkpoints against a best-of-N control, and the reward changes ablated one at a time.`);
}

// 15. Takeaways
{
  const z1 = zeroAdvMean(G1);
  const s = base("Takeaways", { dark: true });
  const rows = [
    ["2 casts", `FAIL → PASS and a ${f(BK.speedup, 1)}× reward${P ? ` for ${sgnPct(castGain)} speed` : ""}: the reference shared the kernel's overflow, and speed was measured against PyTorch, not the starter.`],
    [`${KB.valid}/${KB.n}`, "An identity model is the cheapest harness test; about half of KernelBench L1 failed it locally. Test the verifier before trusting it."],
    [`${LV.best_pass_records_cached}/${LV.best_pass_records}`, "best-PASS timings were cache replays. Timing claims need uncached, repeated, per-shape measurement with an explicit L2 policy."],
    [pc(z1, 0), "zero-advantage groups under the naive reward: verification quality bounded training-signal quality."],
  ];
  rows.forEach(([v, d], i) => {
    const y = 1.2 + i * 1.3;
    card(s, M, y, CW, 1.1, "2A2858");
    s.addText(v, { x: M + 0.3, y, w: 2.9, h: 1.1, fontFace: HEAD, fontSize: 30, bold: true, color: ACC, margin: 0, isTextBox: true, valign: "middle" });
    s.addText(d, { x: M + 3.3, y, w: CW - 3.6, h: 1.1, fontFace: BODY, fontSize: 17, color: WHITE, margin: 0, isTextBox: true, valign: "middle" });
  });
  s.addText("All artifacts open: results/ (rollouts, numbers.json, v3/), modal_app/, presentation/v3/.",
    { x: M, y: 6.45, w: CW, h: 0.45, fontFace: BODY, fontSize: 14, color: DMUTED, margin: 0, isTextBox: true, valign: "middle" });
  s.addNotes(`Four takeaways. Two casts flipped a kernel from FAIL to PASS and earned a ${f(BK.speedup)}x reward${P ? `, while contributing ${sgnPct(castGain)} of speed` : ""}: the reference shared the kernel's overflow, and speed was measured against PyTorch rather than the starter. The verifier needs an independent truth and an attributed baseline. ` +
    `An identity model is the cheapest verifier test, and only ${KB.valid} of ${KB.n} KernelBench problems passed it in the local bridge. ` +
    `${LV.best_pass_records_cached} of ${LV.best_pass_records} best-PASS timings in the suspicious groups were cache replays; timing claims need fresh, repeated, per-shape measurement with a stated L2 policy. ` +
    `And under the naive reward ${pc(z1, 0)} of GRPO groups carried no gradient: in RL for kernels, the verifier sets the ceiling on the signal.`);
}

// ============================== Appendix ==============================
// A1. Reward v1 vs v2
{
  const s = base("Appendix: reward v1 vs v2", { appendix: true });
  const red = V1.per_kernel.reduce.best_speedup, ce = V1.per_kernel.cross_entropy.best_speedup;
  const lr = GE.rollouts[0];
  const [swOk, swTot] = lr.best_stage.match(/sweep (\d+)\/(\d+)/).slice(1).map(Number);
  bullets(s, [
    [{ text: "r_v1: ", options: { bold: true } }, { text: "clip(log2 s, −1, 3) if any turn PASSes, else 0" }],
    [{ text: "floor bug: ", options: { bold: true } }, { text: `reduce PASS at ${f(red)}× → log2 = ${fm(log2(red))}, below a failure at 0; ${V1.ablations.clip.slower_than_pytorch_pass}/${V1.ablations.clip.n_pass_traj} passing episodes negative` }],
    [{ text: "r_v2: ", options: { bold: true } }, { text: "PASS: 0.5 + clip(log2 s, 0, 3); else 0.3 · max-turn progress" }],
    [{ text: "progress: ", options: { bold: true } }, { text: "0 if smoke fails, else 0.25 + 0.5 · sweep fraction + 0.25 · [stability]" }],
    [{ text: "limits: ", options: { bold: true } }, { text: "all PASSes ≤ 1× tie at 0.5; weights heuristic; determinism and edges get no credit; cutoff σ ≤ 0.02 added at the same time" }],
  ], M, 1.35, 7.0, 4.5, { fontSize: 15 });
  const rx = 7.9, rw = W - M - rx;
  table(s, [
    ["episode", "r_v1", "r_v2"],
    ["crash / no code block", f(0, 2), f(0, 2)],
    [`layernorm: smoke ok, sweep ${swOk}/${swTot}, stab ok`, f(0, 2), f(lr.reward_v2, 3)],
    [`reduce PASS at ${f(red)}×`, { text: fm(rV1(true, red), 3), options: { color: RED, bold: true } }, f(rV2(true, red), 2)],
    [`cross_entropy PASS at ${f(ce)}×`, f(rV1(true, ce), 3), f(rV2(true, ce), 3)],
  ], rx, 1.35, rw, [2.9, 0.95, rw - 3.85], { rowH: 0.6, fontSize: 13, centerCols: true });
  cite(s, "src/autokernel-rlvr/agent_loop/reward.py · modal_app/reward_v2.py");
  s.addNotes(`Reward details for questions. v1 is clipped log2 speedup if any turn passes, else 0, which puts a correct reduce at ${f(red)}x below a crash. v2 floors a pass at 0.5 and gives up to 0.3 partial credit from smoke, sweep and stability. Its limits: every pass at or below 1x ties at 0.5, the weights are heuristic, and the 0.02 cutoff was introduced at the same time.`);
}

// A2. Advantage detail
{
  const s = base("Appendix: ties and advantage mass", { appendix: true });
  fig(s, "v2/figures/x2_dr_grpo_noise_mass.png", M, 1.35, 7.0, 3.6, "l");
  const z2 = V2.ablations.zero_adv_group_frac_by_group_size, z1 = V1.ablations.zero_adv_group_frac_by_group_size;
  const rx = 7.8, rw = W - M - rx;
  table(s, [["tied groups (offline)", "G=2", "G=4", "G=8"],
    ["feedback v2", pc(z2["2"]), pc(z2["4"]), pc(z2["8"])],
    ["feedback v1", pc(z1["2"]), pc(z1["4"]), pc(z1["8"])]], rx, 1.35, rw, [2.1, (rw - 2.1) / 3, (rw - 2.1) / 3, (rw - 2.1) / 3], { rowH: 0.45, fontSize: 13, boldFirstCol: true, centerCols: true });
  const D1 = DR.grpo_v1, sh = D1.abs_adv_mass_share_from_noise_groups_pct;
  bullets(s, [
    `v1 |A| from σ < 0.02 groups: ${pp(sh.grpo_std_as_trained)} std-normalised vs ${pp(sh.dr_grpo_no_std)} no-std replay (offline; |A| mass ≠ gradient mass)`,
    `DAPO-style filter drops ${pp(D1.dapo_filter_removed_groups_pct)} of v1 groups: ${f(D1.dapo_eff_batch_trajs.mean, 1)}/${D1.dapo_eff_batch_trajs.nominal} episodes left per step`,
    `group explorer (${GE.task_id}, v1 step ${GE.step}): all 8 v1 rewards 0; v2 replay separates ${f(GE.tied_under_v1_separable_under_v2.grpo_v1.mean, 1)}/9 groups per step`,
    "P(tie) ≥ (1 − p)^G for v1 all-fail groups; bigger G helps slowly",
  ], rx, 2.95, rw, 3.8, { fontSize: 13, color: TXT2 });
  cite(s, "Dr. GRPO 2503.20783 · DAPO 2503.14476 · extra_numbers.json");
  s.addNotes("Detail behind the RL slide: offline regrouping of base-model samples shows ties fall slowly with group size, and the no-std replay shows how much of the absolute advantage mass sat in low-variance groups. All offline replays, no training runs.");
}

// A3. Transfer + time
{
  const s = base("Appendix: transfer and where time goes", { appendix: true });
  fig(s, "figures/10_kernelbench_transfer.png", M, 1.35, 6.1, 3.6, "l");
  fig(s, "v2/figures/x1_time_breakdown.png", 6.9, 1.35, W - M - 6.9, 3.6, "r");
  const api = "Triton API / compile error";
  bullets(s, [
    `KernelBench: ${KBB.meta.tasks} valid L1 tasks (every 2nd), 1 sample × ${KBB.meta.max_turns} turns, ${KBB.meta.bench_gpu}: PASS turns ${KBB.turn_outcomes.PASS || 0}/${turnsTotal(KBB)} → ${KBL.turn_outcomes.PASS || 0}/${turnsTotal(KBL)}; API/compile errors ${KBB.failure_taxonomy[api]} → ${KBL.failure_taxonomy[api]}`,
    `time (v1): benchmarking ${pp(TB.grpo_v1.share_pct.benchmarking)}, generation ${pp(TB.grpo_v1.share_pct.generation)}, update ${pp(TB.grpo_v1.share_pct.policy_update)}; free generation bounds a step speedup at ${f(1 / (1 - TB.grpo_v1.share_pct.generation / 100), 2)}× (Amdahl, derived)`,
    `cost estimates: v1 $${f(G1.est_cost_usd, 2)} (${G1.wall_h} h), v2 $${f(G2.est_cost_usd, 2)} (${G2.wall_h} h), H100 only`,
  ], M, 5.1, CW, 1.8, { fontSize: 13, color: TXT2 });
  cite(s, "KernelBench 2502.10517 · extra_numbers.json time_breakdown");
  s.addNotes("Transfer was zero before and after, dominated by Triton API and compile errors; the test changes task set, interface and GPU at once, so it is scoped to this protocol. Benchmarking dominated wall time.");
}

// A4. Remeasure: all shapes, model / starter
{
  const s = base("Appendix: RMSNorm remeasure, every shape", { appendix: true });
  if (RG) {
    const shapes = [...new Set(RG.table.filter((r) => r.kernel === "best").map((r) => `${r.M}x${r.N}`))]
      .map((k) => k.split("x").map(Number)).sort((a, b) => a[0] * a[1] - b[0] * b[1]);
    const half = Math.ceil(shapes.length / 2);
    const mk = (list) => {
      const rows = [["M×N", "fp16 model/starter", "bf16 model/starter", "fp16 model µs", "GB/s"]];
      list.forEach(([m, n]) => {
        const r = (k, dt) => rtab(k, m, n, dt);
        const rat = (dt) => (r("best", dt) && r("starter", dt) ? f(r("starter", dt).cold_median_us / r("best", dt).cold_median_us, 3) : "–");
        const b = r("best", "float16");
        rows.push([`${m}×${n}`, rat("float16"), rat("bfloat16"), b ? f(b.cold_median_us, 1) : "–", b ? f(gbs(bytesRms(m, n), b.cold_median_us), 0) : "–"]);
      });
      return rows;
    };
    const tw = (CW - 0.3) / 2;
    table(s, mk(shapes.slice(0, half)), M, 1.35, tw, [1.3, 1.45, 1.45, 1.1, tw - 5.3], { rowH: 0.36, fontSize: 12, centerCols: true });
    table(s, mk(shapes.slice(half)), M + tw + 0.3, 1.35, tw, [1.3, 1.45, 1.45, 1.1, tw - 5.3], { rowH: 0.36, fontSize: 12, centerCols: true });
    txt(s, `L4, cold L2, median of ${RG.reps} reps. Speed ratio = starter µs ÷ model µs (> 1 means the casts helped). GB/s from analytic bytes computed here. Geomean ratio ${f(1 / HL.best_vs_starter_cold_geomean_all_shapes, 3)} cold.`,
      M, 1.35 + 0.36 * (half + 1) + 0.15, CW, 0.6, { fontSize: 12, color: MUTED, italic: true });
  } else pending(s, M, 1.6, CW, 2.2, "Per-shape starter/model ratios for fp16 and bf16, warm/cold, from the remeasure job.");
  cite(s, "results/v3/rmsnorm_remeasure.json gpus.L4.table (+ .md)");
  s.addNotes(RG ? "Every shape the remeasure covered, fp16 and bf16. The ratio stays at 1 within noise everywhere: the casts are not a speed optimisation at any shape." : "Placeholder until the remeasure JSON is present.");
}

// A5. Related work
{
  const s = base("Appendix: related work", { appendix: true });
  table(s, [
    ["Area", "Work", "Relevance"],
    ["Verifier", "KernelBench-Verified (2607.16241)", "hidden inputs; realistic baselines and memory"],
    ["", "Measuring the Checker (2609.22220)", "mutation tests measure oracle coverage"],
    ["Kernels", "KernelBench (2502.10517); Kevin-32B", "correctness-aware speed; multi-turn feedback"],
    ["", "AutoTriton (2507.05687); Dr. Kernel (2602.05885)", "SFT cold start; turn-level credit and profiling"],
    ["Methods", "Dr. GRPO (2503.20783); DAPO; GRESO", "std/length normalisation; zero-variance groups"],
    ["Scope", "different models, tasks, budgets", "no cross-paper performance claim"],
  ], M, 1.35, CW, [1.7, 5.0, 5.63], { rowH: 0.55, fontSize: 14, boldFirstCol: true });
  cite(s, "arXiv IDs as listed");
  s.addNotes("These works motivate the tests; their outcomes are not directly comparable to this nine-task experiment.");
}

const out = path.join(__dirname, "deck_v3.pptx");
pres.writeFile({ fileName: out }).then(() => console.log("wrote", out, `(${slideNo} slides)`));
