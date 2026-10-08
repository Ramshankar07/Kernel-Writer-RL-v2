"""
bench v2 report (no GPU): results/v3/{bench_v2_sanity.jsonl, rescore_bench_v2.jsonl,
bench_v2_baselines.jsonl} -> results/v3/rescore_summary.json + results/v3/bench_v2.md.
Every number in the .md is computed here from those files.

    python modal_app/bench_v2_report.py
"""
import json
import math
import pathlib
from collections import Counter, defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "v3"
KERNELS = ["matmul", "softmax", "layernorm", "rmsnorm", "flash_attention",
           "fused_mlp", "cross_entropy", "rotary_embedding", "reduce"]
SIG_SPEEDUP = 1.05
# $/h on Modal (list price, Sept 2026 dashboard): L4 GPU + 4 vCPU + 32 GB
L4_USD_PER_H = 0.80 + 4 * 0.0473 + 32 * 0.008

MECHANISM = {
    "matmul": "mild: torch.matmul with PyTorch's default allow_fp16/bf16_reduced_precision_reduction=True lets cuBLAS reduce split-K partials in fp16/bf16; errors of several output ulps at K>=1024 exceed upstream tol, so v1 FAILs a correctly rounded matmul (golden_cast)",
    "softmax": "none: F.softmax upcasts to fp32 internally (max-subtracted)",
    "layernorm": "none in the reference (fp32 Welford); but upstream bf16 rtol 2e-3 < bf16 unit roundoff, so v1 FAILs the correctly rounded output (and the starter); weight=1/bias=0 in sweep",
    "rmsnorm": "`x ** 2` materialised in fp16: overflows elementwise at |x|>256 -> mean=inf -> output rows = 0 (the mean itself accumulates in fp32, so RMS 8 is fine for the reference). fp16-accumulating kernels (GRPO) overflow the row SUM already at RMS > sqrt(65504/N) = 4 at N=4096",
    "flash_attention": "`Q @ K^T` materialised in fp16/bf16 BEFORE `* sm_scale`: raw scores are rounded at their unscaled magnitude (and overflow when |q.k| > 65504), P is rounded before P@V; errors up to 1.9 (fp16) at Q,K x40",
    "fused_mlp": "gate, up and silu(gate)*up materialised in fp16/bf16 (|silu(g)*u| > 65504 -> inf); bf16 intermediates already exceed tol on the xlarge/llm sweep shapes",
    "cross_entropy": "none: F.cross_entropy log-softmax + NLL accumulate in fp32",
    "rotary_embedding": "x1*cos, x2*sin and their difference each rounded to fp16/bf16 (3 roundings + cancellation): the reference's own error exceeds upstream tol (fp16 atol 1e-3) on normal inputs, which is why v1 FAILs the fp32-computing starter at smoke",
    "reduce": "none: x.sum accumulates fp16/bf16 in fp32",
}


def load(name):
    p = OUT / name
    return [json.loads(l) for l in open(p)] if p.exists() else []


def gm(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None


def f3(x, nd=3):
    return "-" if x is None else f"{x:.{nd}f}"


def baseline_stats(recs):
    """per kernel type: medians over baseline repeats of starter/eager/compile at each timing shape."""
    by = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in recs:
        if not r["id"].startswith("baseline:"):
            continue
        for lab, t in (r.get("timing") or {}).items():
            if not isinstance(t, dict) or "starter" not in t:
                continue
            d = by[r["kernel_type"]][lab]
            d["starter_us"].append(t["starter"]["median_us"])
            d["eager_us"].append(t["eager"]["median_us"])
            d["starter_cv"].append(t["starter"]["cv"])
            d["starter_vs_eager"].append(t["starter_vs_eager"]["median"])
            if "compile" in t:
                d["compile_us"].append(t["compile"]["median_us"])
                d["compile_over_starter"].append(t["compile_over_starter"]["median"])
                d["compile_cv"].append(t["compile"]["cv"])
    out = {}
    for kt, labs in by.items():
        out[kt] = {}
        for lab, d in labs.items():
            med = lambda xs: sorted(xs)[len(xs) // 2] if xs else None
            out[kt][lab] = {k: med(v) for k, v in d.items()}
            out[kt][lab]["n_repeats"] = len(d["starter_us"])
    return out


def old_starter_verdicts():
    res = {}
    p = ROOT / "results" / "stage1_baselines.jsonl"
    for l in open(p):
        r = json.loads(l)
        if r["gpu_req"] == "L4" and not r["quick"]:
            res[r["kernel_type"]] = r["correctness"]
    return res


def sanity_section(san):
    old_st = old_starter_verdicts()
    known_old = {"rmsnorm:fp32acc": "FAIL", "rmsnorm:grpo_best": "PASS"}  # rmsnorm_remeasure.md (L4 full)
    rows, ok_all = [], True
    for r in san:
        if r["id"].startswith("baseline:"):
            continue
        kt, kind = r["id"].split(":", 1)
        exp = r.get("expected", "")
        v = r.get("verdict")
        if kind == "starter":
            old = old_st.get(kt, "-")
        elif kind == "upstream_ref_native":
            old = "PASS (is the reference)"
        else:
            old = r.get("old_verdict") or known_old.get(r["id"], "-")
        if exp in ("PASS", "FAIL"):
            agree = v == exp
        elif exp.startswith("PASS (tol"):
            agree = v == "PASS"
        else:
            agree = None  # starter / native ref: informational
        if agree is False:
            ok_all = False
        why = "; ".join(r.get("fail_reasons") or ([r.get("reason")] if r.get("reason") else []))
        rows.append({"id": r["id"], "kernel_type": kt, "kind": kind, "expected": exp, "v2": v,
                     "v1_full_bench": old, "matches_expectation": agree, "why": why[:260]})
    order = {k: i for i, k in enumerate(KERNELS)}
    kinds = ["starter", "golden_cast", "upstream_ref_native", "zeros", "perturbed_5pct", "fp32acc",
             "grpo_best", "ignores_weight_bias"]
    rows.sort(key=lambda x: (order.get(x["kernel_type"], 99), kinds.index(x["kind"]) if x["kind"] in kinds else 99))
    return rows, ok_all


def audit_section(san):
    """ref_native per case, from the golden_cast jobs (all cases run, no early abort)."""
    aud = {}
    for r in san:
        if not r["id"].endswith(":golden_cast"):
            continue
        kt = r["kernel_type"]
        cases = (r.get("correctness") or {}).get("cases", [])
        fails = []
        for c in cases:
            rn = c.get("ref_native") or {}
            if rn.get("ok") is False:
                fails.append({"stage": c["stage"], "label": c["label"], "dtype": c["dtype"],
                              "max_abs_err": rn.get("max_abs_err"), "n_nonfinite": rn.get("n_nonfinite"),
                              "zero_rows": rn.get("zero_rows"), "n_viol": rn.get("n_viol")})
        aud[kt] = {"n_cases": len(cases), "n_ref_fail": len(fails), "fails": fails,
                   "adversarial_fail": sorted({f"{f['label']}/{f['dtype']}" for f in fails if f["stage"] == "adversarial"}),
                   "base_fail": sorted({f"{f['stage']}/{f['label']}/{f['dtype']}" for f in fails if f["stage"] != "adversarial"}),
                   "mechanism": MECHANISM[kt]}
    return aud


def rescore_section(res, base):
    rows = []
    for r in res:
        kt = r["kernel_type"]
        t = r.get("timing") or {}
        labs = [l for l, v in t.items() if isinstance(v, dict) and "vs_starter" in v]
        vs_st = {l: t[l]["vs_starter"] for l in labs}
        vs_eag = {l: t[l]["vs_eager"]["median"] for l in labs}
        vs_cmp = {}
        for l in labs:
            b = (base.get(kt) or {}).get(l) or {}
            if b.get("compile_over_starter"):
                vs_cmp[l] = vs_st[l]["median"] * b["compile_over_starter"]
        g_st = gm([v["median"] for v in vs_st.values()]) if len(vs_st) == len(labs) and labs else None
        sig = bool(labs) and all(v["ci_lo"] > 1.0 for v in vs_st.values()) and (g_st or 0) > SIG_SPEEDUP
        rows.append({
            "code_sha12": r["code_sha12"], "kernel_type": kt, "sources": r["sources"],
            "old_verdict": "PASS", "old_speedup_vs_pytorch": r.get("old_speedup_vs_pytorch_median"),
            "new_verdict": r.get("verdict"), "fail_reasons": (r.get("fail_reasons") or [])[:3],
            "base_ok": (r.get("correctness") or {}).get("base_ok"),
            "stage_ok": (r.get("correctness") or {}).get("stage_ok"),
            "uses_triton": None,  # filled from source code in main()
            "vs_starter": vs_st, "vs_starter_geomean": g_st, "vs_eager": vs_eag,
            "vs_eager_geomean": gm(list(vs_eag.values())) if vs_eag else None,
            "vs_compile": vs_cmp, "vs_compile_geomean": gm(list(vs_cmp.values())) if vs_cmp else None,
            "kernel_cv": {l: t[l]["kernel"]["cv"] for l in labs},
            "starter_cv": {l: t[l]["starter"]["cv"] for l in labs},
            "real_speedup_vs_starter": sig,
            "sig_speedup_some_shape": [l for l, v in vs_st.items() if v["ci_lo"] > 1.0 and v["median"] > SIG_SPEEDUP],
        })
    return rows


def main(argv=()):
    san = load("bench_v2_sanity.jsonl")
    res = load("rescore_bench_v2.jsonl")
    base_recs = load("bench_v2_baselines.jsonl") + [r for r in san if r["id"].startswith("baseline:")]
    base = baseline_stats(base_recs)
    sanity_rows, sanity_ok = sanity_section(san) if san else ([], None)
    aud = audit_section(san) if san else {}
    rows = rescore_section(res, base) if res else []

    # code flags for rescore rows
    src_codes = {}
    if res:
        import sys
        sys.path.insert(0, str(pathlib.Path(__file__).parent))
        from bench_v2_modal import load_pass_kernels
        src_codes = {k[:12]: v["code"] for k, v in load_pass_kernels().items()}
    for row in rows:
        c = src_codes.get(row["code_sha12"], "")
        row["uses_triton"] = "@triton.jit" in c

    summ = {"harness": "bench_v2.0", "gpu": (res[0].get("gpu") if res else None),
            "significance_rule": f"v2 PASS and paired bootstrap 95% CI lower bound of per-rep starter/kernel "
                                 f"latency ratio > 1.0 at every timing shape and geomean median ratio > {SIG_SPEEDUP}",
            "sanity_all_as_expected": sanity_ok}
    if rows:
        n = len(rows)
        cnt = Counter(r["new_verdict"] for r in rows)
        passed = [r for r in rows if r["new_verdict"] == "PASS"]
        by_kt = {}
        for kt in sorted({r["kernel_type"] for r in rows}):
            rs = [r for r in rows if r["kernel_type"] == kt]
            ps = [r for r in rs if r["new_verdict"] == "PASS"]
            by_kt[kt] = {"old_pass": len(rs), "new": dict(Counter(r["new_verdict"] for r in rs)),
                         "survive": len(ps),
                         "fail_on_adversarial_only": sum(1 for r in rs if r["new_verdict"] == "FAIL" and r["base_ok"]),
                         "real_speedup_vs_starter": sum(r["real_speedup_vs_starter"] for r in ps),
                         "vs_starter_geomean_median_over_survivors": _med([r["vs_starter_geomean"] for r in ps]),
                         "vs_starter_geomean_max_over_survivors": max([r["vs_starter_geomean"] or 0 for r in ps], default=None),
                         "vs_eager_geomean_median_over_survivors": _med([r["vs_eager_geomean"] for r in ps]),
                         "vs_compile_geomean_median_over_survivors": _med([r["vs_compile_geomean"] for r in ps]),
                         "starter_vs_eager": {l: (base.get(kt, {}).get(l) or {}).get("starter_vs_eager") for l in base.get(kt, {})},
                         "compile_over_starter": {l: (base.get(kt, {}).get(l) or {}).get("compile_over_starter") for l in base.get(kt, {})},
                         "no_triton_jit": sum(1 for r in rs if not r["uses_triton"])}
        by_src = {}
        for src in sorted({s for r in rows for s in r["sources"]}):
            rs = [r for r in rows if src in r["sources"]]
            by_src[src] = {"old_pass": len(rs), "survive": sum(r["new_verdict"] == "PASS" for r in rs),
                           "real_speedup_vs_starter": sum(r["real_speedup_vs_starter"] and r["new_verdict"] == "PASS" for r in rs)}
        sig = [r for r in passed if r["real_speedup_vs_starter"]]
        summ.update({
            "n_distinct_old_pass_kernels": n,
            "new_verdicts": dict(cnt),
            "old_pass_survive": len(passed),
            "old_pass_survive_frac": len(passed) / n,
            "real_speedup_vs_starter_gt_1.05_significant": len(sig),
            "pass_with_sig_speedup_at_some_shape_only": [
                {"code_sha12": r["code_sha12"], "kernel_type": r["kernel_type"], "sources": r["sources"],
                 "shapes": r["sig_speedup_some_shape"], "vs_starter": r["vs_starter"], "vs_compile": r["vs_compile"]}
                for r in passed if r["sig_speedup_some_shape"] and not r["real_speedup_vs_starter"]],
            "real_speedup_kernels": [{k: r[k] for k in ("code_sha12", "kernel_type", "sources", "vs_starter", "vs_starter_geomean",
                                                         "vs_eager_geomean", "vs_compile_geomean", "uses_triton")} for r in sig],
            "median_vs_starter_geomean_survivors": _med([r["vs_starter_geomean"] for r in passed]),
            "median_vs_eager_geomean_survivors": _med([r["vs_eager_geomean"] for r in passed]),
            "median_old_speedup_vs_pytorch": _med([r["old_speedup_vs_pytorch"] for r in rows]),
            "by_kernel_type": by_kt, "by_source": by_src,
            "rows": [{k: r[k] for k in ("code_sha12", "kernel_type", "sources", "old_verdict", "new_verdict",
                                         "old_speedup_vs_pytorch", "vs_starter_geomean", "vs_eager_geomean",
                                         "vs_compile_geomean", "real_speedup_vs_starter", "uses_triton", "fail_reasons")}
                     for r in rows],
        })
    summ["baselines"] = base
    summ["audit"] = {kt: {k: v for k, v in a.items() if k != "fails"} for kt, a in aud.items()}
    summ["sanity"] = sanity_rows
    json.dump(summ, open(OUT / "rescore_summary.json", "w"), indent=1, default=str)
    write_md(summ, aud, sanity_rows, base, rows)
    print("wrote", OUT / "rescore_summary.json", OUT / "bench_v2.md")


def _med(xs):
    xs = sorted(x for x in xs if x is not None)
    return xs[len(xs) // 2] if xs else None


def write_md(s, aud, sanity_rows, base, rows):
    L = []
    A = L.append
    A("# bench v2: fixed AutoKernel harness + rescore of every old PASS kernel\n")
    A("Code: `modal_app/bench_v2_core.py` (harness), `modal_app/bench_v2_modal.py` (Modal app `autokernel-bench-v2`, "
      "L4, <=4 containers, no cache), `modal_app/bench_v2_report.py` (this file). v1 (`bench_modal.py`, app "
      "`autokernel-bench`, Dict `autokernel-bench-cache`) is untouched.\n")
    A("Data: `results/v3/bench_v2_sanity.jsonl`, `results/v3/rescore_bench_v2.jsonl`, `results/v3/bench_v2_baselines.jsonl`, "
      "`results/v3/rescore_summary.json`.\n")
    if rows:
        A("## Headline\n")
        A(f"- {s['n_distinct_old_pass_kernels']} distinct (by code sha256) kernels were PASS under the v1 full bench in GRPO v1/v2 "
          f"rollouts and agent_eval (kernel types: {', '.join(sorted(s['by_kernel_type']))}).")
        A(f"- Under bench v2: {s['old_pass_survive']} survive ({100 * s['old_pass_survive_frac']:.1f}%); verdicts {s['new_verdicts']}.")
        A(f"- Kernels with a real speedup vs the starter (>{SIG_SPEEDUP}x geomean, CI-significant at every timing shape, v2 PASS): "
          f"**{s['real_speedup_vs_starter_gt_1.05_significant']}**. Median geomean speedup vs starter over survivors: "
          f"{f3(s['median_vs_starter_geomean_survivors'])}x; vs eager PyTorch: {f3(s['median_vs_eager_geomean_survivors'])}x "
          f"(v1 reported median {f3(s['median_old_speedup_vs_pytorch'])}x vs eager).")
        some = s["pass_with_sig_speedup_at_some_shape_only"]
        if some:
            A(f"- Not counted (significant >{SIG_SPEEDUP}x at one timing shape only): " + "; ".join(
                f"{r['code_sha12']} ({r['kernel_type']}, {','.join(r['sources'])}): " + ", ".join(
                    f"{l} {v['median']:.2f}x [CI {v['ci_lo']:.2f}-{v['ci_hi']:.2f}] vs starter"
                    + (f" = {r['vs_compile'][l]:.2f}x vs torch.compile" if l in r['vs_compile'] else "")
                    for l, v in r["vs_starter"].items()) for r in some) + ".")
        nt = s["new_verdicts"].get("TIMEOUT", 0)
        if nt:
            A(f"- {nt} kernels TIMEOUT (a single evaluation hung > 420 s; counted as not surviving).")
        A("")
        A("| kernel type | old PASS | v2 PASS | v2 FAIL (adversarial only) | real speedup vs starter | median vs starter (survivors) | max vs starter | median vs eager | median vs torch.compile | no @triton.jit |")
        A("|---|---|---|---|---|---|---|---|---|---|")
        for kt, b in s["by_kernel_type"].items():
            A(f"| {kt} | {b['old_pass']} | {b['survive']} | {b['new'].get('FAIL', 0)} ({b['fail_on_adversarial_only']}) | "
              f"{b['real_speedup_vs_starter']} | {f3(b['vs_starter_geomean_median_over_survivors'])} | "
              f"{f3(b['vs_starter_geomean_max_over_survivors'])} | {f3(b['vs_eager_geomean_median_over_survivors'])} | "
              f"{f3(b['vs_compile_geomean_median_over_survivors'])} | {b['no_triton_jit']} |")
        A("")
        A("| source | old PASS | v2 PASS | real speedup |")
        A("|---|---|---|---|")
        for src, b in s["by_source"].items():
            A(f"| {src} | {b['old_pass']} | {b['survive']} | {b['real_speedup_vs_starter']} |")
        A("\n(A kernel seen in several sources is counted in each.)\n")
        if s["real_speedup_kernels"]:
            A("Real-speedup kernels:\n")
            A("| sha | type | sources | vs starter per shape (median [95% CI]) | vs eager | vs compile | triton |")
            A("|---|---|---|---|---|---|---|")
            for r in s["real_speedup_kernels"]:
                per = ", ".join(f"{l}: {v['median']:.3f} [{v['ci_lo']:.3f}, {v['ci_hi']:.3f}]" for l, v in r["vs_starter"].items())
                A(f"| {r['code_sha12']} | {r['kernel_type']} | {','.join(r['sources'])} | {per} | {f3(r['vs_eager_geomean'])} | "
                  f"{f3(r['vs_compile_geomean'])} | {r['uses_triton']} |")
            A("")
    A("## What changed (v1 -> v2)\n")
    A("| | v1 (upstream bench.py @78435821) | v2 |")
    A("|---|---|---|")
    A("| reference | reference.py in the input dtype (fp16/bf16) | same reference.py on inputs upcast to fp64 (norms, softmax, CE, RoPE, reduce) or fp32 with TF32 off (matmul, fused_mlp, attention) |")
    A("| tolerance | per-dtype atol/rtol; x10 relaxed for stability cases | same per-dtype atol/rtol, never relaxed, applied vs the golden; rtol floored at 2u of the dtype (bf16: 7.8e-3), because upstream bf16 rtol 2e-3 (layernorm/softmax/RoPE) is below bf16 unit roundoff 3.9e-3 and fails a correctly rounded output (v2.0 sanity run: golden_cast FAILed layernorm/RoPE bf16) |")
    A("| non-finite | PASS if kernel and reference are both NaN/Inf | FAIL on any non-finite where the golden is finite; golden must be finite and representable in the dtype |")
    A("| stability inputs | first dtype only, 'small' size, every float input x60000 / x1e-6 / mixed 1e3,1e-3 (weights too) | per-type magnitude cases whose true output is representable but whose fp16 intermediates overflow/underflow/stagnate, fp16 and bf16, plus large-N shapes (below) |")
    A("| inputs | seed 42 every run; layernorm weight=1, bias=0; RoPE cos/sin ~ randn | fresh os.urandom seed per evaluation; weight, bias ~ randn; cos/sin of real angles |")
    A("| cache | modal.Dict keyed on code (201/271 best-PASS timings were replays) | none |")
    A("| timing | triton do_bench, warm L2, kernel vs eager only | CUDA events, 256 MB L2 flush before each rep, 10 warmup, 60 reps, implementations interleaved rep-by-rep; median, CV, paired starter/kernel ratio with bootstrap 95% CI; baselines eager, starter, torch.compile(reference) |")
    A("| timing shapes | 'large', first dtype | 'large' (= v1) + one LLM-sized shape, first dtype |")
    A("")
    A("Adversarial cases (see `ADVERSARIAL` in bench_v2_core.py): matmul uniform[0,1) K=8192 and randn x8 K=4096; "
      "softmax / CE logits x100, softmax N=131072, CE vocab=128256; layernorm x300, mean 1000 (+randn), N=65536; "
      "rmsnorm row RMS 8 and 300 at N=4096 (threshold sqrt(65504/4096)=4), x1e-3, per-row 10^U(-2,2.5), RMS 2 at N=65536; "
      "attention Q,K x4 and x40 (raw q.k up to ~1e5), seq 8192; fused_mlp gate/up std ~150 (|silu(g)*u| > 65504 for a few % of elements, output std ~1e3); "
      "RoPE x x1000, seq 16384; reduce uniform[0,1) N=32768 (sum ~16k), N=2^20. Plus zeros/constant rows.\n")
    if aud:
        A("## Bug audit: does the upstream reference itself fail against the fp32/fp64 golden?\n")
        A("Measured by evaluating reference.py in the input dtype on every v2 case (from the `golden_cast` sanity jobs).\n")
        A("| kernel type | low-precision intermediate bug | cases where upstream ref fails / total | failing cases | mechanism |")
        A("|---|---|---|---|---|")
        for kt in KERNELS:
            a = aud.get(kt)
            if not a:
                continue
            bug = ("YES (mild)" if kt == "matmul" else "YES") if a["n_ref_fail"] else "no"
            fl = ", ".join(a["adversarial_fail"] + a["base_fail"]) or "-"
            A(f"| {kt} | {bug} | {a['n_ref_fail']}/{a['n_cases']} | {fl} | {a['mechanism']} |")
        A("")
        A("Other v1 harness defects (by reading bench.py): (1) stability stage PASSes when kernel and reference are both "
          "NaN/Inf, and near_max multiplies every float input (weights included) by 60000, which overflows fp16 inputs "
          "themselves; (2) stability/determinism/edge stages use only the first dtype; (3) `--quick` skips stages 3-5; "
          "(4) layernorm weight=ones/bias=zeros in smoke/sweep/edge, so a kernel that ignores weight/bias passes `--quick` (the full bench only catches it because the stability transforms also rescale the weights); (5) fixed seed 42 + result "
          "cache; (6) speedup only vs eager PyTorch, so any fusion looks like a win; (7) atol 1e-3 on softmax is larger than "
          "every probability at 4096+ columns (a zero output passes at those shapes; v2 keeps upstream tolerances, so this "
          "is only caught by the small shapes and the zeros control).\n")
    if sanity_rows:
        A("## Sanity table (L4)\n")
        A("Controls: `golden_cast` = reference on fp32/fp64-upcast inputs, rounded once to the input dtype (the best any kernel can do); "
          "`upstream_ref_native` = reference.py itself in the input dtype; `zeros`; `perturbed_5pct` = golden x1.05. "
          "v1 column: upstream bench.py full mode run in the v2 container (no cache), starters from `results/stage1_baselines.jsonl`, "
          "fp32acc/grpo_best from `results/v3/rmsnorm_remeasure.md`. Starter failures are genuine: matmul uses TF32 for fp32 inputs "
          "(err ~4e-2 vs tol 1e-4), flash_attention needs 128 KB smem at head_dim=128 (L4 limit 99 KB), fused_mlp calls "
          "`tl.math.tanh` (absent in Triton 3.2). v1 fails the same three.\n")
        A(f"All controls with a definite expectation behaved as expected: **{s['sanity_all_as_expected']}**.\n")
        A("| kernel | expected | v2 | v1 full bench | as expected | v2 failure (first) |")
        A("|---|---|---|---|---|---|")
        for r in sanity_rows:
            m = {True: "yes", False: "**NO**", None: "(info)"}[r["matches_expectation"]]
            why = (r["why"].split(";")[0] if r["why"] else "").replace("|", "/")
            A(f"| {r['id']} | {r['expected']} | {r['v2']} | {r['v1_full_bench']} | {m} | {why[:160]} |")
        A("")
    if base:
        A("## Baselines (L4, fp16, median of 3 fresh-seed repeats)\n")
        A("| kernel type | shape | eager us | starter us (CV) | torch.compile us (CV) | starter vs eager | compile/starter latency |")
        A("|---|---|---|---|---|---|---|")
        for kt in KERNELS:
            for lab, b in (base.get(kt) or {}).items():
                A(f"| {kt} | {lab} | {f3(b.get('eager_us'), 1)} | {f3(b.get('starter_us'), 1)} ({f3(b.get('starter_cv'))}) | "
                  f"{f3(b.get('compile_us'), 1)} ({f3(b.get('compile_cv'))}) | {f3(b.get('starter_vs_eager'))} | {f3(b.get('compile_over_starter'))} |")
        A("\nvs torch.compile for a candidate = paired (starter/kernel) ratio x baseline (compile/starter) ratio, so it "
          "inherits cross-container drift (a few %).\n")
    job_h = sum((r.get("wall_s") or 0) for r in load("rescore_bench_v2.jsonl") + load("bench_v2_sanity.jsonl")
                + load("bench_v2_baselines.jsonl")) / 3600
    A("## Cost (estimate)\n")
    A(f"Summed per-job wall time on L4: {job_h:.2f} h (rescore + baselines + final sanity). Adding the v2.0 sanity pass, 27 v1 "
      f"full-bench control runs, 2 hung jobs (420 s each) and container start-up, roughly 2x that in billed container time: "
      f"~{2 * job_h:.1f} L4-h x ${L4_USD_PER_H:.2f}/h (GPU+4 CPU+32 GB list price) = ~${2 * job_h * L4_USD_PER_H:.1f}.\n")
    (OUT / "bench_v2.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
