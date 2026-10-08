"""
bench v2 core: fixed correctness + timing harness for AutoKernel kernels.

Runs inside the autokernel-bench-v2 container as a subprocess (one process evaluates a list of
jobs; the Modal wrapper restarts it after a crash/hang). The upstream harness (bench.py,
reference.py at the pinned commit) is NOT modified: v2 loads it by path and reuses its sizes,
dtypes and per-dtype tolerances, so v2 differs from v1 only in the points below.

What v2 changes vs upstream bench.py (see results/v3/bench_v2.md):
  1. Golden reference = upstream reference.py applied to inputs upcast to fp64 (elementwise /
     reduction kernels) or fp32 with TF32 off (matmul, fused_mlp, flash_attention). Kernel
     output is compared against it with the upstream per-dtype atol/rtol (no x10 relaxation).
  2. Adversarial magnitude cases where the TRUE output is representable in the input dtype but
     fp16 intermediates overflow / underflow / stagnate. Any non-finite output where the golden
     is finite = FAIL (upstream PASSed "both NaN"). Golden must be finite and representable,
     else the case is invalid (never happens with the cases below; recorded if it does).
  3. Fresh random seed per evaluation (os.urandom), non-trivial layernorm weight/bias
     (upstream uses ones/zeros, so a kernel that ignores them passes), real cos/sin for RoPE.
  4. Timing: CUDA events, L2 flushed (256 MB memset) before every rep, warmup, >=50 reps,
     implementations interleaved rep-by-rep; median, CV, paired per-rep ratio vs the starter
     with a bootstrap 95% CI. Baselines: eager PyTorch reference, starter kernel, and (in
     baseline jobs) torch.compile(reference).
  5. No result cache of any kind.

CLI (used by bench_v2_modal.py):  python bench_v2_core.py jobs.json out.jsonl progress.txt
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import signal
import sys
import tempfile
import time
import traceback

import numpy as np
import torch
import torch._dynamo  # noqa: F401
import torch.nn.functional as F

AK = os.environ.get("AUTOKERNEL_DIR", "/autokernel")
F16, BF16, F32, F64 = torch.float16, torch.bfloat16, torch.float32, torch.float64
DEV = "cuda"
HARNESS_VERSION = "bench_v2.1"


def _load_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclasses (bench.py GPUSpec) look the module up here
    spec.loader.exec_module(mod)
    return mod


# upstream modules, loaded by path (never from cwd: /autokernel/profile.py shadows stdlib)
reference = _load_path("ak_reference", f"{AK}/reference.py")
sys.modules.setdefault("reference", reference)  # bench.py's wrappers do `import reference`
ak_bench = _load_path("ak_bench", f"{AK}/bench.py")
CFG = ak_bench.KERNEL_CONFIGS

# golden precision per kernel type
GOLD_DTYPE = {
    "matmul": F32, "fused_mlp": F32, "flash_attention": F32,
    "softmax": F64, "layernorm": F64, "rmsnorm": F64, "cross_entropy": F64,
    "rotary_embedding": F64, "reduce": F64,
}


def ref_call(kt: str, inp: dict):
    """Upstream reference.py, called exactly as bench.py's _ref_* wrappers do."""
    r = reference
    if kt == "matmul":
        return r.matmul_ref(inp["A"], inp["B"])
    if kt == "softmax":
        return r.softmax_ref(inp["x"])
    if kt == "layernorm":
        return r.layernorm_ref(inp["x"], inp["weight"], inp["bias"])
    if kt == "rmsnorm":
        return r.rmsnorm_ref(inp["x"], inp["weight"])
    if kt == "flash_attention":
        return r.flash_attention_ref(inp["Q"], inp["K"], inp["V"])
    if kt == "fused_mlp":
        return r.fused_mlp_ref(inp["x"], inp["w_gate"], inp["w_up"], inp["w_down"])
    if kt == "cross_entropy":
        return r.cross_entropy_ref(inp["logits"], inp["targets"])
    if kt == "rotary_embedding":
        return r.rotary_embedding_ref(inp["x"], inp["cos"], inp["sin"])
    if kt == "reduce":
        return r.reduce_sum_ref(inp["x"], dim=-1)
    raise KeyError(kt)


def golden(kt: str, inp: dict):
    hp = GOLD_DTYPE[kt]
    prev = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        up = {k: (v.to(hp) if torch.is_tensor(v) and v.is_floating_point() else v)
              for k, v in inp.items()}
        with torch.no_grad():
            return ref_call(kt, up)
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = prev


# ---------------------------------------------------------------------------------------------
# input generation: fp32 draw with a private CUDA generator -> scale -> cast once to dtype
# ---------------------------------------------------------------------------------------------
class Gen:
    def __init__(self, seed):
        self.g = torch.Generator(device=DEV)
        self.g.manual_seed(seed)

    def randn(self, *shape):
        return torch.randn(*shape, device=DEV, dtype=F32, generator=self.g)

    def rand(self, *shape):
        return torch.rand(*shape, device=DEV, dtype=F32, generator=self.g)

    def randint(self, hi, shape):
        return torch.randint(0, hi, shape, device=DEV, generator=self.g)


def make_inputs(kt: str, size: dict, dtype, seed: int, case: str = "base") -> dict:
    """case='base' = upstream distribution (except: layernorm weight/bias ~ randn instead of
    ones/zeros; RoPE cos/sin = cos/sin of real angles instead of randn).
    Other cases = adversarial magnitudes, documented in ADVERSARIAL below."""
    g = Gen(seed)
    c = lambda t: t.to(dtype).contiguous()
    if kt == "matmul":
        M, N, K = size["M"], size["N"], size["K"]
        if case == "pos_uniform":            # no cancellation: fp16 accumulators stagnate
            A, B = g.rand(M, K), g.rand(K, N)
        elif case == "big_scale":            # |A|,|B| ~ 8: outputs ~ 4e3, partial sums large
            A, B = g.randn(M, K) * 8, g.randn(K, N) * 8
        else:
            A, B = g.randn(M, K), g.randn(K, N)
        if case == "zeros":
            A = A * 0
        return {"A": c(A), "B": c(B)}
    if kt == "softmax":
        x = g.randn(size["rows"], size["cols"])
        if case == "big_logits":
            x = x * 100.0                    # exp without max-subtraction overflows
        elif case == "constant":
            x = torch.full_like(x, 3.0)
        return {"x": c(x)}
    if kt == "layernorm":
        B, D = size["batch"], size["dim"]
        x = g.randn(B, D)
        if case == "big_scale":
            x = x * 300.0                    # sum x^2 > 65504: fp16 accumulators overflow
        elif case == "large_mean":
            x = x + 1000.0                   # one-pass E[x^2]-E[x]^2 cancels catastrophically
        elif case == "constant":
            x = torch.full_like(x, 0.5)
        return {"x": c(x), "weight": c(g.randn(D)), "bias": c(g.randn(D))}
    if kt == "rmsnorm":
        M, N = size["M"], size["N"]
        x = g.randn(M, N)
        if case.startswith("rms_"):          # rms_8, rms_300: row RMS >> sqrt(65504/N)
            x = x * float(case.split("_")[1])
        elif case == "tiny":
            x = x * 1e-3                     # x^2 ~ 1e-6: fp16 subnormal range
        elif case == "row_mixed":            # per-row scale 10^U(-2, 2.5)
            x = x * torch.pow(10.0, g.rand(M, 1) * 4.5 - 2.0)
        elif case == "zeros":
            x = x * 0
        return {"x": c(x), "weight": c(g.randn(N))}
    if kt == "flash_attention":
        b, h, s, d = size["batch"], size["heads"], size["seq_len"], size["head_dim"]
        Q, K, V = g.randn(b, h, s, d), g.randn(b, h, s, d), g.randn(b, h, s, d)
        if case.startswith("qk_"):           # qk_4, qk_40: raw Q.K^T up to ~1e5 (> fp16 max)
            f = float(case.split("_")[1])
            Q, K = Q * f, K * f
        return {"Q": c(Q), "K": c(K), "V": c(V)}
    if kt == "fused_mlp":
        B, D, H = size["batch"], size["dim"], size["hidden"]
        x = g.randn(B, D)
        wg, wu, wd = g.randn(H, D) * 0.02, g.randn(H, D) * 0.02, g.randn(D, H) * 0.02
        if case == "big_act":
            # gate/up std ~150 -> silu(g)*u exceeds 65504 for ~1% of elements while the
            # output (std ~ 7e2) stays representable
            x = x * 4.0
            s = 150.0 / (4.0 * math.sqrt(D))
            wg, wu = g.randn(H, D) * s, g.randn(H, D) * s
            wd = g.randn(D, H) * (1e3 / (math.sqrt(H) * 150.0 * 150.0 * 0.7))
        return {"x": c(x), "w_gate": c(wg), "w_up": c(wu), "w_down": c(wd)}
    if kt == "cross_entropy":
        B, V = size["batch"], size["vocab"]
        lg = g.randn(B, V)
        if case == "big_logits":
            lg = lg * 100.0
        return {"logits": c(lg), "targets": g.randint(V, (B,))}
    if kt == "rotary_embedding":
        b, h, s, d = size["batch"], size["heads"], size["seq_len"], size["head_dim"]
        x = g.randn(b, h, s, d)
        if case == "big_x":
            x = x * 1000.0
        ang = g.rand(s, d // 2) * (2 * math.pi)
        return {"x": c(x), "cos": c(torch.cos(ang)), "sin": c(torch.sin(ang))}
    if kt == "reduce":
        M, N = size["M"], size["N"]
        if case == "pos_uniform":            # sum ~ N/2 (< 65504): fp16 running sums stagnate
            x = g.rand(M, N)
        else:
            x = g.randn(M, N)
        return {"x": c(x)}
    raise KeyError(kt)


# adversarial cases: (case, label, size, dtypes)
ADVERSARIAL = {
    "matmul": [("pos_uniform", "pos_uniform_K8192", {"M": 512, "N": 512, "K": 8192}, [F16, BF16]),
               ("big_scale", "big_scale_K4096", {"M": 1024, "N": 1024, "K": 4096}, [F16, BF16]),
               ("zeros", "zeros", {"M": 256, "N": 256, "K": 256}, [F16])],
    "softmax": [("big_logits", "big_logits_x100", {"rows": 1024, "cols": 4096}, [F16, BF16]),
                ("base", "large_N_131072", {"rows": 64, "cols": 131072}, [F16]),
                ("constant", "constant", {"rows": 256, "cols": 1000}, [F16])],
    "layernorm": [("big_scale", "big_scale_x300", {"batch": 1024, "dim": 4096}, [F16, BF16]),
                  ("large_mean", "mean_1000", {"batch": 1024, "dim": 4096}, [F16]),
                  ("base", "large_N_65536", {"batch": 64, "dim": 65536}, [F16]),
                  ("constant", "constant", {"batch": 256, "dim": 1000}, [F16])],
    "rmsnorm": [("rms_8", "rms_8_N4096", {"M": 1024, "N": 4096}, [F16]),
                ("rms_300", "rms_300_N4096", {"M": 1024, "N": 4096}, [F16, BF16]),
                ("tiny", "tiny_1e-3", {"M": 1024, "N": 4096}, [F16]),
                ("row_mixed", "row_mixed_scale", {"M": 1024, "N": 4096}, [F16, BF16]),
                ("rms_2", "rms_2_N65536", {"M": 64, "N": 65536}, [F16]),
                ("zeros", "zeros", {"M": 256, "N": 768}, [F16])],
    "flash_attention": [("qk_4", "qk_x4", {"batch": 2, "heads": 8, "seq_len": 512, "head_dim": 64}, [F16, BF16]),
                        ("qk_40", "qk_x40_overflow", {"batch": 2, "heads": 8, "seq_len": 512, "head_dim": 64}, [F16, BF16]),
                        ("base", "long_8192", {"batch": 1, "heads": 8, "seq_len": 8192, "head_dim": 64}, [F16])],
    "fused_mlp": [("big_act", "big_act", {"batch": 512, "dim": 1024, "hidden": 2048}, [F16, BF16])],
    "cross_entropy": [("big_logits", "big_logits_x100", {"batch": 1024, "vocab": 32000}, [F16, BF16]),
                      ("base", "vocab_128256", {"batch": 256, "vocab": 128256}, [F16])],
    "rotary_embedding": [("big_x", "big_x_x1000", {"batch": 1, "heads": 8, "seq_len": 1024, "head_dim": 128}, [F16, BF16]),
                         ("base", "long_16384", {"batch": 1, "heads": 8, "seq_len": 16384, "head_dim": 128}, [F16])],
    "reduce": [("pos_uniform", "pos_uniform_N32768", {"M": 256, "N": 32768}, [F16, BF16]),
               ("base", "large_N_1M", {"M": 16, "N": 1048576}, [F16])],
}

# timing configs: primary = upstream 'large' (same as v1's speedup), plus one LLM-ish shape
TIMING = {
    "matmul": ["large", "llm_qkv"], "softmax": ["large", "vocab"], "layernorm": ["large", "llm_7b"],
    "rmsnorm": ["large", "medium"], "flash_attention": ["large", "xlarge"],
    "fused_mlp": ["large", "medium"], "cross_entropy": ["large", "gpt2"],
    "rotary_embedding": ["large", "llm_7b"], "reduce": ["large", "wide"],
}


# unit roundoff u of each dtype. A perfectly rounded output already has relative error <= u, so
# an rtol below ~2u is infeasible: upstream's bf16 rtol=2e-3 (layernorm, softmax, RoPE) is below
# bf16's u=3.9e-3 and fails the correctly-rounded golden itself (see sanity golden_cast).
UNIT_ROUNDOFF = {F16: 2.0 ** -11, BF16: 2.0 ** -8, F32: 2.0 ** -24}


def tol_for(kt, dtype):
    """upstream per-dtype atol/rtol, with rtol floored at 2u (two roundings) of the dtype."""
    t = dict(CFG[kt]["tolerances"].get(dtype, {"atol": 1e-2, "rtol": 1e-2}))
    t["rtol"] = max(t["rtol"], 2 * UNIT_ROUNDOFF.get(dtype, 0.0))
    return t


def compare(out, gold, dtype, tol) -> dict:
    """Elementwise |out-gold| <= atol + rtol*|gold| everywhere, no non-finite where gold is
    finite, golden itself finite and representable in `dtype`."""
    if not torch.is_tensor(out):
        return {"ok": False, "reason": f"output is {type(out).__name__}, not a tensor"}
    if tuple(out.shape) != tuple(gold.shape):
        return {"ok": False, "reason": f"shape {tuple(out.shape)} != {tuple(gold.shape)}"}
    g = gold.to(F64)
    o = out.detach().to(F64)
    g_fin = torch.isfinite(g)
    if not bool(g_fin.all()):
        return {"ok": None, "reason": "golden non-finite (invalid case)"}
    if dtype in (F16, BF16) and not bool(torch.isfinite(gold.to(dtype)).all()):
        return {"ok": None, "reason": "golden not representable in dtype (invalid case)"}
    o_fin = torch.isfinite(o)
    n_nonfinite = int((~o_fin).sum())
    err = (o - g).abs()
    lim = tol["atol"] + tol["rtol"] * g.abs()
    viol = (~o_fin) | (err > lim)
    n_viol = int(viol.sum())
    err_f = torch.where(o_fin, err, torch.full_like(err, float("inf")))
    max_abs = float(err_f.max()) if err_f.numel() else 0.0
    rel = err_f / g.abs().clamp_min(max(tol["atol"], 1e-30))
    res = {"ok": n_viol == 0, "max_abs_err": max_abs, "max_rel_err": float(rel.max()) if rel.numel() else 0.0,
           "n_viol": n_viol, "frac_viol": n_viol / max(1, g.numel()), "n_nonfinite": n_nonfinite,
           "atol": tol["atol"], "rtol": tol["rtol"]}
    if g.dim() == 2:
        zero_rows = ((o == 0).all(dim=1) & ~(g == 0).all(dim=1)).sum()
        res["zero_rows"] = int(zero_rows)
        res["rows"] = int(g.shape[0])
    if out.dtype != dtype and dtype is not None:
        res["out_dtype"] = str(out.dtype)
    if not res["ok"]:
        res["reason"] = (f"{n_viol}/{g.numel()} elems out of tol (atol={tol['atol']}, rtol={tol['rtol']}),"
                         f" max_abs_err={max_abs:.3e}, nonfinite={n_nonfinite}"
                         + (f", zero_rows={res.get('zero_rows')}" if res.get("zero_rows") else ""))
    return res


class _Alarm(Exception):
    pass


def _alarm(signum, frame):
    raise _Alarm()


def run_case(kfn, kt, size, dtype, seed, case, with_ref_native=False, timeout=60):
    inp = make_inputs(kt, size, dtype, seed, case)
    gold = golden(kt, inp)
    tol = tol_for(kt, dtype)
    rec = {"dtype": str(dtype).replace("torch.", ""), "size": size, "case": case}
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(timeout)
    try:
        with torch.no_grad():
            out = kfn(**inp)
        torch.cuda.synchronize()
        signal.alarm(0)
        rec.update(compare(out, gold, dtype, tol))
    except _Alarm:
        rec.update({"ok": False, "reason": "TIMEOUT"})
    except torch.cuda.OutOfMemoryError:
        signal.alarm(0)
        rec.update({"ok": None, "reason": "OOM (skipped)"})
    except Exception as e:  # noqa: BLE001
        signal.alarm(0)
        msg = f"{type(e).__name__}: {str(e)[:300]}"
        rec.update({"ok": False, "reason": "EXC " + msg})
        if _is_fatal_cuda(msg):
            rec["fatal"] = True
    finally:
        signal.alarm(0)
    if with_ref_native:  # audit: upstream reference evaluated in the input dtype
        with torch.no_grad():
            rn = ref_call(kt, inp)
        c = compare(rn, gold, dtype, tol)
        rec["ref_native"] = {k: c.get(k) for k in ("ok", "max_abs_err", "n_nonfinite", "n_viol", "zero_rows", "reason")}
    del inp, gold
    return rec


def _is_fatal_cuda(msg):
    m = msg.lower()
    return any(s in m for s in ("illegal memory", "unspecified launch", "device-side assert",
                                "cuda error", "misaligned address", "an illegal instruction"))


def correctness(kfn, kt, seed, with_ref_native=False, early_abort=True):
    cfg = CFG[kt]
    sizes, dtypes = cfg["test_sizes"], cfg["test_dtypes"]
    cases = []
    # smoke + sweep: all upstream sizes x dtypes (fresh seed)
    for label, sz in sizes:
        for dt in dtypes:
            cases.append(("sweep", label, sz, dt, "base"))
    for label, sz in cfg.get("edge_sizes", []):
        for dt in dtypes[:2]:
            cases.append(("edge", label, sz, dt, "base"))
    for case, label, sz, dts in ADVERSARIAL[kt]:
        for dt in dts:
            cases.append(("adversarial", label, sz, dt, case))
    recs, fatal = [], False
    for i, (stage, label, sz, dt, case) in enumerate(cases):
        r = run_case(kfn, kt, sz, dt, seed + i, case, with_ref_native=with_ref_native)
        r.update({"stage": stage, "label": label})
        recs.append(r)
        torch.cuda.empty_cache()
        if r.get("fatal"):
            fatal = True
            break
        if early_abort and i == 0 and r["ok"] is False:
            break  # smoke failed (first sweep case), like upstream
    # determinism: same inputs twice -> bitwise equal
    det = None
    if not fatal and recs and recs[0]["ok"]:
        try:
            lab, sz = sizes[min(1, len(sizes) - 1)]
            inp = make_inputs(kt, sz, dtypes[0], seed + 999, "base")
            with torch.no_grad():
                a = kfn(**inp)
                b = kfn(**inp)
            det = bool(torch.equal(a, b))
        except Exception as e:  # noqa: BLE001
            det = False
            if _is_fatal_cuda(str(e)):
                fatal = True
    stage_ok = {}
    for st in ("sweep", "edge", "adversarial"):
        rs = [r for r in recs if r["stage"] == st]
        stage_ok[st] = None if not rs else all(r["ok"] is not False for r in rs)
    n_cases = len(cases)
    complete = len(recs) == n_cases
    ok = complete and all(r["ok"] is not False for r in recs) and det is True
    base_ok = all(r["ok"] is not False for r in recs if r["stage"] != "adversarial") and complete
    return {"ok": ok, "base_ok": base_ok, "stage_ok": stage_ok, "determinism": det,
            "n_cases": n_cases, "n_run": len(recs), "fatal": fatal,
            "n_fail": sum(1 for r in recs if r["ok"] is False),
            "n_invalid": sum(1 for r in recs if r["ok"] is None), "cases": recs}


# ---------------------------------------------------------------------------------------------
# timing
# ---------------------------------------------------------------------------------------------
_FLUSH = None


def _flush():
    global _FLUSH
    if _FLUSH is None:
        _FLUSH = torch.empty(256 * 2**20 // 4, dtype=torch.float32, device=DEV)  # 256 MB >> L2
    _FLUSH.zero_()


def time_impls(impls: dict, reps=60, warmup=10) -> dict:
    """Interleaved rep-by-rep timing, cold L2. Returns per-impl stats + raw times (us)."""
    names = list(impls)
    with torch.no_grad():
        for n in names:
            for _ in range(warmup):
                impls[n]()
        torch.cuda.synchronize()
        ev = {n: [] for n in names}
        for r in range(reps):
            order = names[r % len(names):] + names[:r % len(names)]
            for n in order:
                _flush()                     # also keeps the GPU busy while fn is enqueued
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record()
                impls[n]()
                e.record()
                ev[n].append((s, e))
        torch.cuda.synchronize()
    out = {}
    for n in names:
        t = np.array([s.elapsed_time(e) * 1000.0 for s, e in ev[n]])
        out[n] = {"median_us": float(np.median(t)), "mean_us": float(t.mean()),
                  "p10_us": float(np.percentile(t, 10)), "p90_us": float(np.percentile(t, 90)),
                  "cv": float(t.std() / t.mean()) if t.mean() > 0 else 0.0, "n": int(t.size),
                  "_t": t}
    return out


def paired_ratio(t_base, t_cand, n_boot=2000, seed=0):
    """speedup = base/cand per rep (reps were interleaved) -> median + bootstrap 95% CI."""
    r = t_base / t_cand
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, r.size, size=(n_boot, r.size))
    meds = np.median(r[idx], axis=1)
    return {"median": float(np.median(r)), "ci_lo": float(np.percentile(meds, 2.5)),
            "ci_hi": float(np.percentile(meds, 97.5))}


def timing(kfn, starter_fn, kt, seed, reps=60, compiled=None) -> dict:
    cfg = CFG[kt]
    sizes = dict(cfg["test_sizes"])
    dtype = cfg["test_dtypes"][0]
    res = {}
    for li, label in enumerate(TIMING[kt]):
        inp = make_inputs(kt, sizes[label], dtype, seed + 1000 * (li + 1), "base")
        impls = {}
        if kfn is not None:
            impls["kernel"] = lambda: kfn(**inp)
        if starter_fn is not None:
            impls["starter"] = lambda: starter_fn(**inp)
        impls["eager"] = lambda: ref_call(kt, inp)
        if compiled is not None:
            impls["compile"] = lambda: compiled(inp)
        try:
            st = time_impls(impls, reps=reps)
        except Exception as e:  # noqa: BLE001
            res[label] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
            if _is_fatal_cuda(str(e)):
                res["fatal"] = True
                break
            continue
        entry = {"dtype": str(dtype).replace("torch.", ""), "size": sizes[label]}
        for n, s in st.items():
            entry[n] = {k: v for k, v in s.items() if k != "_t"}
        if "kernel" in st:
            entry["vs_eager"] = paired_ratio(st["eager"]["_t"], st["kernel"]["_t"])
            if "starter" in st:
                entry["vs_starter"] = paired_ratio(st["starter"]["_t"], st["kernel"]["_t"])
            if "compile" in st:
                entry["vs_compile"] = paired_ratio(st["compile"]["_t"], st["kernel"]["_t"])
        if "starter" in st:
            entry["starter_vs_eager"] = paired_ratio(st["eager"]["_t"], st["starter"]["_t"])
            if "compile" in st:
                entry["compile_over_starter"] = paired_ratio(st["compile"]["_t"], st["starter"]["_t"])
        res[label] = entry
        del inp
        torch.cuda.empty_cache()
    return res


# ---------------------------------------------------------------------------------------------
# job runner
# ---------------------------------------------------------------------------------------------
_MODS = {}


def load_kernel(code: str, tag: str, d: str):
    key = (tag, hash(code))
    if key in _MODS:
        return _MODS[key]
    name = f"k_{tag}_{abs(hash(code)) % 10**10}"
    p = os.path.join(d, name + ".py")
    with open(p, "w") as f:
        f.write(code)
    mod = _load_path(name, p)
    _MODS[key] = mod.kernel_fn
    return mod.kernel_fn


def run_job(job: dict, d: str) -> dict:
    kt = job["kernel_type"]
    seed = int(job.get("seed") or int.from_bytes(os.urandom(4), "little"))
    rec = {"id": job["id"], "kernel_type": kt, "seed": seed, "harness": HARNESS_VERSION,
           "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}
    t0 = time.time()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    starter_fn = None
    if job.get("starter_code"):
        starter_fn = load_kernel(job["starter_code"], "starter_" + kt, d)
    kfn = None
    if job.get("code") is not None:
        try:
            kfn = load_kernel(job["code"], "cand", d)
        except Exception as e:  # noqa: BLE001
            rec.update({"verdict": "CRASH", "reason": f"import: {type(e).__name__}: {str(e)[:300]}",
                        "wall_s": round(time.time() - t0, 2)})
            return rec
    compiled = None
    if job.get("compile"):
        torch._dynamo.config.cache_size_limit = 256
        compiled = torch.compile(lambda inp: ref_call(kt, inp), dynamic=False)
    if kfn is not None:
        c = correctness(kfn, kt, seed, with_ref_native=job.get("ref_native", False),
                        early_abort=job.get("early_abort", True))
        rec["correctness"] = c
        if c["fatal"]:
            rec["verdict"] = "CRASH"
            rec["reason"] = "fatal CUDA error: " + next((r.get("reason", "") for r in c["cases"] if r.get("fatal")), "")
        elif c["n_run"] >= 1 and c["cases"][0]["ok"] is False and c["cases"][0].get("reason", "").startswith("EXC"):
            rec["verdict"] = "CRASH"
            rec["reason"] = c["cases"][0]["reason"]
        else:
            rec["verdict"] = "PASS" if c["ok"] else "FAIL"
            fails = [f"{r['stage']}/{r['label']}/{r['dtype']}: {r.get('reason', '')}" for r in c["cases"] if r["ok"] is False]
            if c["determinism"] is False:
                fails.append("determinism: two runs on identical inputs differ")
            rec["fail_reasons"] = fails[:8]
    if job.get("timing", True) and rec.get("verdict") != "CRASH":
        rec["timing"] = timing(kfn, starter_fn, kt, seed + 7, reps=int(job.get("reps", 60)),
                               compiled=compiled)
    rec["wall_s"] = round(time.time() - t0, 2)
    return rec


def main():
    jobs_path, out_path, prog_path = sys.argv[1:4]
    jobs = json.load(open(jobs_path))
    d = tempfile.mkdtemp()
    for job in jobs:
        with open(prog_path, "a") as f:
            f.write(f"START {job['id']}\n")
        try:
            rec = run_job(job, d)
        except Exception as e:  # noqa: BLE001
            rec = {"id": job["id"], "kernel_type": job["kernel_type"], "verdict": "CRASH",
                   "reason": f"harness: {type(e).__name__}: {str(e)[:300]}",
                   "tb": traceback.format_exc()[-1500:]}
        with open(out_path, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        with open(prog_path, "a") as f:
            f.write(f"DONE {job['id']}\n")
        fatal = rec.get("verdict") == "CRASH" and "fatal" in rec.get("reason", "")
        if fatal or (rec.get("timing") or {}).get("fatal"):
            sys.exit(3)  # CUDA context is poisoned; parent restarts with the next job
    sys.exit(0)


if __name__ == "__main__":
    main()
