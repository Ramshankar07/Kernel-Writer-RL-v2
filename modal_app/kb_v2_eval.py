"""
KernelBench v2 evaluator ("kb v2"): replaces AutoKernel's kernelbench/bench_kb.py for our evals.

Runs in a fresh subprocess per evaluation (see kernelbench_v2_modal.py). One reference, one or
more candidate ModelNew files, one JSON result on stdout-file.

Fixes vs bench_kb.py (see results/v3/kb_v2.md):
  1. weights: torch/cuda/python/numpy RNGs are seeded identically before constructing Model and
     each ModelNew; if ModelNew's state_dict has the same keys/shapes, Model's weights are copied
     in as well. (bench_kb.py never seeds -> every conv problem fails for an identical model.)
  2. inputs: get_inputs() runs under a CUDA default device (tensors are created on the GPU) with a
     per-trial seed, *outside* the timeout / timed region. The correctness timeout covers only the
     forward passes; timing has its own timeout. (bench_kb.py ran CPU torch.rand of up to 2.1B
     elements inside a 30 s per-trial alarm.)
  3. correctness: reference runs in strict fp32 (TF32 off for matmul + cuDNN, globally, so the
     candidate is judged on its kernel, not on global flags); atol=rtol=1e-2 (KernelBench fp32
     convention, same as bench_kb.py), 5 seeded trials + 1 hidden trial with a fresh random seed.
     A strict 1e-4 check is reported alongside. Always the *reference's* get_inputs/get_init_inputs
     (bench_kb.py preferred the candidate's own get_inputs if it defined one).
  4. timing: CUDA events, 3 warmup calls, L2 flushed (256 MB write) before every rep, median of
     >=30 reps (adaptive up to 100 for fast kernels).

Ablation switches (for per-bug attribution): --weight-sync none (unseeded, like bench_kb.py),
--input-mode cpu_timed (CPU get_inputs inside a 30 s per-trial alarm, like bench_kb.py).
"""
import argparse
import importlib.util
import json
import math
import os
import random
import secrets
import signal
import sys
import time
import traceback

import torch

ATOL = RTOL = 1e-2
STRICT = 1e-4
L2_FLUSH_BYTES = 256 * 1024 * 1024   # H100 L2 = 50 MB
CHUNK = 1 << 26                      # chunked compare: 2.1B-element outputs would OOM allclose


class Timeout(Exception):
    pass


class alarm:
    def __init__(self, seconds):
        self.s = int(math.ceil(seconds))

    def __enter__(self):
        def h(*_):
            raise Timeout(f"exceeded {self.s}s")
        self.prev = signal.signal(signal.SIGALRM, h)
        signal.alarm(self.s)

    def __exit__(self, *a):
        signal.alarm(0)
        signal.signal(signal.SIGALRM, self.prev)


def seed_all(s):
    random.seed(s)
    torch.manual_seed(s)            # seeds CPU and all CUDA generators
    try:
        import numpy as np
        np.random.seed(s % (2**32))
    except Exception:
        pass


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def to_cuda(x):
    if isinstance(x, torch.Tensor):
        return x.cuda()
    if isinstance(x, (list, tuple)):
        return type(x)(to_cuda(v) for v in x)
    return x


def build(cls, init_args, seed):
    """Construct on the GPU; seed first (seed=None -> fresh random seed, i.e. unseeded)."""
    seed_all(seed if seed is not None else secrets.randbits(31))
    try:
        with torch.device("cuda"):
            m = cls(*init_args)
    except Exception:
        seed_all(seed if seed is not None else secrets.randbits(31))
        m = cls(*init_args)
    return m.cuda().eval()


def gen_inputs(get_inputs, seed, mode):
    seed_all(seed)
    if mode == "gpu":
        with torch.device("cuda"):
            inp = get_inputs()
    else:
        inp = get_inputs()
    inp = to_cuda(list(inp))
    torch.cuda.synchronize()
    return inp


def compare(out, exp):
    """-> dict(match, strict, max_abs, reason). Recursive for tuple outputs; chunked."""
    if isinstance(exp, (list, tuple)):
        if not isinstance(out, (list, tuple)) or len(out) != len(exp):
            return {"match": False, "strict": False, "max_abs": float("inf"),
                    "reason": f"output structure mismatch: {type(out).__name__} vs {type(exp).__name__}"}
        rs = [compare(o, e) for o, e in zip(out, exp)]
        bad = [r for r in rs if not r["match"]]
        return {"match": not bad, "strict": all(r["strict"] for r in rs),
                "max_abs": max((r["max_abs"] for r in rs), default=0.0),
                "reason": bad[0]["reason"] if bad else ""}
    if not isinstance(exp, torch.Tensor):
        ok = out == exp
        return {"match": bool(ok), "strict": bool(ok), "max_abs": 0.0 if ok else float("inf"),
                "reason": "" if ok else f"non-tensor mismatch {out!r} vs {exp!r}"}
    if not isinstance(out, torch.Tensor):
        return {"match": False, "strict": False, "max_abs": float("inf"),
                "reason": f"expected tensor, got {type(out).__name__}"}
    if out.shape != exp.shape:
        return {"match": False, "strict": False, "max_abs": float("inf"),
                "reason": f"shape {tuple(out.shape)} vs {tuple(exp.shape)}"}
    o, e = out.detach().reshape(-1), exp.detach().reshape(-1)
    max_abs, n_bad, n_bad_strict = 0.0, 0, 0
    for i in range(0, e.numel(), CHUNK):
        oc, ec = o[i:i + CHUNK].float(), e[i:i + CHUNK].float()
        if not torch.equal(torch.isnan(oc), torch.isnan(ec)):
            return {"match": False, "strict": False, "max_abs": float("nan"), "reason": "NaN pattern differs"}
        d = (oc - ec).abs().nan_to_num(0.0)
        max_abs = max(max_abs, d.max().item() if d.numel() else 0.0)
        n_bad += (~torch.isclose(oc, ec, atol=ATOL, rtol=RTOL, equal_nan=True)).sum().item()
        n_bad_strict += (~torch.isclose(oc, ec, atol=STRICT, rtol=STRICT, equal_nan=True)).sum().item()
        del oc, ec, d
    return {"match": n_bad == 0, "strict": n_bad_strict == 0, "max_abs": max_abs,
            "reason": "" if n_bad == 0 else f"{n_bad}/{e.numel()} elements exceed atol=rtol={ATOL} (max_abs={max_abs:.3e})"}


def time_model(model, inputs, flush, timeout_s, min_reps=30, max_reps=100, budget_ms=2000.0):
    with alarm(timeout_s), torch.no_grad():
        for _ in range(3):
            model(*inputs)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(*inputs)
        torch.cuda.synchronize()
        est = (time.perf_counter() - t0) * 1e3
        reps = int(min(max_reps, max(min_reps, budget_ms / max(est, 1e-3))))
        ts = []
        for _ in range(reps):
            flush.zero_()
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            model(*inputs)
            e.record()
            torch.cuda.synchronize()
            ts.append(s.elapsed_time(e))
    ts.sort()
    n = len(ts)
    med = ts[n // 2] if n % 2 else (ts[n // 2 - 1] + ts[n // 2]) / 2
    return {"median_ms": med, "reps": n, "p10_ms": ts[n // 10], "p90_ms": ts[(9 * n) // 10]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--cand", action="append", default=[], help="name=path/to/kernel.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hidden-seed", type=int, default=None)
    ap.add_argument("--n-trials", type=int, default=5)
    ap.add_argument("--weight-sync", choices=["copy", "seed", "none"], default="copy")
    ap.add_argument("--input-mode", choices=["gpu", "cpu_timed"], default="gpu")
    ap.add_argument("--corr-timeout", type=float, default=120)
    ap.add_argument("--perf-timeout", type=float, default=180)
    ap.add_argument("--no-perf", action="store_true")
    a = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    hidden = a.hidden_seed if a.hidden_seed is not None else secrets.randbits(31)
    seeds = [a.seed + i for i in range(a.n_trials)] + [hidden]
    res = {"seeds": seeds, "hidden_seed": hidden, "weight_sync": a.weight_sync,
           "input_mode": a.input_mode, "atol": ATOL, "rtol": RTOL, "cands": {}}

    def dump():
        with open(a.out, "w") as f:
            json.dump(res, f)

    try:
        ref_mod = load_module(a.ref, "kb_reference")
        Model, get_inputs = ref_mod.Model, ref_mod.get_inputs
        get_init = getattr(ref_mod, "get_init_inputs", lambda: [])
        seed_all(a.seed)
        init_args = get_init()
        ref = build(Model, init_args, a.seed)
    except Exception as e:
        res["error"] = f"reference setup: {type(e).__name__}: {e}"
        res["tb"] = traceback.format_exc()[-1500:]
        dump()
        return

    cands = {}
    for spec in a.cand:
        name, path = spec.split("=", 1)
        r = res["cands"][name] = {"status": "PENDING", "trials": []}
        try:
            mod = load_module(path, f"kb_cand_{name}")
            ModelNew = mod.ModelNew
            m = build(ModelNew, init_args, a.seed if a.weight_sync != "none" else None)
            r["weights"] = "seeded" if a.weight_sync != "none" else "unseeded"
            if a.weight_sync == "copy":
                rs, cs = ref.state_dict(), m.state_dict()
                if rs and rs.keys() == cs.keys() and all(rs[k].shape == cs[k].shape for k in rs):
                    m.load_state_dict(rs)
                    r["weights"] = "copied"
            cands[name] = m
        except Exception as e:
            r.update(status="CRASH", stage="instantiate", reason=f"{type(e).__name__}: {e}"[:500])

    # --- correctness: inputs generated outside the timeout (gpu mode) ---
    live = dict(cands)
    t_gen = t_fwd = 0.0
    for ti, s in enumerate(seeds):
        if not live:
            break
        label = "hidden" if ti == a.n_trials else str(ti)
        try:
            if a.input_mode == "gpu":
                t0 = time.perf_counter()
                inp = gen_inputs(get_inputs, s, "gpu")
                t_gen += time.perf_counter() - t0
                ctx = alarm(a.corr_timeout)
            else:   # bench_kb.py behaviour: CPU generation inside a 30 s per-trial alarm
                ctx = alarm(30)
            with ctx:
                t0 = time.perf_counter()
                if a.input_mode != "gpu":
                    inp = gen_inputs(get_inputs, s, "cpu")
                with torch.no_grad():
                    exp = ref(*inp)
                torch.cuda.synchronize()
                t_fwd += time.perf_counter() - t0
        except Exception as e:
            kind = "TIMEOUT" if isinstance(e, Timeout) else "INFRA_ERROR"
            for name in live:
                res["cands"][name].update(status=kind, stage=f"reference trial {label}",
                                          reason=f"{type(e).__name__}: {e}"[:500])
            live = {}
            break
        for name in list(live):
            r = res["cands"][name]
            try:
                with alarm(a.corr_timeout if a.input_mode == "gpu" else 30), torch.no_grad():
                    out = live[name](*inp)
                    torch.cuda.synchronize()
                c = compare(out, exp)
                del out
            except Timeout as e:
                c = {"match": False, "strict": False, "max_abs": float("inf"), "reason": f"TIMEOUT {e}"}
                r["status"] = "TIMEOUT"
            except Exception as e:
                c = {"match": False, "strict": False, "max_abs": float("inf"),
                     "reason": f"{type(e).__name__}: {e}"[:500]}
                r["status"] = "CRASH"
            r["trials"].append({"trial": label, "seed": s, **c})
            if not c["match"]:
                if r["status"] == "PENDING":
                    r["status"] = "FAIL"
                r.update(stage=f"correctness trial {label}", reason=c["reason"])
                del live[name]
        del inp, exp
        torch.cuda.empty_cache()
    res["input_gen_s"] = round(t_gen, 3)
    res["ref_fwd_s"] = round(t_fwd, 3)
    for name in live:
        r = res["cands"][name]
        r["status"] = "PASS"
        r["max_abs"] = max(t["max_abs"] for t in r["trials"])
        r["strict_1e-4"] = all(t["strict"] for t in r["trials"])
        r["hidden_pass"] = r["trials"][-1]["match"]
    dump()

    # --- timing (own timeout; inputs generated untimed) ---
    if a.no_perf or not live:
        return
    try:
        inp = gen_inputs(get_inputs, a.seed, "gpu")
        flush = torch.empty(L2_FLUSH_BYTES // 4, dtype=torch.float32, device="cuda")
        res["ref_timing"] = time_model(ref, inp, flush, a.perf_timeout)
        for name, m in live.items():
            r = res["cands"][name]
            try:
                r["timing"] = time_model(m, inp, flush, a.perf_timeout)
                r["speedup"] = round(res["ref_timing"]["median_ms"] / r["timing"]["median_ms"], 4)
            except Exception as e:
                r["timing_error"] = f"{type(e).__name__}: {e}"[:300]
    except Exception as e:
        res["timing_error"] = f"{type(e).__name__}: {e}"[:300]
    dump()


if __name__ == "__main__":
    main()
