"""
Validation-set harness (Phase 0): bench v2 conventions applied to the held-out problems in
results/v3/val_set/ (built by modal_app/val_set_v1.py). Runs as a subprocess inside the
autokernel-val-bench container, exactly like bench_v2_core.py (restarted after a CUDA fault).

Same as bench v2 (helpers imported from bench_v2_core, unchanged):
  * golden = the problem's reference.py on inputs upcast to fp64 (norms, softmax, CE, RoPE,
    reduce) or fp32 with TF32 off (matmul, fused_mlp, attention), per-dtype atol/rtol from the
    family's upstream bench.py tolerances, rtol floored at 2u, no x10 relaxation;
  * any non-finite output where the golden is finite = FAIL; golden must be representable;
  * fresh os.urandom seed per evaluation; adversarial magnitude cases per family;
  * determinism check (same inputs twice -> bitwise equal);
  * timing: CUDA events, 256 MB L2 flush per rep, interleaved implementations, paired ratio
    vs the starter with a bootstrap CI; eager reference and torch.compile(reference) baselines.
Differences (documented in results/v3/phase0.md):
  * float32 inputs to the matmul-type families use an fp64 golden (bench v2's fp32 golden is
    itself fp32 cuBLAS; an exact fp64 matmul FAILs it at K>=2048, see phase0.md);
  * launch check: PASS requires >= 1 @triton.jit kernel launch during the correctness run
    (a pure-PyTorch kernel_fn is FAIL "no Triton kernel launched");
  * eval mode stops correctness at the first failing case (feedback = that case) and times
    only PASS kernels, at the problem's one timing shape; vs torch.compile comes from the
    per-problem baseline job (compile/starter ratio), as in bench v2.

CLI:  python val_bench_core.py jobs.json out.jsonl progress.txt
"""
from __future__ import annotations

import json
import math
import os
import signal
import sys
import tempfile
import time
import traceback

import torch
import torch._dynamo  # noqa: F401

import bench_v2_core as b2  # noqa: E402  (loads upstream bench.py/reference.py by path)

F16, BF16, F32, F64 = torch.float16, torch.bfloat16, torch.float32, torch.float64
DT = {"float16": F16, "bfloat16": BF16, "float32": F32}
DEV = "cuda"
HARNESS_VERSION = "val_bench_v1"
MATMUL_TYPE = {"matmul", "fused_mlp", "flash_attention"}

# ---------------------------------------------------------------------------------------------
# launch counter: every `kernel[grid](...)` (also via autotune/heuristics) ends in JITFunction.run
# ---------------------------------------------------------------------------------------------
LAUNCHES = [0]
try:
    from triton.runtime.jit import JITFunction as _JF
    _orig_run = _JF.run

    def _counting_run(self, *a, **kw):
        if not kw.get("warmup", False):
            LAUNCHES[0] += 1
        return _orig_run(self, *a, **kw)

    _JF.run = _counting_run
except Exception:  # noqa: BLE001
    pass


def gold_dtype(family, dtype):
    if family in MATMUL_TYPE:
        return F64 if dtype == F32 else F32
    return F64


# torch 2.6.0+cu124 on L4: CUDA float64 softmax / log_softmax / cross_entropy are WRONG (errors up to
# 0.9 in log-probs, sometimes an illegal memory access) when the softmax dim is ~257 < n <= 1024 with
# n % 8 == 1 (e.g. 513, 777, 1017); CPU float64 and CUDA float32 are correct (results/v3/phase0.md).
# Goldens of the softmax-based families are therefore computed on the CPU in float64.
CPU_GOLDEN = {"softmax", "cross_entropy"}


def golden(ref, family, inp, dtype):
    hp = gold_dtype(family, dtype)
    prev = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        if family in CPU_GOLDEN:
            up = {k: (v.detach().cpu().to(hp) if v.is_floating_point() else v.detach().cpu())
                  if torch.is_tensor(v) else v for k, v in inp.items()}
            with torch.no_grad():
                return ref(**up).to(DEV)
        up = {k: (v.to(hp) if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in inp.items()}
        with torch.no_grad():
            return ref(**up)
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = prev


def tol_for(spec, dtype):
    name = str(dtype).replace("torch.", "")
    t = dict(spec["tol"][name])
    t["rtol"] = max(t["rtol"], 2 * b2.UNIT_ROUNDOFF.get(dtype, 0.0))
    return t


# ---------------------------------------------------------------------------------------------
# input generators (fp32 draw from a private CUDA generator -> scale -> cast once)
# ---------------------------------------------------------------------------------------------
def make_inputs(spec, size, dtype, seed, case="base"):
    g = b2.Gen(seed)
    c = lambda t: t.to(dtype).contiguous()
    ga = spec.get("gen_args", {})
    gen = spec["gen"]
    if gen == "softmax":
        x = g.randn(*size["shape"])
        if case == "big_logits":
            x = x * 100.0
        elif case == "constant":
            x = torch.full_like(x, 3.0)
        return {"x": c(x)}
    if gen in ("layernorm", "rmsnorm"):
        shp = size["shape"]
        D = shp[-1]
        x = g.randn(*shp)
        res = g.randn(*shp) if ga.get("residual") else None
        if case == "big_scale":
            x = x * 300.0
            res = None if res is None else res * 300.0
        elif case == "large_mean":
            x = x + 1000.0
        elif case == "constant":
            x = torch.full_like(x, 0.5)
            res = None if res is None else torch.zeros_like(res)
        elif case == "rms_300":
            x = x * 300.0
            res = None if res is None else res * 300.0
        elif case == "tiny":
            x = x * 1e-3
            res = None if res is None else res * 1e-3
        elif case == "row_mixed":
            rows = x.reshape(-1, D)
            sc = torch.pow(10.0, g.rand(rows.shape[0], 1) * 4.5 - 2.0)
            x = (rows * sc).reshape(shp)
            res = None if res is None else (res.reshape(-1, D) * sc).reshape(shp)
        out = {"x": c(x)}
        if res is not None:
            out["residual"] = c(res)
        if ga.get("weight", True):
            out["weight"] = c(g.randn(D) * ga.get("weight_scale", 1.0))
        if ga.get("bias", False):
            out["bias"] = c(g.randn(D))
        return out
    if gen == "reduce":
        shp = size["shape"]
        x = g.rand(*shp) if case == "pos_uniform" else g.randn(*shp)
        return {"x": c(x)}
    if gen == "cross_entropy":
        lead = size["lead"]
        V = size["vocab"]
        lg = g.randn(*lead, V)
        if case == "big_logits":
            lg = lg * 100.0
        t = g.randint(V, tuple(lead))
        if ga.get("ignore_frac"):
            drop = g.rand(*lead) < ga["ignore_frac"]
            t = torch.where(drop, torch.full_like(t, -100), t)
        return {"logits": c(lg), "targets": t}
    if gen == "rotary":
        b, h, s, d = size["b"], size["h"], size["s"], size["d"]
        shp = (b, s, h, d) if ga.get("layout") == "bshd" else (b, h, s, d)
        x = g.randn(*shp)
        if case == "big_x":
            x = x * 1000.0
        ang = g.rand(s, d // 2) * (2 * math.pi)
        return {"x": c(x), "cos": c(torch.cos(ang)), "sin": c(torch.sin(ang))}
    if gen == "matmul":
        mode = ga.get("mode", "mm")
        M, N, K = size["M"], size["N"], size["K"]
        bt = size.get("batch")
        lead = (bt,) if bt else ()
        f = (lambda *s: g.rand(*s)) if case == "pos_uniform" else (lambda *s: g.randn(*s))
        s8 = 8.0 if case == "big_scale" else 1.0
        A = f(*lead, M, K) * s8
        if mode == "linear":
            return {"x": c(A), "w": c(f(N, K) * s8)}
        B = f(*lead, K, N) * s8
        out = {"A": c(A), "B": c(B)}
        if mode == "bias_relu":
            out["bias"] = c(g.randn(N) * s8 * math.sqrt(K) * 0.5)
        return out
    if gen == "attention":
        b, h, s, d = size["b"], size["h"], size["s"], size["d"]
        hkv, sk = size.get("hkv", h), size.get("sk", s)
        Q, K, V = g.randn(b, h, s, d), g.randn(b, hkv, sk, d), g.randn(b, hkv, sk, d)
        if case == "qk_40":
            Q, K = Q * 40.0, K * 40.0
        elif case == "qk_4":
            Q, K = Q * 4.0, K * 4.0
        return {"Q": c(Q), "K": c(K), "V": c(V)}
    if gen == "mlp":
        lead, D, H = size["lead"], size["dim"], size["hidden"]
        x = g.randn(*lead, D)
        wg, wu, wd = g.randn(H, D) * 0.02, g.randn(H, D) * 0.02, g.randn(D, H) * 0.02
        if case == "big_act":   # same construction as bench v2 fused_mlp big_act
            x = x * 4.0
            s = 150.0 / (4.0 * math.sqrt(D))
            wg, wu = g.randn(H, D) * s, g.randn(H, D) * s
            wd = g.randn(D, H) * (1e3 / (math.sqrt(H) * 150.0 * 150.0 * 0.7))
        out = {"x": c(x), "w_gate": c(wg), "w_up": c(wu)}
        if ga.get("down", True):
            out["w_down"] = c(wd)
        return out
    raise KeyError(gen)


def cases_for(spec):
    """[(stage, label, size, dtype, case)] = every correctness shape x its dtypes, then adversarial."""
    out = []
    for label, size, dts in spec["sizes"]:
        for dt in dts:
            out.append(("sweep", label, size, DT[dt], "base"))
    for case, label, size, dts in spec.get("adversarial", []):
        for dt in dts:
            out.append(("adversarial", label, size, DT[dt], case))
    return out


def exc_msg(e, head=300, tail=600):
    """Triton CompilationError puts the source excerpt first and the error last: keep both ends."""
    m = f"{type(e).__name__}: {e}"
    return m if len(m) <= head + tail + 5 else m[:head] + " [...] " + m[-tail:]


def exc_info(e):
    files = [f.filename for f in traceback.extract_tb(e.__traceback__)]
    if any(("/triton/compiler/" in f or "/triton/language/" in f) for f in files):
        where = "triton_compile"
    elif any("/triton/" in f for f in files):
        where = "triton_runtime"
    else:
        where = "python"
    return {"exc_type": type(e).__name__, "exc_where": where}


def run_case(kfn, ref, spec, size, dtype, seed, case, timeout=30):
    inp = make_inputs(spec, size, dtype, seed, case)
    gold = golden(ref, spec["family"], inp, dtype)
    tol = tol_for(spec, dtype)
    rec = {"dtype": str(dtype).replace("torch.", ""), "size": size, "case": case}
    signal.signal(signal.SIGALRM, b2._alarm)
    signal.alarm(timeout)
    try:
        with torch.no_grad():
            out = kfn(**inp)
        torch.cuda.synchronize()
        signal.alarm(0)
        rec.update(b2.compare(out, gold, dtype, tol))
    except b2._Alarm:
        rec.update({"ok": False, "reason": "TIMEOUT"})
    except torch.cuda.OutOfMemoryError:
        signal.alarm(0)
        rec.update({"ok": None, "reason": "OOM (skipped)"})
    except Exception as e:  # noqa: BLE001
        signal.alarm(0)
        rec.update({"ok": False, "reason": "EXC " + exc_msg(e), **exc_info(e)})
        if b2._is_fatal_cuda(rec["reason"]):
            rec["fatal"] = True
    finally:
        signal.alarm(0)
    del inp, gold
    return rec


def correctness(kfn, ref, spec, seed, early_abort=True):
    cases = cases_for(spec)
    recs, fatal = [], False
    LAUNCHES[0] = 0
    for i, (stage, label, sz, dt, case) in enumerate(cases):
        r = run_case(kfn, ref, spec, sz, dt, seed + i, case)
        r.update({"stage": stage, "label": label})
        recs.append(r)
        torch.cuda.empty_cache()
        if r.get("fatal"):
            fatal = True
            break
        if early_abort and r["ok"] is False:
            break
    launches = LAUNCHES[0]
    det = None
    if not fatal and recs and all(r["ok"] is not False for r in recs):
        try:
            label, sz, dts = spec["sizes"][0]
            inp = make_inputs(spec, sz, DT[dts[0]], seed + 999)
            with torch.no_grad():
                a = kfn(**inp)
                b = kfn(**inp)
            det = bool(torch.equal(a, b))
        except Exception as e:  # noqa: BLE001
            det = False
            if b2._is_fatal_cuda(str(e)):
                fatal = True
    complete = len(recs) == len(cases)
    ok = complete and all(r["ok"] is not False for r in recs) and det is True and launches > 0
    return {"ok": ok, "determinism": det, "n_cases": len(cases), "n_run": len(recs), "fatal": fatal,
            "triton_launches": launches, "n_fail": sum(1 for r in recs if r["ok"] is False),
            "n_invalid": sum(1 for r in recs if r["ok"] is None), "cases": recs}


def timing(kfn, starter_fn, ref, spec, seed, reps=40, compiled=None):
    res = {}
    for label, size, dt in spec["timing"]:
        inp = make_inputs(spec, size, DT[dt], seed)
        impls = {}
        if kfn is not None:
            impls["kernel"] = lambda: kfn(**inp)
        if starter_fn is not None:
            impls["starter"] = lambda: starter_fn(**inp)
        impls["eager"] = lambda: ref(**inp)
        if compiled is not None:
            impls["compile"] = lambda: compiled(inp)
        try:
            st = b2.time_impls(impls, reps=reps)
        except Exception as e:  # noqa: BLE001
            res[label] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
            if b2._is_fatal_cuda(str(e)):
                res["fatal"] = True
                break
            continue
        e = {"dtype": dt, "size": size}
        for n, s in st.items():
            e[n] = {k: v for k, v in s.items() if k != "_t"}
        if "kernel" in st:
            e["vs_eager"] = b2.paired_ratio(st["eager"]["_t"], st["kernel"]["_t"])
            if "starter" in st:
                e["vs_starter"] = b2.paired_ratio(st["starter"]["_t"], st["kernel"]["_t"])
            if "compile" in st:
                e["vs_compile"] = b2.paired_ratio(st["compile"]["_t"], st["kernel"]["_t"])
        if "starter" in st:
            e["starter_vs_eager"] = b2.paired_ratio(st["eager"]["_t"], st["starter"]["_t"])
            if "compile" in st:
                e["compile_over_starter"] = b2.paired_ratio(st["compile"]["_t"], st["starter"]["_t"])
        res[label] = e
        del inp
        torch.cuda.empty_cache()
    return res


_MODS = {}


def load_mod(code, tag, d):
    key = (tag, hash(code))
    if key in _MODS:
        return _MODS[key]
    name = f"v_{tag}_{abs(hash(code)) % 10**10}"
    p = os.path.join(d, name + ".py")
    with open(p, "w") as f:
        f.write(code)
    mod = b2._load_path(name, p)
    _MODS[key] = mod
    return mod


def run_job(job, d):
    spec = job["spec"]
    seed = int(job.get("seed") or int.from_bytes(os.urandom(4), "little"))
    rec = {"id": job["id"], "problem_id": spec["id"], "family": spec["family"], "seed": seed,
           "harness": HARNESS_VERSION, "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}
    t0 = time.time()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ref = load_mod(job["reference_code"], "ref", d).reference
    starter_fn = None
    if job.get("starter_code") and job.get("timing", True):
        starter_fn = load_mod(job["starter_code"], "starter", d).kernel_fn
    kfn = None
    if job.get("code") is not None:
        try:
            kfn = load_mod(job["code"], "cand", d).kernel_fn
        except Exception as e:  # noqa: BLE001
            rec.update({"verdict": "CRASH", "reason": "import: " + exc_msg(e), **exc_info(e),
                        "tb": traceback.format_exc()[-800:], "wall_s": round(time.time() - t0, 2)})
            return rec
    compiled = None
    if job.get("compile"):
        torch._dynamo.config.cache_size_limit = 256
        compiled = torch.compile(lambda inp: ref(**inp), dynamic=False)
    if kfn is not None:
        c = correctness(kfn, ref, spec, seed, early_abort=job.get("early_abort", True))
        rec["correctness"] = c
        fails = [f"{r['stage']}/{r['label']}/{r['dtype']}: {r.get('reason', '')}" for r in c["cases"] if r["ok"] is False]
        if c["fatal"]:
            rec["verdict"] = "CRASH"
            rec["reason"] = "fatal CUDA error: " + next((r.get("reason", "") for r in c["cases"] if r.get("fatal")), "")
        elif c["cases"] and c["cases"][0]["ok"] is False and c["cases"][0].get("reason", "").startswith("EXC"):
            rec["verdict"] = "CRASH"
            rec["reason"] = c["cases"][0]["reason"]
            rec["exc_type"], rec["exc_where"] = c["cases"][0].get("exc_type"), c["cases"][0].get("exc_where")
        else:
            rec["verdict"] = "PASS" if c["ok"] else "FAIL"
            if c["determinism"] is False:
                fails.append("determinism: two runs on identical inputs differ")
            if c["triton_launches"] == 0 and not fails:
                fails.append("no Triton kernel launched: kernel_fn must run at least one @triton.jit kernel")
        rec["fail_reasons"] = fails[:8]
    want_t = job.get("timing", True) and (kfn is None or rec.get("verdict") == "PASS" or job.get("time_all"))
    if want_t:
        rec["timing"] = timing(kfn, starter_fn, ref, spec, seed + 7, reps=int(job.get("reps", 40)),
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
            rec = {"id": job["id"], "problem_id": job["spec"]["id"], "verdict": "CRASH",
                   "reason": f"harness: {type(e).__name__}: {str(e)[:300]}", "tb": traceback.format_exc()[-1500:]}
        with open(out_path, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        with open(prog_path, "a") as f:
            f.write(f"DONE {job['id']}\n")
        fatal = rec.get("verdict") == "CRASH" and "fatal" in rec.get("reason", "")
        if fatal or (rec.get("timing") or {}).get("fatal"):
            sys.exit(3)
    sys.exit(0)


if __name__ == "__main__":
    main()
