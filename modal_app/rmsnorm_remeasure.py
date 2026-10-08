"""
Independent re-measurement of the "hero" RMSNorm kernel (deck v3).

Kernels compared (all same launch config: 1 program/row, BLOCK_SIZE=next_pow2(N),
Triton default num_warps=4 unless noted):
  starter      results/autokernel_src/kernels/rmsnorm.py   (load -> fp32, sum/normalize in fp32)
  best         presentation/v2/best_kernel.py              (load -> fp16, sum/normalize in fp16)
  fp32acc      written here: fp32 sum of squares + rsqrt, multiply (no divide),
               num_warps heuristic (8 for N>=4096, 16 for N>=16384)
  torch_eager  reference.rmsnorm_ref (what bench.py divides by: pow, mean, add, sqrt, div, mul)
  torch_compile torch.compile(reference.rmsnorm_ref)  (a fair "fused PyTorch" baseline)

Measures: timing grid (CUDA events, warm and cold L2, >=200 reps, sleep-padded so launch
overhead is excluded), triton.testing.do_bench cross-check, real bench.py quick/full runs,
analytic HBM bytes -> GB/s, register/spill counts, numerics vs fp32/fp64 truth.

    cd modal_app && modal run rmsnorm_remeasure.py            # L4 + H100 (2 GPUs in parallel)
    cd modal_app && modal run rmsnorm_remeasure.py --gpus L4
Writes results/v3/rmsnorm_remeasure.json and .md
"""
import json
import pathlib
import time

import modal

from common import AUTOKERNEL_DIR, bench_image

app = modal.App("autokernel-rmsnorm-remeasure")
image = bench_image

ROOT = pathlib.Path(__file__).resolve().parents[1]

FP32ACC_CODE = '''
"""fp32-accumulate RMSNorm (hand-written for the v3 re-measurement)."""
KERNEL_TYPE = "rmsnorm"
import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_kernel(X_ptr, W_ptr, OUT_ptr, M, N, stride_xm, stride_xn, stride_om, stride_on,
                   eps, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(X_ptr + row * stride_xm + offs * stride_xn, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.math.rsqrt(tl.sum(x * x, axis=0) / N + eps)       # fp32 accumulate
    w = tl.load(W_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(OUT_ptr + row * stride_om + offs * stride_on, x * rstd * w, mask=mask)


def kernel_fn(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    assert x.is_cuda
    M, N = x.shape
    out = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(N)
    nw = 16 if BLOCK_SIZE >= 16384 else (8 if BLOCK_SIZE >= 4096 else 4)
    rmsnorm_kernel[(M,)](x, weight, out, M, N, x.stride(0), x.stride(1),
                         out.stride(0), out.stride(1), eps, BLOCK_SIZE=BLOCK_SIZE, num_warps=nw)
    return out
'''

BENCH_SHAPES = [(1024, 768), (4096, 1024), (4096, 4096), (2048, 4096)]   # bench.py test_sizes
GRID_SHAPES = [(m, n) for m in (1024, 4096, 16384) for n in (512, 1024, 4096, 8192)]
PRIMARY = (4096, 4096)   # bench.py primary ("large"), dtype fp16
PEAK_GBPS = {"L4": 300.0, "H100": 3350.0}   # L4 GDDR6 300 GB/s; H100 SXM HBM3 3.35 TB/s
SLEEP_CYCLES = 1_000_000   # ~0.5 ms GPU spin so the CPU enqueues fn fully before it runs


def _load_module(name, code, d):
    import importlib.util
    p = pathlib.Path(d) / f"{name}.py"
    p.write_text(code)
    spec = importlib.util.spec_from_file_location(f"rk_{name}", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _stats(ts):
    import numpy as np
    a = np.asarray(ts, dtype=float)
    med = float(np.median(a))
    return {"median_us": med, "p10_us": float(np.percentile(a, 10)),
            "p90_us": float(np.percentile(a, 90)), "mean_us": float(a.mean()),
            "cv": float(a.std() / a.mean()) if a.mean() > 0 else 0.0, "n": int(a.size)}


@app.cls(image=image, gpu="L4", timeout=3600, cpu=4, memory=32768, max_containers=1)
class Remeasure:
    @modal.method()
    def run(self, codes: dict, reps: int = 200, warmup: int = 25) -> dict:
        import sys
        import shutil
        import subprocess
        import tempfile
        import torch
        import triton
        import triton.testing

        import torch._dynamo  # noqa: F401  (import before anything that could shadow stdlib `profile`)
        # load reference.py by path: /autokernel has its own profile.py that shadows the stdlib
        import importlib.util as _ilu
        _spec = _ilu.spec_from_file_location("ak_reference", f"{AUTOKERNEL_DIR}/reference.py")
        reference = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(reference)

        dev = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        l2_mb = getattr(props, "L2_cache_size", 0) / 2**20
        td = tempfile.mkdtemp()
        mods = {k: _load_module(k, c, td) for k, c in codes.items()}
        fns = {k: m.kernel_fn for k, m in mods.items()}
        torch._dynamo.config.cache_size_limit = 512   # default 8 -> silent eager fallback after 8 shapes
        if hasattr(torch._dynamo.config, "accumulated_cache_size_limit"):
            torch._dynamo.config.accumulated_cache_size_limit = 4096
        compiled_ref = torch.compile(reference.rmsnorm_ref, dynamic=False)
        impls = {**fns,
                 "torch_eager": lambda x, weight: reference.rmsnorm_ref(x, weight),
                 "torch_compile": lambda x, weight: compiled_ref(x, weight)}

        flush = torch.empty(512 * 2**20 // 4, dtype=torch.float32, device=dev)  # 512 MB > any L2

        def time_events(fn, cold):
            for _ in range(warmup):
                fn()
            torch.cuda.synchronize()
            ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                  for _ in range(reps)]
            for s, e in ev:
                if cold:
                    flush.zero_()
                torch.cuda._sleep(SLEEP_CYCLES)
                s.record()
                fn()
                e.record()
            torch.cuda.synchronize()
            return _stats([s.elapsed_time(e) * 1000.0 for s, e in ev])

        def time_wall(fn, n=200):
            fn(); torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(n):
                fn()
            torch.cuda.synchronize()
            return (time.perf_counter() - t0) / n * 1e6

        def gen(M, N, dtype, seed=42):   # identical to bench.py gen_rmsnorm_inputs
            torch.manual_seed(seed)
            return (torch.randn(M, N, device=dev, dtype=dtype),
                    torch.randn(N, device=dev, dtype=dtype))

        # ---------------- 1. timing grid ----------------
        shapes = list(dict.fromkeys(BENCH_SHAPES + GRID_SHAPES))
        timing = []
        for dtype in (torch.float16, torch.bfloat16):
            for (M, N) in shapes:
                x, w = gen(M, N, dtype)
                esz = x.element_size()
                fused_bytes = (2 * M * N + N) * esz
                eager_bytes = (7 * M * N + 2 * N + 6 * M) * esz  # pow r/w, mean r, div r/w, mul r/w
                for name, f in impls.items():
                    fn = (lambda f=f: f(x, w))
                    try:
                        warm = time_events(fn, cold=False)
                        cold = time_events(fn, cold=True)
                        db = triton.testing.do_bench(fn, warmup=25, rep=100)  # rep is ms, flushes L2
                        wall = time_wall(fn)
                        err = None
                    except Exception as ex:  # noqa: BLE001
                        warm = cold = None; db = wall = None; err = f"{type(ex).__name__}: {ex}"[:300]
                    row = {"kernel": name, "M": M, "N": N, "dtype": str(dtype).split(".")[-1],
                           "bench_shape": (M, N) in BENCH_SHAPES,
                           "fused_bytes": fused_bytes, "eager_bytes_analytic": eager_bytes,
                           "warm": warm, "cold": cold, "do_bench_us": None if db is None else db * 1e3,
                           "wall_per_call_us": wall, "error": err}
                    if cold:
                        row["gbps_cold"] = fused_bytes / (cold["median_us"] * 1e-6) / 1e9
                        row["gbps_warm"] = fused_bytes / (warm["median_us"] * 1e-6) / 1e9
                    timing.append(row)
                    print(gpu_name, row["dtype"], M, N, name,
                          None if not warm else round(warm["median_us"], 1),
                          None if not cold else round(cold["median_us"], 1),
                          None if db is None else round(db * 1e3, 1), err or "", flush=True)
                del x, w
                torch.cuda.empty_cache()

        # ---------------- kernel launches: eager op count ----------------
        from torch.profiler import profile, ProfilerActivity
        x, w = gen(*PRIMARY, torch.float16)
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            reference.rmsnorm_ref(x, w); torch.cuda.synchronize()
        eager_kernels = [e.key for e in prof.key_averages() if e.device_type.name == "CUDA"]

        # ---------------- register / spill / ptx info ----------------
        compile_info = {}
        for name, m in mods.items():
            for N in (1024, 4096, 8192):
                xx, ww = gen(64, N, torch.float16)
                out = torch.empty_like(xx)
                BS = triton.next_power_of_2(N)
                nw = 4
                if name == "fp32acc":
                    nw = 16 if BS >= 16384 else (8 if BS >= 4096 else 4)
                k = m.rmsnorm_kernel[(64,)](xx, ww, out, 64, N, xx.stride(0), xx.stride(1),
                                           out.stride(0), out.stride(1), 1e-6,
                                           BLOCK_SIZE=BS, num_warps=nw)
                info = {"num_warps": nw, "BLOCK_SIZE": BS}
                try:
                    k._init_handles()
                except Exception:  # noqa: BLE001
                    pass
                for a in ("n_regs", "n_spills"):
                    info[a] = getattr(k, a, None)
                try:
                    ptx = k.asm["ptx"]
                    info["ptx_ld_global"] = ptx.count("ld.global")
                    info["ptx_st_global"] = ptx.count("st.global")
                    info["ptx_cvt"] = ptx.count("cvt.")
                    info["ptx_fp32_fma_or_mul"] = ptx.count("mul.f32") + ptx.count("fma.rn.f32")
                    info["ptx_f16_ops"] = ptx.count(".f16")
                    info["ptx_ld_vector_widths"] = {v: ptx.count(f"ld.global.v{v}")
                                                    for v in (2, 4)}
                except Exception as ex:  # noqa: BLE001
                    info["ptx_err"] = repr(ex)[:200]
                compile_info[f"{name}_N{N}"] = info

        # ---------------- 3. numerics ----------------
        def truth(x, w, eps=1e-6):
            xd, wd = x.double(), w.double()
            return (xd * torch.rsqrt((xd * xd).mean(-1, keepdim=True) + eps) * wd)

        numerics = []
        tol = {torch.float16: (1e-2, 1e-2), torch.bfloat16: (1e-1, 5e-2)}
        for dtype in (torch.float16, torch.bfloat16):
            scales = [1, 2, 4, 10, 100, 300, 1000] + ([1e5] if dtype == torch.bfloat16 else [])
            for N in (768, 4096, 8192, 16384):
                for s in scales:
                    torch.manual_seed(0)
                    x = (torch.randn(256, N, device=dev, dtype=torch.float32) * s).to(dtype)
                    w = torch.randn(N, device=dev, dtype=dtype)
                    t = truth(x, w)
                    hr = reference.rmsnorm_ref(x, w)  # harness reference (same dtype as input)
                    tmax = t.abs().max().item()
                    rel_mask = t.abs() > 1e-2 * t.abs().mean()
                    for name in ("starter", "best", "fp32acc", "torch_eager", "torch_compile"):
                        o = hr if name == "torch_eager" else impls[name](x, w)
                        of = o.double()
                        nonfinite = int((~torch.isfinite(of)).sum().item())
                        d = (of - t).abs()
                        d = torch.where(torch.isfinite(d), d, torch.full_like(d, float("inf")))
                        max_abs = d.max().item()
                        rel = (d / t.abs().clamp_min(1e-30))[rel_mask]
                        a, r = tol[dtype]
                        ok_truth = bool(torch.allclose(of, t, atol=a, rtol=r))
                        ok_harness_ref = bool(torch.allclose(o.float(), hr.float(), atol=a, rtol=r))
                        zero_rows = int((o.float().abs().amax(-1) == 0).sum().item())
                        numerics.append({
                            "kernel": name, "dtype": str(dtype).split(".")[-1], "N": N, "scale": s,
                            "sumsq_row_mean": float((x.double() ** 2).sum(-1).mean().item()),
                            "max_abs_err": max_abs, "max_abs_err_over_max_truth": max_abs / tmax,
                            "max_rel_err": float(rel.max().item()) if rel.numel() else None,
                            "median_rel_err": float(rel.median().item()) if rel.numel() else None,
                            "nonfinite": nonfinite, "zero_rows": zero_rows, "rows": 256,
                            "allclose_vs_truth_harness_tol": ok_truth,
                            "allclose_vs_harness_ref": ok_harness_ref})

        # harness stability stage replicated (small 1024x768, fp16, relaxed tol x10)
        stab = []
        cases = {
            "near_max": lambda t: t * 60000.0,
            "near_zero": lambda t: t * 1e-6,
            "mixed_scale": lambda t: t * torch.where(torch.rand_like(t.float()).to(t.dtype) > 0.5,
                                                     torch.tensor(1e3, device=t.device, dtype=t.dtype),
                                                     torch.tensor(1e-3, device=t.device, dtype=t.dtype)),
            "all_zeros": lambda t: torch.zeros_like(t),
            "all_same": lambda t: torch.ones_like(t) * 0.5,
        }
        for cname, tf in cases.items():
            x0, w0 = gen(1024, 768, torch.float16)
            x, w = tf(x0), tf(w0)   # bench.py transforms weight too
            hr = reference.rmsnorm_ref(x, w)
            t = truth(x, w)
            hr_bad = bool((~torch.isfinite(hr)).any()) or False
            for name in ("starter", "best", "fp32acc", "torch_compile"):
                o = impls[name](x, w)
                o_bad = bool((~torch.isfinite(o)).any())
                if o_bad and not hr_bad:
                    verdict = "FAIL (NaN/Inf, ref clean)"
                elif o_bad and hr_bad:
                    verdict = "PASS (both NaN/Inf)"
                else:
                    verdict = "PASS" if torch.allclose(o.float(), hr.float(), atol=0.1, rtol=0.1) else "FAIL"
                fin = torch.isfinite(t) & torch.isfinite(o.double())
                stab.append({"case": cname, "kernel": name, "harness_verdict": verdict,
                             "harness_ref_nonfinite": int((~torch.isfinite(hr)).sum().item()),
                             "harness_ref_zero_rows": int((hr.float().abs().amax(-1) == 0).sum().item()),
                             "kernel_zero_rows": int((o.float().abs().amax(-1) == 0).sum().item()),
                             "max_abs_err_vs_truth_finite": float((o.double() - t).abs()[fin].max().item())
                             if fin.any() else None,
                             "max_abs_err_harness_ref_vs_truth_finite":
                                 float((hr.double() - t).abs()[fin].max().item()) if fin.any() else None})

        # ---------------- real bench.py runs (quick + full) ----------------
        harness = []
        for name in ("starter", "best", "fp32acc"):
            for quick in (True, False):
                sb = pathlib.Path(td) / f"ak_{name}_{int(quick)}"
                shutil.copytree(AUTOKERNEL_DIR, sb, ignore=shutil.ignore_patterns(".git", "workspace"))
                (sb / "kernel.py").write_text(codes[name])
                cmd = ["python", "bench.py", "--kernel", "rmsnorm"] + (["--quick"] if quick else [])
                p = subprocess.run(cmd, cwd=sb, capture_output=True, text=True, timeout=600)
                kv = {}
                for line in p.stdout.splitlines():
                    if ":" in line and not line.startswith(" "):
                        k, v = line.split(":", 1)
                        kv[k.strip()] = v.strip()
                fails = [l.strip() for l in p.stdout.splitlines() if "FAIL" in l][:8]
                harness.append({"kernel": name, "quick": quick,
                                "correctness": kv.get("correctness"),
                                "speedup_vs_pytorch": kv.get("speedup_vs_pytorch"),
                                "latency_us": kv.get("latency_us"),
                                "numerical_stability": kv.get("numerical_stability"),
                                "fail_lines": fails,
                                "perf_lines": [l.strip() for l in p.stdout.splitlines()
                                               if "speedup:" in l][:6]})
                print("harness", name, "quick" if quick else "full", kv.get("correctness"),
                      kv.get("speedup_vs_pytorch"), flush=True)

        return {"gpu": gpu_name, "l2_mb": l2_mb, "torch": torch.__version__,
                "triton": triton.__version__, "reps": reps, "warmup": warmup,
                "timing": timing, "eager_cuda_kernels": eager_kernels,
                "compile_info": compile_info, "numerics": numerics, "stability": stab,
                "harness": harness}


# ------------------------------------------------------------------ local analysis
def _pick(rows, **kw):
    for r in rows:
        if all(r.get(k) == v for k, v in kw.items()):
            return r
    return None


def summarize(res: dict, gpu_key: str) -> dict:
    T = res["timing"]
    peak = PEAK_GBPS[gpu_key]
    M, N = PRIMARY
    g = lambda k, mode, dt="float16", m=M, n=N: (_pick(T, kernel=k, M=m, N=n, dtype=dt) or {}).get(mode) or {}
    h = {}
    for mode in ("warm", "cold"):
        ref = g("torch_eager", mode).get("median_us")
        for k in ("starter", "best", "fp32acc", "torch_compile"):
            v = g(k, mode).get("median_us")
            h[f"{k}_speedup_vs_eager_{mode}_median"] = ref / v if ref and v else None
        h[f"best_vs_starter_{mode}_median"] = (g("starter", mode).get("median_us") /
                                               g("best", mode).get("median_us"))
        h[f"best_vs_torch_compile_{mode}_median"] = (g("torch_compile", mode).get("median_us") /
                                                     g("best", mode).get("median_us"))
    db = lambda k: (_pick(T, kernel=k, M=M, N=N, dtype="float16") or {}).get("do_bench_us")
    h["do_bench_best_speedup"] = db("torch_eager") / db("best")
    h["do_bench_starter_speedup"] = db("torch_eager") / db("starter")

    table = []
    for r in T:
        if not r.get("cold"):
            continue
        e = _pick(T, kernel="torch_eager", M=r["M"], N=r["N"], dtype=r["dtype"])
        s = _pick(T, kernel="starter", M=r["M"], N=r["N"], dtype=r["dtype"])
        table.append({
            "kernel": r["kernel"], "M": r["M"], "N": r["N"], "dtype": r["dtype"],
            "bench_shape": r["bench_shape"],
            "warm_median_us": r["warm"]["median_us"], "warm_p10_us": r["warm"]["p10_us"],
            "warm_p90_us": r["warm"]["p90_us"], "warm_cv": r["warm"]["cv"],
            "cold_median_us": r["cold"]["median_us"], "cold_p10_us": r["cold"]["p10_us"],
            "cold_p90_us": r["cold"]["p90_us"], "cold_cv": r["cold"]["cv"],
            "do_bench_us": r["do_bench_us"], "wall_per_call_us": r["wall_per_call_us"],
            "fused_bytes": r["fused_bytes"],
            "gbps_cold": r["gbps_cold"], "gbps_warm": r["gbps_warm"],
            "pct_peak_cold": 100 * r["gbps_cold"] / peak,
            "speedup_vs_eager_cold": e["cold"]["median_us"] / r["cold"]["median_us"] if e and e.get("cold") else None,
            "speedup_vs_eager_warm": e["warm"]["median_us"] / r["warm"]["median_us"] if e and e.get("warm") else None,
            "speedup_vs_starter_cold": s["cold"]["median_us"] / r["cold"]["median_us"] if s and s.get("cold") else None,
        })
    # geo-mean best/starter over all shapes/dtypes
    import math
    ratios = [t["speedup_vs_starter_cold"] for t in table if t["kernel"] == "best" and t["speedup_vs_starter_cold"]]
    h["best_vs_starter_cold_geomean_all_shapes"] = math.exp(sum(map(math.log, ratios)) / len(ratios))
    ratios_w = []
    for t in table:
        if t["kernel"] == "best":
            s = _pick(table, kernel="starter", M=t["M"], N=t["N"], dtype=t["dtype"])
            ratios_w.append(s["warm_median_us"] / t["warm_median_us"])
    h["best_vs_starter_warm_geomean_all_shapes"] = math.exp(sum(map(math.log, ratios_w)) / len(ratios_w))

    prim = {k: _pick(table, kernel=k, M=M, N=N, dtype="float16") for k in
            ("starter", "best", "fp32acc", "torch_eager", "torch_compile")}
    gb = {k: {"gbps_cold": v["gbps_cold"], "pct_peak_cold": v["pct_peak_cold"],
              "gbps_warm": v["gbps_warm"]} for k, v in prim.items() if v}
    # eager effective traffic
    e = prim["torch_eager"]
    eager_bytes = _pick(T, kernel="torch_eager", M=M, N=N, dtype="float16")["eager_bytes_analytic"]
    gb["torch_eager"]["gbps_cold_on_its_own_analytic_bytes"] = eager_bytes / (e["cold_median_us"] * 1e-6) / 1e9
    return {"headline": h, "primary_rows": prim, "gbps_primary": gb, "table": table,
            "peak_gbps_assumed": peak}


def write_outputs(all_res: dict, codes: dict):
    import difflib
    out_dir = ROOT / "results" / "v3"
    out_dir.mkdir(parents=True, exist_ok=True)
    diff = list(difflib.unified_diff(codes["starter"].splitlines(), codes["best"].splitlines(),
                                     "starter", "best", lineterm="", n=0))
    M, N = PRIMARY
    doc = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "primary_shape": {"M": M, "N": N, "dtype": "float16"},
           "original_claim": {"speedup": 2.926, "gpu": "L4", "mode": "full bench", "source": "grpo_v2 step 9"},
           "starter_stage1_L4_quick_speedups": [2.909, 2.885, 2.917],
           "kernel_diff": diff, "fp32acc_code": FP32ACC_CODE, "gpus": {}}
    for gk, res in all_res.items():
        s = summarize(res, gk)
        doc["gpus"][gk] = {"device": res["gpu"], "l2_mb": res["l2_mb"], "torch": res["torch"],
                           "triton": res["triton"], "reps": res["reps"], **s,
                           "eager_cuda_kernels": res["eager_cuda_kernels"],
                           "compile_info": res["compile_info"], "numerics": res["numerics"],
                           "stability_replica": res["stability"], "harness_runs": res["harness"]}
    L = doc["gpus"].get("L4") or next(iter(doc["gpus"].values()))
    H = L["headline"]
    doc["headline_speedup_warm_median"] = H["best_speedup_vs_eager_warm_median"]
    doc["headline_speedup_cold_median"] = H["best_speedup_vs_eager_cold_median"]
    doc["starter_speedup_warm_median"] = H["starter_speedup_vs_eager_warm_median"]
    doc["starter_speedup_cold_median"] = H["starter_speedup_vs_eager_cold_median"]
    doc["best_vs_starter_cold_median"] = H["best_vs_starter_cold_median"]
    doc["best_vs_starter_warm_median"] = H["best_vs_starter_warm_median"]
    doc["best_vs_torch_compile_cold_median"] = H["best_vs_torch_compile_cold_median"]
    doc["gbps"] = {k: v["gbps_cold"] for k, v in L["gbps_primary"].items()}
    doc["pct_peak"] = {k: v["pct_peak_cold"] for k, v in L["gbps_primary"].items()}

    # numerics: first failing scale per (kernel, dtype, N)
    fails = {}
    for r in L["numerics"]:
        key = f'{r["kernel"]}|{r["dtype"]}|N={r["N"]}'
        bad = (not r["allclose_vs_truth_harness_tol"])
        if bad and key not in fails:
            fails[key] = r["scale"]
    doc["numerics_first_failing_scale_vs_truth"] = fails
    doc["numerics_table"] = [{k: r[k] for k in ("kernel", "dtype", "N", "scale", "max_abs_err",
                              "max_rel_err", "nonfinite", "zero_rows", "allclose_vs_truth_harness_tol",
                              "allclose_vs_harness_ref")} for r in L["numerics"]]
    add_verdicts(doc)
    json.dump(doc, open(out_dir / "rmsnorm_remeasure.json", "w"), indent=1, default=str)
    (out_dir / "rmsnorm_remeasure.md").write_text(render_md(doc))
    return doc


def add_verdicts(doc: dict):
    L = doc["gpus"]["L4"]
    H = L["headline"]
    hr = {f'{r["kernel"]}_{"quick" if r["quick"] else "full"}': r for r in L["harness_runs"]}
    stab = {(r["case"], r["kernel"]): r for r in L["stability_replica"]}
    ms_f, ms_b = stab[("mixed_scale", "fp32acc")], stab[("mixed_scale", "best")]
    fail_line = next((l for l in hr["fp32acc_full"]["fail_lines"] if "mixed_scale" in l), "")
    doc["harness_runs_L4"] = {k: {"correctness": v["correctness"], "speedup": v["speedup_vs_pytorch"],
                                  "numerical_stability": v["numerical_stability"]} for k, v in hr.items()}
    doc["harness_fail_reason"] = {
        "failing_stage": "Stage 3 numerical_stability",
        "failing_case": "mixed_scale (fp16, M=1024 N=768; x and weight each multiplied elementwise by 1e3 or 1e-3; relaxed tol atol=0.1 rtol=0.1)",
        "harness_message_starter_and_fp32acc": fail_line,
        "not_the_cause": ["output dtype (all kernels store into torch.empty_like(x), same dtype)",
                          "shape sweep / edge cases / determinism (all PASS for all three kernels)"],
        "mechanism": ("reference.rmsnorm_ref computes x**2 in the input dtype (fp16). |x|>~256 squares to "
                      "inf, mean -> inf, rms -> inf, x/rms -> 0: the reference returns all-zero rows "
                      f"({ms_f['harness_ref_zero_rows']}/1024 rows zero; max err of the reference vs fp64 truth "
                      f"{ms_f['max_abs_err_harness_ref_vs_truth_finite']:.4g}). The fp16 'best' kernel overflows "
                      f"the same way ({ms_b['kernel_zero_rows']}/1024 zero rows) and so matches the broken "
                      "reference -> PASS. Starter/fp32acc accumulate in fp32, return the correct result "
                      f"(max err vs fp64 truth {ms_f['max_abs_err_vs_truth_finite']:.3g}), and are marked FAIL."),
        "consequence": ("The full-bench verifier rewards reproducing the reference's fp16 overflow. Correct "
                        "fp32-accumulate kernels (starter, fp32acc, and torch.compile of the reference) get reward "
                        "0 under the full bench; the only change GRPO found (fp32 -> fp16 casts) flips FAIL -> PASS "
                        "without any speedup."),
    }
    fs = doc["numerics_first_failing_scale_vs_truth"]
    doc["verdicts"] = {
        "speedup": (f"2.9x vs PyTorch is reproducible (L4 fp16 4096x4096: {H['best_speedup_vs_eager_cold_median']:.2f}x cold-L2, "
                    f"{H['best_speedup_vs_eager_warm_median']:.2f}x warm, do_bench {H['do_bench_best_speedup']:.2f}x) but it is "
                    f"the STARTER's speedup ({H['starter_speedup_vs_eager_cold_median']:.2f}x cold). Best vs starter = "
                    f"{H['best_vs_starter_cold_median']:.3f}x at the primary shape, geomean {H['best_vs_starter_cold_geomean_all_shapes']:.3f}x "
                    "over 32 shape/dtype configs: no kernel speedup. Best vs torch.compile(reference) = "
                    f"{H['best_vs_torch_compile_cold_median']:.3f}x."),
        "where_speedup_comes_from": ("Fusion only. PyTorch eager reference launches 6 CUDA kernels (pow, mean, add, sqrt, div, mul) "
                                     "and moves ~7*M*N elements through HBM vs 2*M*N for any fused kernel; 7/2 = 3.5x ideal, "
                                     "2.9x measured on L4 (small shapes are launch-bound). Starter and best have identical "
                                     "launch config (BLOCK_SIZE=next_pow2(N), num_warps=4), identical vectorized 128-bit "
                                     "loads/stores, no spills; the fp32 casts are register-only and do not change HBM bytes."),
        "bandwidth": (f"All fused kernels hit ~{L['gbps_primary']['best']['gbps_cold']:.0f} GB/s = "
                      f"{L['gbps_primary']['best']['pct_peak_cold']:.0f}% of L4's 300 GB/s at 4096x4096 fp16 (cold L2); "
                      f"eager PyTorch {L['gbps_primary']['torch_eager']['gbps_cold']:.0f} GB/s on the fused-byte count "
                      f"({L['gbps_primary']['torch_eager']['gbps_cold_on_its_own_analytic_bytes']:.0f} GB/s on its own 7*M*N traffic, i.e. it is also bandwidth-bound, it just moves 3.5x the bytes)."),
        "quick_vs_full": ("Quick mode = smoke + shape sweep (randn inputs) + perf at 'large' only; stages 3-5 skipped. "
                          "Timing is identical in both modes (same do_bench on 4096x4096 fp16); the difference is the "
                          "mixed_scale stability case, which starter/fp32acc FAIL and best PASSes."),
        "numerics": ("The fp16-accumulate kernel silently returns all-zero rows whenever a row's sum of squares exceeds "
                     "fp16 max 65504, i.e. row RMS > sqrt(65504/N): RMS>4 at N=4096 (first failing scale "
                     f"{fs.get('best|float16|N=4096')}), RMS>2 at N=16384 (scale {fs.get('best|float16|N=16384')}). Same for bf16 inputs, where "
                     "the bf16 PyTorch reference stays correct up to 1e5 scale, and at bf16 scale 1e5 the fp16 cast itself gives inf/NaN. "
                     "fp32acc is correct at every scale tested (max rel err 4.9e-4 fp16 / 3.9e-3 bf16). The harness would "
                     "catch the scale-4 failure (allclose vs its own reference fails) but never tests it: its sweep uses unit "
                     "randn, stability tests fp16 only, and its one large-magnitude case (x*1e3) overflows in the reference too."),
        "one_line": ("The 2.9x is real but was already in the starter (fusion vs 6-kernel eager PyTorch); GRPO's edit "
                     "(fp32->fp16 casts) adds 0% speed and makes RMSNorm return zeros for RMS>4 at N=4096. It was rewarded "
                     "because the full bench's reference overflows identically, so the verifier passes the broken kernel "
                     "and fails the correct one."),
    }


def render_md(doc: dict) -> str:
    L, Hh = doc["gpus"]["L4"], doc["gpus"].get("H100")
    H = L["headline"]
    v = doc["verdicts"]
    rows = []
    for gk, G in doc["gpus"].items():
        for k in ("starter", "best", "fp32acc", "torch_compile", "torch_eager"):
            t = G["primary_rows"][k]
            rows.append(f"| {gk} | {k} | {t['warm_median_us']:.1f} [{t['warm_p10_us']:.1f}, {t['warm_p90_us']:.1f}] | "
                        f"{t['cold_median_us']:.1f} (cv {t['cold_cv']:.3f}) | {t['do_bench_us']:.1f} | "
                        f"{t['gbps_cold']:.0f} ({t['pct_peak_cold']:.0f}%) | {t['speedup_vs_eager_cold']:.2f}x |")
    hr = doc["harness_runs_L4"]
    num = []
    for r in L["numerics"]:
        if r["N"] in (4096, 16384) and r["dtype"] == "float16" and r["scale"] in (1, 2, 4, 100, 1000) \
                and r["kernel"] in ("best", "fp32acc", "torch_eager"):
            num.append(f"| {r['kernel']} | {r['N']} | {r['scale']} | {r['max_abs_err']:.3g} | {r['max_rel_err']:.3g} | "
                       f"{r['zero_rows']}/256 | {'yes' if r['allclose_vs_truth_harness_tol'] else 'NO'} | "
                       f"{'yes' if r['allclose_vs_harness_ref'] else 'NO'} |")
    f = doc["harness_fail_reason"]
    return f"""# RMSNorm hero kernel: independent re-measurement (v3)

Source: `modal_app/rmsnorm_remeasure.py` (L4 + H100, torch {L['torch']}, triton {L['triton']}). Data: `results/v3/rmsnorm_remeasure.json`.
Timing: CUDA events, 25 warmup + {L['reps']} reps, GPU pre-spin so launch overhead is excluded, cold = 512 MB buffer zeroed before each rep (L4 L2 = {L['l2_mb']:.0f} MB).

## Verdict
{v['one_line']}

- **Speedup:** {v['speedup']}
- **Where it comes from:** {v['where_speedup_comes_from']}
- **Bandwidth:** {v['bandwidth']}
- **Quick vs full:** {v['quick_vs_full']}
- **Numerics:** {v['numerics']}

## The entire GRPO edit
```diff
{chr(10).join(l for l in doc['kernel_diff'] if not l.startswith('@@'))}
```

## Primary shape (bench.py 'large': 4096x4096 fp16), latency in us
| GPU | kernel | warm median [p10, p90] | cold median | do_bench | GB/s cold (% peak) | vs eager (cold) |
|---|---|---|---|---|---|---|
{chr(10).join(rows)}

Best/starter over all 32 L4 shape x dtype configs (cold): geomean {H['best_vs_starter_cold_geomean_all_shapes']:.3f}x. H100 geomean {Hh['headline']['best_vs_starter_cold_geomean_all_shapes']:.3f}x.

## Real bench.py runs (L4)
| kernel | quick | full |
|---|---|---|
""" + chr(10).join(f"| {k} | {hr[k+'_quick']['correctness']} {hr[k+'_quick']['speedup']} | {hr[k+'_full']['correctness']} {hr[k+'_full']['speedup']} |"
                for k in ("starter", "best", "fp32acc")) + f"""

## Why the harness fails the correct kernels (`harness_fail_reason`)
- Stage: {f['failing_stage']}; case: {f['failing_case']}
- Message: `{f['harness_message_starter_and_fp32acc']}`
- {f['mechanism']}
- {f['consequence']}

## Numerics vs fp64 truth (L4, fp16 input, 256 rows, x = randn * scale, harness fp16 tol atol=rtol=1e-2)
| kernel | N | scale | max abs err | max rel err | zero rows | ok vs truth | ok vs harness ref |
|---|---|---|---|---|---|---|---|
{chr(10).join(num)}

"Zero rows" = rows where the kernel output is identically 0 (sum of squares overflowed to inf in fp16).
"""


@app.local_entrypoint()
def rebuild():
    """Regenerate json/md from results/v3/rmsnorm_remeasure_raw.json (no GPU)."""
    codes = {"starter": (ROOT / "results/autokernel_src/kernels/rmsnorm.py").read_text(),
             "best": (ROOT / "presentation/v2/best_kernel.py").read_text(),
             "fp32acc": FP32ACC_CODE}
    all_res = json.load(open(ROOT / "results" / "v3" / "rmsnorm_remeasure_raw.json"))
    doc = write_outputs(all_res, codes)
    print(json.dumps(doc["verdicts"], indent=1))


@app.local_entrypoint()
def main(gpus: str = "L4,H100", reps: int = 200):
    codes = {"starter": (ROOT / "results/autokernel_src/kernels/rmsnorm.py").read_text(),
             "best": (ROOT / "presentation/v2/best_kernel.py").read_text(),
             "fp32acc": FP32ACC_CODE}
    calls = {g: Remeasure.with_options(gpu=g)().run.spawn(codes, reps) for g in gpus.split(",")}
    all_res = {g: c.get() for g, c in calls.items()}
    raw = ROOT / "results" / "v3" / "rmsnorm_remeasure_raw.json"
    raw.parent.mkdir(parents=True, exist_ok=True)
    json.dump(all_res, open(raw, "w"), default=str)
    doc = write_outputs(all_res, codes)
    print(json.dumps({k: v for k, v in doc.items() if k.startswith(("headline", "best_vs", "starter_", "gbps", "pct"))},
                     indent=1, default=str))
