"""
Derive the extra numbers the v3 deck needs -> results/v3/v3_numbers.json.

Sources (all already in the repo, nothing re-run):
  * results/grpo/grpo_v1/rollouts/*.jsonl  -> low-variance group audit (PASS/FAIL mix, cache share)
  * results/stage1_baselines.jsonl        -> rmsnorm starter latency / bandwidth, per GPU and mode
  * results/autokernel_src/bench.py       -> rmsnorm test config, L4 spec, timing + stability settings
  * results/autokernel_src/reference.py   -> rmsnorm reference expression
  * modal_app/common.py, results/grpo_smoke.log -> torch / triton versions

    python3 presentation/v3/make_v3_numbers.py
"""
import glob
import json
import pathlib
import re
import statistics as st
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[2]
R = ROOT / "results"
OUT = R / "v3" / "v3_numbers.json"


def lowvar_audit(run="grpo_v1", lo=0.0, hi=0.02):
    groups = mixed = allpass = allfail = multi_sha = recs = cached = 0
    for f in sorted(glob.glob(str(R / "grpo" / run / "rollouts" / "step_*.jsonl"))):
        g = defaultdict(list)
        for line in open(f):
            r = json.loads(line)
            g[r["task"]["task_id"]].append(r)
        for rs in g.values():
            sd = st.pstdev([r["reward"] for r in rs])
            if not (lo < sd < hi):
                continue
            groups += 1
            ap = [bool(r["any_pass"]) for r in rs]
            mixed += any(ap) and not all(ap)
            allpass += all(ap)
            allfail += not any(ap)
            shas = set()
            for r in rs:
                pt = [t for t in r["turns"] if t["correctness"] == "PASS"]
                if not pt:
                    continue
                b = max(pt, key=lambda t: t["speedup"])
                recs += 1
                cached += bool(b.get("cached"))
                shas.add(b["code_sha"])
            multi_sha += len(shas) > 1
    return {"run": run, "definition": f"{lo} < pstdev(group reward) < {hi}", "groups": groups,
            "mixed_pass_fail": mixed, "all_pass": allpass, "all_fail": allfail,
            "multiple_best_pass_code_sha": multi_sha, "best_pass_records": recs,
            "best_pass_records_cached": cached}


def stage1_rmsnorm():
    out = {}
    for line in open(R / "stage1_baselines.jsonl"):
        r = json.loads(line)
        if r["kernel_type"] != "rmsnorm":
            continue
        key = f'{r["gpu_req"]}_{"quick" if r["quick"] else "full"}'
        out.setdefault(key, []).append({k: r[k] for k in (
            "rep", "correctness", "speedup_vs_pytorch", "latency_us", "pytorch_latency_us",
            "pct_peak_bandwidth", "cached", "stages", "gpu")})
    return out


def _edge_block(src):
    b = src[src.index('"rmsnorm": {'):]
    b = b[b.index('"edge_sizes"'):]
    return b[:b.index("]")]


def bench_config():
    src = (R / "autokernel_src" / "bench.py").read_text()
    lines = src.splitlines()
    blk = src[src.index('"rmsnorm": {'):]
    blk = blk[:blk.index('"input_generator"')]
    sizes = [{"label": m[0], "M": int(m[1]), "N": int(m[2])}
             for m in re.findall(r'\("(\w+)",\s*\{"M":\s*(\d+),\s*"N":\s*(\d+)\}\)', blk)]
    dtypes = re.findall(r"torch\.(float16|bfloat16|float32)", blk.split('"tolerances"')[0].split('"test_dtypes"')[1])
    tols = {m[0]: {"atol": float(m[1]), "rtol": float(m[2])} for m in re.findall(
        r'torch\.(\w+):\s*\{"atol":\s*([\d.e-]+),\s*"rtol":\s*([\d.e-]+)\}', blk)}
    edge = [{"label": m[0], "M": int(m[1]), "N": int(m[2])}
            for m in re.findall(r'\("(edge_\w+)",\s*\{"M":\s*(\d+),\s*"N":\s*(\d+)\}\)',
                                _edge_block(src))]
    l4 = re.search(r'"L4":\s*\(([\d.]+),\s*([\d.]+),\s*([\d.]+)\)', src)
    h100sxm = re.search(r'"H100 SXM":\s*\(([\d.]+),\s*([\d.]+),\s*([\d.]+)\)', src)
    h100fb = re.search(r'"H100":\s*\(([\d.]+),\s*([\d.]+),\s*([\d.]+)\)', src)
    db = re.search(r"def _do_bench\(fn: Callable, warmup: int = (\d+), rep: int = (\d+)\)", src)
    relax = re.search(r'relaxed_atol = tol\["atol"\] \* (\d+)', src)
    nan_rule = next(i + 1 for i, l in enumerate(lines) if "Both have NaN/Inf -- acceptable" in l)
    stab = re.search(r'if label == "(\w+)":\s*\n\s*stab_size = sz', src)
    near_max = re.search(r'\("near_max", lambda t: t \* ([\d.]+)', src)
    mixed = re.search(r'torch\.tensor\((1e\d+)[^)]*\),\s*\n\s*torch\.tensor\((1e-\d+)', src)
    ref = (R / "autokernel_src" / "reference.py").read_text()
    ref_rms = re.search(r"def rmsnorm_ref.*?\n(.*?)\n(.*?)\n", ref.split("def rmsnorm_ref")[0] + "def rmsnorm_ref" +
                        ref.split("def rmsnorm_ref")[1], re.S)
    ref_lines = [l.strip() for l in ref.split("def rmsnorm_ref")[1].splitlines()[1:5] if l.strip() and not l.strip().startswith('"""')]
    return {
        "source": "results/autokernel_src/bench.py (AutoKernel pinned commit)",
        "rmsnorm_test_sizes": sizes, "rmsnorm_edge_sizes": edge, "rmsnorm_test_dtypes": dtypes,
        "rmsnorm_tolerances": tols,
        "rmsnorm_bytes_fn": "(2*M*N + N) * element_size",
        "timed_size_label": "large", "timed_dtype": dtypes[0] if dtypes else None,
        "L4_spec": {"peak_fp16_tflops": float(l4[1]), "peak_bandwidth_gb_s": float(l4[2]), "l2_cache_mb": float(l4[3])},
        "H100_SXM_spec_bw_gb_s": float(h100sxm[2]), "H100_fallback_spec_bw_gb_s": float(h100fb[2]),
        "do_bench_warmup_arg": int(db[1]), "do_bench_rep_arg": int(db[2]),
        "stability_tolerance_relax_factor": int(relax[1]), "stability_size_label": stab[1] if stab else None,
        "stability_near_max_fp16_scale": float(near_max[1]),
        "stability_mixed_scale": [float(mixed[1]), float(mixed[2])] if mixed else None,
        "both_nan_inf_accepted_line": nan_rule,
        "rmsnorm_reference": ref_lines,
    }


def winner_turn():
    bk = json.loads((R / "extra_numbers.json").read_text())["best_kernel"]
    f = R / "grpo" / bk["source"] / "rollouts" / f'step_{bk["step"]:03d}.jsonl'
    for line in open(f):
        r = json.loads(line)
        if r["task"]["kernel_type"] == bk["kernel_type"] and r["sample"] == bk["sample"]:
            t = r["turns"][bk["turn"]]
            assert t["code_sha"] == bk["code_sha"]
            return {"file": str(f.relative_to(ROOT)), **{k: t[k] for k in (
                "turn", "correctness", "speedup", "cached", "bench_wall_s", "code_sha", "stages")}}


def versions():
    common = (ROOT / "modal_app" / "common.py").read_text()
    torch_v = re.search(r'"torch==([\d.]+)"', common)[1]
    trit = re.search(r"triton==([\d.]+)", (R / "grpo_smoke.log").read_text())
    return {"torch": torch_v, "triton": trit[1] if trit else None,
            "autokernel_commit": re.search(r'AUTOKERNEL_COMMIT = "(\w+)"', common)[1][:12]}


if __name__ == "__main__":
    out = {
        "generated_by": "presentation/v3/make_v3_numbers.py",
        "lowvar_audit_grpo_v1": lowvar_audit("grpo_v1"),
        "stage1_rmsnorm": stage1_rmsnorm(),
        "bench_config": bench_config(),
        "versions": versions(),
        "winner_turn": winner_turn(),
        "library_facts": {
            "note": "read from Triton 3.2 source (triton/testing.py, language/standard.py); not re-run here",
            "do_bench_args_are_ms_budgets": True,
            "do_bench_default_return_mode": "mean",
            "do_bench_l2_flush_buffer_mb": 256,
            "tl_sum_widens_fp16": False,
            "fp16_max": 65504.0,
            "triton_default_num_warps": 4,
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    a = out["lowvar_audit_grpo_v1"]
    print("wrote", OUT, "| lowvar", a["groups"], "mixed", a["mixed_pass_fail"], "cached",
          a["best_pass_records_cached"], "/", a["best_pass_records"])
    print(json.dumps(out["bench_config"], indent=None)[:1500])
    print(out["versions"])
