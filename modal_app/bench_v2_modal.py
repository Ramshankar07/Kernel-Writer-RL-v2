"""
bench v2 Modal app (`autokernel-bench-v2`), alongside the untouched v1 app `autokernel-bench`.

Harness logic lives in bench_v2_core.py (run as a subprocess per batch; restarted after a
CUDA fault / hang). No modal.Dict cache: every evaluation draws fresh inputs from a new seed.
GPU: L4 (same as every GRPO / agent_eval bench call), <= BENCH_V2_MAX_CONTAINERS (default 4).

    cd modal_app
    modal deploy bench_v2_modal.py
    modal run bench_v2_modal.py::smoke
    modal run bench_v2_modal.py::sanity      # -> results/v3/bench_v2_sanity.jsonl
    modal run bench_v2_modal.py::rescore     # -> results/v3/rescore_bench_v2.jsonl
    python bench_v2_modal.py report          # -> rescore_summary.json, bench_v2.md (no GPU)
"""
import hashlib
import json
import os
import pathlib
import sys

import modal

from common import AUTOKERNEL_DIR, KERNELS, bench_base_image

MAX_CONTAINERS = int(os.environ.get("BENCH_V2_MAX_CONTAINERS", "4"))
GPU = os.environ.get("BENCH_V2_GPU", "L4")

app = modal.App("autokernel-bench-v2")
image = bench_base_image.add_local_python_source("common", "bench_v2_core")

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "v3"


def _run_jobs(jobs, job_timeout=420):
    import importlib.util
    import subprocess
    import tempfile
    import time
    core = importlib.util.find_spec("bench_v2_core").origin
    td = tempfile.mkdtemp()
    results, remaining, attempt = {}, list(jobs), 0
    while remaining:
        attempt += 1
        jp, op, pp, lp = (os.path.join(td, f"{n}{attempt}") for n in ("jobs", "out", "prog", "log"))
        json.dump(remaining, open(jp, "w"))
        open(pp, "w").close()
        with open(lp, "w") as log:
            proc = subprocess.Popen([sys.executable, core, jp, op, pp], cwd=td, stdout=log,
                                    stderr=subprocess.STDOUT, env={**os.environ, "AUTOKERNEL_DIR": AUTOKERNEL_DIR})
            cur, cur_t, killed = None, time.time(), False
            while proc.poll() is None:
                time.sleep(0.5)
                lines = open(pp).read().split()
                starts = [lines[i + 1] for i in range(0, len(lines) - 1, 2) if lines[i] == "START"]
                dones = {lines[i + 1] for i in range(0, len(lines) - 1, 2) if lines[i] == "DONE"}
                c = starts[-1] if starts and starts[-1] not in dones else None
                if c != cur:
                    cur, cur_t = c, time.time()
                if cur and time.time() - cur_t > job_timeout:
                    proc.kill()
                    killed = True
                    break
            proc.wait()
        if os.path.exists(op):
            for line in open(op):
                r = json.loads(line)
                results[r["id"]] = r
        lines = open(pp).read().split()
        starts = [lines[i + 1] for i in range(0, len(lines) - 1, 2) if lines[i] == "START"]
        tail = open(lp).read()[-1500:]
        for s in starts:
            if s not in results:
                kt = next(j["kernel_type"] for j in remaining if j["id"] == s)
                results[s] = {"id": s, "kernel_type": kt, "verdict": "TIMEOUT" if killed else "CRASH",
                              "reason": ("hang > %ds" % job_timeout) if killed else "process died: " + tail[-400:]}
        before = len(remaining)
        remaining = [j for j in remaining if j["id"] not in results]
        if len(remaining) == before:  # runner died before starting anything
            for j in remaining:
                results[j["id"]] = {"id": j["id"], "kernel_type": j["kernel_type"], "verdict": "CRASH",
                                    "reason": "runner failed to start: " + tail[-400:]}
            remaining = []
    return [results[j["id"]] for j in jobs]


def _old_bench(kernel_type: str, code: str, timeout: int = 600) -> dict:
    """Upstream bench.py full mode, run exactly as v1 does (no cache read or write)."""
    import re
    import shutil
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        sb = os.path.join(td, "autokernel")
        shutil.copytree(AUTOKERNEL_DIR, sb, ignore=shutil.ignore_patterns(".git", "workspace"))
        open(os.path.join(sb, "kernel.py"), "w").write(code)
        try:
            p = subprocess.run([sys.executable, "bench.py", "--kernel", kernel_type], cwd=sb,
                               capture_output=True, text=True, timeout=timeout)
            txt = p.stdout
        except subprocess.TimeoutExpired:
            return {"old_verdict": "TIMEOUT"}
    kv = {}
    for line in txt.splitlines():
        m = re.match(r"^([a-z_]+):\s*(.+?)\s*$", line.strip())
        if m:
            kv[m.group(1)] = m.group(2)
    corr = kv.get("correctness", "").upper()
    v = "PASS" if corr == "PASS" else ("CRASH" if not kv or "Failed to import" in txt else "FAIL")
    sp = kv.get("speedup_vs_pytorch", "0").rstrip("x")
    fails = [l.strip() for l in txt.splitlines() if "FAIL" in l][:4]
    return {"old_verdict": v, "old_speedup_vs_pytorch": float(sp) if v == "PASS" else 0.0,
            "old_stages": {k: kv.get(k) for k in ("smoke_test", "shape_sweep", "numerical_stability",
                                                   "determinism", "edge_cases")},
            "old_fail_lines": fails}


@app.cls(image=image, gpu=GPU, timeout=3600, cpu=4, memory=32768,
         max_containers=MAX_CONTAINERS, scaledown_window=60)
class BenchV2:
    @modal.method()
    def run_batch(self, jobs: list) -> list:
        return _run_jobs(jobs)

    @modal.method()
    def old_bench(self, kernel_type: str, code: str) -> dict:
        return _old_bench(kernel_type, code)


# =============================================================================================
# local side: job construction
# =============================================================================================
def starter_code(kt):
    return (ROOT / "results" / "autokernel_src" / "kernels" / f"{kt}.py").read_text()


SIG = {"matmul": "A, B", "softmax": "x", "layernorm": "x, weight, bias", "rmsnorm": "x, weight",
       "flash_attention": "Q, K, V", "fused_mlp": "x, w_gate, w_up, w_down",
       "cross_entropy": "logits, targets", "rotary_embedding": "x, cos, sin", "reduce": "x"}
REFN = {"matmul": "matmul_ref", "softmax": "softmax_ref", "layernorm": "layernorm_ref",
        "rmsnorm": "rmsnorm_ref", "flash_attention": "flash_attention_ref", "fused_mlp": "fused_mlp_ref",
        "cross_entropy": "cross_entropy_ref", "rotary_embedding": "rotary_embedding_ref",
        "reduce": "reduce_sum_ref"}
HP = {"matmul": "float32", "fused_mlp": "float32", "flash_attention": "float32"}


def control_code(kt: str, kind: str) -> str:
    """Synthetic controls. native = upstream reference in the input dtype; golden = reference on
    fp32/fp64-upcast inputs, cast back; zeros; perturbed = golden * 1.05."""
    args = SIG[kt]
    names = [a.strip() for a in args.split(",")]
    hp = HP.get(kt, "float64")
    up = ", ".join(f"({n}.to(torch.{hp}) if {n}.is_floating_point() else {n})" for n in names)
    dt = names[0]
    head = (f'KERNEL_TYPE = "{kt}"\nimport importlib.util, torch, torch.nn.functional as F\n'
            f'_s = importlib.util.spec_from_file_location("akref_c", "{AUTOKERNEL_DIR}/reference.py")\n'
            f'ref = importlib.util.module_from_spec(_s); _s.loader.exec_module(ref)\n')
    if kind == "native":
        body = f"    return ref.{REFN[kt]}({args})\n"
    elif kind == "golden":
        body = (f"    torch.backends.cuda.matmul.allow_tf32 = False\n"
                f"    return ref.{REFN[kt]}({up}).to({dt}.dtype)\n")
    elif kind == "zeros":
        body = f"    return torch.zeros_like(ref.{REFN[kt]}({up}).to({dt}.dtype))\n"
    elif kind == "perturbed":
        body = f"    return (ref.{REFN[kt]}({up}) * 1.05).to({dt}.dtype)\n"
    elif kind == "noaffine":  # layernorm that ignores weight/bias (upstream tests weight=1, bias=0)
        body = "    return F.layer_norm(x.float(), x.shape[-1:]).to(x.dtype)\n"
    else:
        raise KeyError(kind)
    return head + f"def kernel_fn({args}, **kw):\n" + body


def sanity_kernels():
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from rmsnorm_remeasure import FP32ACC_CODE
    best = (ROOT / "presentation" / "v2" / "best_kernel.py").read_text()
    ks = []
    for kt in KERNELS:
        ks.append((f"{kt}:starter", kt, starter_code(kt), "PASS if starter is correct"))
        ks.append((f"{kt}:golden_cast", kt, control_code(kt, "golden"), "PASS (tolerance calibration)"))
        ks.append((f"{kt}:upstream_ref_native", kt, control_code(kt, "native"), "FAIL iff reference has a low-precision bug"))
        ks.append((f"{kt}:zeros", kt, control_code(kt, "zeros"), "FAIL"))
        ks.append((f"{kt}:perturbed_5pct", kt, control_code(kt, "perturbed"), "FAIL"))
    ks.append(("rmsnorm:fp32acc", "rmsnorm", FP32ACC_CODE, "PASS"))
    ks.append(("rmsnorm:grpo_best", "rmsnorm", best, "FAIL"))
    ks.append(("layernorm:ignores_weight_bias", "layernorm", control_code("layernorm", "noaffine"), "FAIL"))
    return ks


def _batches(jobs, n):
    by = {}
    for j in jobs:
        by.setdefault(j["kernel_type"], []).append(j)
    out = []
    for kt, js in by.items():
        for i in range(0, len(js), n):
            out.append(js[i:i + n])
    return out


def _map(batches):
    res = []
    for r in BenchV2().run_batch.map(batches, return_exceptions=True):
        if isinstance(r, Exception):
            print("batch error:", repr(r)[:300], flush=True)
            continue
        res += r
        for x in r:
            print(x["id"], x.get("verdict"), round(x.get("wall_s", 0), 1), flush=True)
    return res


def baseline_jobs(kts, repeats=3):
    return [{"id": f"baseline:{kt}:{i}", "kernel_type": kt, "code": None, "starter_code": starter_code(kt),
             "compile": True, "timing": True, "reps": 60} for kt in kts for i in range(repeats)]


@app.local_entrypoint()
def smoke():
    jobs = [{"id": "rmsnorm:starter", "kernel_type": "rmsnorm", "code": starter_code("rmsnorm"),
             "starter_code": starter_code("rmsnorm"), "ref_native": True},
            {"id": "rmsnorm:best", "kernel_type": "rmsnorm",
             "code": (ROOT / "presentation" / "v2" / "best_kernel.py").read_text(),
             "starter_code": starter_code("rmsnorm")}]
    for r in BenchV2().run_batch.remote(jobs):
        print(r["id"], r.get("verdict"), r.get("wall_s"), r.get("fail_reasons"), r.get("reason"))
        for lab, t in (r.get("timing") or {}).items():
            if isinstance(t, dict) and "vs_starter" in t:
                print("  ", lab, "vs_starter", t["vs_starter"], "vs_eager", t["vs_eager"],
                      "cv", t["kernel"]["cv"], t["starter"]["cv"])


@app.local_entrypoint()
def sanity(old: bool = True):
    ks = sanity_kernels()
    jobs = [{"id": i, "kernel_type": kt, "code": code, "starter_code": starter_code(kt),
             "ref_native": i.endswith(":golden_cast"), "early_abort": not i.endswith(":golden_cast"),
             "timing": i.endswith((":starter", ":fp32acc", ":grpo_best"))} for i, kt, code, _ in ks]
    jobs += baseline_jobs(KERNELS)
    res = _map(_batches(jobs, 6))
    olds = {}
    if old:  # upstream full bench on the controls whose v1 verdict isn't already on disk
        want = [(i, kt, code) for i, kt, code, _ in ks
                if not i.endswith((":starter", ":fp32acc", ":grpo_best", ":upstream_ref_native"))]
        for (i, _, _), o in zip(want, BenchV2().old_bench.starmap([(kt, c) for _, kt, c in want],
                                                                    return_exceptions=True)):
            olds[i] = o if isinstance(o, dict) else {"old_verdict": "INFRA_ERROR", "err": repr(o)[:200]}
            print("old", i, olds[i].get("old_verdict"), flush=True)
    elif (OUT / "bench_v2_sanity.jsonl").exists():  # keep v1 verdicts from the previous sanity run
        for line in open(OUT / "bench_v2_sanity.jsonl"):
            r = json.loads(line)
            if r.get("old_verdict"):
                olds[r["id"]] = {k: v for k, v in r.items() if k.startswith("old_")}
    exp = {i: e for i, _, _, e in ks}
    with (OUT / "bench_v2_sanity.jsonl").open("w") as f:
        for r in res:
            r["expected"] = exp.get(r["id"], "")
            r.update(olds.get(r["id"], {}))
            f.write(json.dumps(r, default=str) + "\n")
    print("wrote", OUT / "bench_v2_sanity.jsonl")


def load_pass_kernels():
    import glob
    files = sorted(glob.glob(str(ROOT / "results/grpo/*/rollouts/*.jsonl")))
    files += sorted(glob.glob(str(ROOT / "results/agent_eval/autokernel_*.jsonl")))
    ks = {}
    for fp in files:
        p = pathlib.Path(fp)
        src = p.parts[-3] if "grpo" in p.parts else "agent_eval_" + p.stem.split("_")[-1]
        step = p.stem if "grpo" in p.parts else ""
        for line in open(fp):
            r = json.loads(line)
            if r.get("task", {}).get("suite") != "autokernel":
                continue
            kt = r["task"]["kernel_type"]
            for t in r.get("turns", []):
                if t.get("correctness") != "PASS" or not t.get("code"):
                    continue
                h = hashlib.sha256(t["code"].encode()).hexdigest()
                e = ks.setdefault(h, {"sha": h, "kernel_type": kt, "code": t["code"], "sources": set(),
                                      "n_occ": 0, "old_speedups": [], "first": f"{src}/{step}/s{r.get('sample')}/t{t.get('turn')}"})
                e["sources"].add(src)
                e["n_occ"] += 1
                e["old_speedups"].append(t.get("speedup", 0.0))
    return ks


@app.local_entrypoint()
def rescore(limit: int = 0, batch: int = 10):
    ks = load_pass_kernels()
    items = sorted(ks.values(), key=lambda e: (e["kernel_type"], e["sha"]))
    if limit:
        items = items[:limit]
    kts = sorted({e["kernel_type"] for e in items})
    print(len(items), "distinct PASS kernels", kts, flush=True)
    jobs = [{"id": e["sha"][:16], "kernel_type": e["kernel_type"], "code": e["code"],
             "starter_code": starter_code(e["kernel_type"]), "timing": True, "reps": 60} for e in items]
    jobs += baseline_jobs(kts)
    res = _map(_batches(jobs, batch))
    by = {r["id"]: r for r in res}
    with (OUT / "rescore_bench_v2.jsonl").open("w") as f:
        for e in items:
            r = by.get(e["sha"][:16], {"verdict": "INFRA_ERROR"})
            f.write(json.dumps({"sha": e["sha"], "code_sha12": e["sha"][:12], "kernel_type": e["kernel_type"],
                                "sources": sorted(e["sources"]), "n_occurrences": e["n_occ"], "first_seen": e["first"],
                                "old_verdict": "PASS", "old_speedup_vs_pytorch_median": _med(e["old_speedups"]),
                                **{k: v for k, v in r.items() if k not in ("id", "kernel_type")}}, default=str) + "\n")
    with (OUT / "bench_v2_baselines.jsonl").open("w") as f:
        for r in res:
            if r["id"].startswith("baseline:"):
                f.write(json.dumps(r, default=str) + "\n")
    print("wrote", OUT / "rescore_bench_v2.jsonl")


def _med(xs):
    xs = sorted(x for x in xs if x is not None)
    return xs[len(xs) // 2] if xs else None


if __name__ == "__main__":
    import bench_v2_report
    bench_v2_report.main(sys.argv[1:])
