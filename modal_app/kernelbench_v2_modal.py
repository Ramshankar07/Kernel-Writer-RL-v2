"""
KernelBench v1 Level 1 on Modal -- fixed harness ("kb v2"), app `autokernel-kernelbench-v2`.

The old app (kernelbench_modal.py, `autokernel-kernelbench`) is left untouched. This one keeps the
same pinned problem cache (AutoKernel bridge fetch from HF) but replaces bench_kb.py with
modal_app/kb_v2_eval.py (seeded/copied weights, GPU inputs outside the timeout, fp32 reference,
hidden fresh-seed trial, CUDA-event timing with cold L2). See results/v3/kb_v2.md.

    cd modal_app
    modal run kernelbench_v2_modal.py::sanity              # identity x2 + starters, all 100
    modal run kernelbench_v2_modal.py::ablate              # per-bug attribution runs
    modal deploy kernelbench_v2_modal.py                   # KBV2.run(problem_id, code) for evals
"""
import json
import os
import re
import subprocess
import sys
import tempfile
import time

import modal

from common import AUTOKERNEL_DIR, bench_base_image, bench_cache

EVAL = "/kbv2/kb_v2_eval.py"
MAX_CONTAINERS = 4   # shared 10-GPU account cap; another agent runs AutoKernel bench in parallel

kb_image = (
    bench_base_image.pip_install("datasets", "ninja")
    .run_commands(f"cd {AUTOKERNEL_DIR} && python kernelbench/bridge.py fetch --source hf --level 1")
    .add_local_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), "kb_v2_eval.py"), EVAL)
    .add_local_python_source("common")
)

app = modal.App("autokernel-kernelbench-v2")
CACHE_ROOT = os.path.join(AUTOKERNEL_DIR, "workspace", "kb_cache", "level1")
_SUPER = re.compile(r"super\(\s*Model\s*,\s*self\s*\)")
_MARK = "# Optimized implementation (EDIT THIS)"


def _starters(problem_id: int) -> tuple:
    """(bridge starter as upstream generates it, fixed starter)."""
    sys.path.insert(0, os.path.join(AUTOKERNEL_DIR, "kernelbench"))
    from bridge import KernelBenchProblem
    buggy = KernelBenchProblem.load_from_cache(1, problem_id).generate_starter(backend="triton")
    head, sep, tail = buggy.partition(_MARK)
    fixed = head + sep + _SUPER.sub("super(ModelNew, self)", tail)
    return buggy, fixed


def _eval(ref_src: str, cands: dict, timeout: int = 900, **flags) -> dict:
    with tempfile.TemporaryDirectory() as td:
        ref = os.path.join(td, "reference.py")
        open(ref, "w").write(ref_src)
        cmd = [sys.executable, EVAL, "--ref", ref, "--out", os.path.join(td, "out.json")]
        for name, src in cands.items():
            p = os.path.join(td, f"cand_{name}.py")
            open(p, "w").write(src)
            cmd += ["--cand", f"{name}={p}"]
        for k, v in flags.items():
            if v is True:
                cmd.append("--" + k.replace("_", "-"))
            elif v not in (None, False):
                cmd += ["--" + k.replace("_", "-"), str(v)]
        t0 = time.time()
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            tail = (p.stdout + p.stderr)[-2000:]
        except subprocess.TimeoutExpired:
            tail = "subprocess timeout"
        wall = round(time.time() - t0, 2)
        try:
            out = json.load(open(os.path.join(td, "out.json")))
        except Exception:
            out = {"error": "no result", "cands": {}}
        # a subprocess killed mid-way (e.g. segfault / illegal address) leaves PENDING entries
        for name in cands:
            r = out["cands"].setdefault(name, {"status": "CRASH", "reason": "no result"})
            if r.get("status") == "PENDING":
                r.update(status="CRASH", reason="process died: " + tail[-400:])
        out["wall_s"] = wall
        if out.get("error"):
            out["log_tail"] = tail
        return out


@app.cls(image=kb_image, gpu="H100", cpu=8, memory=32768, timeout=3600,
         max_containers=MAX_CONTAINERS, scaledown_window=60)
class KBV2:
    @modal.method()
    def list_problems(self) -> list:
        out = []
        for fn in sorted((f for f in os.listdir(CACHE_ROOT) if f.endswith(".py")),
                         key=lambda s: int(s.split(".")[0])):
            meta = json.load(open(os.path.join(CACHE_ROOT, fn[:-3] + ".json")))
            out.append({"problem_id": int(fn[:-3]), "name": meta.get("name", ""),
                        "code": open(os.path.join(CACHE_ROOT, fn)).read()})
        return out

    def _ref(self, problem_id: int) -> str:
        return open(os.path.join(CACHE_ROOT, f"{problem_id}.py")).read()

    @modal.method()
    def run(self, problem_id: int, code: str | None = None, use_cache: bool = True) -> dict:
        """Evaluate one ModelNew source (None -> identity) on problem `problem_id`."""
        import hashlib
        import torch
        gpu = torch.cuda.get_device_name(0)
        ref = self._ref(problem_id)
        if code is None:
            code, use_cache = ref + "\n\nclass ModelNew(Model):\n    pass\n", False
        key = hashlib.sha256(f"kbv2\0{gpu}\0{problem_id}\0{code}".encode()).hexdigest()
        if use_cache and (hit := bench_cache.get(key)) is not None:
            return {**hit, "cached": True}
        out = _eval(ref, {"kernel": code})
        r = {"problem_id": problem_id, "gpu": gpu, **out["cands"]["kernel"],
             "wall_s": out["wall_s"], "ref_ms": out.get("ref_timing", {}).get("median_ms")}
        if use_cache and r["status"] != "TIMEOUT":
            bench_cache[key] = r
        return {**r, "cached": False}

    @modal.method()
    def sanity_one(self, problem_id: int) -> dict:
        """Identity twice (independent processes) + bridge starter + fixed starter."""
        import torch
        ref = self._ref(problem_id)
        ident = ref + "\n\nclass ModelNew(Model):\n    pass\n"
        buggy, fixed = _starters(problem_id)
        r1 = _eval(ref, {"identity": ident, "starter_fixed": fixed, "starter_bridge": buggy})
        r2 = _eval(ref, {"identity": ident})
        return {"problem_id": problem_id, "gpu": torch.cuda.get_device_name(0), "run1": r1, "run2": r2}

    @modal.method()
    def ablate_one(self, problem_id: int, weight_sync: str, input_mode: str) -> dict:
        ref = self._ref(problem_id)
        out = _eval(ref, {"identity": ref + "\n\nclass ModelNew(Model):\n    pass\n"},
                    weight_sync=weight_sync, input_mode=input_mode, no_perf=True)
        return {"problem_id": problem_id, "weight_sync": weight_sync, "input_mode": input_mode, **out}


def _results_dir():
    import pathlib
    d = pathlib.Path(__file__).resolve().parents[1] / "results" / "v3"
    d.mkdir(parents=True, exist_ok=True)
    return d


@app.local_entrypoint()
def sanity(first: int = 1, last: int = 100, out: str = "kb_v2_sanity.jsonl"):
    B = KBV2()
    ids = [p["problem_id"] for p in B.list_problems.remote() if first <= p["problem_id"] <= last]
    with (_results_dir() / out).open("w") as f:
        for r in B.sanity_one.map(ids, return_exceptions=True):
            if isinstance(r, Exception):
                r = {"problem_id": None, "error": repr(r)}
            f.write(json.dumps(r) + "\n")
            f.flush()
            s = lambda run, c: run.get("cands", {}).get(c, {}).get("status")
            print("kbv2", r.get("problem_id"), s(r.get("run1", {}), "identity"), s(r.get("run2", {}), "identity"),
                  s(r.get("run1", {}), "starter_fixed"), s(r.get("run1", {}), "starter_bridge"),
                  r.get("run1", {}).get("cands", {}).get("identity", {}).get("speedup"), flush=True)


@app.local_entrypoint()
def ablate(ids: str = "", weight_sync: str = "none", input_mode: str = "gpu", out: str = ""):
    """Re-run identity with one fix reverted, on a subset (comma-separated ids)."""
    B = KBV2()
    pids = [int(x) for x in ids.split(",") if x]
    out = out or f"kb_v2_ablate_{weight_sync}_{input_mode}.jsonl"
    with (_results_dir() / out).open("w") as f:
        for r in B.ablate_one.starmap([(p, weight_sync, input_mode) for p in pids], return_exceptions=True):
            if isinstance(r, Exception):
                r = {"error": repr(r)}
            f.write(json.dumps(r) + "\n")
            print("ablate", r.get("problem_id"), r.get("cands", {}).get("identity", {}).get("status"),
                  r.get("cands", {}).get("identity", {}).get("reason", "")[:120], flush=True)
