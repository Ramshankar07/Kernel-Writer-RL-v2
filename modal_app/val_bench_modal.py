"""
Validation-set bench (`autokernel-val-bench`): runs val_bench_core.py (bench v2 conventions) on
the held-out problems in results/v3/val_set/. Same container image, GPU (L4) and subprocess
restart logic as bench_v2_modal.py; no cache.

    cd modal_app
    modal run val_bench_modal.py::verify_starters     # -> results/v3/val_set_starters.jsonl
    modal deploy val_bench_modal.py                   # used by val_eval.py
"""
import json
import os
import pathlib
import sys

import modal

from common import AUTOKERNEL_DIR, bench_base_image

MAX_CONTAINERS = int(os.environ.get("VAL_BENCH_MAX_CONTAINERS", "6"))
GPU = os.environ.get("VAL_BENCH_GPU", "L4")

app = modal.App("autokernel-val-bench")
image = bench_base_image.add_local_python_source("common", "bench_v2_core", "val_bench_core")

ROOT = pathlib.Path(__file__).resolve().parents[1]
VAL = ROOT / "results" / "v3" / "val_set"


def _run_jobs(jobs, job_timeout=150):
    """bench_v2_modal._run_jobs with the val core (restart after crash / hang)."""
    import importlib.util
    import subprocess
    import tempfile
    import time
    core = importlib.util.find_spec("val_bench_core").origin
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
                time.sleep(0.3)
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
                j = next(j for j in remaining if j["id"] == s)
                results[s] = {"id": s, "problem_id": j["spec"]["id"], "verdict": "TIMEOUT" if killed else "CRASH",
                              "reason": ("hang > %ds" % job_timeout) if killed else "process died: " + tail[-600:]}
        before = len(remaining)
        remaining = [j for j in remaining if j["id"] not in results]
        if len(remaining) == before:
            for j in remaining:
                results[j["id"]] = {"id": j["id"], "problem_id": j["spec"]["id"], "verdict": "CRASH",
                                    "reason": "runner failed to start: " + tail[-400:]}
            remaining = []
    return [results[j["id"]] for j in jobs]


@app.cls(image=image, gpu=GPU, timeout=3600, cpu=2, memory=16384,
         max_containers=MAX_CONTAINERS, scaledown_window=60)
class ValBench:
    @modal.method()
    def run_batch(self, jobs: list) -> list:
        return _run_jobs(jobs)

    @modal.method()
    def run_one(self, job: dict) -> dict:
        return _run_jobs([job])[0]


# ---------------------------------------------------------------------------------------------
# local side
# ---------------------------------------------------------------------------------------------
def load_problems(ids=None):
    meta = json.loads((ROOT / "results" / "v3" / "val_set.json").read_text())
    out = []
    for p in meta["problems"]:
        if ids and p["id"] not in ids:
            continue
        d = VAL / p["dir"]
        out.append({"spec": json.loads((d / "problem.json").read_text()),
                    "reference_code": (d / "reference.py").read_text(),
                    "starter_code": (d / "starter.py").read_text()})
    return out


@app.local_entrypoint()
def verify_starters(ids: str = "", out: str = "val_set_starters.jsonl", repeats: int = 1):
    """Each val starter must PASS (full case list); plus the per-problem timing baseline
    (starter vs eager vs torch.compile) that val_eval uses for vs-compile ratios."""
    probs = load_problems(set(ids.split(",")) if ids else None)
    jobs = [{"id": f"{p['spec']['id']}#{i}", "spec": p["spec"], "reference_code": p["reference_code"],
             "starter_code": p["starter_code"], "code": p["starter_code"], "compile": True,
             "timing": True, "time_all": True, "reps": 40, "early_abort": False}
            for p in probs for i in range(repeats)]
    res = []
    for r in ValBench().run_one.map(jobs, return_exceptions=True, wrap_returned_exceptions=False):
        if isinstance(r, Exception):
            print("error:", repr(r)[:300], flush=True)
            continue
        res.append(r)
        t = next((v for v in (r.get("timing") or {}).values() if isinstance(v, dict) and "starter" in v), {})
        print(r["id"], r.get("verdict"), r.get("wall_s"), (r.get("fail_reasons") or [""])[0][:200],
              r.get("reason", "")[:200],
              "launch", (r.get("correctness") or {}).get("triton_launches"),
              "starter/eager %.3f compile/starter %.3f" % (t.get("starter_vs_eager", {}).get("median", 0),
                                                           t.get("compile_over_starter", {}).get("median", 0)),
              flush=True)
    path = ROOT / "results" / "v3" / out
    mode = "a" if ids else "w"
    with path.open(mode) as f:
        for r in res:
            f.write(json.dumps(r, default=str) + "\n")
    print("wrote", path, len(res))
