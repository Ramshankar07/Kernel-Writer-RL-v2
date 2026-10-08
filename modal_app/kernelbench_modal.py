"""
KernelBench v1 (ScalingIntelligence/KernelBench, Level 1) on Modal.

The project started on KernelBench Level 1 as its test set, before moving to
AutoKernel's 9 kernel types for RL. This module rebuilds that eval using
AutoKernel's own KernelBench bridge (kernelbench/bridge.py + bench_kb.py):

  * Level 1 problems are fetched from HF at image build and cached in the image
  * KBBencher.run(problem_id, code) sets up the problem, drops `code` in as
    kernel.py (a ModelNew class) and returns correctness + speedup
  * code=None benches the identity solution (ModelNew subclasses Model), the
    harness sanity test: every problem should PASS at ~1.0x

    modal run modal_app/kernelbench_modal.py::sanity
"""
import json
import os
import re
import shutil
import subprocess
import tempfile
import time

import modal

from common import (AUTOKERNEL_DIR, BENCH_MAX_CONTAINERS, CACHE_VERSION, bench_base_image, bench_cache,
                    extract_failure)

kb_image = (
    bench_base_image.pip_install("datasets", "ninja")
    .run_commands(f"cd {AUTOKERNEL_DIR} && python kernelbench/bridge.py fetch --source hf --level 1")
    .add_local_python_source("common")
)

app = modal.App("autokernel-kernelbench")

_KV = re.compile(r"^([a-z_]+):\s*(.+?)\s*$")


def parse_kb(stdout: str) -> dict:
    kv = {}
    for line in stdout.splitlines():
        m = _KV.match(line.strip())
        if m:
            kv[m.group(1)] = m.group(2)

    def num(k):
        try:
            return float(kv.get(k, "0").rstrip("x"))
        except ValueError:
            return 0.0

    return {
        "correctness": kv.get("correctness", "").upper(),
        "speedup": num("speedup"),
        "kernel_time_ms": num("kernel_time_ms"),
        "reference_time_ms": num("reference_time_ms"),
        "worst_max_abs_error": num("worst_max_abs_error"),
        "name": kv.get("name", ""),
    }


# cpu/memory: bench_kb.py times get_inputs() (CPU torch.rand of up to 1.6B elements)
# inside its 30s trial timeout; Modal's default CPU share makes 45/100 identity
# solutions TIMEOUT.
@app.cls(image=kb_image, gpu="H100", cpu=8, memory=32768, timeout=900,
         max_containers=BENCH_MAX_CONTAINERS, scaledown_window=120)
class KBBencher:
    @modal.method()
    def list_problems(self) -> list:
        root = os.path.join(AUTOKERNEL_DIR, "workspace", "kb_cache", "level1")
        out = []
        # the cache holds N.py (source) and N.json (metadata); only the .py is the problem
        for fn in sorted((f for f in os.listdir(root) if f.endswith(".py")), key=lambda s: int(s.split(".")[0])):
            with open(os.path.join(root, fn)) as f:
                out.append({"problem_id": int(fn.split(".")[0]), "code": f.read()})
        return out

    @modal.method()
    def run(self, problem_id: int, code: str | None = None, quick: bool = True,
            timeout: int = 300, use_cache: bool = True) -> dict:
        import hashlib
        import torch
        gpu = torch.cuda.get_device_name(0)
        key = hashlib.sha256(f"kb1{CACHE_VERSION}\0{gpu}\0{quick}\0{problem_id}\0{code}".encode()).hexdigest()
        if use_cache and (hit := bench_cache.get(key)) is not None:
            return {**hit, "cached": True}

        t0 = time.time()
        with tempfile.TemporaryDirectory() as td:
            sb = os.path.join(td, "autokernel")
            shutil.copytree(AUTOKERNEL_DIR, sb, ignore=shutil.ignore_patterns(".git"))
            setup = subprocess.run(
                ["python", "kernelbench/bridge.py", "setup", "--level", "1",
                 "--problem", str(problem_id), "--backend", "triton"],
                cwd=sb, capture_output=True, text=True)
            if setup.returncode != 0:
                return {"problem_id": problem_id, "correctness": "INFRA_ERROR",
                        "raw": (setup.stdout + setup.stderr)[-2000:]}
            if code is None:
                # Identity solution. Not bridge's starter: it renames the class but keeps
                # `super(Model, self)`, so every starter crashes on instantiation.
                with open(os.path.join(sb, "workspace", "kb_active", "reference.py")) as f:
                    code = f.read() + "\n\nclass ModelNew(Model):\n    pass\n"
                key = None
            with open(os.path.join(sb, "kernel.py"), "w") as f:
                f.write(code)
            cmd = ["python", "kernelbench/bench_kb.py"] + (["--quick"] if quick else [])
            try:
                p = subprocess.run(cmd, cwd=sb, capture_output=True, text=True, timeout=timeout)
                text = p.stdout + "\n" + p.stderr
                res = parse_kb(p.stdout)
                if res["correctness"] != "PASS":
                    crashed = ("Traceback" in text and not res["correctness"]) or \
                              "instantiation failed" in text or "SyntaxError" in text
                    res["correctness"] = "CRASH" if crashed else "FAIL"
            except subprocess.TimeoutExpired:
                text, res = "timeout", {"correctness": "TIMEOUT", "speedup": 0.0}
        res.update({"problem_id": problem_id, "gpu": gpu, "quick": quick,
                    "wall_s": round(time.time() - t0, 2),
                    "fail_lines": [] if res["correctness"] == "PASS" else extract_failure(text),
                    "raw": "" if res["correctness"] == "PASS" else text[-3000:]})
        if key and res["correctness"] != "TIMEOUT":
            bench_cache[key] = res
        return {**res, "cached": False}


@app.local_entrypoint()
def sanity(first: int = 1, last: int = 100, out: str = "kb_sanity.jsonl"):
    """Harness test: ModelNew == Model on Level 1 problems [first, last] -> expect PASS, ~1.0x."""
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1] / "results"
    B = KBBencher()
    allp = B.list_problems.remote()
    (root / "kernelbench_l1_problems.jsonl").write_text(
        "\n".join(json.dumps(p) for p in allp) + "\n")
    probs = [p for p in allp if first <= p["problem_id"] <= last]
    with (root / out).open("w") as f:
        for r in B.run.starmap([(p["problem_id"], None, True, 300, False) for p in probs],
                               return_exceptions=True):
            if isinstance(r, Exception):
                r = {"correctness": "INFRA_ERROR", "raw": repr(r)}
            f.write(json.dumps(r) + "\n")
            print("kb", r.get("problem_id"), r.get("correctness"), r.get("speedup"),
                  r.get("wall_s"), flush=True)
