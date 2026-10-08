"""
Modal bench worker: port of src/autokernel-rlvr/bench_worker/worker.py.

Differences from the SkyPilot worker (both fix real bugs vs upstream bench.py):
  * upstream bench.py takes `--kernel`, not `--kernel-type`
  * upstream bench.py does not write results.tsv; metrics are parsed from the
    greppable `key: value` lines it prints (the `=== FINAL ===` block wins)

Result cache: sha256(gpu, mode, kernel_type, code) -> result, in a modal.Dict,
same semantics as bench_server/app.py:_hash (same code => same reward).

    modal run modal_app/bench_modal.py::smoke
"""
import hashlib
import os
import re
import shutil
import subprocess
import tempfile
import time

import modal

from common import (AUTOKERNEL_DIR, BENCH_MAX_CONTAINERS, CACHE_VERSION, bench_cache, bench_image,
                    extract_failure)

app = modal.App("autokernel-bench")

_KV = re.compile(r"^([a-z_]+):\s*(.+?)\s*$")
_NUM_KEYS = {
    "speedup_vs_pytorch", "throughput_tflops", "pct_peak_compute", "pct_peak_bandwidth",
    "latency_us", "pytorch_latency_us", "kernel_latency_us", "bandwidth_gb_s",
    "bench_time_seconds", "peak_vram_mb",
}


def parse_stdout(text: str) -> dict:
    kv = {}
    for line in text.splitlines():
        m = _KV.match(line.strip())
        if m:
            kv[m.group(1)] = m.group(2)
    out = {}
    for k in _NUM_KEYS:
        if k in kv:
            try:
                out[k] = float(kv[k].rstrip("x%"))
            except ValueError:
                pass
    for k in ("correctness", "smoke_test", "shape_sweep", "numerical_stability",
              "determinism", "edge_cases", "bottleneck", "gpu_name", "kernel_type"):
        if k in kv:
            out[k] = kv[k]
    return out


def classify(parsed: dict, text: str) -> str:
    """Map bench.py output onto the 4-way label the reward/agent expects."""
    corr = parsed.get("correctness", "").upper()
    if corr == "PASS":
        return "PASS"
    if ("Failed to import kernel.py" in text or "syntax error" in text
            or parsed.get("smoke_test") == "CRASH" or not parsed):
        return "CRASH"
    return "FAIL"


def cache_key(gpu: str, quick: bool, kernel_type: str, code: str) -> str:
    h = hashlib.sha256()
    for part in (CACHE_VERSION, gpu, "quick" if quick else "full", kernel_type, code):
        h.update(part.encode())
        h.update(b"\x00")
    return h.hexdigest()


@app.cls(image=bench_image, gpu="H100", timeout=900, max_containers=BENCH_MAX_CONTAINERS,
         scaledown_window=120)
class Bencher:
    @modal.method()
    def run(self, kernel_type: str, code: str, quick: bool = True,
            timeout: int = 180, use_cache: bool = True) -> dict:
        import torch
        gpu = torch.cuda.get_device_name(0)
        key = cache_key(gpu, quick, kernel_type, code)
        if use_cache:
            hit = bench_cache.get(key)
            if hit is not None:
                return {**hit, "cached": True}

        t0 = time.time()
        with tempfile.TemporaryDirectory() as td:
            sandbox = os.path.join(td, "autokernel")
            shutil.copytree(AUTOKERNEL_DIR, sandbox,
                            ignore=shutil.ignore_patterns(".git", "workspace"))
            with open(os.path.join(sandbox, "kernel.py"), "w") as f:
                f.write(code)
            cmd = ["python", "bench.py", "--kernel", kernel_type]
            if quick:
                cmd.append("--quick")
            try:
                proc = subprocess.run(cmd, cwd=sandbox, capture_output=True,
                                      text=True, timeout=timeout)
                text = proc.stdout + "\n" + proc.stderr
                parsed = parse_stdout(proc.stdout)
                label = classify(parsed, text)
            except subprocess.TimeoutExpired as e:
                text = (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
                parsed, label = {}, "TIMEOUT"

        res = {
            "correctness": label,
            "speedup_vs_pytorch": parsed.get("speedup_vs_pytorch", 0.0) if label == "PASS" else 0.0,
            "pct_peak": parsed.get("pct_peak_compute", 0.0),
            "pct_peak_bandwidth": parsed.get("pct_peak_bandwidth", 0.0),
            "latency_us": parsed.get("kernel_latency_us", parsed.get("latency_us", 0.0)),
            "pytorch_latency_us": parsed.get("pytorch_latency_us", 0.0),
            "throughput_tflops": parsed.get("throughput_tflops", 0.0),
            "bottleneck": parsed.get("bottleneck", ""),
            "stages": {k: parsed.get(k) for k in ("smoke_test", "shape_sweep",
                       "numerical_stability", "determinism", "edge_cases")},
            "gpu": gpu,
            "quick": quick,
            "wall_s": round(time.time() - t0, 2),
            "fail_lines": extract_failure(text) if label != "PASS" else [],
            "raw": text[-3000:] if label != "PASS" else "",
        }
        if label != "TIMEOUT":
            bench_cache[key] = res
        return {**res, "cached": False}


def starter_code(kernel_type: str) -> str:
    import pathlib
    here = pathlib.Path(__file__).resolve().parents[1]
    local = here / "results" / "autokernel_src" / "kernels" / f"{kernel_type}.py"
    return local.read_text()


@app.local_entrypoint()
def smoke():
    code = starter_code("matmul")
    r = Bencher().run.remote("matmul", code, quick=True, use_cache=False)
    print({k: v for k, v in r.items() if k != "raw"})
    bad = code.replace("tl.dot(", "0.5 * tl.dot(")
    r2 = Bencher().run.remote("matmul", bad, quick=True, use_cache=False)
    print("broken kernel ->", r2["correctness"])
    syntax = "KERNEL_TYPE = 'matmul'\ndef kernel_fn(:\n"
    r3 = Bencher().run.remote("matmul", syntax, quick=True, use_cache=False)
    print("syntax error ->", r3["correctness"])


@app.local_entrypoint()
def stage1(repeats: int = 3):
    """Starter kernels x GPUs: baseline speedups, noise, bench latency.

    GPUs run one after another so at most BENCH_MAX_CONTAINERS are live.
    """
    import json
    import pathlib
    from common import KERNELS
    out = pathlib.Path(__file__).resolve().parents[1] / "results" / "stage1_baselines.jsonl"
    with out.open("w") as f:
        for gpu in ("H100", "L4"):
            sub = []
            for kt in KERNELS:
                code = starter_code(kt)
                sub += [(kt, True, rep, code) for rep in range(repeats)]
                sub.append((kt, False, 0, code))
            B = Bencher.with_options(gpu=gpu)()
            args = [(kt, code, quick, 300, False) for kt, quick, _, code in sub]
            for (kt, quick, rep, _), r in zip(sub, B.run.starmap(args, return_exceptions=True)):
                if isinstance(r, Exception):
                    r = {"correctness": "INFRA_ERROR", "raw": repr(r)}
                f.write(json.dumps({"gpu_req": gpu, "kernel_type": kt, "quick": quick,
                                    "rep": rep, **r}) + "\n")
                print(gpu, kt, "quick" if quick else "full", rep, r.get("correctness"),
                      r.get("speedup_vs_pytorch"), r.get("wall_s"), flush=True)
