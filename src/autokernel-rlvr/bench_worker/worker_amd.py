"""
AutoKernel RLVR — AMD MI300X bench worker.

Identical loop structure to worker.py (NVIDIA) with three AMD-specific changes:
  1. Uses ROCR_VISIBLE_DEVICES instead of CUDA_VISIBLE_DEVICES to select the
     target GPU.  Also exports HIP_VISIBLE_DEVICES for older ROCm stacks.
  2. Default timeout bumped from 180 s → 300 s: ROCm JIT compilation (HIP-Clang
     or Triton-ROCm) has higher first-run overhead than CUDA nvrtc.
  3. Passes --backend rocm to bench.py (when the AutoKernel fork supports it).

Workers are stateless and spot-safe: the queue TTL re-enqueues on timeout.
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx


def parse_latest_result(tsv_path: Path) -> dict:
    """Read the most recent (non-baseline) row of results.tsv."""
    if not tsv_path.exists():
        return {}
    with tsv_path.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)
    if not rows:
        return {}
    last = rows[-1]
    return {
        "correctness":        last.get("correctness", "CRASH").upper(),
        "speedup_vs_pytorch": float(last.get("speedup_vs_pytorch") or 0.0),
        "pct_peak":           float(last.get("pct_peak")           or 0.0),
        "latency_us":         float(last.get("latency_us")         or 0.0),
        "throughput_tflops":  float(last.get("throughput_tflops")  or 0.0),
    }


def run_one_job(job: dict, autokernel_dir: Path, timeout: int,
                device_id: str) -> dict:
    code        = job["code"]
    kernel_type = job["kernel_type"]

    with tempfile.TemporaryDirectory() as td:
        sandbox = Path(td) / "autokernel"
        shutil.copytree(
            autokernel_dir, sandbox,
            ignore=shutil.ignore_patterns("workspace", ".venv", ".git"),
            symlinks=True,
        )
        (sandbox / "kernel.py").write_text(code)

        env = os.environ.copy()
        # AMD GPU selection — set all three for maximum ROCm stack compatibility.
        env["ROCR_VISIBLE_DEVICES"] = device_id
        env["HIP_VISIBLE_DEVICES"]  = device_id
        # Keep CUDA_VISIBLE_DEVICES empty so any stray CUDA calls fail fast
        # rather than accidentally running on a wrong device.
        env.pop("CUDA_VISIBLE_DEVICES", None)

        cmd = [
            "uv", "run", "bench.py",
            "--kernel-type", kernel_type,
            "--backend",     "rocm",   # AutoKernel AMD fork flag
        ]
        try:
            proc = subprocess.run(
                cmd, cwd=sandbox, env=env,
                capture_output=True, text=True, timeout=timeout,
            )
            stdout, stderr = proc.stdout, proc.stderr
            rc = proc.returncode
        except subprocess.TimeoutExpired as e:
            return {
                "correctness":        "TIMEOUT",
                "speedup_vs_pytorch": 0.0,
                "pct_peak":           0.0,
                "latency_us":         0.0,
                "throughput_tflops":  0.0,
                "raw": (e.stderr or "") if isinstance(e.stderr, str) else "timeout",
            }

        tsv = sandbox / "results.tsv"
        parsed = parse_latest_result(tsv)
        if not parsed:
            return {
                "correctness":        "CRASH",
                "speedup_vs_pytorch": 0.0,
                "pct_peak":           0.0,
                "latency_us":         0.0,
                "throughput_tflops":  0.0,
                "raw": (stderr or stdout)[-4000:],
            }
        if rc != 0 and parsed.get("correctness") == "PASS":
            parsed["raw"] = (stderr or stdout)[-2000:]
        return parsed


def main():
    ap = argparse.ArgumentParser(description="AMD MI300X bench worker for AutoKernel RLVR.")
    ap.add_argument("--bench-url",      required=True,
                    help="Base URL of the bench queue service.")
    ap.add_argument("--autokernel-dir", required=True, type=Path,
                    help="Path to AutoKernel repo with AMD/HIP backend.")
    ap.add_argument("--timeout",        type=int, default=300,
                    help="Wall-clock timeout per job (s). Default 300 for ROCm JIT.")
    ap.add_argument("--device-id",      default="0",
                    help="ROCR_VISIBLE_DEVICES value (default: '0').")
    args = ap.parse_args()

    if not args.autokernel_dir.exists():
        print(f"FATAL: autokernel dir not found: {args.autokernel_dir}", file=sys.stderr)
        sys.exit(1)

    client  = httpx.Client(timeout=30)
    backoff = 1.0

    print(
        f"AMD worker online — polling {args.bench_url} "
        f"(ROCR_VISIBLE_DEVICES={args.device_id}, timeout={args.timeout}s)",
        flush=True,
    )
    while True:
        try:
            r = client.get(f"{args.bench_url}/bench/next")
            if r.status_code == 204:
                time.sleep(min(backoff, 5.0))
                backoff = min(backoff * 1.5, 5.0)
                continue
            backoff = 1.0
            job    = r.json()
            job_id = job["job_id"]
            print(
                f"[{time.strftime('%H:%M:%S')}] job {job_id[:8]} "
                f"kernel={job['kernel_type']}",
                flush=True,
            )

            t0     = time.time()
            result = run_one_job(job, args.autokernel_dir, args.timeout, args.device_id)
            dt     = time.time() - t0
            print(
                f"  -> {result.get('correctness')} "
                f"speedup={result.get('speedup_vs_pytorch', 0.0):.2f}x "
                f"pct_peak={result.get('pct_peak', 0.0):.1f}% "
                f"in {dt:.1f}s",
                flush=True,
            )

            client.post(f"{args.bench_url}/bench/{job_id}/result", json=result)
        except httpx.HTTPError as e:
            print(f"network error: {e}", flush=True)
            time.sleep(5)
        except Exception as e:
            print(f"unexpected error: {e!r}", flush=True)
            time.sleep(2)


if __name__ == "__main__":
    main()
