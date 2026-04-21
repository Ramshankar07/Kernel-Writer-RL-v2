"""
AutoKernel RLVR — bench worker.

Loop:
    1. GET /bench/next  →  job or 204
    2. Write `code` into a throwaway copy of AUTOKERNEL_DIR/kernel.py
    3. Run `uv run bench.py` with a timeout
    4. Parse the appended row from results.tsv (or fall back to stdout)
    5. POST /bench/{job_id}/result with parsed metrics
    6. Sleep briefly if queue was empty; else go again.

Workers are stateless. If they crash mid-job, the queue TTL lets the trainer
re-enqueue. We deliberately do NOT retry failed jobs here — a crash IS the
signal that the kernel is broken, and that's a legitimate reward of CRASH.
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
        "correctness": last.get("correctness", "CRASH").upper(),
        "speedup_vs_pytorch": float(last.get("speedup_vs_pytorch") or 0.0),
        "pct_peak": float(last.get("pct_peak") or 0.0),
        "latency_us": float(last.get("latency_us") or 0.0),
        "throughput_tflops": float(last.get("throughput_tflops") or 0.0),
    }


def run_one_job(job: dict, autokernel_dir: Path, timeout: int) -> dict:
    code = job["code"]
    kernel_type = job["kernel_type"]

    # Sandbox: copy the AutoKernel tree into a temp dir so parallel workers
    # on the same host don't stomp each other. Symlink models/ and kernels/
    # to avoid copying hundreds of MB.
    with tempfile.TemporaryDirectory() as td:
        sandbox = Path(td) / "autokernel"
        shutil.copytree(
            autokernel_dir, sandbox,
            ignore=shutil.ignore_patterns("workspace", ".venv", ".git"),
            symlinks=True,
        )
        (sandbox / "kernel.py").write_text(code)

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0")

        try:
            proc = subprocess.run(
                ["uv", "run", "bench.py", "--kernel-type", kernel_type],
                cwd=sandbox, env=env,
                capture_output=True, text=True, timeout=timeout,
            )
            stdout, stderr = proc.stdout, proc.stderr
            rc = proc.returncode
        except subprocess.TimeoutExpired as e:
            return {
                "correctness": "TIMEOUT",
                "speedup_vs_pytorch": 0.0, "pct_peak": 0.0,
                "latency_us": 0.0, "throughput_tflops": 0.0,
                "raw": (e.stderr or "") if isinstance(e.stderr, str) else "timeout",
            }

        tsv = sandbox / "results.tsv"
        parsed = parse_latest_result(tsv)
        if not parsed:
            return {
                "correctness": "CRASH",
                "speedup_vs_pytorch": 0.0, "pct_peak": 0.0,
                "latency_us": 0.0, "throughput_tflops": 0.0,
                "raw": (stderr or stdout)[-4000:],
            }
        if rc != 0 and parsed.get("correctness") == "PASS":
            # bench.py exit nonzero but row says PASS — trust the row, attach log
            parsed["raw"] = (stderr or stdout)[-2000:]
        return parsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-url", required=True)
    ap.add_argument("--autokernel-dir", required=True, type=Path)
    ap.add_argument("--timeout", type=int, default=180)
    args = ap.parse_args()

    if not args.autokernel_dir.exists():
        print(f"FATAL: autokernel dir not found: {args.autokernel_dir}", file=sys.stderr)
        sys.exit(1)

    client = httpx.Client(timeout=30)
    backoff = 1.0

    print(f"worker online — polling {args.bench_url}", flush=True)
    while True:
        try:
            r = client.get(f"{args.bench_url}/bench/next")
            if r.status_code == 204:
                time.sleep(min(backoff, 5.0))
                backoff = min(backoff * 1.5, 5.0)
                continue
            backoff = 1.0
            job = r.json()
            job_id = job["job_id"]
            print(f"[{time.strftime('%H:%M:%S')}] job {job_id[:8]} kernel={job['kernel_type']}", flush=True)

            t0 = time.time()
            result = run_one_job(job, args.autokernel_dir, args.timeout)
            dt = time.time() - t0
            print(f"  -> {result.get('correctness')} speedup={result.get('speedup_vs_pytorch'):.2f}x in {dt:.1f}s", flush=True)

            client.post(f"{args.bench_url}/bench/{job_id}/result", json=result)
        except httpx.HTTPError as e:
            print(f"network error: {e}", flush=True)
            time.sleep(5)
        except Exception as e:
            print(f"unexpected error: {e!r}", flush=True)
            time.sleep(2)


if __name__ == "__main__":
    main()
