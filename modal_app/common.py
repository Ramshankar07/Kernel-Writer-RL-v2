"""
Shared Modal resources for the AutoKernel-RLVR replication.

Replaces the three SkyPilot clusters in src/autokernel-rlvr/skypilot/:
  * bench queue (FastAPI + Redis)  -> modal.Dict content-hash cache + Function.map
  * bench workers (spot GPUs)      -> modal_app/bench.py  (autoscaled containers)
  * trainer (8xH100 verl)          -> modal_app/train_grpo.py (2xH100, LoRA)
"""
import modal

# Pinned so re-runs benchmark against the exact same harness.
AUTOKERNEL_REPO = "https://github.com/RightNow-AI/autokernel.git"
AUTOKERNEL_COMMIT = "78435821cc3d5756ba6ee1785c397f6d8fa8c90d"
AUTOKERNEL_DIR = "/autokernel"

# Account quota: <=10 concurrent GPUs. Bench gets 8, policy/trainer the rest.
BENCH_MAX_CONTAINERS = 8

KERNELS = [
    "matmul", "softmax", "layernorm", "rmsnorm", "flash_attention",
    "fused_mlp", "cross_entropy", "rotary_embedding", "reduce",
]

volume = modal.Volume.from_name("autokernel-rlvr", create_if_missing=True)
VOL = "/vol"
bench_cache = modal.Dict.from_name("autokernel-bench-cache", create_if_missing=True)

bench_base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch==2.6.0", "numpy", "pandas", "matplotlib")
    .run_commands(
        f"git clone {AUTOKERNEL_REPO} {AUTOKERNEL_DIR}",
        f"cd {AUTOKERNEL_DIR} && git checkout {AUTOKERNEL_COMMIT}",
    )
)
bench_image = bench_base_image.add_local_python_source("common")


CACHE_VERSION = "v2"   # v2: results carry fail_lines (v1 fed the model only the log tail)

_FAIL_PAT = ("FAIL:", "FAIL --", "Trial ", "exceeds tol", "Error:", "Error ", "OutOfResources",
             "error:", "FATAL", "illegal memory", "WARNING: bench")


def extract_failure(text: str, max_lines: int = 12) -> list:
    """The lines that say *why* a kernel failed (bench.py / bench_kb.py print them mid-log;
    the tail is only the summary block)."""
    out, seen = [], set()
    for line in text.splitlines():
        s = line.strip()
        if s and any(p in s for p in _FAIL_PAT) and "PASS" not in s[:6] and s not in seen:
            seen.add(s)
            out.append(s[:300])
    return out[:max_lines]
