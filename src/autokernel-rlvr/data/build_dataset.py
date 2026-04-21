"""
Build the RLVR prompt dataset from AutoKernel's kernel specs.

Each row is one (kernel_type, shape, dtype) triple. The prompt tells the
model:
  * which kernel type to produce
  * the PyTorch reference (from reference.py)
  * the starter Triton kernel (from kernels/{kernel_type}.py)
  * shape / dtype constraints
  * instructions to iterate using the autokernel_bench tool

The ground_truth field is unused (reward is purely from bench). agent_name
is set to "autokernel_agent" so verl dispatches to our AgentLoop.

Run on the trainer node before training:
    python3 data/build_dataset.py \\
        --autokernel-dir ./autokernel \\
        --out ~/data/autokernel \\
        --max-turns 20
"""
import argparse
import itertools
import json
from pathlib import Path

import pandas as pd

KERNELS = [
    "matmul",
    "softmax",
    "layernorm",
    "rmsnorm",
    "flash_attention",
    "fused_mlp",
    "cross_entropy",
    "rotary_embedding",
    "reduce",
]

# Conservative shape sweep per kernel. Expand after v1 works end-to-end.
# These shapes must be ones bench.py already supports; AutoKernel's
# bench.py takes the kernel-type's default shape set unless overridden.
SHAPE_SWEEP = {
    "matmul": [(1024, 1024, 1024), (2048, 2048, 2048), (4096, 4096, 4096)],
    "softmax": [(1024, 1024), (1024, 4096), (4096, 1024)],
    "layernorm": [(1024, 2048), (4096, 2048), (8192, 4096)],
    "rmsnorm": [(1024, 4096), (2048, 4096), (4096, 4096)],
    "flash_attention": [(1, 8, 1024, 64), (1, 16, 2048, 64), (2, 8, 4096, 128)],
    "fused_mlp": [(1024, 4096, 11008), (2048, 4096, 11008)],
    "cross_entropy": [(1024, 32000), (4096, 32000)],
    "rotary_embedding": [(1, 2048, 32, 64), (2, 2048, 32, 128)],
    "reduce": [(1024 * 1024,), (16 * 1024 * 1024,), (64 * 1024 * 1024,)],
}
DTYPES = ["float16", "bfloat16"]


SYSTEM_PROMPT = """You are an expert GPU kernel engineer. You optimize Triton kernels for NVIDIA GPUs.

You have access to one tool:
  autokernel_bench(kernel_type, code) — evaluates your proposed kernel.py on a real GPU.
    Returns: correctness (PASS/FAIL/TIMEOUT/CRASH), speedup_vs_pytorch, pct_peak, latency_us.

Rules:
  * Only PASS results count. A fast but wrong kernel scores zero.
  * Each turn, propose the FULL contents of kernel.py (Triton kernel + PyTorch wrapper).
  * After each bench result, revise. Think about what the result tells you:
    crashed shapes imply indexing bugs; low pct_peak on memory-bound kernels implies
    poor coalescing; low pct_peak on compute-bound kernels implies poor tile sizing.
  * Memory-bound kernels (softmax, layernorm, rmsnorm, reduce) win from coalescing
    and fused reductions. Compute-bound kernels (matmul, flash_attention, fused_mlp)
    win from bigger tiles and better warp specialization.

Your goal: maximize speedup_vs_pytorch while maintaining PASS. You have a limited
number of edits per kernel."""


def user_prompt(kernel_type: str, shape, dtype: str, reference_src: str,
                starter_src: str, max_turns: int) -> str:
    return f"""Kernel type: {kernel_type}
Target shape: {shape}
Target dtype: {dtype}
Edit budget: {max_turns} proposals.

PyTorch reference (ground truth — your kernel must match this numerically):
```python
{reference_src}
```

Starter Triton kernel (feel free to rewrite from scratch):
```python
{starter_src}
```

Start by benchmarking the starter so you see the baseline. Then iterate.
Call autokernel_bench with the full kernel.py contents."""


def _read(path: Path, default: str = "") -> str:
    if not path.exists():
        return default
    return path.read_text()


def build(autokernel_dir: Path, out_dir: Path, max_turns: int,
          val_frac: float = 0.15):
    ref_src = _read(autokernel_dir / "reference.py")

    rows = []
    for ktype in KERNELS:
        starter = _read(autokernel_dir / "kernels" / f"{ktype}.py",
                        default=f"# TODO: starter for {ktype}\n")
        for shape, dtype in itertools.product(SHAPE_SWEEP[ktype], DTYPES):
            prompt_messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt(
                    ktype, shape, dtype, ref_src, starter, max_turns)},
            ]
            rows.append({
                "data_source": "autokernel",
                "prompt": prompt_messages,
                "ability": "gpu_kernels",
                "reward_model": {"style": "rule", "ground_truth": ""},
                "agent_name": "autokernel_agent",
                "extra_info": {
                    "kernel_type": ktype,
                    "shape": list(shape),
                    "dtype": dtype,
                },
            })

    # Deterministic split: val = last val_frac of each kernel_type.
    df = pd.DataFrame(rows)
    val_mask = df.groupby("ability").cumcount(ascending=False) < (
        val_frac * df.groupby("ability")["prompt"].transform("count")
    )
    train = df[~val_mask].reset_index(drop=True)
    val = df[val_mask].reset_index(drop=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    train.to_parquet(out_dir / "train.parquet", index=False)
    val.to_parquet(out_dir / "val.parquet", index=False)
    print(f"wrote {len(train)} train / {len(val)} val rows to {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--autokernel-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-turns", type=int, default=20)
    args = ap.parse_args()
    build(args.autokernel_dir, args.out, args.max_turns)


if __name__ == "__main__":
    main()
