"""
Build the RLVR prompt dataset for AMD MI300X competition kernels.

Targets the six single-GPU MI300X kernel types:
  fp8-gemm, moe, mla-decode,
  mxfp4-mm, moe-mxfp4, mixed-mla

Collective / multi-GPU kernels (all2all, gemm+reducescatter, allgather+gemm)
are excluded — they require Infinity Fabric inter-die communication and
cannot run on a single MI300X node.

Source parquets (from HuggingFace GPUMODE/kernelbot-data):
  submissions.parquet                        — all AMD competition submissions
  successful_submissions.parquet             — passing submissions only
  deduplicated_submissions.parquet           — deduped by (user, code)
  deduplicated_successful_submissions.parquet — deduped passing submissions
  amd_1_1m_competition_submissions.parquet   — amd-mxfp4-mm (763),
                                               amd-moe-mxfp4 (764),
                                               amd-mixed-mla (765)

Usage (on the AMD trainer node):
    python3 data/build_amd_dataset.py \\
        --autokernel-dir ./autokernel \\
        --out ~/data/autokernel_amd \\
        --max-turns 20

Optional: pass --submissions-dir to a local mirror of the HF parquets so
the builder can splice in competition starters / reference impls.
"""
import argparse
import itertools
import json
from pathlib import Path

import pandas as pd

# Import MI300X specs for the system prompt numbers.
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from hardware.mi300x_specs import (
    BF16_TFLOPS_DENSE,
    FP8_TFLOPS_DENSE,
    SUSTAINED_BW_TBS,
    WAVEFRONT_SIZE,
    MAX_LDS_PER_CU_KB,
)

# ---------------------------------------------------------------------------
# Kernel catalogue
# ---------------------------------------------------------------------------

AMD_KERNELS = [
    "fp8-gemm",
    "moe",
    "mla-decode",
    "mxfp4-mm",
    "moe-mxfp4",
    "mixed-mla",
]

# Shapes per kernel.  Format chosen to match what bench.py likely expects:
# scalar dims listed as a tuple, passed via --shape to bench CLI or baked into
# the prompt for context.  Expand after v1 validates end-to-end.
AMD_SHAPE_SWEEP = {
    # (M, K, N) — square and rectangular GEMMs at MI300X-relevant scales
    "fp8-gemm": [
        (4096, 4096, 4096),
        (8192, 4096, 4096),
        (4096, 8192, 8192),
    ],
    # (num_tokens, d_model, ffn_dim, top_k_experts)
    "moe": [
        (512,  4096, 14336, 8),
        (1024, 4096, 14336, 8),
        (2048, 7168, 28672, 8),
    ],
    # (batch, num_heads, kv_seq_len, kv_lora_dim, head_dim)
    "mla-decode": [
        (1, 16, 2048,  512, 64),
        (4, 16, 4096,  512, 64),
        (8, 16, 8192,  512, 128),
    ],
    # (M, K, N)  — MX FP4 matrix multiply
    "mxfp4-mm": [
        (4096, 4096, 4096),
        (8192, 4096, 4096),
        (4096, 8192, 8192),
    ],
    # (num_tokens, d_model, ffn_dim, top_k_experts)  — FP4 weights
    "moe-mxfp4": [
        (512,  4096, 14336, 8),
        (1024, 4096, 14336, 8),
        (2048, 7168, 28672, 8),
    ],
    # (batch, num_heads, kv_seq_len, kv_lora_dim, head_dim)  — mixed precision
    "mixed-mla": [
        (1, 16, 2048,  512, 64),
        (4, 16, 4096,  512, 64),
        (8, 16, 8192,  512, 128),
    ],
}

# Kernels that operate inherently in a low-precision format.
# We skip float16 sweeps for these and only benchmark in their native dtype.
_NATIVE_DTYPE: dict[str, list[str]] = {
    "fp8-gemm":  ["float8_e4m3fnuz"],
    "mxfp4-mm":  ["mxfp4"],
    "moe-mxfp4": ["mxfp4"],
}

# Default activation/weight dtypes for all other AMD kernels.
_DEFAULT_DTYPES = ["float16", "bfloat16"]


def _kernel_dtypes(ktype: str) -> list[str]:
    return _NATIVE_DTYPE.get(ktype, _DEFAULT_DTYPES)


# ---------------------------------------------------------------------------
# System prompt  (AMD / ROCm / HIP specific)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = f"""You are an expert GPU kernel engineer specialising in AMD Instinct MI300X.

Hardware context:
  FP8  peak compute : {FP8_TFLOPS_DENSE:.1f} TFLOPs (dense), {FP8_TFLOPS_DENSE * 2:.1f} TFLOPs (with sparsity)
  BF16 peak compute : {BF16_TFLOPS_DENSE:.1f} TFLOPs (dense)
  Memory bandwidth  : {SUSTAINED_BW_TBS} TB/s sustained (~5.3 TB/s theoretical)
  Wavefront size    : {WAVEFRONT_SIZE} threads  (not 32 — this is AMD, not NVIDIA)
  LDS per CU        : {MAX_LDS_PER_CU_KB} KB (AMD Local Data Share ≈ CUDA shared memory)

You have access to one tool:
  autokernel_bench(kernel_type, code) — compiles and runs your kernel on a real MI300X.
    Returns: correctness (PASS/FAIL/TIMEOUT/CRASH), speedup_vs_pytorch, pct_peak,
             latency_us, throughput_tflops.

Rules:
  * Only PASS results count. A fast but wrong kernel scores zero.
  * Propose the FULL kernel.py each turn (HIP C++ or Triton with ROCm backend).
  * After each bench result, revise based on what the numbers tell you.

AMD-specific optimisation guidance:
  Memory-bound kernels (mla-decode, moe routing):
    - Coalesce 128-byte cache-line fetches across wavefront lanes.
    - Avoid LDS bank conflicts: 32 banks × 4 bytes; stride accesses carefully.
    - Prefer flat global loads over texture paths for non-spatial data.

  Compute-bound kernels (fp8-gemm, mxfp4-mm, moe-mxfp4, mixed-mla):
    - Target MFMA (Matrix Fused Multiply-Add) intrinsics for peak matrix-core use.
    - Tile sizes: MFMA_F8_16×16, MFMA_F8_32×32, MFMA_BF16_16×16 are the key shapes.
    - Keep the LDS double-buffer sized to hide HBM3 latency (~200 ns).
    - Wavefront-level register pressure: >128 VGPRs halves occupancy.

  FP8 / MXFP4 kernels:
    - Use `__hip_fp8_e4m3_fnuz` or `__hip_fp8_e5m2_fnuz` types.
    - Scale factors must be computed per-tile (MX spec: 32-element groups).
    - Dequantise inside the MFMA pipe, not before — keeps bandwidth FP8-width.

Your goal: maximise speedup_vs_pytorch while maintaining PASS.
You have a limited edit budget per kernel — use it wisely."""


# ---------------------------------------------------------------------------
# User prompt template
# ---------------------------------------------------------------------------

def user_prompt(kernel_type: str, shape, dtype: str,
                reference_src: str, starter_src: str, max_turns: int) -> str:
    shape_str = str(shape) if not isinstance(shape, str) else shape
    return f"""Kernel type : {kernel_type}
Target shape: {shape_str}
Target dtype: {dtype}
Edit budget : {max_turns} proposals.

PyTorch reference (ground truth — your kernel must match this numerically):
```python
{reference_src}
```

Starter kernel (feel free to rewrite from scratch):
```python
{starter_src}
```

Start by benchmarking the starter to establish the baseline, then iterate.
Call autokernel_bench with the full kernel.py contents each turn."""


# ---------------------------------------------------------------------------
# Optional: load competition starters from parquet
# ---------------------------------------------------------------------------

def _load_competition_starters(submissions_dir: Path) -> dict[str, str]:
    """Return {kernel_type: best_starter_code} from deduplicated successful submissions.

    Heuristic: pick the submission with the highest throughput_tflops as the
    starter so the model begins from a strong baseline rather than scratch.
    Falls back to an empty string if the parquet is absent.
    """
    parquet = submissions_dir / "deduplicated_successful_submissions.parquet"
    if not parquet.exists():
        return {}

    df = pd.read_parquet(parquet)
    if "kernel_type" not in df.columns or "code" not in df.columns:
        return {}

    starters: dict[str, str] = {}
    perf_col = next(
        (c for c in ("throughput_tflops", "speedup_vs_pytorch", "score") if c in df.columns),
        None,
    )
    for ktype, grp in df.groupby("kernel_type"):
        if perf_col:
            best = grp.loc[grp[perf_col].idxmax()]
        else:
            best = grp.iloc[0]
        starters[str(ktype)] = str(best["code"])

    return starters


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def _read(path: Path, default: str = "") -> str:
    if not path.exists():
        return default
    return path.read_text()


def build(
    autokernel_dir: Path,
    out_dir: Path,
    max_turns: int,
    val_frac: float = 0.15,
    submissions_dir: Path | None = None,
):
    ref_src = _read(autokernel_dir / "reference.py")
    competition_starters = (
        _load_competition_starters(submissions_dir) if submissions_dir else {}
    )

    rows = []
    for ktype in AMD_KERNELS:
        # Starter: prefer competition best, then repo kernel file, then stub.
        starter = competition_starters.get(
            ktype,
            _read(
                autokernel_dir / "kernels" / f"{ktype}.py",
                default=f"# TODO: write HIP/Triton-ROCm kernel for {ktype}\n",
            ),
        )
        dtypes = _kernel_dtypes(ktype)
        shapes = AMD_SHAPE_SWEEP[ktype]

        for shape, dtype in itertools.product(shapes, dtypes):
            prompt_messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt(
                    ktype, shape, dtype, ref_src, starter, max_turns)},
            ]
            rows.append({
                "data_source": "amd_mi300",
                "prompt":      prompt_messages,
                "ability":     "gpu_kernels_amd",
                "reward_model": {"style": "rule", "ground_truth": ""},
                "agent_name":  "autokernel_agent",
                "extra_info": {
                    "kernel_type": ktype,
                    "shape":       list(shape),
                    "dtype":       dtype,
                    "hardware":    "amd_mi300x",
                },
            })

    df = pd.DataFrame(rows)
    # Deterministic train/val split: last val_frac rows of each kernel type.
    val_mask = df.groupby("ability").cumcount(ascending=False) < (
        val_frac * df.groupby("ability")["prompt"].transform("count")
    )
    train = df[~val_mask].reset_index(drop=True)
    val   = df[val_mask].reset_index(drop=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    train.to_parquet(out_dir / "train.parquet", index=False)
    val.to_parquet(out_dir   / "val.parquet",   index=False)
    print(
        f"AMD MI300X single-GPU dataset: {len(train)} train / {len(val)} val rows "
        f"({len(AMD_KERNELS)} kernels × shapes × dtypes) → {out_dir}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Build AMD MI300X RLVR dataset from AutoKernel + competition starters."
    )
    ap.add_argument("--autokernel-dir",  type=Path, required=True,
                    help="Path to cloned AutoKernel repo (needs reference.py + kernels/).")
    ap.add_argument("--out",             type=Path, required=True,
                    help="Output directory for train.parquet and val.parquet.")
    ap.add_argument("--max-turns",       type=int,  default=20,
                    help="Edit budget per episode (max_assistant_turns in verl).")
    ap.add_argument("--submissions-dir", type=Path, default=None,
                    help="Local mirror of GPUMODE/kernelbot-data parquets (optional).")
    args = ap.parse_args()
    build(args.autokernel_dir, args.out, args.max_turns,
          submissions_dir=args.submissions_dir)


if __name__ == "__main__":
    main()
