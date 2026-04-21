"""
AMD MI300X reward function for AutoKernel RLVR.

Philosophy: same correctness-first structure as reward.py (NVIDIA) but with
clipping tuned for AMD FP8/MXFP4 kernels which can deliver larger speedups
over BF16 PyTorch baselines than typical NVIDIA Triton kernels.

  FP8 peak on MI300X  : 2614.9 TFLOPs
  BF16 peak on MI300X : 1307.4 TFLOPs
  → FP8 kernel vs BF16 PyTorch can theoretically achieve ≥ 2× precision gain
    on top of algorithmic improvements, pushing realistic max past log2(8).

Clip schedule (log2 scale):
  FP8 / MXFP4 kernels  : high = 4.0  (log2(16) — allows up to 16× speedup)
  Collective kernels    : high = 3.5  (log2(11.3) — all2all, gather/scatter)
  All others            : high = 3.0  (log2(8)  — same as NVIDIA baseline)

Terminal reward per trajectory:
    r = 0.0                     if no PASS ever
        log2(best_speedup)      if ≥ 1 PASS  (best across all turns)
        clipped to [low, high]  per kernel class above

Optional step shaping (disabled by default — same as NVIDIA):
    +0.02 per PASS, −0.02 per CRASH
"""
import math
from typing import Any

USE_STEP_SHAPING   = False
SPEEDUP_CLIP_LOW   = -1.0

# Kernel-class → log2 clip ceiling
_HIGH_CLIP: dict[str, float] = {
    "fp8-gemm":   4.0,
    "mxfp4-mm":   4.0,
    "moe-mxfp4":  4.0,
    "mixed-mla":  4.0,   # mixed FP8 precision path
    "all2all":            3.5,
    "gemm+reducescatter": 3.5,
    "allgather+gemm":     3.5,
    # moe, mla-decode → default 3.0
}
_DEFAULT_HIGH = 3.0


def _clip_high(kernel_type: str) -> float:
    return _HIGH_CLIP.get(kernel_type, _DEFAULT_HIGH)


def autokernel_reward_amd(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict,
) -> float:
    """Called by verl once per trajectory.

    extra_info keys populated by AutoKernelAgentLoop:
      any_pass       : bool
      best_speedup   : float
      pass_count     : int
      crash_count    : int
      kernel_type    : str   (from the prompt's extra_info, propagated by verl)
    """
    info          = extra_info or {}
    any_pass      = bool(info.get("any_pass",     False))
    best_speedup  = float(info.get("best_speedup", 0.0))
    pass_count    = int(info.get("pass_count",     0))
    crash_count   = int(info.get("crash_count",    0))
    kernel_type   = str(info.get("kernel_type",    ""))

    if not any_pass or best_speedup <= 0:
        reward = 0.0
    else:
        high   = _clip_high(kernel_type)
        reward = math.log2(max(best_speedup, 1e-6))
        reward = max(SPEEDUP_CLIP_LOW, min(high, reward))

    if USE_STEP_SHAPING:
        reward += 0.02 * pass_count
        reward -= 0.02 * crash_count

    return float(reward)
