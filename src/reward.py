"""
AutoKernel reward function.

Philosophy: keep the terminal reward simple and correctness-first.
Per-step shaping is optional and small. No partial credit for
"almost-correct" output — RLVR's strength is an unforgiving verifier.

Terminal reward per trajectory:
    r = 0.0                    if no PASS ever
        log2(best_speedup)     if at least one PASS (best across all turns)
        clipped to [-1.0, +3.0] so a 8x outlier doesn't dominate a batch

Tiny step shaping (optional, disabled by default — flip USE_STEP_SHAPING):
    +0.02 per PASS bench, -0.02 per CRASH
    This is SUMMED into the terminal reward (verl will broadcast it).

The "correctness > 0" structure means the policy MUST produce at least one
working kernel to get any reward. This avoids degenerate behaviors like
returning the PyTorch reference unchanged (0% speedup but correct) being
equivalently rewarded to crashing — both get 0.
"""
import math
from typing import Any

USE_STEP_SHAPING = False
SPEEDUP_CLIP_LOW = -1.0
SPEEDUP_CLIP_HIGH = 3.0      # log2(8) = 3


def autokernel_reward(data_source: str, solution_str: str,
                      ground_truth: Any, extra_info: dict) -> float:
    """Called by verl once per trajectory."""
    info = extra_info or {}
    any_pass: bool = bool(info.get("any_pass", False))
    best_speedup: float = float(info.get("best_speedup", 0.0))
    pass_count: int = int(info.get("pass_count", 0))
    crash_count: int = int(info.get("crash_count", 0))

    if not any_pass or best_speedup <= 0:
        reward = 0.0
    else:
        reward = math.log2(max(best_speedup, 1e-6))
        reward = max(SPEEDUP_CLIP_LOW, min(SPEEDUP_CLIP_HIGH, reward))

    if USE_STEP_SHAPING:
        reward += 0.02 * pass_count
        reward -= 0.02 * crash_count

    return float(reward)
