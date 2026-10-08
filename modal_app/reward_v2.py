"""
Reward v2: fixes the two problems GRPO v1 exposed (results/grpo/grpo_v1).

  1. Sparse signal. With the original terminal reward 6/9 kernels scored 0 for all
     8 samples at every step, so their groups never produced gradient. v2 gives
     partial credit for correctness *progress* using bench.py's own stage results.
  2. Reward floor. Original: a PASS slower than PyTorch scores log2(s) < 0, i.e. below
     a crash. v2: any PASS scores >= 0.5, above every non-PASS (max 0.3).

    PASS in any turn:  r = 0.5 + clip(log2(best_speedup), 0, 3)
    otherwise:         r = 0.3 * max over turns of progress
    progress = 0.25*[smoke_test PASS] + 0.5*(shape-sweep configs passed / total)
             + 0.25*[numerical_stability PASS]

The original src/autokernel-rlvr/agent_loop/reward.py is unchanged and still logged
for every episode, so v1/v2 curves are compared on the same metric.
"""
import math
import re

_FRAC = re.compile(r"\((\d+)/(\d+) failed\)")


def progress(stages: dict | None) -> float:
    if not stages:
        return 0.0
    smoke = str(stages.get("smoke_test") or "")
    if not smoke.startswith("PASS"):
        return 0.0
    p = 0.25
    sweep = str(stages.get("shape_sweep") or "")
    if sweep.startswith("PASS"):
        p += 0.5
    else:
        m = _FRAC.search(sweep)
        if m and int(m.group(2)):
            p += 0.5 * (1 - int(m.group(1)) / int(m.group(2)))
    if str(stages.get("numerical_stability") or "").startswith("PASS"):
        p += 0.25
    return p


def reward_v2(turns: list) -> float:
    passes = [x for x in turns if x.get("correctness") == "PASS"]
    if passes:
        best = max(x.get("speedup", 0.0) for x in passes)
        return 0.5 + max(0.0, min(3.0, math.log2(max(best, 1e-6))))
    return 0.3 * max((progress(x.get("stages")) for x in turns), default=0.0)
