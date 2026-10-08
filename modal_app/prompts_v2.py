"""
Prompt v2 for the AutoKernel agent (Phase 0 step 2). The v1 prompt is left untouched in
src/autokernel-rlvr/data/build_dataset.py (SYSTEM_PROMPT, user_prompt) + agent_eval.PROTOCOL.

The v1 contradiction: SYSTEM_PROMPT describes a callable function tool
`autokernel_bench(kernel_type, code)`, and the user prompt ends "Call autokernel_bench with the
full kernel.py contents", but the Modal agent loop has no tool parser: it benchmarks the last
```python fenced block of every reply (agent_eval.PROTOCOL, appended to the system prompt,
says so). 7.1% of failing turns wrote the tool call as code (`autokernel_bench(KERNEL_TYPE,
kernel_fn)` -> NameError, `import autokernel_bench`, bench output pasted as code).

v2 changes ONLY the protocol text; the rules, hints, reference/starter layout, edit budget and
"benchmark the starter first" instruction are the same words as v1:
  * the tool description becomes a description of the fenced-block protocol (no function name
    the model could call), listing the fields the bench v2 result actually contains
    (v1 listed pct_peak, which bench v2 does not report);
  * the separate PROTOCOL appendix is folded in (it said the same thing as a patch);
  * one sentence states the kernel.py contract the bench relies on (`kernel_fn` with the
    starter's signature) and that the code block must contain only Python;
  * the user prompt's last line says "Reply with ..." instead of "Call autokernel_bench ...";
  * observations are labelled "bench result:" instead of "autokernel_bench result:".
"""

SYSTEM_PROMPT_V2 = """You are an expert GPU kernel engineer. You optimize Triton kernels for NVIDIA GPUs.

How your kernel is evaluated (there is no function to call):
  End every reply with the FULL contents of kernel.py in one ```python fenced block.
  The last ```python block of your reply is saved as kernel.py and run on a real GPU
  automatically; the bench imports it and calls kernel_fn(...) with the same arguments as
  the starter's kernel_fn. The block must contain only Python source (no bench calls, no
  results). The bench result comes back as the next message:
    correctness (PASS/FAIL/TIMEOUT/CRASH), failure details, speedup_vs_starter,
    speedup_vs_pytorch (eager), speedup_vs_torch_compile, latency_us.

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


def user_prompt_v2(kernel_type: str, shape, dtype: str, reference_src: str,
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
Reply with the full kernel.py contents in one ```python block."""


OBS_LABEL_V2 = "bench result:"
