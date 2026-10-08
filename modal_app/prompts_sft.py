"""
SFT prompts for the ModelNew format (Phase 1 step 6). The AutoKernel format reuses prompts_v2
unchanged (SYSTEM_PROMPT_V2, user_prompt_v2, OBS_LABEL_V2).

SYSTEM_PROMPT_MODELNEW is agent_eval.KB_SYSTEM (the prompt the KernelBench eval used) rewritten
in prompt v2's voice and contract style:
  * the "How your ... is evaluated (there is no function to call)" block from SYSTEM_PROMPT_V2:
    the last ```python block is saved and run automatically, contains only Python, and the bench
    result comes back as the next message (labelled OBS_LABEL_V2, "bench result:");
  * the same Rules bullets as v2 where they apply (only PASS counts, full file every turn, revise
    after each result), with the ModelNew contract (same constructor signature and outputs as
    Model, atol=rtol=1e-2, Triton kernels for the heavy ops) instead of the kernel_fn contract;
  * KB_SYSTEM's "speedup vs the PyTorch Model" is kept as the reported metric.
The user message is the module source + "Write ModelNew." (agent_eval.kb_tasks' layout, without
the "KernelBench Level 1, problem N." header, since SFT rows are not KernelBench problems).
"""
from prompts_v2 import OBS_LABEL_V2

SYSTEM_PROMPT_MODELNEW = """You are an expert GPU kernel engineer. You write Triton kernels for NVIDIA GPUs.

You are given a PyTorch `Model` class plus get_inputs()/get_init_inputs(). Write a `ModelNew`
class with the same constructor signature and outputs (atol=rtol=1e-2) that runs faster, using
custom Triton kernels (@triton.jit) for the heavy ops.

How your code is evaluated (there is no function to call):
  End every reply with the FULL file in one ```python fenced block: imports, your Triton
  kernels, and class ModelNew. The last ```python block of your reply is saved and run on a
  real GPU automatically; the bench builds ModelNew(*get_init_inputs()), calls it on
  get_inputs() and compares the outputs with Model's. The block must contain only Python
  source (no bench calls, no results). The bench result comes back as the next message:
    correctness (PASS/FAIL/TIMEOUT/CRASH), failure details, speedup vs the PyTorch Model.

Rules:
  * Only PASS results count. A fast but wrong kernel scores zero.
  * Each turn, propose the FULL file (Triton kernels + class ModelNew).
  * The heavy ops must run in your Triton kernels, not in a PyTorch fallback.
  * After each bench result, revise. Think about what the result tells you:
    crashes imply indexing or launch bugs; wrong values imply a numerics or masking bug.

Your goal: maximize speedup while maintaining PASS."""


def user_prompt_modelnew(module_src: str) -> str:
    return f"```python\n{module_src.rstrip()}\n```\n\nWrite ModelNew."


OBS_LABEL = OBS_LABEL_V2
