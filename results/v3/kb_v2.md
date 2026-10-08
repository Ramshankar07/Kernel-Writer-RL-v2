# KernelBench harness v2 ("kb v2"): identity sanity 55/100 → 100/100

Date: 2026-09-30, H100 80GB. New Modal app `autokernel-kernelbench-v2` (`modal_app/kernelbench_v2_modal.py`,
evaluator `modal_app/kb_v2_eval.py`, summary `modal_app/kb_v2_summarize.py`). The old app
(`kernelbench_modal.py` / `autokernel-kernelbench`, upstream `bench_kb.py`) is untouched. Problems
come from the same pinned cache (AutoKernel commit 7843582, bridge `fetch --source hf --level 1`).

## What changed vs `bench_kb.py`
| # | Old behaviour | kb v2 |
|---|---|---|
| a | weight init never seeded: Model and ModelNew get different random conv weights | python/numpy/torch/CUDA RNGs seeded identically (seed 42) before building Model and each ModelNew. If ModelNew's `state_dict` has the same keys and shapes, Model's weights are also copied in |
| b | CPU `get_inputs()` (up to 2.1B elements) runs inside the 30 s per-trial alarm | `get_inputs()` runs under `torch.device("cuda")`, with a per-trial seed, **outside** any timeout. The correctness timeout (120 s) covers only the forward passes. Timing has its own timeout (180 s) |
| c | bridge starter renames `class Model(` → `class ModelNew(` but keeps `super(Model, self)` | fixed starter rewrites `super(Model, self)` → `super(ModelNew, self)` in the ModelNew section only. Identity test uses `class ModelNew(Model): pass` |
| – | candidate's own `get_inputs` preferred if it defines one | always the reference's `get_inputs` / `get_init_inputs` |
| – | 5 trials, same process RNG state | 5 seeded trials (seeds 42..46) + 1 **hidden** trial with a fresh `secrets.randbits(31)` seed per run |
| – | reference under global defaults | strict fp32 reference: TF32 off for matmul and cuDNN, `float32_matmul_precision("highest")` (set globally, so the candidate is judged on its kernel and not on the flags). atol=rtol=1e-2 (KernelBench fp32 convention, same as bench_kb). A strict 1e-4 check is also reported. Comparison is chunked, so 2.1B-element outputs don't OOM |
| – | CUDA events, warm L2, trimmed median of 100 | CUDA events, 3 warmup calls, L2 flushed (256 MB write) before every rep, median of ≥30 reps (adaptive, up to 100) |

Each evaluation runs in a fresh subprocess. Run 1 and run 2 are independent processes.

## Result: identity `ModelNew(Model)` on all 100 Level 1 problems, two runs
- **Valid (PASS in both runs): 100/100** (before: 55/100, `results/kb_valid_problems.json`). No regressions.
- Max |error| across all 200 runs × 6 trials: 2.98e-7. All 100 also pass the strict 1e-4 check.
  The hidden fresh-seed trial passed in all 200 runs.
- Identity speedup (200 measurements): median 1.000, range 0.982–1.005. None fall outside 0.95–1.05.
- Weights: 38 problems had parameters that were copied from Model. The other 62 have no parameters (seeded only).
- Input generation on the GPU: ≤0.76 s per problem across all 6 trials. Max per-process wall time was 11.2 s.
- Starters (run 1): bridge starter CRASH 100/100
  (`TypeError: super(type, obj): obj must be an instance or subtype of type`); fixed starter PASS 100/100.

## Per-bug attribution (45 recovered problems)
| Fix | Recovered | Evidence |
|---|---|---|
| (a) seeded / copied weights | **35** (all conv: 50, 54–87) | Ablation `kb_v2_ablate_none_gpu.jsonl` reverts only this fix (unseeded) → 35/35 FAIL, with 50–99% of elements off and max_abs 0.55–2.9. `kb_v2_ablate_seed_gpu.jsonl` (seeding alone, no state_dict copy) → 35/35 PASS, so the seed by itself is sufficient |
| (b) inputs on GPU, outside the timeout | **10** (flaky 21, 23, 24, 27, 29, 31, 32, 35, 37 + #38 which failed every old run) | Per trial, generation time drops from 7.0–13.1 s (CPU, measured) to 0.006–0.09 s (GPU). The ablation `kb_v2_ablate_copy_cpu_timed.jsonl` (CPU generation inside a 30 s alarm, 8 CPUs) did **not** reproduce the timeouts: 10/10 PASS, worst trial 13.1 s (#38). So the old failures depended on contention. Examples: bench_kb also runs CPU generation in its stability/determinism stages, and with Modal's default CPU share 45/100 timed out (`kb_sanity_defaultcpu.jsonl`). Attribution is by category, not by a reproduced failure |
| (c) starter `super()` | 0 identity problems (the identity test already bypassed it) | Affects every agent run that starts from the bridge starter: 100/100 crash → 100/100 pass |

## Remaining failures
None. With the identity solution, all 100 Level 1 problems are harness-valid and repeatable.
Caveat: this validates the harness, not candidate kernels. The LoRA agent eval was **not** rerun on v2.
The earlier 0/28 before/after result was measured with the old harness and its 55-problem valid list.

## Cost
About 0.46 H100-hours of evaluation (sanity 928 s + ablations 722 s + smoke ≈ 60 s, plus container
start-up) at ≤4 containers. That is roughly $2–2.5 including CPU and memory (estimate).

## Files
- `results/v3/kb_v2_sanity.jsonl`: per problem, run1 (identity + fixed + bridge starters) and run2 (identity)
- `results/v3/kb_v2_valid_problems.json`: valid ids, attribution, starter and ablation summaries
- `results/v3/kb_v2_ablate_{none_gpu,seed_gpu,copy_cpu_timed}.jsonl`, logs `kb_v2_sanity.log`, `kb_v2_ablate.log`
- Reproduce: `cd modal_app && modal run kernelbench_v2_modal.py::sanity && python kb_v2_summarize.py`
