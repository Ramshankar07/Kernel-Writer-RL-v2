# Phase 0: fixed starters, prompt v2, held-out validation set, base-model baseline

Date: 2026-09-30. Stack: torch 2.6.0+cu124, Triton 3.2.0. Bench GPU: L4. Policy GPU: L40S (vLLM 0.8.5).
Plan: `sft_plan.md` Phase 0 and Phase 1 step 5. No old harness, prompt or result file was edited. Every change below is a new file.

## What changed (new files)

| file | purpose |
|---|---|
| `modal_app/starters_v2/*.py` | all 9 AutoKernel starters: 6 copied unchanged, 3 fixed (matmul, flash_attention, fused_mlp) |
| `modal_app/phase0_starters.py` | runs the bench v2 app (`bench_v2_modal.py`, unmodified) on the v2 starters; also `probe` for ad-hoc kernels |
| `modal_app/prompts_v2.py` | prompt v2 (system + user + observation label) |
| `modal_app/val_set_v1.py` | builds the validation set (`results/v3/val_set/`, `results/v3/val_set.json`) |
| `modal_app/val_bench_core.py`, `modal_app/val_bench_modal.py` | validation harness (app `autokernel-val-bench`), using bench v2 conventions |
| `modal_app/val_eval.py` | multi-turn baseline eval plus summary (`summarize`) |

Results: `phase0_starters.jsonl` (bench v2, 9 starters), `phase0_matmul_exact64_control.jsonl`, `val_set_starters.jsonl` (62 val starters and their per-problem timing baselines), `phase0_runs/*.jsonl` (full trajectories with messages), `phase0_baseline.jsonl` / `phase0_baseline.json`, `phase0_prompt_diff.txt`, `logs/`.

## 1. Starters under bench v2 (L4; fresh seed; full case list; 60 reps with cold L2)

| kernel | bench v2 | cases | failing cases | timing shape | starter_v2 us | upstream starter us | eager us | torch.compile us | vs eager | vs compile |
|---|---|---|---|---|---|---|---|---|---|---|
| matmul | FAIL* | 38/41 | sweep/{xlarge, deep_k, llm_mlp}/float32 | large | 350.2 | 327.7 | 289.8 | 290.8 | 0.829 | 0.830 |
|  |  |  |  | llm_qkv | 394.2 | 353.3 | 299.0 | 299.0 | 0.756 | 0.758 |
| softmax | PASS | 34/34 |  | large | 287.7 | 287.7 | 287.7 | 287.7 | 1.000 | 1.000 |
|  |  |  |  | vocab | 42708.5 | 42711.8 | 3526.7 | 3973.1 | 0.083 | 0.093 |
| layernorm | PASS | 33/33 |  | large | 151.0 | 150.5 | 155.6 | 150.5 | 1.034 | 1.000 |
|  |  |  |  | llm_7b | 293.4 | 290.8 | 300.5 | 291.8 | 1.023 | 0.993 |
| rmsnorm | PASS | 20/20 |  | large | 292.9 | 292.9 | 841.7 | 291.8 | 2.874 | 0.997 |
|  |  |  |  | medium | 76.8 | 80.9 | 136.2 | 76.8 | 1.768 | 1.000 |
| flash_attention | PASS | 25/25 |  | large | 242.7 | 399.4 | 5903.4 | 2482.7 | 24.337 | 10.257 |
|  |  |  |  | xlarge | 844.3 | 1414.7 | 27420.2 | 9142.8 | 32.415 | 10.843 |
| fused_mlp | PASS | 27/27 |  | large | 7390.2 | - (crashed) | 3001.3 | 2820.1 | 0.406 | 0.381 |
|  |  |  |  | medium | 559.1 | - (crashed) | 341.5 | 323.6 | 0.610 | 0.578 |
| cross_entropy | PASS | 28/28 |  | large | 1166.8 | 1166.3 | 2373.6 | 1176.6 | 2.034 | 1.008 |
|  |  |  |  | gpt2 | 6421.0 | 6504.2 | 3655.7 | 1959.9 | 0.569 | 0.306 |
| rotary_embedding | PASS | 28/28 |  | large | 223.2 | 223.7 | 459.8 | 148.0 | 2.055 | 0.664 |
|  |  |  |  | llm_7b | 224.3 | 225.3 | 461.8 | 151.6 | 2.062 | 0.674 |
| reduce | PASS | 15/15 |  | large | 662.5 | 663.0 | 653.3 | 571.4 | 0.986 | 0.861 |
|  |  |  |  | wide | 363.5 | 363.0 | 361.5 | 321.5 | 0.994 | 0.885 |

The "upstream starter us" column comes from the bench v2 baseline jobs (`bench_v2_sanity.jsonl`) and is shown for comparison. The flash_attention and matmul upstream starters were timed there even though they fail correctness.

### The three fixes
- **fused_mlp**
  - Cause of the crash: Triton 3.2's JIT resolves every attribute in the kernel body, so the unused GELU branch's `tl.math.tanh` raised `AttributeError` at launch.
  - Fix: GELU uses tanh(y) = 2·sigmoid(2y) − 1.
  - Two more fixes were needed to PASS:
    - fp32 inputs use `input_precision="ieee"`.
    - The gate·up intermediate is kept in fp32. In fp16 it overflowed on bench v2 `big_act` (|silu(g)·u| > 65504). The down projection runs in fp32 with TF32 off.
  - Cost: the starter is 0.41× eager at `large`, because the down projection is an fp32 SGEMM.
- **flash_attention**
  - Q·Kᵀ now runs on the fp16/bf16 operands with fp32 accumulation. The old fp32 upcast needed 128 KB of shared memory and made `tl.dot` use TF32.
  - head_dim ≥ 128 uses BLOCK_N=32 and num_stages=2 to fit the L4 limit of 101,376 B.
  - Side effect: the starter is 1.6–1.7× faster than the upstream one.
- **matmul**
  - fp32 inputs:
    - Fix: IEEE `tl.dot` per 32-wide K tile, with the partials summed in fp64 so the result is essentially exact.
    - Why: TF32 gave max error 4e-2 against a tolerance of 1e-4.
  - fp16/bf16 inputs:
    - The upstream starter also failed the bench v2 adversarial case `big_scale_K4096`: 60/1M elements, max error 7.9 near zero outputs.
    - Cause: tensor-core fp32 accumulation truncates, and the bias grows linearly over K=4096. An exact fp64 matmul passes this case.
    - Fix: accumulate each 2 K-tiles (`tl.static_range`, so the loop still pipelines) in a fresh fp32 block, then add the blocks with round-to-nearest fp32 adds.
    - Cost: 350 vs 328 us at `large`. A plain `range` inner loop cost 3×, which is why `tl.static_range` is used.

### Tolerance decision (matmul fp32)
matmul still FAILs 3 bench v2 fp32 cases: xlarge, deep_k and llm_mlp, all with K = 4096–8192.
- **An exact result fails them too.** The control `torch.matmul(A.double(), B.double())` cast back to the input dtype FAILs exactly these 3 cases (`phase0_matmul_exact64_control.jsonl`).
- **Cause:**
  - For matmul, bench v2's golden is itself fp32 (cuBLAS SGEMM, TF32 off).
  - Its own rounding error at K ≥ 4096 exceeds the upstream fp32 atol of 1e-4 on near-zero outputs.
  - Only a kernel that reproduces cuBLAS's summation order can pass.
- **Decision:**
  - bench v2 is left unchanged, for comparability.
  - The matmul starter is treated as PASS on every feasible case (38/41; the only failures are the 3 cases the exact control also fails).
  - The validation harness uses an fp64 golden for fp32 inputs of the matmul-type families (next section). All 7 matmul validation starters PASS there, including fp32 at K up to 4096.

### Harness finding: torch 2.6 CUDA fp64 softmax is wrong
On L4 with torch 2.6.0+cu124, CUDA **float64** `F.softmax`, `F.log_softmax` and `F.cross_entropy` return wrong values when the softmax dim is in roughly 257 < n ≤ 1024 and n % 8 == 1:
- errors reach 0.02 in probabilities and 0.9 in log-probs, for example at n = 513, 777 or 1017;
- one run hit an illegal memory access;
- CPU fp64 and CUDA fp32 are correct.

It showed up as a correct Triton starter FAILing `softmax/nonpow2_cols` at 333×777.

Impact:
- **bench v2:** its softmax and CE shapes (128, 512, 1000, 1023, 1024, 4096, 4097, 50257, …) avoid the affected range.
- **KB v2:** not checked.
- **val harness:** computes softmax-family and cross-entropy goldens on the CPU in fp64.

## 2. Prompt v2 (`modal_app/prompts_v2.py`; full diff in `phase0_prompt_diff.txt`)

**Problem.** v1 had a protocol contradiction:
- The system prompt described a callable tool, `autokernel_bench(kernel_type, code)`.
- The user prompt ended "Call autokernel_bench with the full kernel.py contents".
- An appended PROTOCOL paragraph said the opposite: end every reply with a fenced block.

**Change.** v2 replaces the tool description with a description of the fenced-block protocol:
- The last ```python block is run automatically, and there is no function to call.
- The bench calls `kernel_fn` with the starter's arguments.
- The block must contain only Python.
- The result fields listed are the ones the bench now returns.

The appendix is folded in. The user prompt's last line becomes "Reply with the full kernel.py contents in one ```python block.", and observations are labelled `bench result:`.

**Kept identical.** The rules, the hints (including the stale "pct_peak" hint), the reference/starter layout, the edit budget and "Start by benchmarking the starter" are word-for-word v1.

**Effect.** Tool calls written as code fell from 7.1% of failing turns (v1 rollouts) to 2/497 = 0.4% here. Malformed or missing code fences did not fall: 90 turns (18.1% of failures). 81 of those 90 replies contain a fence, but it is unterminated or garbled (for example "``" or "``````"), so the parser, which is unchanged, finds no complete block. This is degenerate sampling, not protocol confusion. This comparison is not like-for-like: the problems, bench and starters are all new.

## 3. Validation set v1 (`results/v3/val_set/`, `results/v3/val_set.json`)

The set has 62 problems and 376 correctness cases, and no KernelBench problems. Each problem is a directory with three files:
- `reference.py`: `reference(**inputs)`, shown to the model;
- `starter.py`: a Triton `kernel_fn` that PASSes;
- `problem.json`: generator, shapes × dtypes, adversarial cases, timing shape, tolerances and tags.

The tags (`family` plus variant tags) are there so SFT data can be decontaminated at the op level and results can be split into families seen and unseen in SFT.

| family | n | variants | cases |
|---|---|---|---|
| softmax | 7 | nonpow2_cols, bf16_wide, online_large_N (rows of 300k–1M), dim0, scaled, log_softmax, 4d_scores (197×197) | 46 |
| layernorm | 7 | no_bias, no_affine, bf16, 3d_eps1e-6, large_D (100k–300k), residual_add, fp32 | 48 |
| rmsnorm | 7 | N16384, nonpow2 (5120/3000), bf16, 3d, gemma_offset (1+w), residual_add, fp32 | 46 |
| reduce | 7 | sum_nonpow2, max, mean, sum_dim0, sum_dim1_3d, sum_bf16_long (4M), sum_of_squares | 39 |
| cross_entropy | 7 | nonpow2_vocab, vocab_151936, reduction_none, ignore_index, label_smoothing, 3d_logits, bf16_vocab_100277 | 42 |
| rotary_embedding | 7 | head_dim 96 / 80 / 256 / 32, neox_rotate_half, bshd_layout, bf16 | 49 |
| matmul | 7 | nonpow2, bf16, fp32_exact, linear_xwT, bias_relu, batched_bmm, skinny_M (M=1–16) | 42 |
| flash_attention | 7 | non_causal, head_dim_128, head_dim_32, nonpow2_seq, bf16, gqa (32q/8kv), cross_attention (256×2048) | 30 |
| fused_mlp | 6 | gelu_tanh, nonpow2, bf16, relu2, gate_up_only, 3d_input | 34 |

### Harness (`val_bench_core.py`)
It reuses bench v2's helpers unchanged: golden precision, per-dtype upstream tolerances with rtol ≥ 2u, the non-finite rule, fresh seeds, adversarial magnitudes, determinism, and cold-L2 interleaved timing with a paired bootstrap.

Differences from bench v2:
- fp64 golden for fp32 inputs of matmul, attention and MLP;
- CPU fp64 golden for softmax and cross-entropy (see the finding above);
- a launch check: PASS requires at least one `@triton.jit` launch;
- in eval mode, it stops at the first failing case;
- only PASS kernels are timed, at one timing shape with 40 reps;
- vs torch.compile = paired vs_starter × the per-problem compile/starter baseline from `val_set_starters.jsonl`.

**Status:** all 62 starters PASS (with the launch check), as recorded in `val_set_starters.jsonl`.

## 4. Base-model baseline (Qwen2.5-Coder-7B-Instruct, no LoRA; prompt v2; val set; val harness)

**Setup:**
- n=4 samples × 62 problems = 248 episodes, 4 turns each (992 benchmarked turns).
- T = 0.8, top_p = 0.95, 4096 new tokens.
- Run as two independent n=2 runs.
- The 7 matmul problems were re-run in both halves after the matmul starter's speed fix (`half1+mmfix1`, `half2+mmfix2`). The first matmul episodes used the 3× slower interim starter and were discarded; they are still in `phase0_runs/half*.jsonl`.

**Solve definition:** a PASS whose code is not AST-identical to the starter.

| metric | value |
|---|---|
| pass@1 (solve) | **0.383** (per run: 0.387 / 0.379) |
| pass@4 (problems with ≥ 1 solving episode) | **0.774** (48/62) |
| pass@1 on turn 0 only | 0.145 |
| pass@1 counting unchanged-starter resubmits | 0.778 |
| beat starter (vs_starter > 1.05, CI lower bound > 1) @1 / @4 | 0.028 / 0.048 |
| compile-error rate (Triton compile errors / all turns) | **0.081** (16.1% of failing turns) |
| failing-turn mix | python/syntax/contract 39.0%, numerics 19.1%, format 18.1%, compile 16.1%, shared-memory OOR 6.0%, CUDA runtime 1.6% |
| repeat-error rate (fail→fail pairs with the same error) | **0.338** (287 pairs) |
| P(solve next turn \| failed turn) | 0.072 |
| starter resubmitted unchanged | 293/992 turns |
| solving turns within difflib 0.9 of the starter | 92.1% (186/202) |
| best speedup per solving episode vs starter (geomean / median, 95 episodes) | **0.950 / 1.000** |
| … vs torch.compile (geomean / median) | **0.846 / 0.988** |
| … vs eager (geomean) | 1.641 |

| family | pass@1 | pass@4 | compile-error rate |
|---|---|---|---|
| softmax | 0.321 | 6/7 | 0.036 |
| layernorm | 0.464 | 7/7 | 0.054 |
| rmsnorm | 0.679 | 7/7 | 0.196 |
| reduce | 0.643 | 7/7 | 0.089 |
| cross_entropy | 0.500 | 7/7 | 0.062 |
| rotary_embedding | 0.321 | 5/7 | 0.062 |
| matmul | 0.321 | 5/7 | 0.045 |
| flash_attention | 0.107 | 3/7 | 0.089 |
| fused_mlp | 0.042 | 1/6 | 0.094 |

**Reading:**
- The base model's "solves" are almost all near-copies of the starter (92% are within difflib ratio 0.9). Only 2.8% of episodes beat the starter.
- The median solve is exactly as fast as the starter.
- This is the "before" number for SFT. The pass@1 seed spread is 0.008, so under the gate's "≥ 2× the seed-to-seed spread" criterion, pass@1 must rise by at least 0.016. The beat-starter rate and the near-starter share are the more informative targets.

## 5. Cost

This is actual Modal billing, from `modal.billing.workspace_billing_report` hourly rows for this session's apps.

| item | cost |
|---|---|
| policy on L40S (smoke test + 4 runs) | $1.45 |
| `autokernel-val-bench`: starter verification (3 runs) and eval bench | $3.62 |
| `autokernel-bench-v2` (starter verification, probes, exact control) | ~$0.15 |
| debug apps (fp64 softmax isolation) | ~$0.01 |
| **total** | **≈ $5.3**, under the $6 cap |

The last ~3 minutes after 16:00 EDT (about $0.1) are not yet in the report. The ~$1.36 `autokernel-bench-v2` row from 14:00 EDT is the earlier rescore session and is not counted. The matmul re-run (2 × 7 problems) is included in the total.
