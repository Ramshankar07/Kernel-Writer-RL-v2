# SFT-before-RL: failure analysis of GRPO v1/v2 + agent_eval rollouts

Data: `results/v3/sft_failure_analysis.json`. This was a local analysis only; no GPU work was run.
- **Unit**: one benchmarked turn, meaning one proposed `kernel.py`.
- **Turns**: 9,393 turns in 2,225 trajectories (GRPO v1: 18 steps; GRPO v2: 10 steps; agent_eval fbv1, fbv2, kb_base, kb_lora and smoke).
- **Failing turns**: 8,267. These verdicts come from the **original** bench (`bench.py` via `bench_modal.py`), not bench v2.
- **Stage** is the earliest stage that failed, in this order: format > python > compile > runtime > numerics > timeout.
- **Backfilled rows**: the fbv1 rows have no `fail_lines`. 108 of them were filled in from an identical (kernel_type, code_sha) that was benched elsewhere. The other 132 stay as `numerics/wrong_output(detail_not_logged)`. They reached the perf table, so they did not crash.

**Environment**: Triton **3.2.0** and torch 2.6.0+cu124. `modal_app/common.py` pins `pip_install("torch==2.6.0")`, which pulls in Triton 3.2.0. `results/v3/rmsnorm_remeasure.json` records `triton.__version__ = 3.2.0`. The GPUs are an L4 (shared memory limit 101,376 B) for GRPO and the autokernel eval, and an H100 for the KB eval.

## 1. Failure buckets

### By stage (all 8,267 failing turns)
| stage | n | % | % excl. starter resubmits (6,534) |
|---|---|---|---|
| numerics (ran, wrong output) | 3,770 | 45.6 | 37.3 |
| compile (Triton frontend / codegen) | 2,590 | 31.3 | 36.6 |
| format / protocol | 745 | 9.0 | 11.4 |
| python (syntax, wrapper, contract) | 613 | 7.4 | 9.4 |
| runtime (launch/GPU) | 501 | 6.1 | 4.6 |
| timeout | 24 | 0.3 | 0.4 |
| unknown / infra | 24 | 0.3 | 0.4 |

**Resubmitted starters**: 1,733 failing turns (21.0%) are the starter resubmitted with no changes. The user prompt says "Start by benchmarking the starter". Under the original full bench, only the cross_entropy, reduce and softmax starters PASS. The other 6 starters FAIL.

### Buckets (% of all failing turns)
| bucket | n | % |
|---|---|---|
| numerics/exceeds_tolerance | 3,195 | 38.6 |
| compile/missing_tl_api_in_3.2 | 780 | 9.4 |
| format/tool_call_or_result_text_in_code (`autokernel_bench(...)`, `import autokernel_bench`, `PASS` as a name) | 585 | 7.1 |
| compile/wrong_kwarg_or_signature | 439 | 5.3 |
| numerics/nan_or_inf | 436 | 5.3 |
| compile/unsupported_python_in_jit (tensor item assignment, `x[i]`, `**`, python `min()`, f-string print, calling a non-jit fn) | 410 | 5.0 |
| runtime/shared_mem_out_of_resources | 376 | 4.5 |
| compile/pointer_arithmetic_or_ptr_type | 251 | 3.0 |
| python/syntax_error (genuine) | 169 (+54 unterminated string, +5 truncated) | 2.0 |
| compile/constexpr_misuse | 150 | 1.8 |
| compile/shape_broadcast | 150 | 1.8 |
| compile/undefined_name_in_jit | 138 | 1.7 |
| python/triton_launch_or_autotune_misuse | 136 | 1.6 |
| runtime/illegal_memory_access (OOB) | 125 | 1.5 |
| compile/unknown (message truncated by the 600-char tail) | 124 | 1.5 |
| python/name_error, other wrapper errors, kernel_fn contract, bad imports, module-level self-test asserts | 106 / 115 / 89 / 24 / 33 | 4.5 total |
| compile/dtype_mismatch (loop-carried fp16 to fp32, operand dtype, LLVM `inElemTy.isF32` conversion assert) | 77 + 15 | 1.1 |
| compile/tl_dot_constraints (M/N/K >= 16, 2D/3D, K mismatch) | 46 | 0.6 |
| format/no_code_block (+16 truncated at max_tokens) | 75 + 16 | 1.1 |
| format/prose_or_fake_bench_output_inside_code_block | 107 | 1.3 |

**Compile sub-buckets.** There are 2,590 compile failures:
- **Missing API** (30%). This covers APIs removed in 3.x or never real:
  - `tl.math.tanh` ×281 (almost all fused_mlp GELU; in 3.2 it is `tl.extra.cuda.libdevice.tanh`)
  - `tl.tanh` ×36, `tl.barrier` ×18, `tl.math.pi` ×16, `tl.bool` ×16, `tl.syncthreads` ×15
  - `tl.mean`, `tl.shared`, `tl.sync_mem`, `tl.debug_print`, `tl.isnan`, `tl.reduce_sum`, `tl.all_reduce`, `tl.clip`
- **Wrong kwarg or signature** (17%):
  - `tl.sum(dtype=)` ×58, `tl.sum(mask=)` ×54, `tl.sum/max(keepdim=)` ×50
  - `tl.zeros(N)` with an int instead of a tuple ×41, invalid `axis` ×40, `tl.arange(BLOCK)` ×34
- **Python in `@jit`** (16%).
- **Pointer** (10%). Mostly `tl.load` on an int64 offset tensor that is not a pointer (reduce ×135), or a block mask on a scalar pointer.
- **Constexpr** (6%). `arange` args not constexpr ×58, and globals used without constexpr.
- **Shape/broadcast** (6%).
- **Undefined name in the kernel** (5%).

**Numerics caveat.** 35% of numerics failures are unchanged starters. Another 52% come from matmul, layernorm and rotary_embedding, where the original bench has known false negatives (see the `results/v3/bench_v2.md` audit: reduced-precision cuBLAS split-K reference, bf16 rtol below unit roundoff). Only 1,515 numerics failures are non-starter code in the other kernel types. Treat numerics labels as unreliable for those 3 types.

### Per kernel type (% of that type's failing turns)
| kernel type | turns | PASS % | fails | format | python | compile | runtime | numerics | timeout | starter-resubmit % of fails | top bucket |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cross_entropy | 1025 | 35.1 | 665 | 10.4 | 8.0 | 38.4 | 2.4 | 40.6 | 0.3 | 0.0 | numerics/exceeds_tolerance (39.9%) |
| flash_attention | 1025 | 0.0 | 1025 | 10.6 | 4.4 | 22.6 | 24.3 | 37.7 | 0 | 19.9 | numerics/nan_or_inf (34.5%); smem OOR 24.1% (starter itself OORs on L4 gqa) |
| fused_mlp | 1025 | 0.0 | 1025 | 9.2 | 7.8 | 44.5 | 3.8 | 34.5 | 0 | 19.2 | compile/missing_tl_api (37.5%, `tl.math.tanh`) |
| layernorm | 1025 | 0.0 | 1025 | 7.5 | 3.5 | 16.9 | 0.1 | 71.6 | 0 | 34.7 | numerics/exceeds_tolerance (67.6%) |
| matmul | 1025 | 0.0 | 1025 | 7.3 | 4.4 | 28.5 | 10.8 | 48.4 | 0.6 | 27.2 | numerics/exceeds_tolerance (46.4%) |
| reduce | 1025 | 42.3 | 591 | 10.8 | 9.3 | 61.3 | 5.1 | 13.5 | 0 | 0.0 | compile/pointer (24.5%) |
| rmsnorm | 1025 | 1.3 | 1012 | 8.6 | 5.2 | 29.9 | 0 | 56.2 | 0 | 29.1 | numerics/exceeds_tolerance (54.1%) |
| rotary_embedding | 1025 | 0.0 | 1025 | 7.9 | 8.5 | 8.6 | 3.2 | 71.3 | 0 | 39.3 | numerics/exceeds_tolerance (70.2%) |
| softmax | 1025 | 31.1 | 706 | 12.2 | 8.1 | 52.7 | 2.7 | 20.8 | 2.3 | 0.0 | compile/unsupported_python_in_jit (13.5%) |
| KernelBench L1 | 168 | 0.0 | 168 | 1.8 | 60.7 | 33.9 | 1.8 | 1.8 | 0 | n/a | compile/unknown (24.4%), launch misuse (16.7%) |

### Per source (% of that source's failing turns)
| source | turns | PASS | format | python | compile | runtime | numerics | timeout |
|---|---|---|---|---|---|---|---|---|
| grpo_v1 | 5184 | 600 | 8.6 | 5.9 | 32.8 | 6.2 | 45.9 | 0.3 |
| grpo_v2 | 2880 | 364 | 9.1 | 6.5 | 30.6 | 6.9 | 46.3 | 0.2 |
| eval autokernel fbv1 | 576 | 59 | 16.6 | 11.0 | 23.0 | 2.7 | 46.6 | 0 |
| eval autokernel fbv2 | 576 | 101 | 7.0 | 3.8 | 29.7 | 5.1 | 53.5 | 1.1 |
| eval kb_base | 84 | 0 | 3.6 | 45.2 | 45.2 | 3.6 | 2.4 | 0 |
| eval kb GRPO-v2 LoRA | 84 | 0 | 0 | 76.2 | 22.6 | 0 | 1.2 | 0 |

GRPO did not change the shape of the failure distribution: v1 and v2 match within about 2 points on every stage.

### Top 15 recurring error messages
Each message is normalized (digits replaced by N). The example snippets are exact lines from the rollouts; the error marker was added.

| # | n | % fails | bucket | message | main kernel types |
|---|---|---|---|---|---|
| 1 | 425 | 5.1 | format/tool_call | `NameError: name 'autokernel_bench' is not defined` | fused_mlp, rotary, flash_attn, layernorm, matmul |
| 2 | 376 | 4.5 | runtime/smem OOR | `OutOfResources: out of resource: shared memory, Required: N, Hardware limit: 101376` | flash_attention 247, matmul 98, fused_mlp 31 |
| 3 | 281 | 3.4 | compile/missing API | `AttributeError: module 'triton.language.math' has no attribute 'tanh'` | fused_mlp 279 |
| 4 | 179 | 2.2 | compile/python-in-jit | `ValueError: Did you forget to add @triton.jit ? (_builder argument must be provided outside of JIT functions.)` | cross_entropy 88, softmax 27, rmsnorm 23 |
| 5 | 143 | 1.7 | compile/unknown | `CompilationError: at N:N:` (message cut from the tail) | kb 41, reduce, softmax, rmsnorm |
| 6 | 140 | 1.7 | compile/pointer | ``ValueError: Unsupported ptr type <[N], int64> in `tl.load` `` | reduce 135 |
| 7 | 126 | 1.5 | runtime/OOB | `RuntimeError: CUDA error: an illegal memory access was encountered` | rotary 33, reduce 30, softmax 19 |
| 8 | 107 | 1.3 | format/prose-in-block | `SyntaxError: invalid syntax` (fake bench output such as `PASS, 3.8, 97%, 4.6` inside the code block) | softmax, flash_attn, layernorm, rmsnorm |
| 9 | 99 | 1.2 | compile/python-in-jit | bare `AssertionError:` on tensor item assignment `C_ptr[i, j] = acc[i, j]` | softmax 37, matmul 17 |
| 10 | 96 | 1.2 | compile/shape | `ValueError: Cannot make_shape_compatible: incompatible dimensions at index N: N and N` | matmul 50, reduce 22 |
| 11 | 67 | 0.8 | python/launch | `RuntimeError: Cannot call @triton.jit'd outside of the scope of a kernel` (e.g. `tl.cdiv` in the host wrapper) | kb 22, softmax 14 |
| 12 | 63 | 0.8 | format/tool_call | `SyntaxError: ... Perhaps you forgot a comma?` (`autokernel_bench(KERNEL_TYPE, [ """...`) | rmsnorm 24 |
| 13 | 58 | 0.7 | compile/constexpr | `ValueError: arange's arguments must be of type tl.constexpr` | softmax 20, cross_entropy 18 |
| 14 | 58 | 0.7 | compile/kwarg | `TypeError: sum() got an unexpected keyword argument 'dtype'` | rmsnorm 43, layernorm 15 |
| 15 | 55 | 0.7 | python/contract | `AttributeError: module 'kernel' has no attribute 'kernel_fn'` | spread |

Example snippets:
```python
# 1  tool call written as code
    return output.view(orig_shape)
autokernel_bench(KERNEL_TYPE, kernel_fn)          # <-- error
# 3  removed API (3.2: tl.extra.cuda.libdevice.tanh, or 2*sigmoid(2x)-1)
gate_activated = 0.5 * acc_gate * (1.0 + tl.math.tanh(0.7978845608 * (acc_gate + 0.044715 * acc_gate*acc_gate*acc_gate)))
# 6  loading an offset tensor, not base_ptr + offsets
x = tl.load(col_idx, mask=mask, other=0.0)
# 9  tensor item assignment inside @jit
C_ptr[pid_m * BLOCK_SIZE_M + tid_m, pid_n * BLOCK_SIZE_N + tid_n] = acc[tid_m, tid_n]
# 11 device fn on host
rmsnorm_kernel[(M, tl.cdiv(N, BLOCK_SIZE))](...)
# 14 torch-ism kwargs
sq_mean = tl.sum(x * x, axis=0, dtype=tl.float32) / N
```
The per-message JSON keeps an example location, the raw message and a snippet (`top15_error_messages`).

### Stuck loops
- **Same error**: in 6,052 consecutive fail→fail pairs within a trajectory, turn k repeated turn k-1's error **54.5%** of the time (same stage, bucket and normalized message; numerics compared by bucket only). They matched on bucket 58.2% of the time. **29.2% of pairs resubmitted byte-identical code.**
- **By source**:
  - GRPO v1/v2: 52% same error, 24% identical code.
  - Eval fbv1: 71% / 61%. Eval fbv2: 65% / 52%.
  - KB: 61–63% / 32–36%.
- **Long runs**: 41.8% of trajectories with 3 or more turns contain a run of at least 3 identical errors.
- **Recovery**: P(next turn PASS | this turn failed) is 6.0% after format, 2.4% after python, 1.2% after compile, 0.7% after runtime and 1.0% after numerics. The model rarely repairs from feedback.

## 2. SFT-usable positives (bench v2 PASS)
Source is `results/v3/rescore_bench_v2.jsonl`: 346 PASS rows. All 346 code bodies were recovered from the rollouts by `code_sha12`, and each was verified against the full sha256.

| kernel type | PASS (exact sha) | distinct whitespace-norm | **distinct AST** (comments/docstrings/format stripped) | distinct AST, numeric literals masked | rows == starter (AST) | distinct non-starter | … with starter similarity < 0.9 | median starter similarity |
|---|---|---|---|---|---|---|---|---|
| reduce | 182 | 177 | **107** | 78 | 23 | 106 | 28 | 0.965 |
| softmax | 94 | 91 | **67** | 62 | 17 | 66 | 18 | 0.971 |
| cross_entropy | 70 | 63 | **40** | 35 | 17 | 39 | 8 | 0.979 |
| **total** | 346 | 331 | **214** | 175 | 57 | 211 | 54 | |

- **Starter copies**: 57 of the 346 rows are the starter unchanged (whitespace or comment differences only). They reduce to 3 distinct programs, one per type. That leaves **211 distinct non-starter programs**. Masking numeric literals, which merges block-size or num_warps variants, collapses them to about 172. Only 54 differ materially from the starter (difflib ratio < 0.9), and only 5 have ratio < 0.7.
- **Speed**: none is significantly faster than the starter (`rescore_summary.json`: 0 with CI > 1.05×; survivors' median vs-starter geomean is about 0.998–0.999). These positives teach correct **format and API usage**, not optimization.
- **Kernel types with 0 positives**: rmsnorm, layernorm, matmul, flash_attention, fused_mlp and rotary_embedding. rmsnorm's 12 old passes all FAIL v2 (fp16 overflow on the adversarial inputs).
- **Starters under bench v2**:
  - PASS: softmax, layernorm, rmsnorm, cross_entropy, rotary_embedding and reduce.
  - FAIL: matmul (fp32 tolerance) and flash_attention (shared-memory OOR on L4 gqa).
  - CRASH: fused_mlp.

  So matmul, flash_attention and fused_mlp have neither a positive nor a working starter. SFT data for them has to come from outside the repo (for example Triton tutorials adapted to 3.2 and verified with bench v2).

## 3. Prompt and format the policy sees (so SFT data can match it)
- **Model and sampling**: `Qwen/Qwen2.5-Coder-7B-Instruct` via vLLM `llm.chat` with the HF Qwen2.5 chat template and top_p=0.95.
  - Eval: T=0.8, max_new=4096, 8 turns.
  - GRPO: T=1.0, max_new=3072, group 8, max_turns 4, LoRA r=64.
  - Context is capped at 32k: a trajectory stops when prompt_tokens + 2·max_new > 32768.
- **System (autokernel)**: `build_dataset.SYSTEM_PROMPT` followed by `agent_eval.PROTOCOL`.
  - SYSTEM_PROMPT describes a *function tool* `autokernel_bench(kernel_type, code)` returning PASS/FAIL/TIMEOUT/CRASH, speedup_vs_pytorch, pct_peak and latency_us. It also gives rules (only PASS counts; send the FULL kernel.py each turn) and hints on memory-bound vs compute-bound kernels.
  - PROTOCOL: "instead of a function call, end every reply with the FULL kernel.py in one ```python fenced block".
  - These two parts contradict each other. That is the likely source of the 7.1% tool-call-in-code failures.
- **User (autokernel)**: `user_prompt(kt, SHAPE_SWEEP[kt][-1], "float16", reference.py, starter, max_turns)`. It contains:
  - `Kernel type / Target shape / Target dtype / Edit budget: N proposals`
  - the whole `reference.py` (all 10 `*_ref` functions, 3 KB) in ```python
  - the starter `kernels/<kt>.py` in ```python
  - "Start by benchmarking the starter so you see the baseline. Then iterate. Call autokernel_bench with the full kernel.py contents."

  This comes to about 2,028 prompt tokens.
- **User (KB)**: `KB_SYSTEM` plus "KernelBench Level 1, problem N." + ```python Model``` + "Write ModelNew." About 413 tokens.
- **Answer extraction**: regex ``` ```(?:python|py)?\s*\n(.*?)``` ```. The **last** fenced block is used. If there is none, the turn is a CRASH with "no ```python code block in reply".
- **File contract**: the file must define `KERNEL_TYPE = "<kt>"` and `kernel_fn(...)` with the starter's signature, implemented with `@triton.jit` kernels. `bench.py` imports it as module `kernel`.
- **Observation (feedback v2)**: this is what GRPO v1/v2 and eval fbv2 used. It is a user message:
  ```
  autokernel_bench result:
  {"correctness": "FAIL", "speedup_vs_pytorch": 0.0, "pct_peak": ..., "latency_us": ..., "pytorch_latency_us": ..., "bottleneck": "...", "stages": {"smoke_test": ..., "shape_sweep": ..., "numerical_stability": ..., "determinism": ..., "edge_cases": ...}}

  failures:
  <up to 12 fail_lines, e.g. "FAIL: large torch.float16 -> max_abs_error=... exceeds tol(...)" / "ERROR: small -> CompilationError: at 24:31:">

  log tail:            # CRASH only
  <last 800 chars of bench log>
  ```
  - Empty keys are dropped. A PASS gets only the JSON line.
  - Feedback v1 (eval fbv1) sends only the last 1,500 chars of the log tail.
  - KB observations use the same `autokernel_bench result:` header.
  - The CompilationError fail_line holds only `at L:C:`. The real cause, such as `TypeError: sum() got an unexpected keyword argument 'dtype'`, reaches the model only through the CRASH log tail.
- **Loss mask** (`train_grpo.build_masked`): every assistant turn is trained, including `<|im_end|>`. System, user and observation tokens are masked. SFT should use the same multi-turn masking.

## Implications for SFT data
1. **Format**: fix the contract, which accounts for about 9% of failures plus 1.1% contract/import errors. Every target reply should end with exactly one full `kernel.py` block. It should never contain `autokernel_bench(...)`, fake result text, or module-level tests.
2. **API**: 3.2-correct API examples would cover about 17% of all failures.
   - GELU with `tl.extra.cuda.libdevice.tanh` or sigmoid.
   - `tl.sum(x, axis=0)` with a `.to(tl.float32)` before it, and `keep_dims` rather than `keepdim`.
   - `tl.zeros((B,), ...)`, `base_ptr + offsets`, constexpr BLOCK sizes, and `triton.cdiv` rather than `tl.cdiv` on the host.
   - No tensor item assignment inside `@jit`.
3. **Resources**: `num_stages` and block sizes sized for the L4's 101 KB shared memory (the flash_attention starter itself OORs there).
4. **Repair turns**: include multi-turn repair demonstrations (error → targeted fix) to break the 54% same-error loop. Drop "resubmit the starter" turns, or at most keep one.
5. **Positives**: the 211 distinct non-starter positives cover only reduce, softmax and cross_entropy. They are mostly near-starter. The other 6 types need external, v2-verified kernels.
