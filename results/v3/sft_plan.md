# SFT-before-RL plan (drafted 2026-09-30)

Inputs: `sft_failure_analysis.md` (what breaks), `sft_data_research.md` (data, cutoff < 2026-03-01),
`bench_v2.md` / `kb_v2.md` (fixed verifiers). Stack: torch 2.6.0, Triton 3.2.0, Qwen2.5-Coder-7B-Instruct + LoRA.

## Why SFT, not more GRPO
- Under bench v2, 0 GRPO kernels beat their starter; the old reward never saw a real speedup.
- 31.3% of failing turns are Triton compile errors; after a compile failure the next turn passes 1.2% of the time.
- Stuck loops: 54.5% of consecutive failures repeat the same error. RL can't learn from groups that never succeed.

## Phase 0: Remove failures that aren't the model's fault (~2 h work, ~$2)
1. Fix the 3 starters that fail under bench v2:
   - fused_mlp: `tl.math.tanh` is missing in 3.2; use `libdevice` or an exp-based form
   - flash_attention: out of shared memory on L4; shrink the block sizes
   - matmul: the fp32 path uses TF32, so either disable TF32 or match the tolerance
2. Fix the prompt protocol contradiction: the system prompt describes a function tool, but the parser wants a fenced block. This causes 7.1% of failures.
3. Re-baseline the base model on the fixed prompt, bench v2, and a fixed validation set (Phase 1 step 5). This is the "before" number for SFT.

## Phase 1: Build the SFT set (~1 day work, ~$11 verification)
1. Pull the pinned, pre-cutoff revisions:
   - `hkust-nlp/drkernel-coldstart-8k` @ `cba0ef06…`: take the best round as single-turn; keep the multi-turn rows as repair data
   - `GPUMODE/KernelBook` @ `1576375b…` (MIT): pure-Triton rows only, no cuBLAS/cuDNN fallbacks
   - the in-repo bench v2 positives: 211 non-starter, reduce/softmax/cross_entropy only
   - exclude `ppbhatt500/kernelbook-triton-reasoning-traces`, which is built from KernelBench
   - exclude `siro1/kernelbook-kimi_k2_thinking-evals-filtered` (user decision 2026-09-30: undocumented license and verification)
2. Static filter:
   - parses as Python
   - has ≥1 `@triton.jit` kernel that is actually launched
   - no `torch.nn.functional` or `torch.matmul` in the forward path
   - only `tl.*` symbols that exist in 3.2 (build the allowlist from `dir(triton.language)` in the bench image)
3. Dynamic verification with the bench v2 checker:
   - fp32 golden reference
   - 3 seeds × 2 shapes
   - extreme-magnitude inputs where the op has a reduction
   - a launch check (the Triton kernel ran; no PyTorch fallback)
   - use ≤4 containers
4. Decontamination:
   - no verbatim or near-duplicate (normalized-AST) match with KernelBench L1
   - tag every example with its op family, so results can be reported separately for families seen and unseen in SFT
5. Hold out a fixed validation set before training and never tune on KernelBench:
   - ~60 problems covering the 9 AutoKernel families
   - use new shapes and variants: layernorm without bias, rmsnorm at N=16384, online softmax, rotary with other head dims, bf16 inputs
   - remove their op-level near-duplicates from the SFT data
6. Convert every example to the policy's exact chat format: system prompt, user message with reference.py (and the starter if present), assistant reply ending in one full `kernel.py` block.
7. Add repair data aimed at the stuck loops:
   - failing turn + real bench feedback → verified fix
   - sources: Dr. Kernel multi-turn rows, and in-repo failures paired with a verified kernel of the same type
   - target ~15% of the mix
8. Mix: 4–6k examples, rebalanced toward the 9 families. Rotary and fused_mlp are scarce, so upsample them ≤3×.

## Phase 2: Train (~$13)
1. Pilot: 100 steps to measure throughput, then re-estimate the full cost before launching.
2. Full run: LoRA r=64, alpha 128, lr 1e-4 cosine, 2 epochs, max length 4k, loss on assistant tokens only. Launch with `spawn`, not `modal run`.
3. Save a checkpoint at every half-epoch, and evaluate each checkpoint on the validation set.

## Phase 3: Evaluate (~$5)
Base (Phase 0) vs SFT checkpoints on:
- the validation set under bench v2: compile rate, pass@1, pass@8, repeat-error rate, speedup vs starter and vs torch.compile
- KernelBench L1, all 100 problems under kb v2, once and at the end only, split into families seen and unseen in SFT

## Gate: resume GRPO only if all of these hold
- validation pass@1 improves by at least 2× the seed-to-seed spread
- the compile-error share falls clearly (target < 15% of failures)
- ≥30% of GRPO groups would have mixed outcomes (nonzero advantage) under bench v2 at the SFT checkpoint
- then resume GRPO from the SFT LoRA, rewarded by bench v2: fp32 golden, launch check, speedup vs starter, and fresh inputs with no cache

If the gate fails: move to easier curriculum tasks the SFT model sometimes solves, rather than extending RL.

## Budget
| Phase | Modal cost (estimate) |
|---|---|
| 0 | ~$2 |
| 1 | ~$11 |
| 2 | ~$13 (pilot + full) |
| 3 | ~$5 |
| **Total** | **~$31** |

The throughput behind the training estimate is an assumption; the pilot replaces it.
