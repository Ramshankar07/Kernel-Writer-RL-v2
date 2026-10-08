# SFT data and recipes for Triton kernel generation (research note, 2026-09-30)

Context: GRPO on Qwen2.5-Coder-7B-Instruct (LoRA) for AutoKernel's 9 kernel types, with KernelBench
Level 1 as the transfer test. The reward never saw a real speedup, and the policy kept making the
same Triton compile mistakes. Plan: run SFT on correct Triton code first, then go back to RL.
Our stack is `torch==2.6.0` (`modal_app/common.py`, `modal_app/train_grpo.py`), which ships Triton 3.2.

**Data cutoff (user constraint):** recommended data must have been publicly released **before 2026-03-01**.
Papers and recipes from any date can be cited. Release dates below come from the Hugging Face
(HF) commit history (`/api/datasets/<id>/commits/main`) or the GitHub `created_at`, both checked on 2026-09-30.

Measured locally (scratchpad scripts, not committed): numbers marked **[measured]** come from
downloading the parquet and running pandas over it.

---

## 1. Datasets

### 1a. Eligible (released before 2026-03-01)

| Dataset | First public release | Size | License (at pinned rev) | How correctness was established | Triton / torch assumption | KernelBench contamination |
|---|---|---|---|---|---|---|
| **GPUMODE/KernelBook** `dataset_permissive` ([HF](https://huggingface.co/datasets/GPUMODE/KernelBook)) | 2025-03-14 (initial commit `76b2a2ce`). Data files last changed 2025-06-24 | 18,162 rows [measured]. Model card says ~25k pairs incl. synthetic ([KernelLLM card](https://huggingface.co/facebook/KernelLLM)). The released file has `synthetic=False` for every row [measured] | MIT at rev `1576375bc92745b490e1cdf2fce01eba76d9f847` (2026-02-05, LICENSE = MIT). **HEAD (2026-06-09) switched to a "Researcher Reciprocity" license.** Pin the Feb revision; the parquet LFS oid is identical at both revisions (`64af2baa…`) | Not verified per sample. The Triton code is **TorchInductor output** (`torch.compile`) from GitHub nn.Modules (The Stack). It is correct by construction, not written by a model. TritonRL found some rows are "invalid Triton" because the core op runs through `extern_kernels` ([TritonRL App. G.2](https://arxiv.org/html/2510.17891v2)) | Generated with **PyTorch 2.5.0** (Triton 3.1), per the card. All 18,162 rows import `torch._inductor` internals [measured], so they are version-fragile under torch 2.6 | **No verbatim KernelBench-L1 copies** [measured: 0/100 L1 problems have ≥80% of their distinctive lines in any single row; 0 repos named KernelBench]. **Op-level overlap is unavoidable** (softmax, norms, matmul, reductions, activations). 10,747/18,162 rows (59%) call `extern_kernels` (cuBLAS/cuDNN) for mm/conv [measured] |
| **hkust-nlp/drkernel-coldstart-8k** ([HF](https://huggingface.co/datasets/hkust-nlp/drkernel-coldstart-8k)) | 2026-02-05. Pin `cba0ef06a5b1e3c307b7acfa8b6acb7a46578105` (2026-02-06) | 8,920 five-turn trajectories. Best-round answer ≈2.1k tokens p50; full trajectory ≈14.5k tokens p50 [measured, chars/3.5] | MIT tag. **Caveat:** the paper says it was distilled from GPT-5 ([Dr. Kernel §4.1](https://arxiv.org/html/2602.05885v1)), so OpenAI output-use terms may apply | Every turn was executed in KernelGym (correctness, runtime, profiling feedback, and a hacking check that requires a Triton kernel to actually run). `final_speedup` quantiles p10/p50/p90 = 0.44/1.70/8.18; 5,405 ≥1.0×; 458 NaN or ≤0 [measured]. All best-round answers contain `@triton.jit` [measured] | Not stated. Hand-written (LLM) Triton, so much less tied to a torch version than Inductor output. Re-verify anyway | Queries come from ByteDance cudaLLM (aten/nn op compositions). **0/100 L1 verbatim matches** [measured]. Op-level overlap exists. Dr. Kernel's validation set is KernelBench **L2**, not L1 |
| **siro1/kernelbook-kimi_k2_thinking-evals-filtered** ([HF](https://huggingface.co/datasets/siro1/kernelbook-kimi_k2_thinking-evals-filtered)) | 2026-01-30. Pin `86681bb24ad398f863efd98e0499321b344ec948` (2026-02-01) | 3,883 train + 432 val. Completions ≈720 tokens p50, 1.5k p90 [measured] | **No license tag, and the card is undocumented.** Kimi-K2 outputs fall under a modified-MIT license. Treat as research-only unless the author clarifies | Scored in an eval env (a `reward` field, and `speedup_reward` p50 0.96, p90 1.36 [measured]). The verification method is not documented, so **re-verification is mandatory** | Unknown; idiomatic hand-written Triton. 3,864/3,883 have `@triton.jit`; 357 still call `F.*` [measured] | Prompts are KernelBook modules, so the KernelBook analysis applies (no verbatim L1) |
| siro1/kernelbook-glm4-evals-filtered ([HF](https://huggingface.co/datasets/siro1/kernelbook-glm4-evals-filtered)) | 2026-01-13. Pin `db0810d72f75ab84b58cc902ea506f43e948be93` | 6,462 + 719 val. Completions ≈9k tokens p50 (reasoning inline) [measured] | No tag (GLM-4.x outputs are MIT) | Same env as above. `reward` p50 1.00, p90 1.04 [measured], i.e. mostly ~1× | Unknown. 1,951 use `F.*` [measured], so there is a lot of partial "lazy" conversion | As KernelBook |
| ByteDance-Seed/cudaLLM-data ([HF](https://huggingface.co/datasets/ByteDance-Seed/cudaLLM-data)) | 2025-08-03 (`8387cbea…`) | SFT file 139 MB, RL file 394 MB | Apache-2.0 | AST checks plus CUDA execution (card) | **CUDA, not Triton.** Use only as a pool of prompts/tasks | Op compositions from aten/nn/HF Transformers. No KernelBench sourcing claimed |
| Teen-Different/Code_Opt_Triton ([HF](https://huggingface.co/datasets/Teen-Different/Code_Opt_Triton)) | 2025-03-27 | ~2× KernelBook (each row duplicated with the input swapped) | MIT | Same as KernelBook (Inductor) | PyTorch 2.5 Inductor | As KernelBook. Adds nothing new |
| thunlp/TritonBench ([GitHub](https://github.com/thunlp/TritonBench), [paper](https://arxiv.org/abs/2502.14752)) | 2025-02-20 | 184 (G, GitHub kernels) + 166 (T, PyTorch-aligned) | Apache-2.0 | Benchmark with reference kernels and tests | Older Triton | Not KernelBench, but it **is** a benchmark, so use it only as a second held-out eval, never for training |

### 1b. Not usable / contaminated / no data released

- **Kevin-32B (Cognition):** CUDA. Trained on **180 of the 200 KernelBench L1+L2 tasks** with 20 held out ([blog](https://cognition.com/blog/kevin-32b)), so it is contaminated by design. No training data released.
- **CUDA-L1:** CUDA. SFT/RL data is built from the **KernelBench reference code for all 250 tasks** ([arXiv 2507.14111](https://arxiv.org/abs/2507.14111)), so it is contaminated by design.
- **AutoTriton:** only the model was released ([GitHub](https://github.com/AI9Stars/AutoTriton), [HF model](https://huggingface.co/ai9stars/AutoTriton)). The 14,102-sample SFT set was not released.
- **KernelLLM:** model only ([card](https://huggingface.co/facebook/KernelLLM)). Its training data is KernelBook.
- **ppbhatt500/kernelbook-triton-reasoning-traces** (2026-02-13, 170 rows): the schema has `level` and `problem_id` fields, i.e. KernelBench problems, so it is **contaminated. Exclude.**

### 1c. Excluded as too recent (first release on or after 2026-03-01)

| Dataset / release | Date | Note |
|---|---|---|
| KernelBook HEAD with Researcher-Reciprocity license | 2026-06-09 | Same data as the pinned Feb revision; use rev `1576375b…` |
| TritonRL code repo (`jiinw21/TritonRL`) | GitHub created 2026-08-07 | The paper (Oct 2025) says data and code are released, but the only public repo post-dates the cutoff |
| DRTriton (CSP-DAG synthetic data) | paper 2026-03-23 ([arXiv](https://arxiv.org/abs/2603.21465)) | |
| CoopReason/Kernel-Smith-SFT-71K | 2026-07-23 | [paper 2603.28342](https://arxiv.org/abs/2603.28342) |
| autummata/kernelbook-verified | 2026-06-07 | |
| AMDKernelVault (amd/AIG-Datasets) | 2026-09-11 ([arXiv](https://arxiv.org/abs/2609.12471)) | AMD/HIP focus |
| AnodeAI/advanced-triton-kernel-traces, AnodeAI/Agent-traces-triton, beatsprom/* | Sept 2026 | |

---

## 2. Published SFT → RL recipes

| Work | Base / size | SFT data | SFT hyper-params | Full vs LoRA | RL | Result | Anti-hacking / verification |
|---|---|---|---|---|---|---|---|
| **KernelLLM** (Meta, [card](https://huggingface.co/facebook/KernelLLM)) | Llama-3.1-8B-Instruct | KernelBook (~25k pairs + synthetic) | 10 epochs, bs 32, ~12 h on 16 GPUs (192 GPU-h) | Full (standard SFT) | none | KernelBench-Triton L1 pass@1 score 20.2 | Unit tests only. The card itself says outputs "structurally resemble compiler-generated output… often fail to implement a meaningful kernel" |
| **AutoTriton** ([arXiv 2507.05687](https://arxiv.org/html/2507.05687v1)) | Seed-Coder-8B-Reasoning | 14,102 samples: GitHub/HF PyTorch → LLM-distilled Triton, compile- and execution-validated | 3 epochs, lr 1e-5, 16k ctx, ~16 h on 8×A800 | Full | GRPO, 6,302 tasks, lr 1e-6, 1 epoch, ~32 h on 16×A800 | ≈Claude-4-Sonnet / DeepSeek-R1 on TritonBench + KernelBench correctness | Rule reward requiring `@triton.jit`; invalid outputs dropped from 25→6 on KB-L1. Dr. Kernel still measured ~10% hacking in its outputs (kernel defined but never called) |
| **TritonRL** ([arXiv 2510.17891](https://arxiv.org/html/2510.17891v2), ICML'26) | Qwen3-8B | 11,621 executable KernelBook modules → **~58–60k DeepSeek-R1 (or GPT-OSS-120B) CoT+code traces** (5 per task) | 2 epochs (R1, 12k ctx) / 3 epochs (GPT-OSS, 16k), bs 16, lr 1e-5, 8×A100 | Full | GRPO with hierarchical reward decomposition, bs 32, lr 1e-6, 2 epochs | SOTA at 8B on KB L1/L2. **RL adds >20 pts correctness over SFT alone.** Found that KernelLLM and AutoTriton *increase* functional invalidity vs. their base models | "Robust verifier": `@triton.jit` linter, a check that a Triton kernel is actually called, a flag on torch.nn/`@`/`torch.matmul` fallbacks, and an LLM judge (Qwen3-235B) |
| **Dr. Kernel** ([arXiv 2602.05885](https://arxiv.org/html/2602.05885v1), ICML'26) | Qwen3-8B-Base / 14B | **8,920 five-turn GPT-5 trajectories** in KernelGym (the drkernel-coldstart-8k set) | multi-turn SFT, max_len 18,432 ([card](https://huggingface.co/datasets/hkust-nlp/drkernel-coldstart-8k)) | Full | Multi-turn RL with TRLOO (leave-one-out, unbiased vs GRPO self-inclusion), profiling-based reward and rejection sampling | 14B: 31.6% of KB-L2 kernels ≥1.2× (Claude-4.5-Sonnet 26.7%, GPT-5 28.6%) | Instruments Triton's launch path. A candidate is marked incorrect if **no Triton kernel executes** in train or eval mode. Hacking fell from ~20% to ~3% during training; 1.7% on L1 vs AutoTriton's ~10% |
| **Kevin-32B** ([blog](https://cognition.com/blog/kevin-32b)) | QwQ-32B | none (RL only) | – | – | multi-turn GRPO on 180 KernelBench tasks | correctness 56→82%, mean speedup 0.53→1.10× | Reward 0 for responses using PyTorch functions or lacking CUDA kernels. Observed hacks: copying the reference, try/except fallback, inheriting the reference class. "Reward hacking occurs when the gap between the model capabilities and the dataset difficulty is significant", which matches our GRPO failure |
| CUDA-L1 ([arXiv](https://arxiv.org/abs/2507.14111)) | – | SFT via augmentation over KernelBench refs | – | – | contrastive RL | 3.12× mean on KB (A100) | Trains on the benchmark, so not comparable for transfer |
| KernelBench-Verified ([arXiv 2607.16241](https://arxiv.org/abs/2607.16241), 2026-06) | eval paper | – | – | – | – | Frontier-model "speedups" are often inflated | Adds a TF32 baseline and checks for hard-coded bypasses on narrow test distributions. Worth adopting in our eval |

Takeaways for us:
1. Every successful Triton RL run started from an SFT checkpoint trained on **thousands** of verified examples (9k–60k), always as full fine-tunes on 8–16 GPUs. No published run used a 7B LoRA with a budget like ours, so expectations should be modest.
2. SFT on Inductor output (KernelLLM) teaches the "compiler style plus `extern_kernels`" shortcut. TritonRL shows this raises functional invalidity. **Do not SFT on raw KernelBook `triton_code`.** At most, use its *prompts* (PyTorch modules) and its pure-Triton rows.
3. The anti-hacking checks that matter are cheap and deterministic: (a) at least one `@triton.jit` kernel is **actually launched** during `forward` (hook Triton's launcher, as Dr. Kernel does); (b) no `torch.nn.functional`/`torch.matmul`/`@`/reference-class calls on the hot path; (c) no try/except fallback; (d) random seeds and several shapes, so outputs can't be hard-coded. Apply the same filter to SFT data *and* to the RL reward.

---

## 3. Recommendation for our budget

Assumptions: Modal H100 $3.95/h (L40S $1.95/h, A100-80GB $2.50/h per [modal.com/pricing](https://modal.com/pricing), checked 2026-09-30). Qwen2.5-Coder-7B-Instruct with LoRA (r=64, all linear layers), bf16, packed sequences. **Training throughput is an assumption: ~3k tokens/s on one H100 with HF+PEFT+FlashAttention-2. Measure it on a 200-step pilot before spending more.**

### Data (ranked)

1. **drkernel-coldstart-8k @ `cba0ef06…` → converted to single-turn** (prompt = original PyTorch module, target = the **best-round** assistant answer). This is the only pre-cutoff set that was execution-verified with an actual hacking check and has real speedups (median 1.7×). Filter to `final_speedup ≥ 1.0` (5,405 rows), then re-verify. Caveat: the GPT-5 distillation terms.
2. **siro1 Kimi-K2 filtered @ `86681bb2…`**: short (≈720 tokens), idiomatic, cheap to train on. Undocumented verifier and license, so re-verify everything and keep it research-only.
3. **KernelBook @ `1576375b…` (MIT revision)**: use only the 7,415 rows that are **pure Triton** (no `extern_kernels`) [measured], and ideally just as a *prompt pool*. Its Inductor style differs from what AutoKernel wants. Re-verification under torch 2.6 will drop rows whose `torch._inductor` imports broke.
4. Do not use the GLM-4 traces: they are mostly ~1× and very long (9k tokens), which is bad value per token.

### Filtering / re-verification pipeline (under our Triton 3.2 / torch 2.6)

1. Static filter: must define a `@triton.jit` kernel. Reject `extern_kernels`, `torch.matmul`/`@`/`F.linear`/`F.softmax`/`F.layer_norm` inside `forward`, `try:`, `self.training` branches, and inheritance from the reference class.
2. Dynamic check on **L40S** (cheap, correctness-only): run the reference and the candidate on 3 random seeds × 2 shapes (the dataset shape plus a 2× or odd shape). Require `allclose` (fp32 atol/rtol 1e-4, bf16 1e-2), and require that a Triton kernel launch happens (hook `triton.runtime.jit.JITFunction.run` or `CompiledKernel.__getitem__`). Batch ~50 candidates per container to amortize imports.
3. Decontamination: drop any row whose PyTorch module matches a KernelBench L1 file with ≥0.5 distinctive-line overlap (our check found none verbatim). Also keep a list of the **op-overlap subset** so KB-L1 results can be reported separately for ops seen and unseen in SFT.
4. Rebalance toward AutoKernel's 9 types. KernelBook prompt counts (regex, approximate) [measured]: softmax 2,783; layernorm 1,181; rmsnorm 516; cross_entropy 959; rotary 171; attention 2,090; gelu/silu MLP 470; matmul/linear ~7.5k. Rotary and fused-MLP are thin. Include all of them and oversample 2×.
5. Rewrite every target into **our agent's exact output format** (the AutoKernel `kernel.py` contract and ModelNew for KB). Train prompt → final-code only, with no multi-turn history, to keep sequences under 4k tokens.

### Size, time, cost

- **4k–6k verified examples**, **2 epochs**, lr 1e-4 (LoRA), cosine, 3% warmup, max_len 4,096, loss on completions only.
  Tokens: ~5k × ~3k tokens (Dr.K best-round ≈2.1k + prompt, Kimi ≈1.3k total) ≈ 15M/epoch → 30M total → **~2.8 h on 1×H100 ≈ $11** at 3k tok/s. A leaner 3k-example / 2-epoch run is ≈1 h ≈ **$4–5**.
- Re-verification: ~10k candidates × ~2 s (warm container) on L40S ≈ 5.5 GPU-h ≈ **$11**. Run it at `BENCH_MAX_CONTAINERS=8`.
- Gates before resuming RL (`results/` numbers only): (a) compile-error rate on our 9 types drops versus the base model; (b) pass@8 on AutoKernel ≥ the base model's; (c) KB-L1 (55 harness-valid) correctness > 0/28. Then resume GRPO **with the Dr. Kernel launch-hook check in the reward**, plus Dr. GRPO/leave-one-out advantages (our v1 analysis: 78.9% of |advantage| came from timing-noise groups).

Expected total: **~$15–25**, which is within the "tens of dollars" budget.

---

## Sources
- KernelBook: https://huggingface.co/datasets/GPUMODE/KernelBook (commit history: https://huggingface.co/api/datasets/GPUMODE/KernelBook/commits/main)
- KernelLLM: https://huggingface.co/facebook/KernelLLM
- AutoTriton: https://arxiv.org/abs/2507.05687 · https://github.com/AI9Stars/AutoTriton
- TritonRL: https://arxiv.org/abs/2510.17891 · https://icml.cc/virtual/2026/82775
- Dr. Kernel: https://arxiv.org/abs/2602.05885 · https://github.com/hkust-nlp/KernelGYM · https://huggingface.co/datasets/hkust-nlp/drkernel-coldstart-8k · https://huggingface.co/datasets/hkust-nlp/drkernel-rl-data · https://huggingface.co/datasets/hkust-nlp/drkernel-validation-data
- Kevin-32B: https://cognition.com/blog/kevin-32b · https://arxiv.org/abs/2507.11948
- CUDA-L1: https://arxiv.org/abs/2507.14111
- cudaLLM-data: https://huggingface.co/datasets/ByteDance-Seed/cudaLLM-data
- siro1 traces: https://huggingface.co/datasets/siro1/kernelbook-kimi_k2_thinking-evals-filtered · https://huggingface.co/datasets/siro1/kernelbook-glm4-evals-filtered
- Code_Opt_Triton: https://huggingface.co/datasets/Teen-Different/Code_Opt_Triton
- TritonBench: https://github.com/thunlp/TritonBench · https://arxiv.org/abs/2502.14752
- DRTriton: https://arxiv.org/abs/2603.21465 · Kernel-Smith: https://arxiv.org/abs/2603.28342 · AMDKernelVault: https://arxiv.org/abs/2609.12471
- KernelBench-Verified: https://arxiv.org/abs/2607.16241
- ConCuR/KernelCoder (CUDA, not assessed in depth): https://arxiv.org/abs/2510.07356
- Modal pricing: https://modal.com/pricing
