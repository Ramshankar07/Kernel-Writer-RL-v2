# Kernel-Writer-RL-v2 — AutoKernel RLVR

Original design (SkyPilot + verl, April 2026) lives in `src/autokernel-rlvr/`. All original
results were lost with an old disk; everything measured now comes from the Modal re-run
(Sept 2026) in `modal_app/`, with artifacts in `results/` and slides in `presentation/`.

## Rules
- Work on branch `modal-replication` (or another feature branch). Never commit/push to `main`.
- Modal account cap: 10 concurrent GPUs/containers. Bench pools use `BENCH_MAX_CONTAINERS=8`
  (`modal_app/common.py`); run one GPU stage at a time.
- Every number in the deck/Q&A must come from `results/` (via `presentation/numbers.json`),
  never typed by hand.

## Modal apps (deploy once, then call)
```
cd modal_app
modal deploy bench_modal.py        # autokernel-bench: Bencher.run(kernel_type, code, quick)
modal deploy kernelbench_modal.py  # autokernel-kernelbench: KBBencher.run(problem_id, code)
modal deploy policy.py             # autokernel-policy: vLLM Qwen2.5-Coder, LoRA hot-swap
```
Volume `autokernel-rlvr` (HF cache, GRPO checkpoints/metrics), Dict `autokernel-bench-cache`.

## Stages
| Stage | Command | Output |
|---|---|---|
| 1 bench baselines | `modal run bench_modal.py::stage1` | `results/stage1_baselines.jsonl` |
| 1b KernelBench v1 harness test | `modal run kernelbench_modal.py::sanity` | `results/kb_sanity.jsonl` |
| 2 agent eval (no training) | `modal run agent_eval.py --suite autokernel --n 8 --max-turns 8` | `results/agent_eval/*.jsonl` |
| 2b KernelBench L1 agent eval | `modal run agent_eval.py --suite kb --n 1 --max-turns 4 --bench-gpu H100` | `results/agent_eval/kb_*.jsonl` |
| 3 GRPO | `modal deploy train_grpo.py`, then `modal.Function.from_name("autokernel-grpo","train").spawn(run=..., reward="v1"|"v2", adv_min_std=...)` | Volume `grpo/<run>/` → `results/grpo/<run>/` |
| 4 analysis + materials | `python modal_app/analyze.py && python presentation/make_figures.py && python presentation/make_docs.py && node presentation/build_deck.js` | `results/numbers.json`, `presentation/*` |
| 5 deck render (QA) | `modal run modal_app/render_deck.py` | `presentation/_render/slide-NN.png` |

## Findings that changed the code
- Upstream `bench.py` takes `--kernel` (not `--kernel-type`) and prints metrics to stdout; it
  writes no `results.tsv`. `bench_modal.py:parse_stdout` handles this.
- `--quick` skips numerical-stability checks: rmsnorm/flash_attention starters PASS quick but
  FAIL full. Reward uses the full bench.
- AutoKernel's KernelBench starter keeps `super(Model, self)` after renaming to ModelNew → every
  starter crashes; identity test uses `class ModelNew(Model): pass`.
- `bench_kb.py` times CPU `get_inputs()` inside a 30 s trial timeout; KB containers need
  `cpu=8, memory=32768`, and even then 9 huge-input problems time out intermittently.
- `bench_kb.py` never seeds weight init, so all 35 conv problems fail even for an identical model.
  Only 55/100 Level 1 problems are harness-valid → `results/kb_valid_problems.json` (used by agent_eval).
- The kb_cache dir holds `N.py` **and** `N.json`; list only `.py` (an early run read 1–50 twice;
  those two runs are kept as `kb_sanity_1_50_run{1,2}.jsonl` for repeatability).
- No MI300X on Modal: the AMD path (`reward_amd.py`, `worker_amd.py`) is design-only.

## Status (2026-09-23)
All stages done. GRPO v1 (original reward, 18 steps) and v2 (`modal_app/reward_v2.py`, 10 steps)
in `results/grpo/`. KernelBench before/after: 0/28 both. Total Modal spend ≈ $95 (estimate).
Use spawn (not `modal run`) for long jobs: a `modal run --detach` job got cancelled at step 5.

## Deck v2 (researcher-oriented analysis)
Extra analyses: `presentation/v2/make_extra.py` → `results/extra_numbers.json`, figures in `presentation/v2/figures/`.
Key finding: 78.9% of GRPO v1's |advantage| came from timing-noise groups (std < 0.02); Dr. GRPO 6.2%.

## SFT-before-RL (v3, `results/v3/sft_plan.md`)
- Phase 0 done 2026-09-30 (`results/v3/phase0.md`): fixed starters, prompt v2, 62-problem val set, base pass@1 0.383.
- Phase 1 done 2026-10-06 (`results/v3/sft/`): ingest → static filter → GPU verify (`modal_app/sft_verify_modal.py`,
  ≤4 L40S) → decontam (`sft_decontam.py`) → `sft_assemble.py`. Mix: **3,022 train / 60 dev**, 15% repair,
  3.6M trained tokens/epoch. Verify pass rates: Dr.Kernel 91.3%, in-repo 100%, KernelBook 69/1029 (Inductor kernels
  hard-code shapes → FAIL_SHAPE2, excluded by choice). Phase 1 GPU ≈ $8.
- Gotchas: KernelBook `get_init_inputs()` returns `[args, kwargs]` (paritybench) and imports `_paritybench_helpers`;
  verifier batches run serially per container (use small `--batch` for slow autokernel rows); zsh doesn't word-split
  `$var` in loops (use `bash -c`). `candidates_drkernel_repair.jsonl` (160 MB) is gitignored; rebuild with
  `python3 modal_app/sft_ingest_drkernel.py`.
- Phase 2 done 2026-10-07: `modal_app/sft_train_modal.py` (pilot-resumable), run `sft_v1` 378 steps, dev loss
  0.601 → 0.302, ~4.6k tok/s on 1×H100, ≈ $3.9. Adapter on Volume `sft/sft_v1/step_378`.
- Phase 3 done 2026-10-07 (`results/v3/phase3.md`, `python3 modal_app/phase3_report.py`): val pass@1 0.383 → 0.508,
  compile errors 8.1% → 1.9%, near-starter solves 92% → 20%, beat-starter flat. All 3 GRPO gates pass. KernelBench L1
  (kb v2, `modal_app/kb_v2_agent_eval.py`): Triton-launch PASS 0/100 → 20/100 (op-seen 13/41, op-unseen 7/59).
- GRPO NOT resumed (user decision, budget ~$50 left). Offline check: only 2/126 solved SFT val episodes beat the
  starter by >5%, so a speed reward would mostly re-teach correctness.
- Budget: read real spend with `modal.billing.workspace_billing_report` before any GPU job.
